// ---------------------------------------------------------------------------
//  main.cpp  --  nanochat-core C++ driver: train / sample a character-level
//  modern transformer (RoPE, RMSNorm, QK-norm, GQA, ReLU^2, softcap) on CPU.
//
//    nanochat train [input.txt] [--steps N --lr F --batch N --block N
//                    --layers N --embd N --heads N --kv-heads N --out FILE
//                    --init scratch|resume|finetune --ckpt FILE
//                    --grad-clip F --grad-accum N --tokenizer char|bpe --vocab N]
//    nanochat sample [ckpt.bin] [--tokens N --temp F --topk N --prompt STR]
//      resume:   continue training (params + Muon/AdamW state + step)
//      finetune: keep weights, fresh optimiser + step, train on new data
//
//  Gradient check: build/run the nanochat_gradcheck target.
// ---------------------------------------------------------------------------
#include "gpt.h"
#include "tokenizer.h"
#include "bpe.h"
#include "tool.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
using namespace gpt;

// Tokenizer wrapper: char-level (default) or self-trained byte-level BPE, with a
// uniform encode/decode/vocab_size interface used throughout the driver.
struct Tok {
    int kind = 0;            // 0 = char, 1 = bpe
    CharTokenizer ch;
    BPE bpe;
    int vocab_size() const { return kind ? bpe.n_vocab() : ch.vocab_size(); }
    std::vector<int> encode(const std::string& s) const { return kind ? bpe.encode(s) : ch.encode(s); }
    std::string decode(const std::vector<int>& v) const { return kind ? bpe.decode(v) : ch.decode(v); }
    int seed_token() const {  // a sensible context seed for empty prompts
        if (kind) return bpe.bos_id;
        return ch.stoi.count('\n') ? ch.stoi.at('\n') : 0;
    }
};

//  NCp1 = params only.  NCp2 = params, then AdamW/Muon optimizer state + step,
//  so training can resume with momentum intact. NCp1 files still load (sample).
static const char MAGIC1[4] = {'N', 'C', 'p', '1'};
static const char MAGIC2[4] = {'N', 'C', 'p', '2'};

static bool save_ckpt(const std::string& path, GPT& m, const Tok& tok, int32_t iter = 0) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hdr[7] = {m.cfg.sequence_len, m.cfg.vocab_size, m.cfg.n_layer,
                      m.cfg.n_head, m.cfg.n_kv_head, m.cfg.n_embd, (int32_t)m.cfg.rope_base};
    f.write(MAGIC2, 4); f.write((char*)hdr, sizeof(hdr));
    // tokenizer: kind (0 char, 1 bpe) then its vocab data
    int32_t kind = tok.kind; f.write((char*)&kind, 4);
    if (kind == 0) {
        int32_t vn = (int32_t)tok.ch.itos.size(); f.write((char*)&vn, 4);
        if (vn) f.write(tok.ch.itos.data(), vn);
    } else {
        int32_t nv = (int32_t)tok.bpe.vocab.size(); f.write((char*)&nv, 4);
        for (auto& v : tok.bpe.vocab) { int32_t l = (int32_t)v.size(); f.write((char*)&l, 4); f.write(v.data(), l); }
    }
    auto wr = [&](std::vector<real>& v) {
        std::vector<float> b(v.size()); for (size_t i = 0; i < v.size(); i++) b[i] = (float)v[i];
        f.write((char*)b.data(), (std::streamsize)(b.size() * sizeof(float)));
    };
    wr(m.wte); wr(m.lm_head);
    for (auto& lp : m.layers) { wr(lp.Wq); wr(lp.Wk); wr(lp.Wv); wr(lp.Wo); wr(lp.Wfc); wr(lp.Wproj); if (lp.ve) { wr(lp.ve_emb); wr(lp.ve_gate); } }
    wr(m.resid_l); wr(m.x0_l); wr(m.smear_w);
    float sc[2] = {(float)m.smear_lambda, (float)m.backout_lambda}; f.write((char*)sc, sizeof(sc));
    // NCp2: optimizer state (step, scalar moments, then all state vectors)
    int32_t adam_t = m.adam_t; f.write((char*)&adam_t, 4); f.write((char*)&iter, 4);
    float sm[4] = {(float)m.m_sl, (float)m.v_sl, (float)m.m_bo, (float)m.v_bo};
    f.write((char*)sm, sizeof(sm));
    for (auto* v : m.opt_state_ptrs()) wr(*v);
    return (bool)f;
}
// Loads params (NCp1/NCp2). If out_iter != nullptr and the file is NCp2, also
// restores optimizer momentum and returns the saved step via *out_iter.
static bool load_ckpt(const std::string& path, GPT& m, Tok& tok, int32_t* out_iter = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    char mg[4]; int32_t hdr[7]; f.read(mg, 4); f.read((char*)hdr, sizeof(hdr));
    bool v2 = std::memcmp(mg, MAGIC2, 4) == 0;
    if (!v2 && std::memcmp(mg, MAGIC1, 4) != 0) { std::fprintf(stderr, "bad ckpt\n"); return false; }
    GPTConfig c; c.sequence_len = hdr[0]; c.vocab_size = hdr[1]; c.n_layer = hdr[2];
    c.n_head = hdr[3]; c.n_kv_head = hdr[4]; c.n_embd = hdr[5]; c.rope_base = hdr[6];
    int32_t kind = 0; f.read((char*)&kind, 4); tok.kind = kind;
    if (kind == 0) {
        int32_t vn; f.read((char*)&vn, 4); std::vector<char> ch(vn); if (vn) f.read(ch.data(), vn);
        tok.ch.set_vocab(ch);
    } else {
        int32_t nv = 0; f.read((char*)&nv, 4);
        tok.bpe.vocab.assign(nv, ""); tok.bpe.ranks.clear();
        for (int i = 0; i < nv; i++) { int32_t l = 0; f.read((char*)&l, 4); std::string s(l, '\0'); if (l) f.read(&s[0], l); tok.bpe.vocab[i] = s; tok.bpe.ranks[s] = i; }
        tok.bpe.special.clear();
        for (int s = 0; s < NANOCHAT_NUM_SPECIAL; s++) tok.bpe.special[NANOCHAT_SPECIALS[s]] = nv + s;
        tok.bpe.bos_id = tok.bpe.special["<|bos|>"];
    }
    m.build(c);
    auto rd = [&](std::vector<real>& v) {
        std::vector<float> b(v.size()); f.read((char*)b.data(), (std::streamsize)(b.size() * sizeof(float)));
        for (size_t i = 0; i < v.size(); i++) v[i] = (real)b[i];
    };
    rd(m.wte); rd(m.lm_head);
    for (auto& lp : m.layers) { rd(lp.Wq); rd(lp.Wk); rd(lp.Wv); rd(lp.Wo); rd(lp.Wfc); rd(lp.Wproj); if (lp.ve) { rd(lp.ve_emb); rd(lp.ve_gate); } }
    rd(m.resid_l); rd(m.x0_l); rd(m.smear_w);
    float sc[2]; f.read((char*)sc, sizeof(sc)); m.smear_lambda = (real)sc[0]; m.backout_lambda = (real)sc[1];
    if (out_iter) *out_iter = 0;
    if (v2) {
        int32_t adam_t = 0, iter = 0; f.read((char*)&adam_t, 4); f.read((char*)&iter, 4);
        float sm[4]; f.read((char*)sm, sizeof(sm));
        m.m_sl = (real)sm[0]; m.v_sl = (real)sm[1]; m.m_bo = (real)sm[2]; m.v_bo = (real)sm[3];
        for (auto* v : m.opt_state_ptrs()) rd(*v);
        if (f) { m.adam_t = adam_t; if (out_iter) *out_iter = iter; }
    }
    return (bool)f;
}

// sample one token id from raw logits with temperature + top-k (in-place scratch).
static int sample_token(std::vector<real>& l, real temp, int topk, int V, std::mt19937& rng) {
    std::uniform_real_distribution<double> uni(0, 1);
    for (auto& x : l) x /= (temp > 0 ? temp : (real)1);
    if (topk > 0 && topk < V) {
        std::vector<real> s2(l); std::nth_element(s2.begin(), s2.end() - topk, s2.end());
        real th = s2[s2.size() - topk]; for (auto& x : l) if (x < th) x = (real)-1e30;
    }
    real mx = *std::max_element(l.begin(), l.end());
    double sum = 0; for (auto& x : l) { x = (real)std::exp((double)(x - mx)); sum += x; }
    double r = uni(rng) * sum, acc = 0; int nx = V - 1;
    for (int i = 0; i < V; i++) { acc += l[i]; if (acc >= r) { nx = i; break; } }
    return nx;
}

// Generation with a KV cache within the context window (one O(1)-context step
// per token); falls back to the recompute-with-sliding path beyond it.
static std::vector<int> generate(GPT& m, std::vector<int> idx, int n, real temp, int topk, std::mt19937& rng) {
    int V = m.V, bs = m.cfg.sequence_len;
    int produced = 0;
    if ((int)idx.size() <= bs) {
        KVCache kv; kv.init(m.L, m.C, m.Ckv, m.cfg.sequence_len);
        std::vector<real> logits;
        for (size_t i = 0; i < idx.size() && kv.pos < bs; i++) m.forward_one(idx[i], kv, logits);
        while (produced < n && kv.pos < bs) {
            int nx = sample_token(logits, temp, topk, V, rng);
            idx.push_back(nx); produced++;
            if (kv.pos < bs && produced < n) m.forward_one(nx, kv, logits);
        }
    }
    for (; produced < n; produced++) {   // fallback beyond the context window
        int t = (int)idx.size(), start = std::max(0, t - bs), tc = t - start;
        std::vector<int> cond(idx.begin() + start, idx.end());
        m.forward(cond.data(), nullptr, 1, tc);
        std::vector<real> l(m.logits.begin() + (size_t)(tc - 1) * V, m.logits.begin() + (size_t)tc * V);
        idx.push_back(sample_token(l, temp, topk, V, rng));
    }
    return idx;
}

static double est_loss(GPT& m, const std::vector<int>& d, int B, int T, int iters, std::mt19937& rng) {
    if ((int)d.size() < T + 2) return 0;   // split too small for one window (avoids OOB)
    std::uniform_int_distribution<int> pick(0, (int)d.size() - T - 1);
    std::vector<int> inp(B * T), tgt(B * T); double s = 0;
    for (int it = 0; it < iters; it++) {
        for (int b = 0; b < B; b++) { int o = pick(rng); for (int t = 0; t < T; t++) { inp[b*T+t]=d[o+t]; tgt[b*T+t]=d[o+t+1]; } }
        m.forward(inp.data(), tgt.data(), B, T); s += m.mean_loss;
    }
    return s / iters;
}

static int argi(int c, char** v, const char* n, int d) { for (int i=1;i<c-1;i++) if(!strcmp(v[i],n)) return atoi(v[i+1]); return d; }
static double argf(int c, char** v, const char* n, double d) { for (int i=1;i<c-1;i++) if(!strcmp(v[i],n)) return atof(v[i+1]); return d; }
static std::string args(int c, char** v, const char* n, const std::string& d) { for (int i=1;i<c-1;i++) if(!strcmp(v[i],n)) return v[i+1]; return d; }

static bool argflag(int c, char** v, const char* n) { for (int i=1;i<c;i++) if(!strcmp(v[i],n)) return true; return false; }

static int cmd_train(int argc, char** argv) {
    std::string input = (argc > 2 && argv[2][0] != '-') ? argv[2] : "input.txt";
    int steps = argi(argc, argv, "--steps", 2000);
    int B = argi(argc, argv, "--batch", 32), T = argi(argc, argv, "--block", 128);
    int nl = argi(argc, argv, "--layers", 4), ne = argi(argc, argv, "--embd", 256);
    int nh = argi(argc, argv, "--heads", 4), nkv = argi(argc, argv, "--kv-heads", 2);
    double lr = argf(argc, argv, "--lr", 1.0);   // global multiplier on nanochat's per-group LRs
    int ev = argi(argc, argv, "--eval-every", 250);
    uint32_t seed = (uint32_t)argi(argc, argv, "--seed", 1337);
    std::string out = args(argc, argv, "--out", "ckpt.bin");
    double grad_clip = argf(argc, argv, "--grad-clip", 0.0);   // 0 = disabled (nanochat has no default clip)
    int accum = argi(argc, argv, "--grad-accum", 1); if (accum < 1) accum = 1;
    std::string init = args(argc, argv, "--init", "scratch");  // scratch | resume | finetune
    std::string ckpt = args(argc, argv, "--ckpt", "");
    std::string tkkind = args(argc, argv, "--tokenizer", "char");  // char | bpe
    int bpe_vocab = argi(argc, argv, "--vocab", 2048);             // target BPE vocab (bpe only)
    bool is_resume = (init == "resume"), is_finetune = (init == "finetune");
    if ((is_resume || is_finetune) && ckpt.empty()) { std::fprintf(stderr, "--init %s requires --ckpt FILE\n", init.c_str()); return 2; }
    if (init != "scratch" && !is_resume && !is_finetune) { std::fprintf(stderr, "--init must be scratch, resume or finetune\n"); return 2; }

    std::ifstream f(input, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", input.c_str()); return 1; }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::printf("dataset: %zu chars\n", text.size());

    GPT m; Tok tok; int32_t start_iter = 0;
    if (is_resume || is_finetune) {
        if (!load_ckpt(ckpt, m, tok, &start_iter)) return 1;
        T = m.cfg.sequence_len;   // model's context length is fixed by the checkpoint
        std::printf("%s from %s: %s vocab %d, %.2fM params, saved step %d\n",
                    init.c_str(), ckpt.c_str(), tok.kind ? "bpe" : "char", tok.vocab_size(), m.num_params()/1e6, start_iter);
        if (is_finetune) {   // keep weights, start a fresh optimiser + step counter
            for (auto* v : m.opt_state_ptrs()) std::fill(v->begin(), v->end(), (real)0);
            m.m_sl = m.v_sl = m.m_bo = m.v_bo = 0; m.adam_t = 0; start_iter = 0;
        }
    } else {
        if (tkkind == "bpe") {
            std::printf("training BPE tokenizer (target vocab %d)...\n", bpe_vocab);
            tok.kind = 1; tok.bpe.train(text, bpe_vocab);
            std::printf("  -> %zu BPE tokens + %d special\n", tok.bpe.vocab.size(), NANOCHAT_NUM_SPECIAL);
        } else {
            tok.kind = 0; tok.ch.build_from_text(text);
        }
        GPTConfig c; c.sequence_len = T; c.vocab_size = tok.vocab_size();
        c.n_layer = nl; c.n_head = nh; c.n_kv_head = nkv; c.n_embd = ne;
        m.build(c); m.init(seed);
        std::printf("nanochat-core: %d layers, %d embd, %d/%d heads (GQA), block %d, %s vocab %d  (%.2fM params)\n",
                    nl, ne, nh, nkv, T, tok.kind ? "bpe" : "char", tok.vocab_size(), m.num_params()/1e6);
    }
    std::printf("vocab: %d\n", tok.vocab_size());
    std::vector<int> data = tok.encode(text);
    std::printf("encoded %zu tokens\n", data.size());
    size_t n = data.size();
    std::vector<int> tr(data.begin(), data.begin() + (size_t)(n*0.9)), va(data.begin()+(size_t)(n*0.9), data.end());
    if ((int)tr.size() < T + 2) {   // the 90% train split must hold at least one window
        std::fprintf(stderr, "dataset too small: %zu train tokens < block %d+2 "
                     "(use a bigger corpus or a smaller --block)\n", tr.size(), T);
        return 1;
    }
    if (accum > 1) std::printf("grad accumulation: %d micro-batches -> effective batch %d\n", accum, B * accum);
    if (grad_clip > 0) std::printf("grad clip: %.2g\n", grad_clip);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, (int)tr.size() - T - 1);
    std::vector<int> inp(B*T), tgt(B*T);
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step <= steps; step++) {
        if (step % ev == 0 || step == steps) {
            std::mt19937 er(999);
            double tl = est_loss(m, tr, B, T, 20, er), vl = est_loss(m, va, B, T, 20, er);
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("step %5d | train %.4f | val %.4f | %.1fs\n", start_iter + step, tl, vl, sec);
            std::fflush(stdout);
        }
        if (step == steps) break;
        // gradient accumulation: sum grads over `accum` micro-batches (backward
        // accumulates into the param grads; activation grads are backward-local).
        m.zero_grad();
        for (int micro = 0; micro < accum; micro++) {
            for (int b = 0; b < B; b++) { int o = pick(rng); for (int t = 0; t < T; t++) { inp[b*T+t]=tr[o+t]; tgt[b*T+t]=tr[o+t+1]; } }
            m.forward(inp.data(), tgt.data(), B, T);
            m.backward();
        }
        if (accum > 1) m.scale_grads((real)(1.0 / accum));
        if (grad_clip > 0) {
            real gnorm = m.grad_global_norm();
            if (gnorm > grad_clip) m.scale_grads((real)(grad_clip / (gnorm + 1e-6)));
        }
        // warmup + cosine LR schedule (multiplier on nanochat's per-group base LRs)
        int gstep = start_iter + step, gtotal = start_iter + steps;
        int warm = std::max(1, gtotal / 50);
        double mult = gstep < warm ? (double)gstep / warm
            : 0.1 + 0.9 * 0.5 * (1 + std::cos(3.14159265358979 * (double)(gstep - warm) / std::max(1, gtotal - warm)));
        m.optimize((real)(lr * mult));
    }
    if (save_ckpt(out, m, tok, start_iter + steps)) std::printf("saved %s (step %d)\n", out.c_str(), start_iter + steps);
    std::mt19937 gr(seed + 1);
    std::vector<int> ctx = {tok.seed_token()};
    std::printf("\n--- sample ---\n%s\n", tok.decode(generate(m, ctx, 300, (real)0.8, 40, gr)).c_str());
    return 0;
}

static int cmd_sample(int argc, char** argv) {
    std::string ckpt = (argc > 2 && argv[2][0] != '-') ? argv[2] : "ckpt.bin";
    int toks = argi(argc, argv, "--tokens", 500);
    double temp = argf(argc, argv, "--temp", 0.8);
    int topk = argi(argc, argv, "--topk", 40);
    uint32_t seed = (uint32_t)argi(argc, argv, "--seed", 1337);
    std::string prompt = args(argc, argv, "--prompt", "");
    GPT m; Tok tok;
    if (!load_ckpt(ckpt, m, tok)) return 1;
    std::printf("loaded %s: %d layers, %d embd, %d/%d heads, %s vocab %d\n",
                ckpt.c_str(), m.cfg.n_layer, m.cfg.n_embd, m.cfg.n_head, m.cfg.n_kv_head, tok.kind ? "bpe" : "char", m.V);
    std::vector<int> ctx = prompt.empty() ? std::vector<int>{} : tok.encode(prompt);
    if (ctx.empty()) ctx.push_back(tok.seed_token());
    std::mt19937 rng(seed);
    std::printf("%s\n", tok.decode(generate(m, ctx, toks, (real)temp, topk, rng)).c_str());
    return 0;
}

// ===========================================================================
//  Chat: template rendering, SFT (assistant-only loss), and an interactive CLI
//  with the calculator tool. Needs a BPE tokenizer (the special tokens live
//  there). The chat format is:
//     <|bos|> <|user_start|>..<|user_end|> <|assistant_start|>..<|assistant_end|> ...
//  SFT computes loss only on assistant content + <|assistant_end|>.
// ===========================================================================
// encode text, but recognise literal special-token markers ("<|python_start|>",
// etc.) and emit their ids instead of BPE-encoding them as characters.
static std::vector<int> encode_with_specials(const BPE& b, const std::string& text) {
    std::vector<int> out; size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '<' && i + 1 < text.size() && text[i + 1] == '|') {
            size_t e = text.find("|>", i);
            if (e != std::string::npos) {
                std::string tok = text.substr(i, e + 2 - i);
                int sid = b.special_id(tok);
                if (sid >= 0) { out.push_back(sid); i = e + 2; continue; }
            }
        }
        size_t next = text.find("<|", i + 1);
        std::string chunk = text.substr(i, next == std::string::npos ? std::string::npos : next - i);
        for (int id : b.encode(chunk)) out.push_back(id);
        i = (next == std::string::npos) ? text.size() : next;
    }
    return out;
}

static void render_turn(const BPE& b, std::vector<int>& ids, std::vector<int>& mask,
                        const std::string& text, bool assistant) {
    ids.push_back(b.special_id(assistant ? "<|assistant_start|>" : "<|user_start|>")); mask.push_back(0);
    for (int id : encode_with_specials(b, text)) { ids.push_back(id); mask.push_back(assistant ? 1 : 0); }
    ids.push_back(b.special_id(assistant ? "<|assistant_end|>" : "<|user_end|>")); mask.push_back(assistant ? 1 : 0);
}

// Parse a simple chat corpus: lines "U: ..." / "A: ..."; a blank line ends a
// conversation. Render everything into one (ids, mask) stream (each convo led
// by <|bos|>).
static bool load_chat(const std::string& path, const BPE& b, std::vector<int>& ids, std::vector<int>& mask) {
    std::ifstream f(path); if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::string line; bool started = false;
    auto bos = b.special_id("<|bos|>");
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        bool blank = line.find_first_not_of(" \t") == std::string::npos;
        if (blank) { started = false; continue; }
        bool asst = (line.size() >= 2 && (line[0] == 'A' || line[0] == 'a') && line[1] == ':');
        bool user = (line.size() >= 2 && (line[0] == 'U' || line[0] == 'u') && line[1] == ':');
        if (!asst && !user) continue;
        std::string text = line.substr(2);
        size_t s = text.find_first_not_of(" \t"); if (s != std::string::npos) text = text.substr(s);
        if (!started) { ids.push_back(bos); mask.push_back(0); started = true; }
        render_turn(b, ids, mask, text, asst);
    }
    return !ids.empty();
}

static int cmd_sft(int argc, char** argv) {
    std::string input = (argc > 2 && argv[2][0] != '-') ? argv[2] : "chat.txt";
    std::string ckpt = args(argc, argv, "--ckpt", "");
    if (ckpt.empty()) { std::fprintf(stderr, "sft needs a base model: --ckpt FILE (a bpe checkpoint)\n"); return 2; }
    std::string initm = args(argc, argv, "--init", "finetune");   // finetune | resume
    int steps = argi(argc, argv, "--steps", 400);
    int B = argi(argc, argv, "--batch", 16);
    double lr = argf(argc, argv, "--lr", 1.0);
    int ev = argi(argc, argv, "--eval-every", 50);
    uint32_t seed = (uint32_t)argi(argc, argv, "--seed", 1337);
    std::string out = args(argc, argv, "--out", "sft.bin");
    double grad_clip = argf(argc, argv, "--grad-clip", 0.0);
    int accum = argi(argc, argv, "--grad-accum", 1); if (accum < 1) accum = 1;

    GPT m; Tok tok; int32_t start_iter = 0;
    if (!load_ckpt(ckpt, m, tok, &start_iter)) return 1;
    if (tok.kind != 1) { std::fprintf(stderr, "sft requires a BPE model (train with --tokenizer bpe)\n"); return 1; }
    if (initm == "finetune") {
        for (auto* v : m.opt_state_ptrs()) std::fill(v->begin(), v->end(), (real)0);
        m.m_sl = m.v_sl = m.m_bo = m.v_bo = 0; m.adam_t = 0; start_iter = 0;
    }
    int T = m.cfg.sequence_len, V = m.V;
    std::vector<int> ids, mask;
    if (!load_chat(input, tok.bpe, ids, mask)) { std::fprintf(stderr, "no chat data in %s (use 'U:'/'A:' lines)\n", input.c_str()); return 1; }
    std::printf("sft %s: %zu tokens (%d assistant), model %.2fM, block %d\n",
                initm.c_str(), ids.size(), (int)std::count(mask.begin(), mask.end(), 1), m.num_params()/1e6, T);
    if ((int)ids.size() < T + 2) { std::fprintf(stderr, "chat data too small for block %d\n", T); return 1; }
    if (accum > 1) std::printf("grad accumulation: eff batch %d\n", B * accum);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, (int)ids.size() - T - 1);
    std::vector<int> inp(B*T), tgt(B*T);
    auto fill = [&]() {
        for (int b = 0; b < B; b++) { int o = pick(rng);
            for (int t = 0; t < T; t++) { inp[b*T+t] = ids[o+t];
                tgt[b*T+t] = mask[o+t+1] ? ids[o+t+1] : -1;   // assistant-only loss
            } }
    };
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step <= steps; step++) {
        if (step % ev == 0 || step == steps) {
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::mt19937 er(step); double s = 0; int it = 5;
            for (int k = 0; k < it; k++) { fill(); m.forward(inp.data(), tgt.data(), B, T); s += m.mean_loss; }
            std::printf("step %5d | sft loss %.4f | %.1fs\n", step, s/it, sec); std::fflush(stdout);
        }
        if (step == steps) break;
        m.zero_grad();
        for (int micro = 0; micro < accum; micro++) { fill(); m.forward(inp.data(), tgt.data(), B, T); m.backward(); }
        if (accum > 1) m.scale_grads((real)(1.0/accum));
        if (grad_clip > 0) { real g = m.grad_global_norm(); if (g > grad_clip) m.scale_grads((real)(grad_clip/(g+1e-6))); }
        int warm = std::max(1, steps/50);
        double mult = step < warm ? (double)step/warm
            : 0.1 + 0.9*0.5*(1 + std::cos(3.14159265358979*(double)(step-warm)/std::max(1, steps-warm)));
        m.optimize((real)(lr * mult));
    }
    if (save_ckpt(out, m, tok, 0)) std::printf("saved %s\n", out.c_str());
    return 0;
}

// generate an assistant reply into the KV cache, streaming decoded text; runs the
// calculator tool when the model emits <|python_start|>expr<|python_end|>.
static int cmd_chat(int argc, char** argv) {
    std::string ckpt = (argc > 2 && argv[2][0] != '-') ? argv[2] : "sft.bin";
    double temp = argf(argc, argv, "--temp", 0.7);
    int topk = argi(argc, argv, "--topk", 40);
    int maxtok = argi(argc, argv, "--max-tokens", 256);
    uint32_t seed = (uint32_t)argi(argc, argv, "--seed", 1337);
    std::string oneshot = args(argc, argv, "--message", "");
    GPT m; Tok tok;
    if (!load_ckpt(ckpt, m, tok)) return 1;
    if (tok.kind != 1) { std::fprintf(stderr, "chat requires a BPE model\n"); return 1; }
    BPE& b = tok.bpe;
    int A_end = b.special_id("<|assistant_end|>"), py0 = b.special_id("<|python_start|>");
    int py1 = b.special_id("<|python_end|>"), o0 = b.special_id("<|output_start|>"), o1 = b.special_id("<|output_end|>");
    std::mt19937 rng(seed);
    std::vector<int> convo = { b.special_id("<|bos|>") };
    std::printf("nanochat (%s). %s\n", tok.kind ? "bpe" : "char",
                oneshot.empty() ? "Type a message (Ctrl-C to quit)." : "");
    std::string line; bool interactive = oneshot.empty();
    for (;;) {
        std::string user;
        if (interactive) { std::printf("\nyou> "); std::fflush(stdout); if (!std::getline(std::cin, line)) break; user = line; }
        else user = oneshot;
        // build context: convo so far + this user turn + <|assistant_start|>
        std::vector<int> ctx = convo, mask;
        render_turn(b, ctx, mask, user, false);
        ctx.push_back(b.special_id("<|assistant_start|>"));
        int bs = m.cfg.sequence_len;
        if ((int)ctx.size() > bs) { std::fprintf(stderr, "\n[context full]\n"); break; }
        KVCache kv; kv.init(m.L, m.C, m.Ckv, m.cfg.sequence_len);
        std::vector<real> logits;
        for (int id : ctx) m.forward_one(id, kv, logits);
        std::vector<int> reply;
        std::printf("bot> "); std::fflush(stdout);
        bool in_tool = false; std::vector<int> tool_expr;
        for (int step = 0; step < maxtok && kv.pos < bs; step++) {
            int nx = sample_token(logits, (real)temp, topk, m.V, rng);
            if (nx == A_end) break;
            if (nx == py0) { in_tool = true; tool_expr.clear(); m.forward_one(nx, kv, logits); reply.push_back(nx); continue; }
            if (nx == py1 && in_tool) {   // run the calculator, feed the output back
                in_tool = false; reply.push_back(nx);
                std::string expr = b.decode(tool_expr), res;
                bool ok = eval_calculator(expr, res);
                std::printf("[calc %s = %s]", expr.c_str(), ok ? res.c_str() : "?"); std::fflush(stdout);
                std::vector<int> inj = { o0 }; for (int id : b.encode(ok ? res : "?")) inj.push_back(id); inj.push_back(o1);
                m.forward_one(py1, kv, logits);
                for (int id : inj) { if (kv.pos >= bs) break; reply.push_back(id); m.forward_one(id, kv, logits); }
                continue;
            }
            reply.push_back(nx);
            if (in_tool) tool_expr.push_back(nx);
            else { std::string piece = b.decode({nx}); std::fputs(piece.c_str(), stdout); std::fflush(stdout); }
            if (kv.pos < bs && step + 1 < maxtok) m.forward_one(nx, kv, logits);
        }
        std::printf("\n");
        // append the assistant turn (incl. markers) to the running conversation
        convo.insert(convo.end(), ctx.begin() + convo.size(), ctx.end());
        convo.insert(convo.end(), reply.begin(), reply.end());
        convo.push_back(A_end);
        if (!interactive) break;
    }
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "train") return cmd_train(argc, argv);
    if (mode == "sample") return cmd_sample(argc, argv);
    if (mode == "sft") return cmd_sft(argc, argv);
    if (mode == "chat") return cmd_chat(argc, argv);
    std::fprintf(stderr,
        "usage:\n"
        "  %s train [input.txt] [--steps N --lr F --batch N --block N --layers N --embd N --heads N --kv-heads N --out FILE\n"
        "                        --init scratch|resume|finetune --ckpt FILE --grad-clip F --grad-accum N\n"
        "                        --tokenizer char|bpe --vocab N]\n"
        "  %s sample [ckpt.bin] [--tokens N --temp F --topk N --prompt STR]\n"
        "  %s sft chat.txt --ckpt base.bin [--init finetune|resume --steps N --lr F --batch N --out FILE]\n"
        "  %s chat [sft.bin] [--message STR --temp F --topk N --max-tokens N]   (BPE model; calculator tool)\n"
        "\n(gradient check: build/run the nanochat_gradcheck target)\n", argv[0], argv[0], argv[0], argv[0]);
    return 2;
}
