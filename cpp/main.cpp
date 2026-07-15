// ---------------------------------------------------------------------------
//  main.cpp  --  nanochat-core C++ driver: train / sample a character-level
//  modern transformer (RoPE, RMSNorm, QK-norm, GQA, ReLU^2, softcap) on CPU.
//
//    nanochat train [input.txt] [--steps N --lr F --batch N --block N
//                    --layers N --embd N --heads N --kv-heads N --out FILE]
//    nanochat sample [ckpt.bin] [--tokens N --temp F --topk N --prompt STR]
//
//  Gradient check: build/run the nanochat_gradcheck target.
// ---------------------------------------------------------------------------
#include "gpt.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>
using namespace gpt;

static const char MAGIC[4] = {'N', 'C', 'p', '1'};

static bool save_ckpt(const std::string& path, GPT& m, const CharTokenizer& tok) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hdr[7] = {m.cfg.sequence_len, m.cfg.vocab_size, m.cfg.n_layer,
                      m.cfg.n_head, m.cfg.n_kv_head, m.cfg.n_embd, (int32_t)m.cfg.rope_base};
    f.write(MAGIC, 4); f.write((char*)hdr, sizeof(hdr));
    int32_t vn = (int32_t)tok.itos.size(); f.write((char*)&vn, 4);
    if (vn) f.write(tok.itos.data(), vn);
    auto wr = [&](std::vector<real>& v) {
        std::vector<float> b(v.size()); for (size_t i = 0; i < v.size(); i++) b[i] = (float)v[i];
        f.write((char*)b.data(), (std::streamsize)(b.size() * sizeof(float)));
    };
    wr(m.wte); wr(m.lm_head);
    for (auto& lp : m.layers) { wr(lp.Wq); wr(lp.Wk); wr(lp.Wv); wr(lp.Wo); wr(lp.Wfc); wr(lp.Wproj); if (lp.ve) { wr(lp.ve_emb); wr(lp.ve_gate); } }
    wr(m.resid_l); wr(m.x0_l); wr(m.smear_w);
    float sc[2] = {(float)m.smear_lambda, (float)m.backout_lambda}; f.write((char*)sc, sizeof(sc));
    return (bool)f;
}
static bool load_ckpt(const std::string& path, GPT& m, CharTokenizer& tok) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    char mg[4]; int32_t hdr[7]; f.read(mg, 4); f.read((char*)hdr, sizeof(hdr));
    if (std::memcmp(mg, MAGIC, 4) != 0) { std::fprintf(stderr, "bad ckpt\n"); return false; }
    GPTConfig c; c.sequence_len = hdr[0]; c.vocab_size = hdr[1]; c.n_layer = hdr[2];
    c.n_head = hdr[3]; c.n_kv_head = hdr[4]; c.n_embd = hdr[5]; c.rope_base = hdr[6];
    int32_t vn; f.read((char*)&vn, 4); std::vector<char> ch(vn); if (vn) f.read(ch.data(), vn);
    tok.set_vocab(ch);
    m.build(c);
    auto rd = [&](std::vector<real>& v) {
        std::vector<float> b(v.size()); f.read((char*)b.data(), (std::streamsize)(b.size() * sizeof(float)));
        for (size_t i = 0; i < v.size(); i++) v[i] = (real)b[i];
    };
    rd(m.wte); rd(m.lm_head);
    for (auto& lp : m.layers) { rd(lp.Wq); rd(lp.Wk); rd(lp.Wv); rd(lp.Wo); rd(lp.Wfc); rd(lp.Wproj); if (lp.ve) { rd(lp.ve_emb); rd(lp.ve_gate); } }
    rd(m.resid_l); rd(m.x0_l); rd(m.smear_w);
    float sc[2]; f.read((char*)sc, sizeof(sc)); m.smear_lambda = (real)sc[0]; m.backout_lambda = (real)sc[1];
    return (bool)f;
}

static std::vector<int> generate(GPT& m, std::vector<int> idx, int n, real temp, int topk, std::mt19937& rng) {
    int V = m.V, bs = m.cfg.sequence_len;
    std::uniform_real_distribution<double> uni(0, 1);
    for (int s = 0; s < n; s++) {
        int t = (int)idx.size(), start = std::max(0, t - bs), tc = t - start;
        std::vector<int> cond(idx.begin() + start, idx.end());
        m.forward(cond.data(), nullptr, 1, tc);
        std::vector<real> l(m.logits.begin() + (size_t)(tc - 1) * V, m.logits.begin() + (size_t)tc * V);
        for (auto& x : l) x /= (temp > 0 ? temp : (real)1);
        if (topk > 0 && topk < V) {
            std::vector<real> s2(l); std::nth_element(s2.begin(), s2.end() - topk, s2.end());
            real th = s2[s2.size() - topk]; for (auto& x : l) if (x < th) x = (real)-1e30;
        }
        real mx = *std::max_element(l.begin(), l.end());
        double sum = 0; for (auto& x : l) { x = (real)std::exp((double)(x - mx)); sum += x; }
        double r = uni(rng) * sum, acc = 0; int nx = V - 1;
        for (int i = 0; i < V; i++) { acc += l[i]; if (acc >= r) { nx = i; break; } }
        idx.push_back(nx);
    }
    return idx;
}

static double est_loss(GPT& m, const std::vector<int>& d, int B, int T, int iters, std::mt19937& rng) {
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

    std::ifstream f(input, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", input.c_str()); return 1; }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::printf("dataset: %zu chars\n", text.size());
    CharTokenizer tok; tok.build_from_text(text);
    std::printf("vocab: %d\n", tok.vocab_size());
    std::vector<int> data = tok.encode(text);
    size_t n = data.size();
    std::vector<int> tr(data.begin(), data.begin() + (size_t)(n*0.9)), va(data.begin()+(size_t)(n*0.9), data.end());

    GPTConfig c; c.sequence_len = T; c.vocab_size = tok.vocab_size();
    c.n_layer = nl; c.n_head = nh; c.n_kv_head = nkv; c.n_embd = ne;
    GPT m; m.build(c); m.init(seed);
    std::printf("nanochat-core: %d layers, %d embd, %d/%d heads (GQA), block %d  (%.2fM params)\n",
                nl, ne, nh, nkv, T, m.num_params()/1e6);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, (int)tr.size() - T - 1);
    std::vector<int> inp(B*T), tgt(B*T);
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step <= steps; step++) {
        if (step % ev == 0 || step == steps) {
            std::mt19937 er(999);
            double tl = est_loss(m, tr, B, T, 20, er), vl = est_loss(m, va, B, T, 20, er);
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("step %5d | train %.4f | val %.4f | %.1fs\n", step, tl, vl, sec);
            std::fflush(stdout);
        }
        if (step == steps) break;
        for (int b = 0; b < B; b++) { int o = pick(rng); for (int t = 0; t < T; t++) { inp[b*T+t]=tr[o+t]; tgt[b*T+t]=tr[o+t+1]; } }
        m.forward(inp.data(), tgt.data(), B, T);
        m.zero_grad(); m.backward();
        // warmup + cosine LR schedule (multiplier on nanochat's per-group base LRs)
        int warm = std::max(1, steps / 50);
        double mult = step < warm ? (double)step / warm
            : 0.1 + 0.9 * 0.5 * (1 + std::cos(3.14159265358979 * (double)(step - warm) / std::max(1, steps - warm)));
        m.optimize((real)(lr * mult));
    }
    if (save_ckpt(out, m, tok)) std::printf("saved %s\n", out.c_str());
    std::mt19937 gr(seed + 1);
    std::vector<int> ctx = {tok.stoi.count('\n') ? tok.stoi.at('\n') : 0};
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
    GPT m; CharTokenizer tok;
    if (!load_ckpt(ckpt, m, tok)) return 1;
    std::printf("loaded %s: %d layers, %d embd, %d/%d heads, vocab %d\n",
                ckpt.c_str(), m.cfg.n_layer, m.cfg.n_embd, m.cfg.n_head, m.cfg.n_kv_head, m.V);
    std::vector<int> ctx = prompt.empty() ? std::vector<int>{} : tok.encode(prompt);
    if (ctx.empty()) ctx.push_back(tok.stoi.count('\n') ? tok.stoi.at('\n') : 0);
    std::mt19937 rng(seed);
    std::printf("%s\n", tok.decode(generate(m, ctx, toks, (real)temp, topk, rng)).c_str());
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "train") return cmd_train(argc, argv);
    if (mode == "sample") return cmd_sample(argc, argv);
    std::fprintf(stderr,
        "usage:\n"
        "  %s train [input.txt] [--steps N --lr F --batch N --block N --layers N --embd N --heads N --kv-heads N --out FILE]\n"
        "  %s sample [ckpt.bin] [--tokens N --temp F --topk N --prompt STR]\n"
        "\n(gradient check: build/run the nanochat_gradcheck target)\n", argv[0], argv[0]);
    return 2;
}
