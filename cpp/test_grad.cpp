// Gradient check for the extended nanochat model (resid/x0, smear, backout,
// value embeddings). Build with -DGPT_USE_DOUBLE.
#include "gpt.h"
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
using namespace gpt;

int main() {
    GPTConfig c;
    c.sequence_len = 16; c.vocab_size = 13; c.n_layer = 3;   // 3 layers -> ve on layers 0,2
    c.n_head = 4; c.n_kv_head = 2; c.n_embd = 16;
    c.force_window = 3;   // exercise the sliding-window attention path (window < T)
    int B = 2, T = 6;

    GPT m; m.build(c); m.init_test(7);
    std::mt19937 rng(1);
    std::vector<int> inp(B * T), tgt(B * T);
    for (auto& v : inp) v = std::uniform_int_distribution<int>(0, c.vocab_size - 1)(rng);
    for (auto& v : tgt) v = std::uniform_int_distribution<int>(0, c.vocab_size - 1)(rng);

    m.forward(inp.data(), tgt.data(), B, T);
    m.zero_grad(); m.backward();

    struct Grp { const char* n; real* p; real* g; size_t sz; };
    std::vector<Grp> gs = {
        {"wte", m.wte.data(), m.dwte.data(), m.wte.size()},
        {"lm_head", m.lm_head.data(), m.dlm.data(), m.lm_head.size()},
        {"resid_l", m.resid_l.data(), m.d_resid.data(), m.resid_l.size()},
        {"x0_l", m.x0_l.data(), m.d_x0.data(), m.x0_l.size()},
        {"smear_w", m.smear_w.data(), m.d_smear_w.data(), m.smear_w.size()},
        {"smear_lam", &m.smear_lambda, &m.d_smear_lambda, 1},
        {"backout", &m.backout_lambda, &m.d_backout, 1},
    };
    for (int l = 0; l < c.n_layer; l++) {
        auto& lp = m.layers[l];
        gs.push_back({l==0?"Wq0":(l==1?"Wq1":"Wq2"), lp.Wq.data(), lp.dWq.data(), lp.Wq.size()});
        gs.push_back({l==0?"Wv0":(l==1?"Wv1":"Wv2"), lp.Wv.data(), lp.dWv.data(), lp.Wv.size()});
        gs.push_back({l==0?"Wo0":(l==1?"Wo1":"Wo2"), lp.Wo.data(), lp.dWo.data(), lp.Wo.size()});
        gs.push_back({l==0?"Wfc0":(l==1?"Wfc1":"Wfc2"), lp.Wfc.data(), lp.dWfc.data(), lp.Wfc.size()});
        gs.push_back({l==0?"Wproj0":(l==1?"Wproj1":"Wproj2"), lp.Wproj.data(), lp.dWproj.data(), lp.Wproj.size()});
        if (lp.ve) {
            gs.push_back({l==0?"ve_emb0":"ve_emb2", lp.ve_emb.data(), lp.dve_emb.data(), lp.ve_emb.size()});
            gs.push_back({l==0?"ve_gate0":"ve_gate2", lp.ve_gate.data(), lp.dve_gate.data(), lp.ve_gate.size()});
        }
    }

    const real eps = (real)1e-5;
    real max_rel = 0;
    std::printf("gradient check (%s, eps=%.0e):\n", sizeof(real)==8?"double":"float", (double)eps);
    for (auto& gr : gs) {
        real gm = 0; size_t stride = std::max((size_t)1, gr.sz / 9);
        for (size_t i = 0; i < gr.sz; i += stride) {
            real o = gr.p[i];
            gr.p[i] = o + eps; m.forward(inp.data(), tgt.data(), B, T); real lp = m.mean_loss;
            gr.p[i] = o - eps; m.forward(inp.data(), tgt.data(), B, T); real lm = m.mean_loss;
            gr.p[i] = o;
            real num = (lp - lm) / (2 * eps), ana = gr.g[i];
            real rel = std::fabs(num - ana) / std::max((real)1e-9, std::fabs(num) + std::fabs(ana));
            gm = std::max(gm, rel);
        }
        max_rel = std::max(max_rel, gm);
        std::printf("  %-9s max rel err = %.3e\n", gr.n, gm);
    }
    real tol = sizeof(real)==8 ? (real)1e-4 : (real)5e-2;
    std::printf("overall max rel err = %.3e  ->  %s\n", max_rel, max_rel < tol ? "PASS" : "FAIL");
    return max_rel < tol ? 0 : 1;
}
