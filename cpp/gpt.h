// ---------------------------------------------------------------------------
//  gpt.h  --  dependency-free C++ port of nanochat's GPT (nanochat/gpt.py),
//  now including the "modded-nanogpt" refinements that the core port omitted:
//    - per-layer learnable resid / x0 scalars (blend the initial embedding)
//    - "smear": mix the previous token's embedding via a learned gate
//    - "backout": subtract a mid-layer residual before the final norm
//    - value embeddings (ResFormer) on alternating layers, with a per-head gate
//    - the Muon optimizer for matrix params (AdamW for embeddings/scalars)
//
//  On top of the core features: RoPE, RMSNorm (no params), QK-norm + 1.2 scale,
//  ReLU^2 MLP, GQA, untied embeddings, no bias, logit softcap.
//
//  Faithful to nanochat's architecture. Simplifications (documented): Muon here
//  implements Nesterov momentum + Polar-Express orthogonalization + Muon+ renorm
//  + per-matrix LR scaling + cautious weight decay, but NOT NorMuon variance
//  reduction or row equilibration. Sliding-window attention and vocab padding
//  are omitted (both are inactive/irrelevant at toy CPU scale).
//
//  No PyTorch/Eigen/BLAS. `real` is float; define GPT_USE_DOUBLE for gradcheck.
// ---------------------------------------------------------------------------
#ifndef NANOCHAT_GPT_H
#define NANOCHAT_GPT_H

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <algorithm>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gpt {

#ifdef GPT_USE_DOUBLE
using real = double;
#else
using real = float;
#endif

struct GPTConfig {
    int sequence_len = 256;
    int vocab_size = 65;
    int n_layer = 6;
    int n_head = 6;
    int n_kv_head = 6;
    int n_embd = 384;
    double rope_base = 100000.0;
    bool use_muon = true;   // Muon for matrices (AdamW otherwise)
    const char* window_pattern = "SSSL"; // sliding-window pattern, tiled across layers; last layer always L
    int force_window = 0;   // gradcheck only: if >0, force this window (left) on every layer
    int vocab_pad = 64;     // pad vocab_size up to a multiple of this (GPU-alignment; no effect on CPU results)
};

inline bool has_ve(int l, int L) { return l % 2 == (L - 1) % 2; }
inline real sigmoidf(real z) { return (real)1.0 / ((real)1.0 + (real)std::exp(-(double)z)); }

// ---- ops (forward + backward), same as the core port -----------------------
inline void rmsnorm_fwd(real* out, real* rstd, const real* in, int N, int D) {
    const real eps = (real)1e-6;
    for (int n = 0; n < N; n++) {
        const real* x = in + (size_t)n * D;
        real ms = 0; for (int i = 0; i < D; i++) ms += x[i] * x[i]; ms /= D;
        real r = (real)1.0 / std::sqrt(ms + eps);
        real* o = out + (size_t)n * D;
        for (int i = 0; i < D; i++) o[i] = x[i] * r;
        rstd[n] = r;
    }
}
inline void rmsnorm_bwd(real* din, const real* dout, const real* in,
                        const real* rstd, int N, int D) {
    for (int n = 0; n < N; n++) {
        const real* x = in + (size_t)n * D; const real* g = dout + (size_t)n * D;
        real* di = din + (size_t)n * D; real r = rstd[n];
        real dot = 0; for (int i = 0; i < D; i++) dot += g[i] * x[i];
        real c = r * r * r * dot / D;
        for (int i = 0; i < D; i++) di[i] += r * g[i] - c * x[i];
    }
}
inline void matmul_fwd(real* out, const real* in, const real* W, int N, int C, int OC) {
    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        const real* x = in + (size_t)n * C; real* o = out + (size_t)n * OC;
        for (int oc = 0; oc < OC; oc++) {
            const real* w = W + (size_t)oc * C;
            real acc = 0; for (int c = 0; c < C; c++) acc += x[c] * w[c];
            o[oc] = acc;
        }
    }
}
inline void matmul_bwd(real* din, real* dW, const real* dout, const real* in,
                       const real* W, int N, int C, int OC) {
    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        const real* d = dout + (size_t)n * OC; real* di = din + (size_t)n * C;
        for (int oc = 0; oc < OC; oc++) {
            const real* w = W + (size_t)oc * C; real dd = d[oc];
            for (int c = 0; c < C; c++) di[c] += w[c] * dd;
        }
    }
    #pragma omp parallel for
    for (int oc = 0; oc < OC; oc++) {
        real* dw = dW + (size_t)oc * C;
        for (int n = 0; n < N; n++) {
            const real* x = in + (size_t)n * C; real dd = dout[(size_t)n * OC + oc];
            for (int c = 0; c < C; c++) dw[c] += x[c] * dd;
        }
    }
}
inline void rope_fwd(real* out, const real* in, const real* cos, const real* sin,
                     int B, int T, int H, int hd) {
    int d = hd / 2;
    for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
        const real* cs = cos + (size_t)t * d; const real* sn = sin + (size_t)t * d;
        for (int h = 0; h < H; h++) {
            const real* x = in + (((size_t)(b * T + t) * H) + h) * hd;
            real* o = out + (((size_t)(b * T + t) * H) + h) * hd;
            for (int j = 0; j < d; j++) {
                real x1 = x[j], x2 = x[j + d];
                o[j] = x1 * cs[j] + x2 * sn[j]; o[j + d] = -x1 * sn[j] + x2 * cs[j];
            }
        }
    }
}
inline void rope_bwd(real* din, const real* dout, const real* cos, const real* sin,
                     int B, int T, int H, int hd) {
    int d = hd / 2;
    for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
        const real* cs = cos + (size_t)t * d; const real* sn = sin + (size_t)t * d;
        for (int h = 0; h < H; h++) {
            const real* g = dout + (((size_t)(b * T + t) * H) + h) * hd;
            real* di = din + (((size_t)(b * T + t) * H) + h) * hd;
            for (int j = 0; j < d; j++) {
                real g1 = g[j], g2 = g[j + d];
                di[j] += g1 * cs[j] - g2 * sn[j]; di[j + d] += g1 * sn[j] + g2 * cs[j];
            }
        }
    }
}
// window = max tokens of left context (query t attends keys [t-window, t]); <=0 or >=T means full causal.
inline void attention_fwd(real* y, real* att, const real* q, const real* k,
                          const real* v, int B, int T, int nh, int nkv, int hd, int window) {
    int group = nh / nkv; real scale = (real)(1.0 / std::sqrt((double)hd));
    int BNHT = B * nh * T;
    #pragma omp parallel for
    for (int idx = 0; idx < BNHT; idx++) {
        int b = idx / (nh * T), h = (idx / T) % nh, t = idx % T, hk = h / group;
        int lo = (window > 0 && window < T) ? std::max(0, t - window) : 0;
        const real* qv = q + (((size_t)(b * T + t) * nh) + h) * hd;
        real* a = att + (((size_t)b * nh + h) * T + t) * T;
        real mx = -1e30f;
        for (int t2 = lo; t2 <= t; t2++) {
            const real* kv = k + (((size_t)(b * T + t2) * nkv) + hk) * hd;
            real dot = 0; for (int i = 0; i < hd; i++) dot += qv[i] * kv[i];
            dot *= scale; a[t2] = dot; if (dot > mx) mx = dot;
        }
        real sum = 0; for (int t2 = lo; t2 <= t; t2++) { real e = std::exp(a[t2] - mx); a[t2] = e; sum += e; }
        real inv = sum > 0 ? (real)1.0 / sum : (real)0;
        for (int t2 = lo; t2 <= t; t2++) a[t2] *= inv;
        for (int t2 = 0; t2 < lo; t2++) a[t2] = 0;
        for (int t2 = t + 1; t2 < T; t2++) a[t2] = 0;
        real* o = y + (((size_t)(b * T + t) * nh) + h) * hd;
        for (int i = 0; i < hd; i++) o[i] = 0;
        for (int t2 = lo; t2 <= t; t2++) {
            const real* vv = v + (((size_t)(b * T + t2) * nkv) + hk) * hd;
            real aw = a[t2]; for (int i = 0; i < hd; i++) o[i] += aw * vv[i];
        }
    }
}
inline void attention_bwd(real* dq, real* dk, real* dv, const real* dy, const real* att,
                          const real* q, const real* k, const real* v,
                          int B, int T, int nh, int nkv, int hd, int window) {
    int group = nh / nkv; real scale = (real)(1.0 / std::sqrt((double)hd));
    std::vector<real> da(T);
    for (int b = 0; b < B; b++) for (int h = 0; h < nh; h++) {
        int hk = h / group;
        for (int t = 0; t < T; t++) {
            int lo = (window > 0 && window < T) ? std::max(0, t - window) : 0;
            const real* a = att + (((size_t)b * nh + h) * T + t) * T;
            const real* dO = dy + (((size_t)(b * T + t) * nh) + h) * hd;
            const real* qv = q + (((size_t)(b * T + t) * nh) + h) * hd;
            real* dqv = dq + (((size_t)(b * T + t) * nh) + h) * hd;
            for (int t2 = lo; t2 <= t; t2++) {
                const real* vv = v + (((size_t)(b * T + t2) * nkv) + hk) * hd;
                real* dvv = dv + (((size_t)(b * T + t2) * nkv) + hk) * hd;
                real acc = 0; for (int i = 0; i < hd; i++) { acc += dO[i] * vv[i]; dvv[i] += a[t2] * dO[i]; }
                da[t2] = acc;
            }
            real dsum = 0; for (int t2 = lo; t2 <= t; t2++) dsum += a[t2] * da[t2];
            for (int t2 = lo; t2 <= t; t2++) {
                real dpre = a[t2] * (da[t2] - dsum) * scale;
                const real* kv = k + (((size_t)(b * T + t2) * nkv) + hk) * hd;
                real* dkv = dk + (((size_t)(b * T + t2) * nkv) + hk) * hd;
                for (int i = 0; i < hd; i++) { dqv[i] += dpre * kv[i]; dkv[i] += dpre * qv[i]; }
            }
        }
    }
}
inline void relu2_fwd(real* out, const real* in, int N) {
    for (int i = 0; i < N; i++) { real x = in[i] > 0 ? in[i] : (real)0; out[i] = x * x; }
}
inline void relu2_bwd(real* din, const real* in, const real* dout, int N) {
    for (int i = 0; i < N; i++) { real x = in[i]; din[i] += (x > 0 ? (real)2 * x : (real)0) * dout[i]; }
}
inline void softcap_fwd(real* out, const real* in, real cap, int N) {
    for (int i = 0; i < N; i++) out[i] = cap * (real)std::tanh(in[i] / cap);
}
inline void softcap_bwd(real* din, const real* in, const real* dout, real cap, int N) {
    for (int i = 0; i < N; i++) { real th = (real)std::tanh(in[i] / cap); din[i] = (1 - th * th) * dout[i]; }
}
inline real softmax_ce(real* dlogits, real* probs, const real* logits,
                       const int* targets, int N, int V) {
    double loss = 0; int cnt = 0;
    for (int n = 0; n < N; n++) {
        const real* l = logits + (size_t)n * V; real* p = probs + (size_t)n * V;
        real mx = -1e30f; for (int i = 0; i < V; i++) if (l[i] > mx) mx = l[i];
        real sum = 0; for (int i = 0; i < V; i++) { real e = std::exp(l[i] - mx); p[i] = e; sum += e; }
        real inv = (real)1.0 / sum; for (int i = 0; i < V; i++) p[i] *= inv;
        if (targets[n] >= 0) { loss += -std::log((double)p[targets[n]] + 1e-30); cnt++; }
    }
    if (dlogits) {
        real dl = (real)(1.0 / (cnt > 0 ? cnt : 1));
        for (int n = 0; n < N; n++) {
            const real* p = probs + (size_t)n * V; real* d = dlogits + (size_t)n * V; int tg = targets[n];
            if (tg < 0) { for (int i = 0; i < V; i++) d[i] = 0; continue; }
            for (int i = 0; i < V; i++) d[i] = (p[i] - (i == tg ? (real)1 : (real)0)) * dl;
        }
    }
    return (real)(loss / (cnt > 0 ? cnt : 1));
}

// ---- Muon: orthogonalized-momentum update for a 2D matrix W[rows, cols] -----
// Polar-Express coefficients (num_iters=5), from nanochat/optim.py.
inline void muon_update(real* W, const real* G, real* buf, real* buf2, int rows, int cols,
                        real lr, real momentum, real wd, int ns_steps, real beta2) {
    static const double coeff[5][3] = {
        {8.156554524902461, -22.48329292557795, 15.878769915207462},
        {4.042929935166739, -2.808917465908714, 0.5000178451051316},
        {3.8916678022926607, -2.772484153217685, 0.5060648178503393},
        {3.285753657755655, -2.3681294933425376, 0.46449024233003106},
        {2.3465413258596377, -1.7097828382687081, 0.42323551169305323},
    };
    size_t n = (size_t)rows * cols;
    // Nesterov momentum: buf = buf*mom + g*(1-mom); g = g*(1-mom) + buf*mom
    std::vector<double> X(n);
    for (size_t i = 0; i < n; i++) {
        buf[i] = momentum * buf[i] + (1 - momentum) * G[i];
        X[i] = (1 - momentum) * (double)G[i] + momentum * (double)buf[i];
    }
    auto fro = [&](const std::vector<double>& M) { double s = 0; for (double v : M) s += v * v; return std::sqrt(s); };
    // MuonEq row equilibration: rescale each row to the mean row norm
    {
        double target = fro(X) / std::sqrt((double)rows);
        for (int r = 0; r < rows; r++) {
            double rn = 0; for (int c = 0; c < cols; c++) { double v = X[(size_t)r * cols + c]; rn += v * v; }
            rn = std::sqrt(rn); if (rn < 1e-6) rn = 1e-6;
            double s = target / rn;
            for (int c = 0; c < cols; c++) X[(size_t)r * cols + c] *= s;
        }
    }
    // normalize
    double nrm = fro(X) * 1.01 + 1e-6;
    for (auto& v : X) v /= nrm;
    // Polar Express iterations
    bool tall = rows > cols;
    for (int it = 0; it < ns_steps; it++) {
        double a = coeff[it][0], b = coeff[it][1], c = coeff[it][2];
        if (tall) { // A = X^T X (cols x cols); B = bA + cA^2; X = aX + X B
            std::vector<double> A((size_t)cols * cols, 0);
            for (int i = 0; i < cols; i++) for (int j = 0; j < cols; j++) {
                double s = 0; for (int r = 0; r < rows; r++) s += X[(size_t)r * cols + i] * X[(size_t)r * cols + j];
                A[(size_t)i * cols + j] = s;
            }
            std::vector<double> AA((size_t)cols * cols, 0);
            for (int i = 0; i < cols; i++) for (int j = 0; j < cols; j++) {
                double s = 0; for (int m = 0; m < cols; m++) s += A[(size_t)i * cols + m] * A[(size_t)m * cols + j];
                AA[(size_t)i * cols + j] = s;
            }
            std::vector<double> Bm((size_t)cols * cols);
            for (size_t i = 0; i < Bm.size(); i++) Bm[i] = b * A[i] + c * AA[i];
            std::vector<double> XB((size_t)rows * cols, 0);
            for (int r = 0; r < rows; r++) for (int j = 0; j < cols; j++) {
                double s = 0; for (int m = 0; m < cols; m++) s += X[(size_t)r * cols + m] * Bm[(size_t)m * cols + j];
                XB[(size_t)r * cols + j] = s;
            }
            for (size_t i = 0; i < n; i++) X[i] = a * X[i] + XB[i];
        } else { // A = X X^T (rows x rows); B = bA + cA^2; X = aX + B X
            std::vector<double> A((size_t)rows * rows, 0);
            for (int i = 0; i < rows; i++) for (int j = 0; j < rows; j++) {
                double s = 0; for (int m = 0; m < cols; m++) s += X[(size_t)i * cols + m] * X[(size_t)j * cols + m];
                A[(size_t)i * rows + j] = s;
            }
            std::vector<double> AA((size_t)rows * rows, 0);
            for (int i = 0; i < rows; i++) for (int j = 0; j < rows; j++) {
                double s = 0; for (int m = 0; m < rows; m++) s += A[(size_t)i * rows + m] * A[(size_t)m * rows + j];
                AA[(size_t)i * rows + j] = s;
            }
            std::vector<double> Bm((size_t)rows * rows);
            for (size_t i = 0; i < Bm.size(); i++) Bm[i] = b * A[i] + c * AA[i];
            std::vector<double> BX((size_t)rows * cols, 0);
            for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) {
                double s = 0; for (int m = 0; m < rows; m++) s += Bm[(size_t)i * rows + m] * X[(size_t)m * cols + j];
                BX[(size_t)i * cols + j] = s;
            }
            for (size_t i = 0; i < n; i++) X[i] = a * X[i] + BX[i];
        }
    }
    // Muon+ renorm: snap Frobenius norm to sqrt(min(rows,cols))
    {
        double target = std::sqrt((double)std::min(rows, cols));
        double cur = fro(X); if (cur < 1e-6) cur = 1e-6;
        double rescale = target / cur;
        for (auto& v : X) v *= rescale;
    }
    // NorMuon variance reduction: per-row (rows>=cols) or per-col second-moment
    // adaptive scaling. buf2 stores the EMA second moment (size max(rows,cols)).
    bool byrow = rows >= cols;              // red_dim reduces the OTHER axis
    int m = byrow ? rows : cols;            // number of reduction groups
    int rdsz = byrow ? cols : rows;         // elements per group
    std::vector<double> vmean(m, 0);
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
        double v = X[(size_t)r * cols + c]; int g = byrow ? r : c; vmean[g] += v * v;
    }
    for (int i = 0; i < m; i++) vmean[i] /= rdsz;
    double vnorm_sq = 0; for (double v : vmean) vnorm_sq += v; vnorm_sq *= rdsz;
    double vnorm = std::sqrt(vnorm_sq);
    for (int i = 0; i < m; i++) buf2[i] = beta2 * buf2[i] + (1 - beta2) * (real)vmean[i];
    std::vector<double> step(m); for (int i = 0; i < m; i++) { double b = std::max((double)buf2[i], 1e-10); step[i] = 1.0 / std::sqrt(b); }
    double vnew_sq = 0; for (int i = 0; i < m; i++) vnew_sq += (vmean[i] * rdsz) * step[i] * step[i];
    double vnew = std::sqrt(vnew_sq); if (vnew < 1e-10) vnew = 1e-10;
    std::vector<double> fscale(m); for (int i = 0; i < m; i++) fscale[i] = step[i] * (vnorm / vnew);
    // cautious weight decay + update (lr already includes the max(1,rows/cols)^0.5 factor)
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
        size_t i = (size_t)r * cols + c; int gi = byrow ? r : c;
        real g = (real)(X[i] * fscale[gi]);
        real mask = (g * W[i]) >= 0 ? (real)1 : (real)0;
        W[i] -= lr * g + lr * wd * W[i] * mask;
    }
}

// ---- one transformer layer ------------------------------------------------
struct LayerParams {
    std::vector<real> Wq, Wk, Wv, Wo, Wfc, Wproj;
    std::vector<real> dWq, dWk, dWv, dWo, dWfc, dWproj;
    std::vector<real> bufWq, bufWk, bufWv, bufWo, bufWfc, bufWproj;    // muon momentum
    std::vector<real> b2Wq, b2Wk, b2Wv, b2Wo, b2Wfc, b2Wproj;         // muon NorMuon 2nd moment
    int window = 0;   // sliding-window left context for this layer (0 = full)
    // value-embedding params (only if has_ve)
    bool ve = false;
    std::vector<real> ve_emb, dve_emb, m_ve, v_ve;      // (V, Ckv)
    std::vector<real> ve_gate, dve_gate, m_veg, v_veg;  // (nkv, 12)
};

struct GPT {
    GPTConfig cfg;
    int C, hd, nh, nkv, Ckv, V, Vp, L, T = 0, B = 0, backout_layer = 0;
    static const int VE_GATE_CH = 12, SMEAR_CH = 24;

    std::vector<real> wte, dwte, m_wte, v_wte;
    std::vector<real> lm_head, dlm, m_lm, v_lm;
    std::vector<LayerParams> layers;
    std::vector<real> cos, sin;

    // scalars (AdamW)
    std::vector<real> resid_l, x0_l, d_resid, d_x0, m_resid, v_resid, m_x0, v_x0; // (L)
    std::vector<real> smear_w, d_smear_w, m_sw, v_sw;   // (SMEAR_CH)
    real smear_lambda = 0, d_smear_lambda = 0, m_sl = 0, v_sl = 0;
    real backout_lambda = (real)0.2, d_backout = 0, m_bo = 0, v_bo = 0;

    int adam_t = 0;
    real mean_loss = 0;
    std::vector<int> inputs, targets;

    // caches
    std::vector<real> emb, emb_n, rstd_emb, xs, sg_smear;
    std::vector<std::vector<real>> res_in, res_a, an, rstd_an, q0, k0, v0;
    std::vector<std::vector<real>> ve_val, ve_sig, vused, q1, k1, rstd_q, rstd_k, q3, k3;
    std::vector<std::vector<real>> att, yatt, yproj, mn, rstd_mn, fc, hg, mproj;
    std::vector<real> x_backout, xf, xf_n, rstd_f, logits_raw, logits, probs;

    void build(const GPTConfig& c) {
        cfg = c; C = c.n_embd; nh = c.n_head; nkv = c.n_kv_head; hd = C / nh;
        Ckv = nkv * hd; V = c.vocab_size; L = c.n_layer; backout_layer = L / 2;
        int pad = c.vocab_pad > 0 ? c.vocab_pad : 1;
        Vp = ((V + pad - 1) / pad) * pad;   // padded vocab (GPU alignment; no effect on CPU results)
        auto z = [](std::vector<real>& w, size_t n) { w.assign(n, 0); };
        z(wte, (size_t)Vp * C); z(dwte, wte.size()); z(m_wte, wte.size()); z(v_wte, wte.size());
        z(lm_head, (size_t)Vp * C); z(dlm, lm_head.size()); z(m_lm, lm_head.size()); z(v_lm, lm_head.size());
        layers.resize(L);
        for (int l = 0; l < L; l++) {
            LayerParams& lp = layers[l];
            auto a2 = [&](std::vector<real>& w, size_t n) { w.assign(n, 0); };
            a2(lp.Wq, (size_t)C * C); a2(lp.dWq, lp.Wq.size()); a2(lp.bufWq, lp.Wq.size()); a2(lp.b2Wq, C);
            a2(lp.Wk, (size_t)Ckv * C); a2(lp.dWk, lp.Wk.size()); a2(lp.bufWk, lp.Wk.size()); a2(lp.b2Wk, C);
            a2(lp.Wv, (size_t)Ckv * C); a2(lp.dWv, lp.Wv.size()); a2(lp.bufWv, lp.Wv.size()); a2(lp.b2Wv, C);
            a2(lp.Wo, (size_t)C * C); a2(lp.dWo, lp.Wo.size()); a2(lp.bufWo, lp.Wo.size()); a2(lp.b2Wo, C);
            a2(lp.Wfc, (size_t)4 * C * C); a2(lp.dWfc, lp.Wfc.size()); a2(lp.bufWfc, lp.Wfc.size()); a2(lp.b2Wfc, 4 * C);
            a2(lp.Wproj, (size_t)C * 4 * C); a2(lp.dWproj, lp.Wproj.size()); a2(lp.bufWproj, lp.Wproj.size()); a2(lp.b2Wproj, 4 * C);
            // sliding-window size for this layer
            {
                int longw = cfg.sequence_len;
                int shortw = ((longw / 4 + 127) / 128) * 128;
                int plen = (int)std::string(cfg.window_pattern).size(); if (plen < 1) plen = 1;
                char ch = cfg.window_pattern[l % plen];
                lp.window = (ch == 'S' || ch == 's') ? shortw : longw;
                if (l == L - 1) lp.window = longw;          // last layer always full
                if (cfg.force_window > 0) lp.window = cfg.force_window;
            }
            lp.ve = has_ve(l, L);
            if (lp.ve) {
                a2(lp.ve_emb, (size_t)V * Ckv); a2(lp.dve_emb, lp.ve_emb.size()); a2(lp.m_ve, lp.ve_emb.size()); a2(lp.v_ve, lp.ve_emb.size());
                a2(lp.ve_gate, (size_t)nkv * VE_GATE_CH); a2(lp.dve_gate, lp.ve_gate.size()); a2(lp.m_veg, lp.ve_gate.size()); a2(lp.v_veg, lp.ve_gate.size());
            }
        }
        auto z1 = [](std::vector<real>& w, int n) { w.assign(n, 0); };
        z1(resid_l, L); z1(x0_l, L); z1(d_resid, L); z1(d_x0, L);
        z1(m_resid, L); z1(v_resid, L); z1(m_x0, L); z1(v_x0, L);
        z1(smear_w, SMEAR_CH); z1(d_smear_w, SMEAR_CH); z1(m_sw, SMEAR_CH); z1(v_sw, SMEAR_CH);
        precompute_rope(c.sequence_len);
    }

    void precompute_rope(int maxT) {
        int d = hd / 2; cos.assign((size_t)maxT * d, 0); sin.assign((size_t)maxT * d, 0);
        for (int t = 0; t < maxT; t++) for (int j = 0; j < d; j++) {
            double inv = 1.0 / std::pow(cfg.rope_base, (double)(2 * j) / hd), f = t * inv;
            cos[(size_t)t * d + j] = (real)std::cos(f); sin[(size_t)t * d + j] = (real)std::sin(f);
        }
    }

    void init(uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> nwte(0.0, 0.8), nlm(0.0, 0.001);
        for (auto& w : wte) w = (real)nwte(rng);
        for (auto& w : lm_head) w = (real)nlm(rng);
        real s = (real)(std::sqrt(3.0) / std::sqrt((double)C));
        std::uniform_real_distribution<double> u(-s, s), ufc(-0.4 * s, 0.4 * s), ug(0.0, 0.02);
        for (auto& lp : layers) {
            for (auto& w : lp.Wq) w = (real)u(rng);
            for (auto& w : lp.Wk) w = (real)u(rng);
            for (auto& w : lp.Wv) w = (real)u(rng);
            for (auto& w : lp.Wfc) w = (real)ufc(rng);
            // Wo, Wproj stay zero
            if (lp.ve) { for (auto& w : lp.ve_emb) w = (real)u(rng); for (auto& w : lp.ve_gate) w = (real)ug(rng); }
        }
        for (int i = 0; i < L; i++) {
            resid_l[i] = (real)(1.15 - 0.10 * i / std::max(L - 1, 1));
            x0_l[i] = (real)(0.20 - 0.15 * i / std::max(L - 1, 1));
        }
        for (auto& w : smear_w) w = (real)ug(rng);
        smear_lambda = 0; backout_lambda = (real)0.2;
    }

    // gradcheck-only: randomize every weight so all paths are exercised
    void init_test(uint32_t seed) {
        std::mt19937 rng(seed); std::normal_distribution<double> nd(0.0, 0.1);
        auto f = [&](std::vector<real>& w) { for (auto& v : w) v = (real)nd(rng); };
        f(wte); f(lm_head);
        for (auto& lp : layers) { f(lp.Wq); f(lp.Wk); f(lp.Wv); f(lp.Wo); f(lp.Wfc); f(lp.Wproj); if (lp.ve) { f(lp.ve_emb); f(lp.ve_gate); } }
        f(resid_l); f(x0_l); f(smear_w);
        smear_lambda = (real)0.3; backout_lambda = (real)0.25;
    }

    size_t num_params() const {
        size_t n = wte.size() + lm_head.size() + resid_l.size() + x0_l.size() + smear_w.size() + 2;
        for (auto& lp : layers) { n += lp.Wq.size()+lp.Wk.size()+lp.Wv.size()+lp.Wo.size()+lp.Wfc.size()+lp.Wproj.size(); if (lp.ve) n += lp.ve_emb.size()+lp.ve_gate.size(); }
        return n;
    }

    void ensure_acts(int B_, int T_) {
        if (B == B_ && T == T_ && !emb.empty()) return;
        B = B_; T = T_;
        size_t bt = (size_t)B * T, btc = bt * C, btkv = bt * Ckv, btnhT = (size_t)B * nh * T * T;
        emb.assign(btc, 0); emb_n.assign(btc, 0); rstd_emb.assign(bt, 0); xs.assign(btc, 0); sg_smear.assign(bt, 0);
        auto R = [&](std::vector<std::vector<real>>& v, size_t sz) { v.assign(L, std::vector<real>(sz, 0)); };
        R(res_in, btc); R(res_a, btc); R(an, btc); R(rstd_an, bt);
        R(q0, btc); R(k0, btkv); R(v0, btkv);
        R(ve_val, btkv); R(ve_sig, bt * nkv); R(vused, btkv);
        R(q1, btc); R(k1, btkv); R(rstd_q, bt * nh); R(rstd_k, bt * nkv); R(q3, btc); R(k3, btkv);
        R(att, btnhT); R(yatt, btc); R(yproj, btc);
        R(mn, btc); R(rstd_mn, bt); R(fc, bt * 4 * C); R(hg, bt * 4 * C); R(mproj, btc);
        x_backout.assign(btc, 0); xf.assign(btc, 0); xf_n.assign(btc, 0); rstd_f.assign(bt, 0);
        logits_raw.assign(bt * V, 0); logits.assign(bt * V, 0); probs.assign(bt * V, 0);
        inputs.assign(bt, 0); targets.assign(bt, -1);
    }

    void forward(const int* idx, const int* tgt, int B_, int T_) {
        ensure_acts(B_, T_);
        int bt = B * T;
        for (int i = 0; i < bt; i++) inputs[i] = idx[i];
        for (int n = 0; n < bt; n++) { const real* w = wte.data() + (size_t)idx[n] * C; real* e = emb.data() + (size_t)n * C; for (int c = 0; c < C; c++) e[c] = w[c]; }
        rmsnorm_fwd(emb_n.data(), rstd_emb.data(), emb.data(), bt, C);
        // smear: xs[t] = emb_n[t] + smear_lambda*sigmoid(<emb_n[t,:24], smear_w>) * emb_n[t-1]
        std::copy(emb_n.begin(), emb_n.end(), xs.begin());
        int SM = std::min((int)SMEAR_CH, C);
        for (int b = 0; b < B; b++) for (int t = 1; t < T; t++) {
            int n = b * T + t; const real* e = emb_n.data() + (size_t)n * C;
            real z = 0; for (int c = 0; c < SM; c++) z += e[c] * smear_w[c];
            real sg = sigmoidf(z); sg_smear[n] = sg; real gate = smear_lambda * sg;
            const real* ep = emb_n.data() + (size_t)(n - 1) * C; real* o = xs.data() + (size_t)n * C;
            for (int c = 0; c < C; c++) o[c] += gate * ep[c];
        }
        std::vector<real> x(xs.begin(), xs.end());   // residual stream
        for (int l = 0; l < L; l++) {
            LayerParams& lp = layers[l];
            std::copy(x.begin(), x.end(), res_in[l].begin());
            for (size_t i = 0; i < (size_t)bt * C; i++) res_a[l][i] = resid_l[l] * x[i] + x0_l[l] * xs[i];
            rmsnorm_fwd(an[l].data(), rstd_an[l].data(), res_a[l].data(), bt, C);
            matmul_fwd(q0[l].data(), an[l].data(), lp.Wq.data(), bt, C, C);
            matmul_fwd(k0[l].data(), an[l].data(), lp.Wk.data(), bt, C, Ckv);
            matmul_fwd(v0[l].data(), an[l].data(), lp.Wv.data(), bt, C, Ckv);
            // value embedding + gate
            if (lp.ve) {
                for (int n = 0; n < bt; n++) {
                    const real* vemb = lp.ve_emb.data() + (size_t)inputs[n] * Ckv;
                    real* vv = ve_val[l].data() + (size_t)n * Ckv;
                    for (int i = 0; i < Ckv; i++) vv[i] = vemb[i];
                    const real* a = an[l].data() + (size_t)n * C;
                    int VG = std::min((int)VE_GATE_CH, C);
                    for (int hk = 0; hk < nkv; hk++) {
                        real z = 0; for (int c = 0; c < VG; c++) z += a[c] * lp.ve_gate[(size_t)hk * VE_GATE_CH + c];
                        real sg = sigmoidf(z); ve_sig[l][(size_t)n * nkv + hk] = sg; real gate = (real)3 * sg;
                        real* vu = vused[l].data() + (size_t)n * Ckv + hk * hd;
                        const real* v0p = v0[l].data() + (size_t)n * Ckv + hk * hd;
                        const real* vep = vv + hk * hd;
                        for (int i = 0; i < hd; i++) vu[i] = v0p[i] + gate * vep[i];
                    }
                }
            } else std::copy(v0[l].begin(), v0[l].end(), vused[l].begin());
            rope_fwd(q1[l].data(), q0[l].data(), cos.data(), sin.data(), B, T, nh, hd);
            rope_fwd(k1[l].data(), k0[l].data(), cos.data(), sin.data(), B, T, nkv, hd);
            rmsnorm_fwd(q3[l].data(), rstd_q[l].data(), q1[l].data(), bt * nh, hd);
            rmsnorm_fwd(k3[l].data(), rstd_k[l].data(), k1[l].data(), bt * nkv, hd);
            for (auto& g : q3[l]) g *= (real)1.2;
            for (auto& g : k3[l]) g *= (real)1.2;
            attention_fwd(yatt[l].data(), att[l].data(), q3[l].data(), k3[l].data(), vused[l].data(), B, T, nh, nkv, hd, lp.window);
            matmul_fwd(yproj[l].data(), yatt[l].data(), lp.Wo.data(), bt, C, C);
            for (size_t i = 0; i < (size_t)bt * C; i++) res_a[l][i] += yproj[l][i]; // res_a now = bx + attn (== residual after attn)
            rmsnorm_fwd(mn[l].data(), rstd_mn[l].data(), res_a[l].data(), bt, C);
            matmul_fwd(fc[l].data(), mn[l].data(), lp.Wfc.data(), bt, C, 4 * C);
            relu2_fwd(hg[l].data(), fc[l].data(), bt * 4 * C);
            matmul_fwd(mproj[l].data(), hg[l].data(), lp.Wproj.data(), bt, 4 * C, C);
            for (size_t i = 0; i < (size_t)bt * C; i++) x[i] = res_a[l][i] + mproj[l][i];
            if (l == backout_layer) std::copy(x.begin(), x.end(), x_backout.begin());
        }
        for (size_t i = 0; i < (size_t)bt * C; i++) xf[i] = x[i] - backout_lambda * x_backout[i];
        rmsnorm_fwd(xf_n.data(), rstd_f.data(), xf.data(), bt, C);
        matmul_fwd(logits_raw.data(), xf_n.data(), lm_head.data(), bt, C, V);
        softcap_fwd(logits.data(), logits_raw.data(), (real)15, bt * V);
        if (tgt) {
            for (int i = 0; i < bt; i++) targets[i] = tgt[i];
            mean_loss = softmax_ce(nullptr, probs.data(), logits.data(), targets.data(), bt, V);
        } else mean_loss = -1;
    }

    void zero_grad() {
        auto z = [](std::vector<real>& v) { std::fill(v.begin(), v.end(), (real)0); };
        z(dwte); z(dlm); z(d_resid); z(d_x0); z(d_smear_w); d_smear_lambda = 0; d_backout = 0;
        for (auto& lp : layers) { z(lp.dWq); z(lp.dWk); z(lp.dWv); z(lp.dWo); z(lp.dWfc); z(lp.dWproj); if (lp.ve) { z(lp.dve_emb); z(lp.dve_gate); } }
    }

    void backward() {
        int bt = B * T;
        std::vector<real> dlogits(bt * V), dlraw(bt * V), dxf_n((size_t)bt * C, 0), dxf((size_t)bt * C, 0);
        softmax_ce(dlogits.data(), probs.data(), logits.data(), targets.data(), bt, V);
        softcap_bwd(dlraw.data(), logits_raw.data(), dlogits.data(), (real)15, bt * V);
        matmul_bwd(dxf_n.data(), dlm.data(), dlraw.data(), xf_n.data(), lm_head.data(), bt, C, V);
        rmsnorm_bwd(dxf.data(), dxf_n.data(), xf.data(), rstd_f.data(), bt, C);
        // backout: xf = x - backout_lambda * x_backout
        std::vector<real> dx(dxf), dx_backout_inj((size_t)bt * C, 0);
        for (size_t i = 0; i < (size_t)bt * C; i++) {
            d_backout += -x_backout[i] * dxf[i];
            dx_backout_inj[i] = -backout_lambda * dxf[i];
        }
        std::vector<real> dx0_total((size_t)bt * C, 0);
        for (int l = L - 1; l >= 0; l--) {
            LayerParams& lp = layers[l];
            if (l == backout_layer) for (size_t i = 0; i < dx.size(); i++) dx[i] += dx_backout_inj[i];
            std::vector<real> dres_a(dx), dmproj(dx);         // x = res_a(after attn) + mproj
            std::vector<real> dhg((size_t)bt * 4 * C, 0);
            matmul_bwd(dhg.data(), lp.dWproj.data(), dmproj.data(), hg[l].data(), lp.Wproj.data(), bt, 4 * C, C);
            std::vector<real> dfc((size_t)bt * 4 * C, 0);
            relu2_bwd(dfc.data(), fc[l].data(), dhg.data(), bt * 4 * C);
            std::vector<real> dmn((size_t)bt * C, 0);
            matmul_bwd(dmn.data(), lp.dWfc.data(), dfc.data(), mn[l].data(), lp.Wfc.data(), bt, C, 4 * C);
            rmsnorm_bwd(dres_a.data(), dmn.data(), res_a[l].data(), rstd_mn[l].data(), bt, C); // res_a here = post-attn residual
            // res_a(after attn) = bx + yproj
            std::vector<real> dbx(dres_a), dyproj(dres_a);
            std::vector<real> dyatt((size_t)bt * C, 0);
            matmul_bwd(dyatt.data(), lp.dWo.data(), dyproj.data(), yatt[l].data(), lp.Wo.data(), bt, C, C);
            std::vector<real> dq3((size_t)bt * C, 0), dk3((size_t)bt * Ckv, 0), dvused((size_t)bt * Ckv, 0);
            attention_bwd(dq3.data(), dk3.data(), dvused.data(), dyatt.data(), att[l].data(), q3[l].data(), k3[l].data(), vused[l].data(), B, T, nh, nkv, hd, lp.window);
            for (auto& g : dq3) g *= (real)1.2;
            for (auto& g : dk3) g *= (real)1.2;
            std::vector<real> dq1((size_t)bt * C, 0), dk1((size_t)bt * Ckv, 0);
            rmsnorm_bwd(dq1.data(), dq3.data(), q1[l].data(), rstd_q[l].data(), bt * nh, hd);
            rmsnorm_bwd(dk1.data(), dk3.data(), k1[l].data(), rstd_k[l].data(), bt * nkv, hd);
            std::vector<real> dq0((size_t)bt * C, 0), dk0((size_t)bt * Ckv, 0);
            rope_bwd(dq0.data(), dq1.data(), cos.data(), sin.data(), B, T, nh, hd);
            rope_bwd(dk0.data(), dk1.data(), cos.data(), sin.data(), B, T, nkv, hd);
            std::vector<real> dan((size_t)bt * C, 0);
            // value embedding backward: vused = v0 + gate*ve ; gate = 3*sig
            std::vector<real> dv0(dvused);
            if (lp.ve) {
                for (int n = 0; n < bt; n++) {
                    const real* a = an[l].data() + (size_t)n * C;
                    real* dve = lp.dve_emb.data() + (size_t)inputs[n] * Ckv;
                    real* danr = dan.data() + (size_t)n * C;
                    for (int hk = 0; hk < nkv; hk++) {
                        real sg = ve_sig[l][(size_t)n * nkv + hk]; real gate = (real)3 * sg;
                        const real* du = dvused.data() + (size_t)n * Ckv + hk * hd;
                        const real* vep = ve_val[l].data() + (size_t)n * Ckv + hk * hd;
                        real dgate = 0;
                        for (int i = 0; i < hd; i++) { dgate += du[i] * vep[i]; dve[hk * hd + i] += gate * du[i]; }
                        real dz = (real)3 * dgate * sg * (1 - sg);
                        int VG = std::min((int)VE_GATE_CH, C);
                        for (int c = 0; c < VG; c++) {
                            lp.dve_gate[(size_t)hk * VE_GATE_CH + c] += a[c] * dz;
                            danr[c] += lp.ve_gate[(size_t)hk * VE_GATE_CH + c] * dz;
                        }
                    }
                }
            }
            matmul_bwd(dan.data(), lp.dWq.data(), dq0.data(), an[l].data(), lp.Wq.data(), bt, C, C);
            matmul_bwd(dan.data(), lp.dWk.data(), dk0.data(), an[l].data(), lp.Wk.data(), bt, C, Ckv);
            matmul_bwd(dan.data(), lp.dWv.data(), dv0.data(), an[l].data(), lp.Wv.data(), bt, C, Ckv);
            // an = norm(bx) ; note res_a[l] currently holds post-attn residual, but norm used bx.
            // We need bx = res_a_before_attn = res_a[l] - yproj[l]. Reconstruct.
            std::vector<real> bx((size_t)bt * C);
            for (size_t i = 0; i < bx.size(); i++) bx[i] = res_a[l][i] - yproj[l][i];
            rmsnorm_bwd(dbx.data(), dan.data(), bx.data(), rstd_an[l].data(), bt, C);
            // bx = resid_l[l]*res_in[l] + x0_l[l]*xs
            for (size_t i = 0; i < bx.size(); i++) {
                d_resid[l] += res_in[l][i] * dbx[i];
                d_x0[l] += xs[i] * dbx[i];
                dx0_total[i] += x0_l[l] * dbx[i];
            }
            for (size_t i = 0; i < dx.size(); i++) dx[i] = resid_l[l] * dbx[i];
        }
        // dxs = dx (grad wrt res_in[0]=xs) + dx0_total
        std::vector<real> dxs((size_t)bt * C);
        for (size_t i = 0; i < dxs.size(); i++) dxs[i] = dx[i] + dx0_total[i];
        // smear backward
        std::vector<real> demb_n((size_t)bt * C, 0);
        for (size_t i = 0; i < demb_n.size(); i++) demb_n[i] += dxs[i]; // emb_n[t] direct term
        for (int b = 0; b < B; b++) for (int t = 1; t < T; t++) {
            int n = b * T + t; real sg = sg_smear[n]; real gate = smear_lambda * sg;
            const real* dxsr = dxs.data() + (size_t)n * C;
            const real* ep = emb_n.data() + (size_t)(n - 1) * C;
            real* dprev = demb_n.data() + (size_t)(n - 1) * C;
            real dgate = 0;
            for (int c = 0; c < C; c++) { dgate += dxsr[c] * ep[c]; dprev[c] += gate * dxsr[c]; }
            d_smear_lambda += sg * dgate;
            real dsig = smear_lambda * dgate; real dz = dsig * sg * (1 - sg);
            const real* e = emb_n.data() + (size_t)n * C; real* dcur = demb_n.data() + (size_t)n * C;
            int SM = std::min((int)SMEAR_CH, C);
            for (int c = 0; c < SM; c++) { d_smear_w[c] += e[c] * dz; dcur[c] += smear_w[c] * dz; }
        }
        std::vector<real> demb((size_t)bt * C, 0);
        rmsnorm_bwd(demb.data(), demb_n.data(), emb.data(), rstd_emb.data(), bt, C);
        for (int n = 0; n < bt; n++) { real* dw = dwte.data() + (size_t)inputs[n] * C; const real* d = demb.data() + (size_t)n * C; for (int c = 0; c < C; c++) dw[c] += d[c]; }
    }

    // AdamW for one parameter group
    void adamw_group(std::vector<real>& p, std::vector<real>& g, std::vector<real>& m,
                     std::vector<real>& v, real lr, real b1, real b2, real eps, real wd) {
        real bc1 = (real)(1.0 - std::pow((double)b1, adam_t)), bc2 = (real)(1.0 - std::pow((double)b2, adam_t));
        for (size_t i = 0; i < p.size(); i++) {
            real gr = g[i];
            m[i] = b1 * m[i] + (1 - b1) * gr; v[i] = b2 * v[i] + (1 - b2) * gr * gr;
            real mh = m[i] / bc1, vh = v[i] / bc2;
            p[i] -= lr * (mh / ((real)std::sqrt((double)vh) + eps) + wd * p[i]);
        }
    }
    void muon_group(std::vector<real>& p, std::vector<real>& g, std::vector<real>& buf,
                    std::vector<real>& buf2, int rows, int cols, real base_lr, real mult) {
        real lr = base_lr * mult * (real)std::sqrt(std::max(1.0, (double)rows / cols));
        muon_update(p.data(), g.data(), buf.data(), buf2.data(), rows, cols, lr, (real)0.95, (real)0.0, 5, (real)0.9);
    }

    // One optimizer step. lr_mult is a schedule multiplier (e.g. cosine).
    // Base LRs follow nanochat, scaled by 1/sqrt(n_embd/768).
    void optimize(real lr_mult) {
        adam_t++;
        real scale = (real)std::sqrt(768.0 / C);
        // AdamW groups (embeddings / head / scalars / value-embeddings / gates)
        adamw_group(wte, dwte, m_wte, v_wte, (real)0.2 * scale * lr_mult, (real)0.8, (real)0.995, (real)1e-10, (real)0.001);
        adamw_group(lm_head, dlm, m_lm, v_lm, (real)0.004 * scale * lr_mult, (real)0.8, (real)0.96, (real)1e-10, (real)0.01);
        adamw_group(resid_l, d_resid, m_resid, v_resid, (real)0.005 * lr_mult, (real)0.8, (real)0.95, (real)1e-10, (real)0.05);
        adamw_group(x0_l, d_x0, m_x0, v_x0, (real)0.5 * lr_mult, (real)0.96, (real)0.95, (real)1e-10, (real)0.0);
        adamw_group(smear_w, d_smear_w, m_sw, v_sw, (real)0.2 * lr_mult, (real)0.8, (real)0.95, (real)1e-10, (real)0.0);
        { std::vector<real> P{smear_lambda}, G{d_smear_lambda}, M{m_sl}, Vv{v_sl};
          adamw_group(P, G, M, Vv, (real)0.2 * lr_mult, (real)0.8, (real)0.95, (real)1e-10, (real)0.0);
          smear_lambda = P[0]; m_sl = M[0]; v_sl = Vv[0]; }
        { std::vector<real> P{backout_lambda}, G{d_backout}, M{m_bo}, Vv{v_bo};
          adamw_group(P, G, M, Vv, (real)0.2 * lr_mult, (real)0.8, (real)0.95, (real)1e-10, (real)0.0);
          backout_lambda = P[0]; m_bo = M[0]; v_bo = Vv[0]; }
        for (auto& lp : layers) {
            if (lp.ve) {
                adamw_group(lp.ve_emb, lp.dve_emb, lp.m_ve, lp.v_ve, (real)0.1 * scale * lr_mult, (real)0.8, (real)0.995, (real)1e-10, (real)0.01);
                adamw_group(lp.ve_gate, lp.dve_gate, lp.m_veg, lp.v_veg, (real)0.2 * lr_mult, (real)0.8, (real)0.95, (real)1e-10, (real)0.0);
            }
        }
        // Matrix params: Muon. (cfg.use_muon reserved; matrices always use Muon here.)
        for (auto& lp : layers) {
            muon_group(lp.Wq, lp.dWq, lp.bufWq, lp.b2Wq, C, C, (real)0.02, lr_mult);
            muon_group(lp.Wk, lp.dWk, lp.bufWk, lp.b2Wk, Ckv, C, (real)0.02, lr_mult);
            muon_group(lp.Wv, lp.dWv, lp.bufWv, lp.b2Wv, Ckv, C, (real)0.02, lr_mult);
            muon_group(lp.Wo, lp.dWo, lp.bufWo, lp.b2Wo, C, C, (real)0.02, lr_mult);
            muon_group(lp.Wfc, lp.dWfc, lp.bufWfc, lp.b2Wfc, 4 * C, C, (real)0.02, lr_mult);
            muon_group(lp.Wproj, lp.dWproj, lp.bufWproj, lp.b2Wproj, C, 4 * C, (real)0.02, lr_mult);
        }
    }

    int argmax_logit(int n) const {
        const real* l = logits.data() + (size_t)n * V; int best = 0;
        for (int i = 1; i < V; i++) if (l[i] > l[best]) best = i; return best;
    }

    // ---- gradient / optimizer-state access (for grad-clip, accumulation, resume) ----
    // all gradient *vectors* (the two scalar grads d_smear_lambda/d_backout are
    // handled separately by grad_global_norm / scale_grads).
    std::vector<std::vector<real>*> grad_ptrs() {
        std::vector<std::vector<real>*> p = { &dwte, &dlm, &d_resid, &d_x0, &d_smear_w };
        for (auto& lp : layers) {
            p.insert(p.end(), { &lp.dWq, &lp.dWk, &lp.dWv, &lp.dWo, &lp.dWfc, &lp.dWproj });
            if (lp.ve) p.insert(p.end(), { &lp.dve_emb, &lp.dve_gate });
        }
        return p;
    }
    // all optimizer-state vectors in a fixed canonical order (Muon momentum +
    // NorMuon 2nd moment + AdamW moments), for checkpoint save/resume. The four
    // scalar moments (m_sl/v_sl/m_bo/v_bo) and adam_t are serialized separately.
    std::vector<std::vector<real>*> opt_state_ptrs() {
        std::vector<std::vector<real>*> p = { &m_wte, &v_wte, &m_lm, &v_lm, &m_resid,
                                              &v_resid, &m_x0, &v_x0, &m_sw, &v_sw };
        for (auto& lp : layers) {
            p.insert(p.end(), { &lp.bufWq, &lp.bufWk, &lp.bufWv, &lp.bufWo, &lp.bufWfc, &lp.bufWproj,
                                &lp.b2Wq, &lp.b2Wk, &lp.b2Wv, &lp.b2Wo, &lp.b2Wfc, &lp.b2Wproj });
            if (lp.ve) p.insert(p.end(), { &lp.m_ve, &lp.v_ve, &lp.m_veg, &lp.v_veg });
        }
        return p;
    }
    real grad_global_norm() {
        double s = (double)d_smear_lambda * d_smear_lambda + (double)d_backout * d_backout;
        for (auto* v : grad_ptrs()) for (real g : *v) s += (double)g * g;
        return (real)std::sqrt(s);
    }
    void scale_grads(real f) {
        d_smear_lambda *= f; d_backout *= f;
        for (auto* v : grad_ptrs()) for (real& g : *v) g *= f;
    }
};

} // namespace gpt

#endif // NANOCHAT_GPT_H
