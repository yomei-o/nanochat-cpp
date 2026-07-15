[English](README.md) | [日本語](README.ja.md)

# nanochat — C++ port

A dependency-free **standard C++17** port of the model + optimizer from Andrej
Karpathy's [nanochat](https://github.com/karpathy/nanochat). It implements
nanochat's modern transformer — **forward AND backward** — plus the **Muon**
optimizer, so you can train a small language model on **CPU** and sample from
it. No PyTorch, no Rust, no CUDA. OpenMP is used if present (optional).

Ports the core of [`nanochat/gpt.py`](../nanochat/gpt.py) and
[`nanochat/optim.py`](../nanochat/optim.py).

## Can nanochat train an LLM on CPU?

**A toy one, yes; a capable one, no.** The real nanochat trains a GPT-2-grade
model (depth 20–26) on an **8×H100** node in ~2 hours (~4×10¹⁹ FLOPs). A CPU is
~5–6 orders of magnitude slower → training the real model on CPU would take on
the order of a **decade**. This port lets you train and understand nanochat's
architecture on CPU at toy scale, dependency-free — not a way to get a capable
chatbot. (nanochat itself ships `runs/runcpu.sh`, an explicitly educational
depth-6 demo.)

## What's implemented

Architecture (nanochat's `gpt.py`):
- **Rotary position embeddings (RoPE)**, no learned positional embeddings
- **RMSNorm** with no learnable params (norm after embedding, pre-norm blocks, final norm)
- **QK-norm** (RMSNorm on per-head q,k) with the 1.2 sharpening scale
- **ReLU² MLP**, **Grouped-Query Attention (GQA)**, untied embeddings, no bias
- **logit softcap** `15·tanh(logits/15)`
- **per-layer learnable resid / x0 scalars** (blend the initial embedding back in)
- **"smear"**: mix the previous token's embedding via a learned gate
- **"backout"**: subtract a mid-layer residual before the final norm
- **value embeddings (ResFormer)** on alternating layers, with a per-head input gate

- **sliding-window attention** (per-layer window pattern, e.g. `SSSL`; last layer
  full context)
- **vocab padding** to a multiple of 64 (a GPU-alignment detail; no effect on
  CPU results)

Optimizer (nanochat's `optim.py`) — the **full Muon**:
- Nesterov momentum → **MuonEq row equilibration** → Polar-Express
  orthogonalization → **Muon+ Frobenius renorm** → **NorMuon variance reduction**
  → per-matrix LR scaling → cautious weight decay
- **AdamW** for embeddings / lm_head / value-embeddings / gates / scalars, with
  nanochat's per-group learning rates (scaled by 1/√(n_embd/768)) and betas
- warmup + cosine learning-rate schedule

All forward/backward paths are verified against numerical gradients (below).

## Tokenizer

Two tokenizers, both dependency-free:

- **char-level** (default) — simplest, fully self-contained.
- **byte-level BPE** (`--tokenizer bpe`) — a self-contained port of nanochat's
  tokenizer (nanochat trains a Rust BPE + tiktoken; here we **train our own**
  byte-level BPE on the corpus in pure C++). Same tiktoken-style greedy merge and
  GPT-4-ish split pattern, plus nanochat's special tokens (`<|bos|>`,
  `<|user_start|>`, …, `<|python_start|>`, …). Because training and inference
  share one pre-tokenizer it is self-consistent (round-trip is exact); it won't
  match a real tiktoken vocab bit-for-bit, but nanochat trains its own anyway.
  The trained vocab is baked into the checkpoint. Everything else in `gpt.py` /
  `optim.py` is implemented.

## Files

| File | Purpose |
|---|---|
| `gpt.h` | Model (all ops fwd+bwd), value embeddings, smear, backout, Muon + AdamW |
| `tokenizer.h` | Character-level tokenizer |
| `bpe.h` | Self-contained byte-level BPE (train + encode/decode + special tokens) |
| `tool.h` | Calculator tool (pure-C++ arithmetic evaluator) for chat tool use |
| `main.cpp` | `train` / `sample` / `sft` / `chat` driver |
| `test_grad.cpp` | Gradient check (double precision) |
| `bpe_test.cpp` | BPE self-test (train on a corpus, verify exact round-trip) |
| `CMakeLists.txt` | Cross-platform build (auto-detects OpenMP) |

## Build

```bash
cmake -S . -B build && cmake --build build --config Release
# or directly:
g++ -O3 -ffast-math -fopenmp -std=c++17 -o nanochat main.cpp
g++ -O3 -ffast-math -DGPT_USE_DOUBLE -std=c++17 -o nanochat_gradcheck test_grad.cpp
```
In Visual Studio 2022: Open Folder (CMake auto-detected) → `x64-Release` → Build All.

## Train & sample (CPU)

```bash
# input.txt (tiny-shakespeare) is included
nanochat train input.txt --steps 3000 --layers 6 --embd 384 --heads 6 --kv-heads 2 --block 128 --out ckpt.bin
nanochat sample ckpt.bin --tokens 500 --temp 0.8 --topk 40 --prompt "ROMEO:"
```
`train` options: `--steps --lr(global LR multiplier) --batch --block --layers
--embd --heads --kv-heads --out --eval-every --seed --init --ckpt --grad-clip
--grad-accum --tokenizer --vocab`. `--heads` must be divisible by `--kv-heads`
(GQA). Per-group base LRs follow nanochat; `--lr` scales them all.

Train with the **BPE** tokenizer (trains it on the corpus first, then bakes it
into the checkpoint):

```bash
nanochat train input.txt --tokenizer bpe --vocab 2048 --steps 3000 --layers 6 --embd 384 --out ckpt.bin
nanochat sample ckpt.bin --prompt "ROMEO:" --tokens 200
```

### Resume / fine-tune, grad clip, grad accumulation

```bash
# resume: continue a run — restores weights, Muon+AdamW momentum and the step.
nanochat train input.txt --init resume --ckpt ckpt.bin --steps 3000 --out ckpt.bin
# finetune: keep the weights, fresh optimiser + step counter, new data.
nanochat train other.txt --init finetune --ckpt ckpt.bin --steps 300 --out ft.bin
```

- `--init scratch` (default) / `resume` (params **+** optimizer state + step) /
  `finetune` (params only; optimizer + step reset). Checkpoints are **NCp2**
  (params + Muon momentum + NorMuon 2nd moments + AdamW moments + step); old
  **NCp1** files (params only) still load for `sample`.
- `--grad-clip F` clips gradients to global L2 norm `F` (`0` = off, the default,
  matching upstream nanochat which doesn't clip).
- `--grad-accum N` sums grads over `N` micro-batches per optimizer step
  (effective batch `--batch × N`). On `resume` the LR schedule continues at the
  restored global step.

## Chat: SFT + calculator tool (BPE models)

The port can also turn a base model into a **chat** model and talk to it — fully
dependency-free. Needs a **BPE** model (the chat/tool special tokens live in the
tokenizer): `<|bos|>`, `<|user_start|>`/`<|user_end|>`,
`<|assistant_start|>`/`<|assistant_end|>`, and the tool markers
`<|python_start|>`/`<|python_end|>`, `<|output_start|>`/`<|output_end|>`.

```bash
# 1) base-train a BPE model, 2) SFT on a chat corpus, 3) chat with it
nanochat train corpus.txt --tokenizer bpe --vocab 512 --out base.bin
nanochat sft chat.txt --ckpt base.bin --steps 200 --out sft.bin
nanochat chat sft.bin --message "What is 37 + 45?"
```

- **`sft`** trains on a simple chat corpus — lines `U: ...` / `A: ...`, a blank
  line ends a conversation — rendered with the special tokens. **Loss is
  computed only on the assistant tokens** (`target = -1` elsewhere). Assistant
  text may embed the tool markers literally; they are encoded as their special
  ids. `--init finetune` (default) starts a fresh optimizer; `resume` continues.
- **`chat`** renders the conversation, generates the reply with the **KV cache**,
  streams it, and stops at `<|assistant_end|>`. `--message` does one turn;
  otherwise it's an interactive REPL. `--temp --topk --max-tokens`.
- **Calculator tool** (`tool.h`): when the model emits
  `<|python_start|>EXPR<|python_end|>`, a pure-C++ arithmetic evaluator computes
  `EXPR` and feeds it back as `<|output_start|>RESULT<|output_end|>`. This is
  exactly nanochat's "python" tool, which only allows arithmetic (no builtins,
  no `**`) — so no Python is needed. (True arbitrary-code tools / HumanEval
  would need a real interpreter and are intentionally out of scope.)

Worked end-to-end here: SFT on synthetic arithmetic Q&A (assistant-only loss
10.98 → 0.29) taught a 0.9M-param model to call the tool, and the C++ calculator
answers correctly:

```
you> What is 37 + 45?
bot> [calc 37+45 = 82]The answer is 82.
```

The Muon optimizer + architecture refinements converge markedly faster than
plain AdamW on the bare architecture. Measured here (4 layers / 96 embd / GQA
4:2, 4 CPU threads, 300 steps):

| | step 0 | step 300 val loss |
|---|---|---|
| bare architecture + AdamW | 4.17 | 2.44 |
| **full nanochat (Muon + VE + smear + backout + …)** | 4.17 | **~1.9** |

(NorMuon variance reduction / MuonEq / sliding-window are large-scale
refinements; at this toy scale they're within noise but are implemented and
gradient-checked.)

## Verification (run on this machine)

| Check | How | Result |
|---|---|---|
| Backprop correctness | `nanochat_gradcheck` (analytic vs numerical, double) | **PASS**, max rel err ~3e-7 across every weight tensor — incl. value embeddings, ve-gate, smear, backout, resid/x0, GQA, RoPE, QK-norm, ReLU², softcap |
| Muon + full model training | `nanochat train` on tiny-shakespeare | loss 4.17 → 1.87, coherent Shakespeare-style text |

## Notes

- Compute type is `float`; `-DGPT_USE_DOUBLE` builds the gradient-check target.
- KV cache: generation caches each layer's QK-normed key and gated value per
  position (`GPT::forward_one`, `KVCache`), so each new token is one
  O(1)-context step — verified **bit-identical** to the full `forward()` across
  every position (RoPE, GQA, sliding-window, value-embeddings, smear and backout
  all included). Measured ~**2.2× faster** than recomputing (6L/384/GQA 6:2, 4
  CPU threads; the gain grows with model size and sequence length). It applies
  within the context window; beyond `sequence_len` it falls back to the
  recompute-with-sliding path.
- The RoPE cache is sized to `sequence_len`; generation crops context to it.
