[English](README.md) | [日本語](README.ja.md)

# nanochat — C++ 移植版

Andrej Karpathy の [nanochat](https://github.com/karpathy/nanochat) の
モデル＋オプティマイザを、依存ライブラリなしの **標準 C++17** だけで移植したものです。
nanochat のモダンな Transformer を **forward だけでなく backward まで** 実装し、
さらに **Muon** オプティマイザも実装しているので、小さな言語モデルを **CPU** で学習して
サンプリングできます。PyTorch も Rust も CUDA も不要。OpenMP は存在すれば使います（任意）。

[`nanochat/gpt.py`](../nanochat/gpt.py) と [`nanochat/optim.py`](../nanochat/optim.py)
の中核を移植しています。

## nanochat は CPU で LLM を学習できる？

**おもちゃ規模なら yes、実用規模は no。** 本物の nanochat は GPT-2 級のモデル（depth 20〜26）を
**8×H100** ノードで約2時間（約 4×10¹⁹ FLOPs）学習します。CPU はおよそ 5〜6 桁遅いので、
本物のモデルを CPU で学習すると **10年** 規模の時間がかかります。この移植版は、nanochat の
アーキテクチャを CPU 上・おもちゃ規模・依存なしで学習して理解するためのものです（実用的な
チャットボットを得る手段ではありません）。nanochat 自身も `runs/runcpu.sh` という明示的に
教育用の depth-6 デモを同梱しています。

## 実装内容

アーキテクチャ（nanochat の `gpt.py`）:
- **回転位置埋め込み（RoPE）**、学習型の位置埋め込みは無し
- **RMSNorm**（学習パラメータ無し。埋め込み後・各ブロック前・最終の正規化）
- **QK-norm**（ヘッドごとの q,k に RMSNorm）＋ 1.2 のシャープ化スケール
- **ReLU² MLP**、**Grouped-Query Attention（GQA）**、埋め込み非共有、bias 無し
- **logit softcap** `15·tanh(logits/15)`
- **層ごとの学習可能 resid / x0 スカラー**（初期埋め込みをブレンドして戻す）
- **"smear"**: 前トークンの埋め込みを学習ゲートで混ぜる
- **"backout"**: 最終正規化の前に中間層の残差を差し引く
- **value embeddings（ResFormer）** を交互の層に、ヘッドごとの入力ゲート付きで

- **sliding-window attention**（層ごとのウィンドウパターン、例 `SSSL`。最終層は全文脈）
- **語彙パディング** を 64 の倍数へ（GPU アライメントの都合。CPU の結果には影響なし）

オプティマイザ（nanochat の `optim.py`）— **フル Muon**:
- Nesterov モーメンタム → **MuonEq 行イコライゼーション** → Polar-Express 直交化 →
  **Muon+ Frobenius リノーム** → **NorMuon 分散低減** → 行列ごとの LR スケーリング →
  cautious weight decay
- 埋め込み / lm_head / value-embeddings / ゲート / スカラーには **AdamW**。nanochat の
  グループ別学習率（1/√(n_embd/768) でスケール）と betas を使用
- warmup ＋ cosine 学習率スケジュール

すべての forward/backward 経路は数値微分と照合して検証済みです（下記）。

## トークナイザ

依存なしのトークナイザが2種類:

- **char レベル**（デフォルト）— 最もシンプルで完全自己完結。
- **byte-level BPE**（`--tokenizer bpe`）— nanochat のトークナイザ（本家は Rust BPE + tiktoken）を
  自己完結で移植したもの。ここでは純 C++ で **コーパスから自前の** byte-level BPE を学習します。
  tiktoken 方式の greedy merge と GPT-4 系の split パターン、そして nanochat の特殊トークン
  （`<|bos|>`、`<|user_start|>`、…、`<|python_start|>`、…）を備えます。学習と推論で同じ
  前トークナイザを共有するため自己整合的（往復は完全一致）で、本物の tiktoken 語彙と
  ビット単位で一致はしませんが、nanochat も自前学習なので問題ありません。学習した語彙は
  チェックポイントに埋め込まれます。`gpt.py` / `optim.py` のそれ以外はすべて実装済みです。
  **byte-level** なので **任意の UTF-8** テキストを往復完全一致で扱え、非ラテン文字でも学習
  できます。例えば日本語も end-to-end で動作します（`train … --tokenizer bpe` → `sample`/`chat`）。
  モデルは言語非依存です。

## ファイル一覧

| ファイル | 役割 |
|---|---|
| `gpt.h` | モデル（全演算 fwd+bwd）、value embeddings、smear、backout、Muon + AdamW |
| `tokenizer.h` | 文字レベルトークナイザ |
| `bpe.h` | 自己完結の byte-level BPE（学習＋encode/decode＋特殊トークン） |
| `tool.h` | 電卓ツール（純 C++ の算術評価器）— chat のツール使用向け |
| `main.cpp` | `train` / `sample` / `sft` / `chat` ドライバ |
| `test_grad.cpp` | 勾配チェック（倍精度） |
| `bpe_test.cpp` | BPE 自己テスト（コーパスで学習し完全往復を検証） |
| `CMakeLists.txt` | クロスプラットフォームビルド（OpenMP 自動検出） |

## ビルド

```bash
cmake -S . -B build && cmake --build build --config Release
# または直接:
g++ -O3 -ffast-math -fopenmp -std=c++17 -o nanochat main.cpp
g++ -O3 -ffast-math -DGPT_USE_DOUBLE -std=c++17 -o nanochat_gradcheck test_grad.cpp
```
Visual Studio 2022 では: フォルダーを開く（CMake 自動検出）→ `x64-Release` → すべてビルド。

## 学習＆サンプリング（CPU）

```bash
# input.txt（tiny-shakespeare）は同梱
nanochat train input.txt --steps 3000 --layers 6 --embd 384 --heads 6 --kv-heads 2 --block 128 --out ckpt.bin
nanochat sample ckpt.bin --tokens 500 --temp 0.8 --topk 40 --prompt "ROMEO:"
```
`train` オプション: `--steps --lr(グローバル LR 倍率) --batch --block --layers
--embd --heads --kv-heads --out --eval-every --seed --init --ckpt --grad-clip
--grad-accum --tokenizer --vocab`。`--heads` は `--kv-heads` で割り切れる必要があります
（GQA）。グループ別の基準 LR は nanochat に準拠し、`--lr` が全体を倍率でスケールします。

**BPE** トークナイザで学習する（先にコーパスで学習し、チェックポイントに埋め込む）:

```bash
nanochat train input.txt --tokenizer bpe --vocab 2048 --steps 3000 --layers 6 --embd 384 --out ckpt.bin
nanochat sample ckpt.bin --prompt "ROMEO:" --tokens 200
```

### 再開（resume）／ファインチューニング、勾配クリップ、勾配累積

```bash
# resume: 続きから学習。重み・Muon+AdamW のモーメンタム・ステップ数を復元。
nanochat train input.txt --init resume --ckpt ckpt.bin --steps 3000 --out ckpt.bin
# finetune: 重みは保持、オプティマイザ＋ステップはリセット、別データで学習。
nanochat train other.txt --init finetune --ckpt ckpt.bin --steps 300 --out ft.bin
```

- `--init scratch`（デフォルト）／`resume`（パラメータ **＋** オプティマイザ状態＋ステップ）／
  `finetune`（パラメータのみ。オプティマイザ＋ステップはリセット）。チェックポイントは
  **NCp2**（パラメータ + Muon モーメンタム + NorMuon 2次モーメント + AdamW モーメント +
  ステップ）。旧 **NCp1**（パラメータのみ）も `sample` で読めます。
- `--grad-clip F` は勾配をグローバル L2 ノルム `F` にクリップ（`0`=無効。デフォルト。本家
  nanochat がクリップしないのに合わせています）。
- `--grad-accum N` は `N` 個のマイクロバッチで勾配を累積してから1回最適化（実効バッチ
  `--batch × N`）。`resume` 時は LR スケジュールが復元されたグローバルステップから継続します。

## Chat: SFT ＋ 電卓ツール（BPE モデル）

この移植版は、ベースモデルを **chat** モデルに仕立てて対話することもできます — 完全に
依存なしで。**BPE** モデルが必要です（chat/ツールの特殊トークンがトークナイザに含まれる）:
`<|bos|>`、`<|user_start|>`/`<|user_end|>`、`<|assistant_start|>`/`<|assistant_end|>`、
そしてツールマーカー `<|python_start|>`/`<|python_end|>`、`<|output_start|>`/`<|output_end|>`。

```bash
# 1) BPE ベース学習、2) chat コーパスで SFT、3) 対話
nanochat train corpus.txt --tokenizer bpe --vocab 512 --out base.bin
nanochat sft chat.txt --ckpt base.bin --steps 200 --out sft.bin
nanochat chat sft.bin --message "What is 37 + 45?"
```

- **`sft`** はシンプルな chat コーパス（`U: ...` / `A: ...` の行、空行で会話を区切る）を
  特殊トークンで整形して学習します。**loss は assistant トークンのみ**（それ以外は
  `target = -1`）。assistant テキストにツールマーカーを直接埋め込め、それらは特殊 id として
  エンコードされます。`--init finetune`（デフォルト）は新しいオプティマイザで開始、`resume`
  は継続。
- **`chat`** は会話を整形し、**KV cache** で応答を生成してストリーム表示し、
  `<|assistant_end|>` で停止します。`--message` は1ターン、無指定なら対話 REPL。
  `--temp --topk --max-tokens`。
- **電卓ツール**（`tool.h`）: モデルが `<|python_start|>EXPR<|python_end|>` を出力すると、
  純 C++ の算術評価器が `EXPR` を計算し、`<|output_start|>RESULT<|output_end|>` として
  返します。これはまさに nanochat の "python" ツール（算術のみ、builtins 無し、`**` 無し）で、
  Python は不要です。（任意コード実行ツール／HumanEval は本物のインタプリタが必要なので
  意図的に対象外。）

ここでは end-to-end で動作確認済み: 合成算術QAでの SFT（assistant のみの loss 10.98 → 0.29）で
0.9M パラメータのモデルがツール呼び出しを学習し、C++ 電卓が正答します:

```
you> What is 37 + 45?
bot> [calc 37+45 = 82]The answer is 82.
```

Muon オプティマイザ＋アーキテクチャの改良により、素のアーキテクチャに単純な AdamW を
使うより明確に速く収束します。実測（4 層 / 96 embd / GQA 4:2、4 CPU スレッド、300 ステップ）:

| | step 0 | step 300 val loss |
|---|---|---|
| 素のアーキテクチャ + AdamW | 4.17 | 2.44 |
| **フル nanochat（Muon + VE + smear + backout + …）** | 4.17 | **約1.9** |

（NorMuon 分散低減 / MuonEq / sliding-window は大規模向けの改良で、このおもちゃ規模では
ノイズの範囲内ですが、実装済みかつ勾配チェック済みです。）

## 検証（このマシンで実施）

| チェック | 方法 | 結果 |
|---|---|---|
| backprop の正しさ | `nanochat_gradcheck`（解析 vs 数値、倍精度） | **PASS**、全重みテンソルで最大相対誤差 約3e-7 — value embeddings・ve-gate・smear・backout・resid/x0・GQA・RoPE・QK-norm・ReLU²・softcap を含む |
| Muon ＋ フルモデル学習 | `nanochat train`（tiny-shakespeare） | loss 4.17 → 1.87、一貫したシェイクスピア調テキスト |

## メモ

- 計算型は `float`。`-DGPT_USE_DOUBLE` で勾配チェックのターゲットをビルド。
- KV キャッシュ: 生成は層ごとに QK 正規化済みの key とゲート済みの value を位置ごとに
  キャッシュし（`GPT::forward_one`、`KVCache`）、新しいトークンごとに O(1) 文脈のステップで
  済みます。フル `forward()` と全位置で **ビット単位一致**を検証済み（RoPE・GQA・
  sliding-window・value-embeddings・smear・backout すべて込み）。再計算に対して実測で約
  **2.2倍高速**（6L/384/GQA 6:2、4 CPU スレッド。モデルが大きく系列が長いほど差は拡大）。
  文脈ウィンドウ内で有効で、`sequence_len` を超えると再計算＋スライドの経路にフォールバック。
- RoPE キャッシュは `sequence_len` サイズ。生成はそこに文脈をクロップします。
