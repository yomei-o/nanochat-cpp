# 例: 日本語で計算する LLM を依存なし C++ で作る

nanochat-cpp だけを使って、「**日本語で計算問題を投げると、電卓ツールを呼んで日本語で
答える**」小さな chat LLM を、**GPU も PyTorch も使わず CPU だけ**で作る手順です。
`train`（BPE 学習＋事前学習）→ `sft`（ツール使用を教える）→ `chat`（KV cache 生成＋
C++ 電卓）という流れで、すべてこのリポジトリの実行ファイル1つで完結します。

> このドキュメントは実際にこのマシン（4 CPU スレッド）で走らせた**実測**に基づきます。
> 末尾に「うまくいくこと／おもちゃ規模の限界」を正直にまとめています。要点を先に言うと、
> **ツール使用の"形式"は一瞬で学習できるが、"数字を正確に写す"のは小さいモデルには難しい**、
> という実 LLM の function calling にも通じる教訓が得られます。

---

## なぜ日本語が扱えるのか

`bpe.h` のトークナイザは **byte-level BPE**（tiktoken 方式）です。日本語の UTF-8
（漢字＝3バイト等）も単なるバイト列として扱い、頻出バイトペアを併合していくだけなので、
`encode → decode` は**バイト単位で完全一致**します。モデルはトークン id を並べるだけの
言語非依存な仕組みなので、英語でも算術でも日本語でも同じコードで動きます。

```bash
# BPE が日本語を完全往復できることの確認（bpe_test）
./bpe_test your_japanese_corpus.txt 2048
#  -> round-trip exact: YES
```

---

## Step 0: ビルド

```bash
cd cpp
g++ -O3 -ffast-math -fopenmp -std=c++17 -o nanochat main.cpp
```

---

## Step 1: 学習データを作る

「平文コーパス」（BPE とベース事前学習用）と「chat コーパス」（SFT 用）を作ります。
chat 側では assistant の答えに**ツールマーカー**を直接埋め込みます:
`<|python_start|>式<|python_end|><|output_start|>答え<|output_end|>答えは…です。`

```python
import random
random.seed(0)
ops = ["+", "-", "*"]
qtmpl = ["{e}はいくつですか？", "{e}を計算してください。", "{e}の答えを教えて。", "{e}は？"]
chat, plain = [], []
for _ in range(6000):
    a, b = random.randint(1, 99), random.randint(1, 99)
    e = f"{a}{random.choice(ops)}{b}"; res = eval(e)
    q = random.choice(qtmpl).format(e=e)
    ans = f"<|python_start|>{e}<|python_end|><|output_start|>{res}<|output_end|>答えは{res}です。"
    chat += [f"U: {q}", f"A: {ans}", ""]        # 空行で 1 会話を区切る
    plain.append(f"{q} {e}は{res}です。")
open("jc_chat.txt", "w", encoding="utf-8").write("\n".join(chat))
open("jc_plain.txt", "w", encoding="utf-8").write("\n".join(plain))
```

chat コーパスの一例:

```
U: 37+45はいくつですか？
A: <|python_start|>37+45<|python_end|><|output_start|>82<|output_end|>答えは82です。
```

`sft` は `U:` / `A:` の行を特殊トークンで整形し、**loss を assistant トークンのみ**に
かけます（質問部分は `target = -1` で無視）。テキスト中の `<|python_start|>` 等は
特殊 id として認識されます。

---

## Step 2: BPE ＋ ベース事前学習

まず平文コーパスで BPE を学習し（語彙に日本語と数字トークンが入る）、モデルを事前学習します。

```bash
./nanochat train jc_plain.txt --tokenizer bpe --vocab 512 \
    --steps 300 --layers 4 --embd 128 --heads 4 --kv-heads 2 --block 96 --batch 32 \
    --out base.bin
```

実測（このマシン）:

```
training BPE tokenizer (target vocab 512)...
  -> 503 BPE tokens + 9 special
nanochat-core: 4 layers, 128 embd, 4/2 heads (GQA), block 96, bpe vocab 512  (0.92M params)
encoded 62996 tokens
step     0 | train 6.24 | val 6.24
step   300 | train ~1.0 | ...
saved base.bin (step 300)
```

---

## Step 3: SFT（ツール使用を教える）

ベースモデルから、chat コーパスで教師ありファインチューニングします。

```bash
./nanochat sft jc_chat.txt --ckpt base.bin --steps 250 --batch 16 --out sft.bin
```

実測 — assistant-only loss が一瞬で下がり、**ツール使用の形式は完全に学習**されます:

```
sft finetune: 138989 tokens (90989 assistant), model 0.92M, block 96
step     0 | sft loss 9.86
step   125 | sft loss 0.41
step   250 | sft loss 0.29
saved sft.bin
```

---

## Step 4: 日本語で計算チャット

```bash
./nanochat chat sft.bin --message "37+45はいくつですか？" --temp 0.2
```

生成の流れ:
1. モデルが `<|python_start|>37+45<|python_end|>` を出力
2. **C++ の電卓**（`tool.h`）が式を評価 → `[calc 37+45 = 82]`
3. `<|output_start|>82<|output_end|>` を文脈に注入
4. モデルが「答えは82です。」と続ける

対話モード（引数なし）なら:

```bash
./nanochat chat sft.bin
you> 8*9を計算してください。
bot> [calc 8*9 = 72]答えは72です。
```

---

## うまくいくこと / おもちゃ規模の限界（正直な観察）

このパイプラインは **end-to-end で機能します**:

- ✅ 日本語のトークナイズ（byte-level BPE、往復完全一致）
- ✅ SFT の assistant-only loss（形式を一瞬で学習、loss → 0.29）
- ✅ **C++ 電卓ツールの実行**（式を正しく評価し、答えを文脈に戻す）
- ✅ 日本語での入出力

一方で、**この規模（0.9M パラメータ・数百ステップ・CPU）では「数字を質問から式へ
正確に写すこと」が不安定**です。実際の出力例（`--temp 0.2`）:

```
Q: 37+45はいくつですか？  ->  [calc 26+40 = 66]答えは66です。   ← 数字が質問と不一致
Q: 76-89は？            ->  [calc 76-50 = 26]答えは26です。   ← 76 と - は合うが 89 が誤り
Q: 54*3は？             ->  [calc 3*36 = 108]答えは108です。  ← 3 と * はあるが位置がずれる
```

### なぜか

- ツール使用の**"形式"**（`<|python_start|>` を出す → 式らしきもの → `<|python_end|>` →
  日本語で答える）は低エントロピーなテンプレートなので、ほぼ即座に学習されます。
- しかし**"中身"**、すなわち質問中の数字を式へ**位置正確にコピー**するには、attention が
  いわゆる **induction head**（「あの位置のトークンをここへ写す」機構）を形成する必要が
  あり、これは形式学習よりずっと多くの規模・データ・ステップを要します。
- 事実、**ベースモデル単体の平文 `sample` でもコピーできません**（「8*9を計算してください。」に
  続けて別の数字を生成する）。つまり chat 実装のバグではなく、モデルの能力限界です。
- 記号演算子（`+`）でも語演算子（`たす`）でも同様で、1桁の数字でも起きます。
  （英語版デモ `What is 37 + 45?` がうまく見えたのは、式が質問のほぼ連続部分文字列で
  短距離コピーだったこと＋少数例だったことによる面が大きいです。）

### 改善するには

- **モデルを大きく**（`--embd` / `--layers` を増やす）— コピー回路の容量を上げる
- **SFT を長く / データを増やす** — induction head が形成されるまで学習
- **カリキュラム**（1桁 → 2桁）や**数字を単一トークン化**して写経を易しくする
- 本気でやるなら本家 nanochat（GPU・大規模）へ

### 教訓

**「ツール使用の"形"はほぼタダで学べる。難しいのは"中身"の正確さ」** — これは
production の LLM function calling でも本質的に同じで、フォーマット遵守と、引数を正確に
埋めることは別の難しさを持ちます。0.9M のおもちゃモデルで、ソースを全部追える形で
その境界を観察できるのが、この依存なし C++ 実装の面白いところです。

---

（関連: [`README.ja.md`](README.ja.md) の「Chat: SFT ＋ 電卓ツール」節。英語版は
[`README.md`](README.md)。）
