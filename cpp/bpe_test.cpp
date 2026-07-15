// standalone test for bpe.h: train on input.txt, verify round-trip, print stats.
#include "bpe.h"
#include <cstdio>
#include <fstream>
#include <string>
using namespace gpt;
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "input.txt";
    int vocab = argc > 2 ? std::atoi(argv[2]) : 1024;
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("cannot open %s\n", path); return 1; }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::printf("corpus: %zu bytes\n", text.size());

    BPE bpe; bpe.train(text, vocab);
    std::printf("trained: %zu BPE tokens + %d special = %d vocab\n",
                bpe.vocab.size(), NANOCHAT_NUM_SPECIAL, bpe.n_vocab());
    std::printf("bos id = %d\n", bpe.bos_id);

    // round-trip on the whole corpus
    auto ids = bpe.encode(text);
    std::string back = bpe.decode(ids);
    std::printf("encoded %zu tokens (%.2f bytes/token, %.2fx vs chars)\n",
                ids.size(), (double)text.size() / ids.size(), (double)text.size() / ids.size());
    std::printf("round-trip exact: %s\n", back == text ? "YES" : "NO");
    if (back != text) {
        size_t k = 0; while (k < back.size() && k < text.size() && back[k] == text[k]) k++;
        std::printf("  first diff at byte %zu\n", k);
    }

    // a couple of small examples
    const char* ex[] = {"Hello world", "ROMEO: What's here?", "The year 2024, indeed."};
    for (auto* s : ex) {
        auto e = bpe.encode(s);
        std::printf("  encode(\"%s\") = %zu tokens, decode ok: %s\n",
                    s, e.size(), bpe.decode(e) == std::string(s) ? "YES" : "NO");
    }
    return back == text ? 0 : 1;
}
