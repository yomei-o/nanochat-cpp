// Character-level tokenizer (same idea as nanochat's toy runs use a BPE, but
// char-level keeps this port fully self-contained).
#ifndef NANOCHAT_TOKENIZER_H
#define NANOCHAT_TOKENIZER_H
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
namespace gpt {
struct CharTokenizer {
    std::vector<char> itos;
    std::unordered_map<char, int> stoi;
    void build_from_text(const std::string& t) {
        std::set<char> u(t.begin(), t.end());
        itos.assign(u.begin(), u.end());
        stoi.clear(); for (int i = 0; i < (int)itos.size(); i++) stoi[itos[i]] = i;
    }
    void set_vocab(const std::vector<char>& v) {
        itos = v; stoi.clear(); for (int i = 0; i < (int)itos.size(); i++) stoi[itos[i]] = i;
    }
    int vocab_size() const { return (int)itos.size(); }
    std::vector<int> encode(const std::string& s) const {
        std::vector<int> o; for (char c : s) { auto it = stoi.find(c); if (it != stoi.end()) o.push_back(it->second); } return o;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string o; for (int i : ids) if (i >= 0 && i < (int)itos.size()) o.push_back(itos[i]); return o;
    }
};
}
#endif
