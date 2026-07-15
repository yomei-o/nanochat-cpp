// ---------------------------------------------------------------------------
//  bpe.h  --  self-contained byte-level BPE tokenizer for the nanochat C++ port.
//
//  Mirrors nanochat's tokenizer (rustbpe train + tiktoken inference) but is
//  fully dependency-free: it TRAINS its own byte-level BPE on a corpus and does
//  encode/decode in pure C++17. Because training and inference share the same
//  pre-tokenizer, the tokenizer is self-consistent (it won't match a real
//  tiktoken vocab bit-for-bit, but nanochat trains its own tokenizer anyway).
//
//  - byte-level: base vocab is the 256 bytes; merges get ids 256, 257, ...
//  - tiktoken-style encode: greedily merge the adjacent pair with the lowest
//    merge rank (= token id), matching the order merges were learned.
//  - special tokens (<|bos|>, chat/tool markers) live above the BPE vocab.
// ---------------------------------------------------------------------------
#ifndef NANOCHAT_BPE_H
#define NANOCHAT_BPE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cctype>
#include <fstream>
#include <algorithm>

namespace gpt {

// nanochat's special tokens, in order (ids assigned above the BPE vocab).
static const char* NANOCHAT_SPECIALS[] = {
    "<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>",
    "<|assistant_end|>", "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>"
};
enum { NANOCHAT_NUM_SPECIAL = 9 };

struct BPE {
    // byte-sequence (as std::string of raw bytes) -> token id (== merge rank)
    std::unordered_map<std::string, int> ranks;
    std::vector<std::string> vocab;                 // id -> byte-sequence
    std::unordered_map<std::string, int> special;   // "<|bos|>" -> id
    int bos_id = -1;

    int n_vocab() const { return (int)vocab.size() + (int)special.size(); }
    int special_id(const std::string& s) const {
        auto it = special.find(s); return it == special.end() ? -1 : it->second;
    }

    // ---- ASCII-faithful approximation of nanochat's GPT-4 split pattern -----
    // Always advances the cursor (never loops). Categories: contraction,
    // (optional single non-alnum lead +) letters, 1-2 digits, punctuation run,
    // whitespace run.
    static bool isL(unsigned char c) { return std::isalpha(c) != 0; }
    static bool isN(unsigned char c) { return std::isdigit(c) != 0; }
    static bool isS(unsigned char c) { return std::isspace(c) != 0; }

    static std::vector<std::string> pretokenize(const std::string& t) {
        std::vector<std::string> out;
        size_t i = 0, n = t.size();
        while (i < n) {
            unsigned char c = t[i];
            // contractions: '(s|d|m|t|ll|ve|re), case-insensitive
            if (c == '\'' && i + 1 < n) {
                auto lo = [](char x){ return (char)std::tolower((unsigned char)x); };
                char a = lo(t[i + 1]); char b = i + 2 < n ? lo(t[i + 2]) : 0;
                if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') || (a == 'r' && b == 'e')) {
                    out.push_back(t.substr(i, 3)); i += 3; continue;
                }
                if (a == 's' || a == 'd' || a == 'm' || a == 't') { out.push_back(t.substr(i, 2)); i += 2; continue; }
            }
            // optional single non-letter/non-digit/non-newline lead, then letters
            size_t start = i, j = i;
            bool lead = (!isL(c) && !isN(c) && c != '\n' && c != '\r' && !isS(c));
            if (lead && j + 1 < n && isL((unsigned char)t[j + 1])) j++;   // consume the lead
            else if (c == ' ' && j + 1 < n && isL((unsigned char)t[j + 1])) { lead = true; j++; }
            if (j < n && isL((unsigned char)t[j])) {
                size_t k = j; while (k < n && isL((unsigned char)t[k])) k++;
                out.push_back(t.substr(start, k - start)); i = k; continue;
            }
            // 1-2 digits (optional single leading space)
            size_t dj = i; bool dlead = false;
            if (c == ' ' && dj + 1 < n && isN((unsigned char)t[dj + 1])) { dlead = true; dj++; }
            if (dj < n && isN((unsigned char)t[dj])) {
                size_t k = dj, cnt = 0; while (k < n && isN((unsigned char)t[k]) && cnt < 2) { k++; cnt++; }
                out.push_back(t.substr(dlead ? i : dj, k - (dlead ? i : dj))); i = k; continue;
            }
            // punctuation run: optional leading space, then non-space/non-alnum, then newlines
            size_t pj = i; bool plead = (c == ' ');
            size_t ps = pj + (plead ? 1 : 0);
            if (ps < n && !isS((unsigned char)t[ps]) && !isL((unsigned char)t[ps]) && !isN((unsigned char)t[ps])) {
                size_t k = ps; while (k < n && !isS((unsigned char)t[k]) && !isL((unsigned char)t[k]) && !isN((unsigned char)t[k])) k++;
                while (k < n && (t[k] == '\r' || t[k] == '\n')) k++;
                out.push_back(t.substr(i, k - i)); i = k; continue;
            }
            // whitespace run (peel a trailing single space to lead the next token)
            if (isS(c)) {
                size_t k = i; while (k < n && isS((unsigned char)t[k])) k++;
                if (k < n && k - i > 1 && t[k - 1] == ' ') { out.push_back(t.substr(i, k - i - 1)); i = k - 1; continue; }
                out.push_back(t.substr(i, k - i)); i = k; continue;
            }
            // fallback single char (guarantees progress)
            out.push_back(std::string(1, (char)c)); i++;
        }
        return out;
    }

    // greedily merge the adjacent pair with the lowest rank (tiktoken order).
    std::vector<std::string> bpe_merge(const std::string& piece) const {
        std::vector<std::string> parts;
        parts.reserve(piece.size());
        for (char ch : piece) parts.push_back(std::string(1, ch));
        if (parts.size() <= 1) return parts;
        while (true) {
            int best_rank = 0x7fffffff; size_t best = SIZE_MAX;
            for (size_t k = 0; k + 1 < parts.size(); k++) {
                auto it = ranks.find(parts[k] + parts[k + 1]);
                if (it != ranks.end() && it->second < best_rank) { best_rank = it->second; best = k; }
            }
            if (best == SIZE_MAX) break;
            parts[best] += parts[best + 1];
            parts.erase(parts.begin() + best + 1);
        }
        return parts;
    }

    // ---- training: learn merges up to `vocab_size` (incl. specials) ---------
    void train(const std::string& text, int vocab_size) {
        int n_special = NANOCHAT_NUM_SPECIAL;
        int target = vocab_size - n_special;
        if (target < 256) target = 256;
        // base vocab: 256 bytes
        ranks.clear(); vocab.assign(256, "");
        for (int b = 0; b < 256; b++) { std::string s(1, (char)b); ranks[s] = b; vocab[b] = s; }
        // pre-tokenize once; represent each word as a list of token-ids
        std::vector<std::vector<int>> words;
        std::vector<int> wfreq;
        {
            std::unordered_map<std::string, int> wmap;
            for (auto& p : pretokenize(text)) {
                auto it = wmap.find(p);
                if (it == wmap.end()) { wmap[p] = (int)words.size(); std::vector<int> w; for (unsigned char ch : p) w.push_back(ch); words.push_back(w); wfreq.push_back(1); }
                else wfreq[it->second]++;
            }
        }
        // iteratively merge the most frequent adjacent id-pair
        while ((int)vocab.size() < target) {
            std::unordered_map<long long, long long> pairc;
            auto key = [](int a, int b){ return ((long long)a << 32) | (unsigned int)b; };
            for (size_t w = 0; w < words.size(); w++) {
                auto& ws = words[w]; int f = wfreq[w];
                for (size_t k = 0; k + 1 < ws.size(); k++) pairc[key(ws[k], ws[k + 1])] += f;
            }
            if (pairc.empty()) break;
            long long bestk = 0, bestc = -1;
            for (auto& kv : pairc) if (kv.second > bestc || (kv.second == bestc && kv.first < bestk)) { bestc = kv.second; bestk = kv.first; }
            int a = (int)(bestk >> 32), b = (int)(bestk & 0xffffffff);
            int nid = (int)vocab.size();
            std::string merged = vocab[a] + vocab[b];
            ranks[merged] = nid; vocab.push_back(merged);
            // apply the merge in every word
            for (auto& ws : words) {
                std::vector<int> nw; nw.reserve(ws.size());
                for (size_t k = 0; k < ws.size(); ) {
                    if (k + 1 < ws.size() && ws[k] == a && ws[k + 1] == b) { nw.push_back(nid); k += 2; }
                    else { nw.push_back(ws[k]); k++; }
                }
                ws.swap(nw);
            }
        }
        // assign special-token ids above the BPE vocab
        special.clear();
        for (int s = 0; s < n_special; s++) special[NANOCHAT_SPECIALS[s]] = (int)vocab.size() + s;
        bos_id = special["<|bos|>"];
    }

    // ---- encode / decode ----------------------------------------------------
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        for (auto& p : pretokenize(text))
            for (auto& tok : bpe_merge(p)) { auto it = ranks.find(tok); if (it != ranks.end()) ids.push_back(it->second); }
        return ids;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        int base = (int)vocab.size();
        for (int id : ids) {
            if (id >= 0 && id < base) out += vocab[id];
            // special tokens decode to their literal text (useful for debugging)
            else for (auto& kv : special) if (kv.second == id) { out += kv.first; break; }
        }
        return out;
    }

    // ---- save / load (compact binary) --------------------------------------
    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary); if (!f) return false;
        int32_t nv = (int32_t)vocab.size(); f.write((char*)&nv, 4);
        for (auto& v : vocab) { int32_t l = (int32_t)v.size(); f.write((char*)&l, 4); f.write(v.data(), l); }
        return (bool)f;
    }
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary); if (!f) return false;
        int32_t nv = 0; f.read((char*)&nv, 4); if (!f) return false;
        vocab.assign(nv, ""); ranks.clear();
        for (int i = 0; i < nv; i++) { int32_t l = 0; f.read((char*)&l, 4); std::string s(l, '\0'); if (l) f.read(&s[0], l); vocab[i] = s; ranks[s] = i; }
        special.clear();
        for (int s = 0; s < NANOCHAT_NUM_SPECIAL; s++) special[NANOCHAT_SPECIALS[s]] = nv + s;
        bos_id = special["<|bos|>"];
        return true;
    }
};

} // namespace gpt

#endif // NANOCHAT_BPE_H
