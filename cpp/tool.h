// ---------------------------------------------------------------------------
//  tool.h  --  the "python" calculator tool, dependency-free.
//
//  nanochat's chat model can offload arithmetic to a tool: it emits
//  <|python_start|> EXPR <|python_end|>, a harness evaluates EXPR, and the
//  result is fed back as <|output_start|> RESULT <|output_end|>. nanochat's
//  actual tool only allows arithmetic (no builtins, no ** power) -- i.e. a
//  calculator -- so it needs no Python: a tiny recursive-descent evaluator over
//  + - * / ( ) and decimals reproduces it in pure C++.
// ---------------------------------------------------------------------------
#ifndef NANOCHAT_TOOL_H
#define NANOCHAT_TOOL_H

#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace gpt {

// Recursive-descent arithmetic evaluator. Returns true on success (result in
// `out`); false on any parse error / division by zero, matching the tool's
// "return nothing on bad usage" behaviour.
struct Calc {
    const char* p; bool ok = true;
    void skip() { while (*p == ' ' || *p == '\t') p++; }
    double expr() {
        double v = term();
        for (;;) { skip();
            if (*p == '+') { p++; v += term(); }
            else if (*p == '-') { p++; v -= term(); }
            else break;
        }
        return v;
    }
    double term() {
        double v = factor();
        for (;;) { skip();
            if (*p == '*') { p++; v *= factor(); }
            else if (*p == '/') { p++; double d = factor(); if (d == 0) ok = false; else v /= d; }
            else break;
        }
        return v;
    }
    double factor() {
        skip();
        if (*p == '+') { p++; return factor(); }
        if (*p == '-') { p++; return -factor(); }
        if (*p == '(') { p++; double v = expr(); skip(); if (*p == ')') p++; else ok = false; return v; }
        // number
        const char* s = p; char* e = nullptr;
        double v = std::strtod(s, &e);
        if (e == s) { ok = false; return 0; }
        p = e; return v;
    }
};

// Evaluate an arithmetic expression like nanochat's use_calculator: strip commas,
// allow only [0-9 * + - / . ( ) space], disallow the ** power operator.
inline bool eval_calculator(const std::string& expr_in, std::string& out) {
    std::string expr;
    for (char c : expr_in) if (c != ',') expr += c;      // remove thousands separators
    if (expr.find("**") != std::string::npos) return false;
    for (char c : expr) if (!std::strchr("0123456789*+-/.() \t", c)) return false;
    Calc calc; calc.p = expr.c_str();
    double v = calc.expr();
    calc.skip();
    if (!calc.ok || *calc.p != '\0') return false;       // trailing garbage / error
    // format: integer if whole, else trim trailing zeros
    char buf[64];
    if (v == std::floor(v) && std::fabs(v) < 1e15) std::snprintf(buf, sizeof(buf), "%lld", (long long)v);
    else std::snprintf(buf, sizeof(buf), "%g", v);
    out = buf;
    return true;
}

} // namespace gpt

#endif // NANOCHAT_TOOL_H
