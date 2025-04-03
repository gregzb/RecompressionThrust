#include "types.hpp"
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

void hash_combine(size_t &seed, size_t hash_value) { seed ^= hash_value * 0x9e3779b9 + (seed << 6) + (seed >> 2); }

std::string to_string(std::vector<symbol_t> input) {
    std::string out;
    for (auto el : input) {
        out.push_back(el);
    }
    return out;
}

std::vector<symbol_t> to_vec(std::string input) {
    std::vector<symbol_t> out;
    for (auto el : input) {
        out.push_back(std::min((unsigned int)el, 255u));
    }
    return out;
}

std::string escapeChar(char c) {
    switch (c) {
    case '\0':
        return "\\0";
    case '\a':
        return "\\a";
    case '\b':
        return "\\b";
    case '\f':
        return "\\f";
    case '\n':
        return "\\n";
    case '\r':
        return "\\r";
    case '\t':
        return "\\t";
    case '\v':
        return "\\v";
    case '\\':
        return "\\\\";
    case '\"':
        return "\\\"";
    case '\'':
        return "\\\'";
        // Add other single-character escapes if needed.
    }

    // If the character is printable, just return it as a single-character string.
    if (std::isprint(static_cast<unsigned char>(c))) {
        return std::string(1, c);
    } else {
        // For non-printable characters, return a \xNN style escape.
        std::ostringstream oss;
        oss << "\\x" << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
            << (static_cast<int>(static_cast<unsigned char>(c)));
        return oss.str();
    }
}