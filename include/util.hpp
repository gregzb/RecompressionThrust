#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "types.hpp"

void hash_combine(size_t &seed, size_t hash_value);

template <class S, class T>
struct pair_hash
{
    auto operator()(const std::pair<S, T> &pair) const noexcept -> uint64_t
    {
        size_t seed = 0;
        hash_combine(seed, (size_t)(pair.first));
        hash_combine(seed, (size_t)(pair.second));
        return seed;
    };
};

std::string to_string(std::vector<symbol_t> input);

std::vector<symbol_t> to_vec(std::string input);

std::string escapeChar(char c);