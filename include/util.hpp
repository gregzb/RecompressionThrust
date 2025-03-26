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

auto time_f_store(auto f, double &millseconds_out)
{
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::nanoseconds;

    auto t1 = high_resolution_clock::now();
    auto ret = f();
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    millseconds_out = ms_double.count();
    return ret;
}

auto time_f_print(auto f, const std::string &str)
{
    double ms;
    auto ret = time_f_store(f, ms);
    std::cout << str << " took " << ms << " ms\n";
    return ret;
}

auto time_f_store_void(auto f, double &millseconds_out)
{
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::nanoseconds;

    auto t1 = high_resolution_clock::now();
    f();
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    millseconds_out = ms_double.count();
}

auto time_f_print_void(auto f, const std::string &str)
{
    double ms;
    time_f_store_void(f, ms);
    std::cout << str << " took " << ms << " ms\n";
}