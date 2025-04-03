#pragma once
#include "types.hpp"
#include <cstdint>
#include <string>
#include <vector>

void hash_combine(size_t &seed, size_t hash_value);

template <class S, class T> struct pair_hash {
    auto operator()(const std::pair<S, T> &pair) const noexcept -> uint64_t {
        size_t seed = 0;
        hash_combine(seed, (size_t)(pair.first));
        hash_combine(seed, (size_t)(pair.second));
        return seed;
    };
};

std::string to_string(std::vector<symbol_t> input);

std::vector<symbol_t> to_vec(std::string input);

std::string escapeChar(char c);

inline int time_print_depth = 0;

inline void print_spaces() {
    for (int i = 0; i < time_print_depth; i++) {
        std::cout << '|';
        std::cout << ' ';
    }
}

template <typename F> auto time_f_store(F f, double &millseconds_out) {
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

template <typename F> auto time_f_print(F f, const std::string &str) {
    print_spaces();
    std::cout << str << ": STARTING\n";
    double ms;
    time_print_depth++;
    auto ret = time_f_store(f, ms);
    time_print_depth--;
    print_spaces();
    std::cout << str << ": " << ms << " ms\n";
    return ret;
}

template <typename F> auto time_f_store_void(F f, double &millseconds_out) {
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

template <typename F> auto time_f_print_void(F f, const std::string &str) {
    print_spaces();
    std::cout << str << ": STARTING\n";
    double ms;
    time_print_depth++;
    time_f_store_void(f, ms);
    time_print_depth--;
    print_spaces();
    std::cout << str << ": " << ms << " ms\n";
}