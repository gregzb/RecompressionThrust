#pragma once
#include <vector>
#include "types.hpp"

namespace Cu
{
    void init();

    template <bool IS_GPU>
    struct Thrust
    {
        static void recompression(symbol_t alphabet_size, std::vector<symbol_t> input);
    };
}