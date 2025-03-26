#pragma once
#include <vector>
#include "types.hpp"

#include "mio/mio.hpp"

namespace Cu
{
    void init();

    using iter_t = const char *;
    template <bool IS_GPU>
    struct Thrust
    {
        static void recompression(symbol_t alphabet_size, iter_t begin, iter_t end);
    };

}