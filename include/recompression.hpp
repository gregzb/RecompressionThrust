#pragma once
#include "rlslp.hpp"
#include "types.hpp"
#include <vector>

#include "mio/mio.hpp"

namespace Cu {
void init();

using iter_t = const unsigned char *;
template <bool IS_GPU> struct Thrust {
    static Rlslp recompression(symbol_t alphabet_size, iter_t begin, iter_t end);
};

} // namespace Cu