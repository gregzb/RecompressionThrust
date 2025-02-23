#include <vector>
#include "types.hpp"

namespace Cu
{
    void init();
    void recompression(symbol_t alphabet_size, std::vector<symbol_t> input);
}