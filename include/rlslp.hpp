#include <array>
#include <vector>
#include "types.hpp"

struct Rlslp
{
    struct Pair
    {
        std::array<symbol_t, 2> children;
        symbol_t &left() { return children[0]; }
        const symbol_t &left() const { return children[0]; }
        symbol_t &right() { return children[1]; }
        const symbol_t &right() const { return children[1]; }
    };

    struct Block
    {
        symbol_t symbol;
        symbol_t count;
    };

    struct Rule
    {
        union
        {
            Pair pair;
            Block block;
        };
    };

    std::vector<Rule> rules;
    std::vector<level_t> levels;
    symbol_t alphabet_size;
    symbol_t root;
    len_t len;

    static bool level_is_block(level_t level) { return level % 2 == 1; }

    static bool level_is_pair(level_t level) { return level % 2 == 0; }
};