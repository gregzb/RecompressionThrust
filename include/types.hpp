#pragma once
#include <cstdint>

// using symbol_t to represent the type that contains the largest symbol we need
// during recompression as well as the original length of the text
using symbol_t = int32_t;
using level_t = int16_t;
