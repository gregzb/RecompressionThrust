#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/zip_function.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/unique.h>

#include <chrono>
#include <iostream>
#include "types.hpp"
#include "arena.cuh"

auto timeF(auto f, const std::string &str)
{
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::nanoseconds;

    auto t1 = high_resolution_clock::now();
    auto ret = f();
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    // std::cout << str << " took " << ms_double.count() << " ms\n";
    return ret;
};

namespace Cu
{
    void init()
    {
        cudaSetDevice(0);
        cudaFree(0);
        cudaDeviceSynchronize();
    }
    template <bool IS_GPU>
    struct Thrust
    {
        struct efficient_random_byte_functor
        {
            unsigned int seed;

            __host__ __device__
            efficient_random_byte_functor(unsigned int _seed) : seed(_seed) {}

            // A simple 32-bit hash function based on xorshift and multiplicative mixing.
            __host__ __device__ unsigned int hash(unsigned int x) const
            {
                x ^= x >> 16;
                x *= 0x85ebca6b;
                x ^= x >> 13;
                x *= 0xc2b2ae35;
                x ^= x >> 16;
                return x;
            }

            // For each index, mix the seed and index to produce a pseudo-random number.
            __host__ __device__ unsigned char operator()(unsigned int idx) const
            {
                unsigned int key = idx * 0x9E3779B9;
                unsigned int h = hash(seed ^ key);
                return static_cast<unsigned char>((h % 17) & 0b1);
            }
        };

        // // using InputIterT = SizedIter<decltype(thrust::make_permutation_iterator(static_cast<symbol_t *>(nullptr), thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride_functor(0))))>;
        // // using RulesIterT = SizedIter<decltype(thrust::make_permutation_iterator(static_cast<thrust::tuple<symbol_t, symbol_t> *>(nullptr), thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride_functor(0))))>;
        // using InputIterT = SizedIter<decltype(thrust::make_permutation_iterator(thrust::device_pointer_cast<symbol_t>(nullptr), thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride_functor(0))))>;
        // using RulesIterT = SizedIter<decltype(thrust::make_permutation_iterator(thrust::device_pointer_cast<thrust::tuple<symbol_t, symbol_t>>(nullptr), thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride_functor(0))))>;

        using InputIterType = typename std::conditional<
            IS_GPU,
            decltype(thrust::make_permutation_iterator(
                thrust::device_pointer_cast<symbol_t>(nullptr),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride_functor(0)))),
            decltype(thrust::make_permutation_iterator(
                static_cast<symbol_t *>(nullptr),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride_functor(0))))>::type;

        // Select the correct Rules iterator type using std::conditional.
        using RulesIterType = typename std::conditional<
            IS_GPU,
            decltype(thrust::make_permutation_iterator(
                thrust::device_pointer_cast<thrust::tuple<symbol_t, symbol_t>>(nullptr),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride_functor(0)))),
            decltype(thrust::make_permutation_iterator(
                static_cast<thrust::tuple<symbol_t, symbol_t> *>(nullptr),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride_functor(0))))>::type;

        // Now wrap the types in SizedIter.
        using InputIterT = SizedIter<InputIterType>;
        using RulesIterT = SizedIter<RulesIterType>;

        enum class CurrentInputPos
        {
            TWO,
            THREE
        };

        static CurrentInputPos other_pos(CurrentInputPos curr_input_pos)
        {
            return curr_input_pos == CurrentInputPos::TWO ? CurrentInputPos::THREE : CurrentInputPos::TWO;
        }

        // template <bool IS_GPU>
        static InputIterT view_for(Arena<IS_GPU> &arena, CurrentInputPos curr_input_pos, symbol_t original_size, symbol_t curr_input_size)
        {
            symbol_t mult = curr_input_pos == CurrentInputPos::TWO ? 2 : 3;
            return arena.template view_start_at_items<symbol_t>(mult * original_size, curr_input_size);
        }

        struct make_key_count_tuple
        {
            __host__ __device__
                thrust::tuple<uint64_t, symbol_t>
                operator()(symbol_t key, symbol_t count, symbol_t index) const
            {
                return thrust::make_tuple((((uint64_t)(uint32_t)key) << 32) | count, index);
            }
        };

        struct count_predicate
        {
            __host__ __device__ bool operator()(symbol_t key, symbol_t count, symbol_t index) const
            {
                return count > 1;
            }
        };

        // 9
        // new input, new num symbols
        // curr input is expected to be a view at symbol item index 2
        // template <bool IS_GPU>

        static symbol_t bcomp(Arena<IS_GPU> &arena, InputIterT &curr_input, CurrentInputPos &curr_input_pos, symbol_t original_size, symbol_t num_symbols, RulesIterT rules)
        {
            using std::chrono::duration;
            using std::chrono::duration_cast;
            using std::chrono::high_resolution_clock;
            using std::chrono::nanoseconds;

            auto keys = view_for(arena, other_pos(curr_input_pos), original_size, curr_input.size);
            auto counts = arena.template view_start_at_items<symbol_t>(4 * original_size, curr_input.size);
            auto indices = view_for(arena, curr_input_pos, original_size, curr_input.size);
            auto keys_counts_double = arena.template view_start_at_bytes<uint64_t>(0 * original_size * sizeof(symbol_t), curr_input.size);

            auto [keys_end, counts_end] = timeF([&]()
                                                { return thrust::reduce_by_key(curr_input.iter, curr_input.iter + curr_input.size, thrust::make_constant_iterator(1), keys.iter, counts.iter); }, "reduce_by_key");

            keys.shrink(keys_end - keys.iter);

            if (keys.size == curr_input.size)
            {
                return num_symbols;
            }

            auto zip_iter = thrust::make_zip_iterator(keys.iter, counts.iter, thrust::make_counting_iterator(0));
            auto keys_counts_and_indices = thrust::make_zip_iterator(keys_counts_double.iter, indices.iter);
            // auto transform_it = thrust::make_transform_output_iterator(keys_counts_and_indices, thrust::make_zip_function([] __host__ __device__(symbol_t key, symbol_t count, symbol_t index) -> thrust::tuple<uint64_t, symbol_t>
            //                                                                                                               { return thrust::make_tuple(((((uint64_t)(uint32_t)key) << 32) | count), index); }));
            auto transform_it = thrust::make_transform_output_iterator(
                keys_counts_and_indices,
                thrust::make_zip_function(make_key_count_tuple()));
            // auto transform_it_end = timeF([&]()
            //                               { return thrust::copy_if(zip_iter, zip_iter + keys.size, transform_it, thrust::make_zip_function([] __host__ __device__(symbol_t key, symbol_t count, symbol_t index) -> bool
            //                                                                                                                                { return count > 1; })); }, "copy if");
            auto transform_it_end = thrust::copy_if(
                zip_iter, zip_iter + keys.size, transform_it,
                thrust::make_zip_function(count_predicate()));

            symbol_t num_blocks = transform_it_end - transform_it;

            if (num_blocks > 0)
            {
                timeF([&]()
                      { thrust::sort_by_key(keys_counts_double.iter, keys_counts_double.iter + num_blocks, indices.iter); return 0; }, "sort");

                auto offsets = arena.template view_start_at_items<symbol_t>(4 * original_size, num_blocks);

                offsets.iter[0] = 0;
                auto prev_and_next = thrust::make_zip_iterator(keys_counts_and_indices, keys_counts_and_indices + 1);
                timeF([&]()
                      { thrust::transform_inclusive_scan(prev_and_next, prev_and_next + offsets.size - 1, offsets.iter + 1, [] __host__ __device__(const thrust::tuple<thrust::tuple<uint64_t, symbol_t>, thrust::tuple<uint64_t, symbol_t>> &item) -> symbol_t
                                             {
            bool eq = thrust::get<0>(thrust::get<0>(item)) == thrust::get<0>(thrust::get<1>(item));
            return (symbol_t)(!eq); }, [] __host__ __device__(symbol_t a, symbol_t b) -> symbol_t
                                             { return a + b; }); return 0; }, "transform_inclusive_scan");

                timeF([&]()
                      { thrust::unique_by_key_copy(offsets.iter, offsets.iter + offsets.size, thrust::make_transform_iterator(keys_counts_double.iter, [] __host__ __device__(uint64_t item) -> thrust::tuple<symbol_t, symbol_t>
                                                                                                                  { return thrust::make_tuple(item >> 32, item & 0xffffffff); }),
                                       thrust::make_discard_iterator(), rules.iter + num_symbols); return 0; }, "unique by key copy");

                symbol_t last_offset = timeF([&]()
                                             { return offsets.iter[offsets.size - 1]; }, "last offset");

                auto offset_iter = thrust::make_transform_iterator(offsets.iter, [num_symbols] __host__ __device__(const symbol_t &offset) -> symbol_t
                                                                   { return num_symbols + offset; });
                symbol_t num_new_symbols = last_offset + 1;
                num_symbols += num_new_symbols;

                timeF([&]()
                      { thrust::scatter(offset_iter, offset_iter + offsets.size, indices.iter, keys.iter); return 0; }, "scatter");
            }
            curr_input = keys;
            curr_input_pos = other_pos(curr_input_pos);
            return num_symbols;
        }

        inline static size_t total_pcomp_sorted = 0;

        // 6
        // new input, new num symbols
        // template <bool IS_GPU>
        static symbol_t pcomp(Arena<IS_GPU> &arena, InputIterT &curr_input, CurrentInputPos curr_input_pos, symbol_t original_size, symbol_t num_symbols, RulesIterT rules, int x)
        {
            auto pseudo_rand_bits = arena.template view_start_at_bytes<uint8_t>(original_size / 2 * sizeof(symbol_t) * 3 + 0 * original_size, num_symbols);
            auto counting_iter = thrust::make_counting_iterator(0);
            timeF([&]()
                  {
            thrust::transform(counting_iter, counting_iter + pseudo_rand_bits.size, pseudo_rand_bits.iter, efficient_random_byte_functor(num_symbols + x));
            return 0; }, "get random bits");

            auto assigned_bit = arena.template view_start_at_bytes<uint8_t>(original_size / 2 * sizeof(symbol_t) * 3 + 1 * original_size, curr_input.size);

            timeF([&]()
                  {
            thrust::gather(curr_input.iter, curr_input.iter + curr_input.size, pseudo_rand_bits.iter, assigned_bit.iter);
            return 0; }, "gather");

            timeF([&]()
                  {
            thrust::adjacent_difference(assigned_bit.iter, assigned_bit.iter + assigned_bit.size, assigned_bit.iter, [] __host__ __device__(uint8_t curr, uint8_t prev) -> uint8_t
                                        { return curr && !prev; });
            return 0; }, "adjacent difference");
            assigned_bit.iter[0] = 0;

            auto pairs = arena.template view_start_at_bytes<uint64_t>(0, assigned_bit.size / 2);
            auto indices = arena.template view_start_at_bytes<symbol_t>(sizeof(uint64_t) * original_size / 2 * 1, assigned_bit.size / 2);

            auto iter_adj = thrust::make_zip_iterator(curr_input.iter, curr_input.iter + 1, thrust::make_counting_iterator(1));

            auto pairs_and_sources_iter = thrust::make_zip_iterator(pairs.iter, indices.iter);
            auto pairs_and_sources_transformed_iter = thrust::make_transform_output_iterator(pairs_and_sources_iter, thrust::make_zip_function(make_key_count_tuple()));
            {
                auto pairs_and_sources_transformed_end = timeF([&]()
                                                               { return thrust::copy_if(iter_adj, iter_adj + assigned_bit.size - 1, assigned_bit.iter + 1, pairs_and_sources_transformed_iter, thrust::identity<uint8_t>()); }, "copy_if");

                pairs.shrink(pairs_and_sources_transformed_end - pairs_and_sources_transformed_iter);
                indices.shrink(pairs_and_sources_transformed_end - pairs_and_sources_transformed_iter);
            }

            // {
            //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input_end);
            //     for (auto el : tmp5)
            //     {
            //         std::cout << el << ", ";
            //     }
            //     std::cout << "\n";
            // }

            if (pairs.size > 0)
            {
                // total_pcomp_sorted += pairs.size;
                timeF([&]()
                      { thrust::sort_by_key(pairs.iter, pairs.iter+pairs.size, indices.iter); return 0; }, "sort");

                auto offsets = arena.template view_start_at_items<symbol_t>(4 * original_size, pairs.size);

                timeF([&]()
                      { return 0; }, "set first element");

                auto prev_and_next = thrust::make_zip_iterator(pairs.iter, pairs.iter + 1);
                timeF([&]()
                      {thrust::transform_inclusive_scan(prev_and_next, prev_and_next + pairs.size - 1, offsets.iter + 1, [] __host__ __device__(const thrust::tuple<uint64_t, uint64_t> &item) -> symbol_t
                                             {
            bool eq = thrust::get<0>(item) == thrust::get<1>(item);
            return (symbol_t)(!eq); }, thrust::plus<symbol_t>()); return 0; }, "transform_inclusive_scan");

                timeF([&]()
                      { thrust::unique_by_key_copy(offsets.iter, offsets.iter + offsets.size, thrust::make_transform_iterator(pairs.iter, [] __host__ __device__(uint64_t item) -> thrust::tuple<symbol_t, symbol_t>
                                                                                                                  { return thrust::make_tuple(item >> 32, item & 0xffffffff); }),
                                       thrust::make_discard_iterator(), rules.iter + num_symbols); return 0; }, "unique by key copy");

                auto offset_iter = thrust::make_transform_iterator(offsets.iter, [num_symbols] __host__ __device__(const symbol_t &offset) -> symbol_t
                                                                   { return num_symbols + offset; });
                symbol_t num_new_symbols = offsets.iter[offsets.size - 1] + 1;
                num_symbols += num_new_symbols;

                timeF([&]()
                      { thrust::scatter(offset_iter, offset_iter + offsets.size, indices.iter, curr_input.iter); return 0; }, "scatter");

                auto curr_input_and_idx = thrust::make_zip_iterator(curr_input.iter, thrust::make_counting_iterator(0));
                if constexpr (IS_GPU)
                {
                    auto assigned_bit_raw_ptr = thrust::device_pointer_cast(&assigned_bit.iter[0]);
                    auto new_end = timeF([&]()
                                         { return thrust::remove_if(curr_input_and_idx, curr_input_and_idx + curr_input.size, [assigned_bit_raw_ptr] __host__ __device__(const thrust::tuple<symbol_t, symbol_t> &item)
                                                                    {
                auto idx = thrust::get<1>(item);
                return *(assigned_bit_raw_ptr + idx + 1) == 1; }); }, "remove_if");

                    curr_input.shrink(new_end - curr_input_and_idx);
                }
                else
                {
                    auto assigned_bit_raw_ptr = &assigned_bit.iter[0];
                    auto new_end = timeF([&]()
                                         { return thrust::remove_if(curr_input_and_idx, curr_input_and_idx + curr_input.size, [assigned_bit_raw_ptr] __host__ __device__(const thrust::tuple<symbol_t, symbol_t> &item)
                                                                    {
                auto idx = thrust::get<1>(item);
                return *(assigned_bit_raw_ptr + idx + 1) == 1; }); }, "remove_if");

                    curr_input.shrink(new_end - curr_input_and_idx);
                }
                // // auto assigned_bit_raw_ptr = &assigned_bit.iter[0];
                // auto assigned_bit_raw_ptr = thrust::device_pointer_cast(&assigned_bit.iter[0]);
                // auto new_end = timeF([&]()
                //                      { return thrust::remove_if(curr_input_and_idx, curr_input_and_idx + curr_input.size, [assigned_bit_raw_ptr] __host__ __device__(const thrust::tuple<symbol_t, symbol_t> &item)
                //                                                 {
                // auto idx = thrust::get<1>(item);
                // return *(assigned_bit_raw_ptr + idx + 1) == 1; }); }, "remove_if");

                // curr_input.shrink(new_end - curr_input_and_idx);
            }
            return num_symbols;
        }

        static void recompression(symbol_t alphabet_size, std::vector<symbol_t> input)
        {
            using std::chrono::duration;
            using std::chrono::duration_cast;
            using std::chrono::high_resolution_clock;
            using std::chrono::nanoseconds;
            symbol_t num_symbols = alphabet_size;

            symbol_t unit_size = std::max(alphabet_size * 4, (symbol_t)input.size());

            Arena<IS_GPU> arena(unit_size * sizeof(symbol_t) * 7);
            auto curr_input = arena.template view_start_at_items<symbol_t>(2 * unit_size, input.size());

            auto curr_input_pos = CurrentInputPos::TWO;

            // thrust::device_vector<thrust::tuple<symbol_t, symbol_t>> rules(curr_input.size);
            auto rules = arena.template view_start_at_bytes<thrust::tuple<symbol_t, symbol_t>>(5 * unit_size * sizeof(symbol_t), input.size());

            thrust::copy(input.begin(), input.end(), curr_input.iter);
            auto ts = high_resolution_clock::now();

            // {
            //     thrust::host_vector<symbol_t> tmp5(curr_input.iter, curr_input.iter + curr_input.size);
            //     for (auto el : tmp5)
            //     {
            //         std::cout << el << ", ";
            //     }
            //     std::cout << "\n";
            // }

            int num_layers = 0;
            while (curr_input.size > 1)
            {
                auto t1 = high_resolution_clock::now();
                auto new_num_symbols = bcomp(arena, curr_input, curr_input_pos, unit_size, num_symbols, rules);
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                std::cout << "bcomp " << (ms_double.count()) << " ms, " << num_symbols << " symbols" << "\n";
                num_symbols = new_num_symbols;
                // {
                //     thrust::host_vector<symbol_t> tmp5(curr_input.iter, curr_input.iter + curr_input.size);
                //     for (auto el : tmp5)
                //     {
                //         std::cout << el << ", ";
                //     }
                //     std::cout << "\n";
                // }

                num_layers++;

                if (curr_input.size == 1)
                    break;

                auto t3 = high_resolution_clock::now();
                auto t4 = high_resolution_clock::now();
                int cnt = 0;
                symbol_t orig_symbols = num_symbols;
                while (true)
                {
                    size_t prev_size = curr_input.size;
                    auto new_num_symbols2 = pcomp(arena, curr_input, curr_input_pos, unit_size, num_symbols, rules, cnt);
                    // {
                    //     thrust::host_vector<symbol_t> tmp5(curr_input.iter, curr_input.iter + curr_input.size);
                    //     for (auto el : tmp5)
                    //     {
                    //         std::cout << el << ", ";
                    //     }
                    //     std::cout << "\n";
                    // }
                    t4 = high_resolution_clock::now();
                    if (curr_input.size != prev_size)
                    {
                        num_symbols = new_num_symbols2;
                        break;
                    }
                    cnt++;
                }
                duration<double, std::milli> ms_double2 = t4 - t3;
                std::cout << "pcomp " << (ms_double2.count()) << " ms, " << orig_symbols << " symbols before, " << num_symbols << " symbols after" << "\n";

                num_layers++;

                std::cout << "so far: " << (duration<double, std::milli>(high_resolution_clock::now() - ts)).count() << " ms\n";
            }

            rules.shrink(num_symbols);

            std::cout << "layers: " << num_layers << std::endl;
            // std::cout << "total pcomp sorted " << total_pcomp_sorted << std::endl;
        }
    };
    template struct Thrust<false>;
    template struct Thrust<true>;
}