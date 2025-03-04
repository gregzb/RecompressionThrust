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
    std::cout << str << " took " << ms_double.count() << " ms\n";
    return ret;
};

namespace Cu
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

    // template <typename T>
    struct stride_functor
    {
        size_t stride;
        stride_functor(size_t stride) : stride(stride) {}
        __host__ __device__
            size_t
            operator()(size_t i) const
        {
            return i * stride;
        }
    };

    // template <class T>
    // struct SizedIter
    // {
    //     T iter;
    //     size_t size;
    //     SizedIter(T iter, size_t size) : iter(iter), size(size)
    //     {
    //     }

    //     void shrink(size_t new_size)
    //     {
    //         if (new_size <= size)
    //         {
    //             size = new_size;
    //         }
    //         else
    //         {
    //             std::cout << "EXITING! New size must be same or less than current size! Current size: " << size << ", new size: " << new_size << std::endl;
    //             exit(1);
    //         }
    //     }
    // };

    template <class T>
    struct SizedIter
    {
        T iter;
        size_t size;
        SizedIter(T iter, size_t size) : iter(iter), size(size)
        {
        }

        void shrink(size_t new_size)
        {
            if (new_size <= size)
            {
                size = new_size;
            }
            else
            {
                std::cout << "EXITING! New size must be same or less than current size! Current size: " << size << ", new size: " << new_size << std::endl;
                exit(1);
            }
        }
    };

    struct Arena
    {
        thrust::device_vector<unsigned char> buffer;
        Arena(size_t num_bytes) : buffer(num_bytes)
        {
        }

        // start specified in bytes
        // stride specified in items
        template <typename TYPE>
        auto view_start_at_bytes(size_t start, size_t size, size_t stride = 1)
        {
            auto start_ptr = thrust::device_pointer_cast<TYPE>((TYPE *)(thrust::raw_pointer_cast(buffer.data()) + start));
            auto stride_iter =
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride_functor(stride));
            auto iter = thrust::make_permutation_iterator(start_ptr, stride_iter);
            return SizedIter(iter, size);
        }

        // start specified in items
        // stride specified in items
        template <typename TYPE>
        auto view_start_at_items(size_t start, size_t size, size_t stride = 1)
        {
            return view_start_at_bytes<TYPE>(start * sizeof(TYPE), size, stride);
        }
    };

    using InputIterT = SizedIter<decltype(thrust::make_permutation_iterator(thrust::device_pointer_cast<symbol_t>(nullptr), thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride_functor(0))))>;

    // TODO(bgruber): we can drop thrust::less etc. when they truly alias to the ::cuda::std ones
    template <class Key, class CompareOp>
    using can_use_primitive_sort = ::cuda::std::integral_constant<
        bool,
        (::cuda::std::is_arithmetic<Key>::value
#if defined(_CCCL_HAS_NVFP16) && !defined(__CUDA_NO_HALF_OPERATORS__) && !defined(__CUDA_NO_HALF_CONVERSIONS__)
         || ::cuda::std::is_same<Key, __half>::value
#endif // defined(_CCCL_HAS_NVFP16) && !defined(__CUDA_NO_HALF_OPERATORS__) && !defined(__CUDA_NO_HALF_CONVERSIONS__)
#if defined(_CCCL_HAS_NVBF16) && !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__) && !defined(__CUDA_NO_BFLOAT16_OPERATORS__)
         || ::cuda::std::is_same<Key, __nv_bfloat16>::value
#endif // defined(_CCCL_HAS_NVBF16) && !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__) &&
       // !defined(__CUDA_NO_BFLOAT16_OPERATORS__)
         ) &&
            (::cuda::std::is_same<CompareOp, thrust::less<Key>>::value || ::cuda::std::is_same<CompareOp, ::cuda::std::less<Key>>::value || ::cuda::std::is_same<CompareOp, thrust::less<void>>::value || ::cuda::std::is_same<CompareOp, ::cuda::std::less<void>>::value || ::cuda::std::is_same<CompareOp, thrust::greater<Key>>::value || ::cuda::std::is_same<CompareOp, ::cuda::std::greater<Key>>::value || ::cuda::std::is_same<CompareOp, thrust::greater<void>>::value || ::cuda::std::is_same<CompareOp, ::cuda::std::greater<void>>::value)>;

    // 9
    // new input, new num symbols
    // curr input is expected to be a view at symbol item index 2
    symbol_t bcomp(Arena &arena, InputIterT &curr_input, symbol_t original_size, symbol_t num_symbols, thrust::device_vector<thrust::tuple<symbol_t, symbol_t>> &rules)
    {
        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::nanoseconds;

        auto keys = arena.view_start_at_items<symbol_t>(3 * original_size, curr_input.size);
        auto counts = arena.view_start_at_items<symbol_t>(4 * original_size, curr_input.size);
        auto indices = arena.view_start_at_items<symbol_t>(2 * original_size, curr_input.size);
        auto keys_counts_double = arena.view_start_at_bytes<uint64_t>(0 * original_size * sizeof(symbol_t), curr_input.size);
        // std::cout << (3 * curr_input.size * sizeof(symbol_t)) << std::endl;

        auto [keys_end, counts_end] = timeF([&]()
                                            { return thrust::reduce_by_key(curr_input.iter, curr_input.iter + curr_input.size, thrust::make_constant_iterator(1), keys.iter, counts.iter); }, "reduce_by_key");

        keys.shrink(keys_end - keys.iter);

        if (keys.size == curr_input.size)
        {
            return num_symbols;
        }

        auto zip_iter = thrust::make_zip_iterator(keys.iter, counts.iter, thrust::make_counting_iterator(0));
        auto keys_counts_and_indices = thrust::make_zip_iterator(keys_counts_double.iter, indices.iter);
        auto transform_it = thrust::make_transform_output_iterator(keys_counts_and_indices, thrust::make_zip_function([] __device__(symbol_t key, symbol_t count, symbol_t index) -> thrust::tuple<uint64_t, symbol_t>
                                                                                                                      { return thrust::make_tuple(((((uint64_t)(uint32_t)key) << 32) | count), index); }));
        auto transform_it_end = timeF([&]()
                                      { return thrust::copy_if(zip_iter, zip_iter + keys.size, transform_it, thrust::make_zip_function([] __device__(symbol_t key, symbol_t count, symbol_t index) -> bool
                                                                                                                                       { return count > 1; })); }, "copy if");

        symbol_t num_blocks = transform_it_end - transform_it;

        // std::cout << "blocks: " << num_blocks << std::endl;

        if (num_blocks > 0)
        {
            std::cout << "Sorting " << (num_blocks) << " items\n";
            timeF([&]()
                  { thrust::sort_by_key(keys_counts_double.iter, keys_counts_double.iter + num_blocks, indices.iter); return 0; }, "sort");

            auto offsets = arena.view_start_at_items<symbol_t>(4 * original_size, num_blocks);

            offsets.iter[0] = 0;
            auto prev_and_next = thrust::make_zip_iterator(keys_counts_and_indices, keys_counts_and_indices + 1);
            timeF([&]()
                  { thrust::transform_inclusive_scan(prev_and_next, prev_and_next + offsets.size - 1, offsets.iter + 1, [] __device__(const thrust::tuple<thrust::tuple<uint64_t, symbol_t>, thrust::tuple<uint64_t, symbol_t>> &item) -> symbol_t
                                             {
            bool eq = thrust::get<0>(thrust::get<0>(item)) == thrust::get<0>(thrust::get<1>(item));
            return (symbol_t)(!eq); }, [] __device__(symbol_t a, symbol_t b) -> symbol_t
                                             { return a + b; }); return 0; }, "transform_inclusive_scan");

            timeF([&]()
                  { thrust::unique_by_key_copy(offsets.iter, offsets.iter + offsets.size, thrust::make_transform_iterator(keys_counts_double.iter, [] __device__(uint64_t item) -> thrust::tuple<symbol_t, symbol_t>
                                                                                                                  { return thrust::make_tuple(item >> 32, item & 0xffffffff); }),
                                       thrust::make_discard_iterator(), rules.begin() + num_symbols); return 0; }, "unique by key copy");

            symbol_t last_offset = timeF([&]()
                                         { return offsets.iter[offsets.size - 1]; }, "last offset");

            auto offset_iter = thrust::make_transform_iterator(offsets.iter, [num_symbols] __device__(const symbol_t &offset) -> symbol_t
                                                               { return num_symbols + offset; });
            symbol_t num_new_symbols = last_offset + 1;
            num_symbols += num_new_symbols;

            timeF([&]()
                  { thrust::scatter(offset_iter, offset_iter + offsets.size, indices.iter, keys.iter); return 0; }, "scatter");
        }
        timeF([&]()
              {
            curr_input.shrink(keys.size);
            return 0; }, "resize");

        return num_symbols;
    }

    // 6
    // new input, new num symbols
    std::tuple<thrust::device_vector<symbol_t>, symbol_t> pcomp(Arena &arena, InputIterT &curr_input, symbol_t original_size, symbol_t num_symbols, thrust::device_vector<thrust::tuple<symbol_t, symbol_t>> &rules, int x)
    {
        // thrust::device_vector<uint8_t> pseudo_rand_bits = timeF([&]()
        //                                                         { return thrust::device_vector<uint8_t>(num_symbols); }, "alloc pseudo rand bits");
        auto pseudo_rand_bits = arena.view_start_at_bytes<uint8_t>(original_size / 2 * sizeof(symbol_t) * 3 + 0 * original_size, num_symbols);
        auto counting_iter = thrust::make_counting_iterator(0);
        timeF([&]()
              {
            thrust::transform(counting_iter, counting_iter + pseudo_rand_bits.size, pseudo_rand_bits.iter, efficient_random_byte_functor(num_symbols + x));
            return 0; }, "get random bits");

        // thrust::device_vector<uint8_t> assigned_bit = timeF([&]()
        //                                                     { return thrust::device_vector<uint8_t>(curr_input_end - curr_input.begin()); }, "alloc assigned bits");

        auto assigned_bit = arena.view_start_at_bytes<uint8_t>(original_size / 2 * sizeof(symbol_t) * 3 + 1 * original_size, curr_input.size);

        timeF([&]()
              {
            thrust::gather(curr_input.iter, curr_input.iter + curr_input.size, pseudo_rand_bits.iter, assigned_bit.iter);
            return 0; }, "gather");

        timeF([&]()
              {
            thrust::adjacent_difference(assigned_bit.iter, assigned_bit.iter + assigned_bit.size, assigned_bit.iter, [] __device__(uint8_t curr, uint8_t prev) -> uint8_t
                                        { return curr && !prev; });
            return 0; }, "adjacent difference");
        assigned_bit.iter[0] = 0;

        // if (thrust::count(assigned_bit.begin(), assigned_bit.end(), 1) != 0) {

        // pair, index
        // thrust::device_vector<thrust::tuple<symbol_t, symbol_t, symbol_t>> pairs_and_sources = timeF([&]()
        //                                                                                              { return thrust::device_vector<thrust::tuple<symbol_t, symbol_t, symbol_t>>(assigned_bit.size / 2); }, "alloc pairs and sources");
        // auto pairs_and_sources = arena.view_start_at_bytes<thrust::tuple<symbol_t, symbol_t, symbol_t>>(0, assigned_bit.size / 2);
        auto pairs = arena.view_start_at_bytes<uint64_t>(0, assigned_bit.size / 2);
        auto indices = arena.view_start_at_bytes<symbol_t>(sizeof(uint64_t) * original_size * 1, assigned_bit.size / 2);

        auto iter_adj = thrust::make_zip_iterator(curr_input.iter, curr_input.iter + 1, thrust::make_counting_iterator(1));

        // auto iter_adj = thrust::make_zip_iterator(assigned_bit.begin() + 1, curr_input.begin(), curr_input.begin() + 1, thrust::make_counting_iterator(1));
        // auto pairs_and_sources_end = thrust::transform_if(iter_adj, iter_adj + assigned_bit.size() - 1, pairs_and_sources.begin(), [] __device__(const thrust::tuple<uint8_t, symbol_t, symbol_t, symbol_t> &item) -> thrust::tuple<symbol_t, symbol_t, symbol_t>
        //                                                   { return thrust::make_tuple(thrust::get<1>(item), thrust::get<2>(item), thrust::get<3>(item)); }, [] __device__(const thrust::tuple<uint8_t, symbol_t, symbol_t, symbol_t> &item) -> bool
        //                                                   { return false; });
        auto pairs_and_sources_iter = thrust::make_zip_iterator(pairs.iter, indices.iter);
        auto pairs_and_sources_transformed_iter = thrust::make_transform_output_iterator(pairs_and_sources_iter, thrust::make_zip_function([] __device__(symbol_t key1, symbol_t key2, symbol_t index) -> thrust::tuple<uint64_t, symbol_t>
                                                                                                                                           { return thrust::make_tuple(((((uint64_t)(uint32_t)key1) << 32) | key2), index); }));
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
            std::cout << "Sorting " << (pairs.size) << " items\n";
            timeF([&]()
                  { thrust::sort_by_key(pairs.iter, pairs.iter+pairs.size, indices.iter); return 0; }, "sort");

            // // new offset, index to scatter back to
            // thrust::device_vector<thrust::tuple<symbol_t, symbol_t>> offsets_and_indices = timeF([&]()
            //                                                                                      { return thrust::device_vector<thrust::tuple<symbol_t, symbol_t>>(pairs_and_sources.size); }, "alloc offsets and indices"); // this can be made to share space with keys/counts

            auto offsets = arena.view_start_at_items<symbol_t>(4 * original_size, pairs.size);

            timeF([&]()
                  { 
            // thrust::tuple<symbol_t, symbol_t, symbol_t> x = pairs_and_sources.iter[0];
            // offsets_and_indices[0] = thrust::make_tuple(0, thrust::get<2>(x));
            
             return 0; }, "set first element");
            // auto prev_and_next = thrust::make_zip_iterator(pairs_and_sources_iter, pairs_and_sources_iter + 1);
            // timeF([&]()
            //       {thrust::transform_inclusive_scan(prev_and_next, prev_and_next + indices.size - 1, offsets.iter + 1, [] __device__(const thrust::tuple<thrust::tuple<uint64_t, symbol_t>, thrust::tuple<uint64_t, symbol_t>> &item) -> thrust::tuple<symbol_t, symbol_t>
            //                                  {
            // bool eq = thrust::get<0>(thrust::get<0>(item)) == thrust::get<0>(thrust::get<1>(item)) && thrust::get<1>(thrust::get<0>(item)) == thrust::get<1>(thrust::get<1>(item));
            // return thrust::make_tuple((symbol_t)(!eq), thrust::get<2>(thrust::get<1>(item))); }, [] __device__(const thrust::tuple<symbol_t, symbol_t> &a, const thrust::tuple<symbol_t, symbol_t> &b) -> thrust::tuple<symbol_t, symbol_t>
            //                                  { return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b), thrust::get<1>(b)); }); return 0; }, "transform_inclusive_scan");

            // auto prev_and_next = thrust::make_zip_iterator(pairs_and_sources_iter, pairs_and_sources_iter + 1);
            // timeF([&]()
            //       {thrust::transform_inclusive_scan(prev_and_next, prev_and_next + indices.size - 1, offsets.iter + 1, [] __device__(const thrust::tuple<thrust::tuple<uint64_t, symbol_t>, thrust::tuple<uint64_t, symbol_t>> &item) -> thrust::tuple<symbol_t, symbol_t>
            //                                  {
            // bool eq = thrust::get<0>(thrust::get<0>(item)) == thrust::get<0>(thrust::get<1>(item)) && thrust::get<1>(thrust::get<0>(item)) == thrust::get<1>(thrust::get<1>(item));
            // return thrust::make_tuple((symbol_t)(!eq), thrust::get<2>(thrust::get<1>(item))); }, [] __device__(const thrust::tuple<symbol_t, symbol_t> &a, const thrust::tuple<symbol_t, symbol_t> &b) -> thrust::tuple<symbol_t, symbol_t>
            //                                  { return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b), thrust::get<1>(b)); }); return 0; }, "transform_inclusive_scan");

            auto prev_and_next = thrust::make_zip_iterator(pairs.iter, pairs.iter + 1);
            timeF([&]()
                  {thrust::transform_inclusive_scan(prev_and_next, prev_and_next + pairs.size - 1, offsets.iter + 1, [] __device__(const thrust::tuple<uint64_t, uint64_t> &item) -> symbol_t
                                             {
            bool eq = thrust::get<0>(item) == thrust::get<1>(item);
            return (symbol_t)(!eq); }, thrust::plus<symbol_t>()); return 0; }, "transform_inclusive_scan");

            // thrust::tuple<symbol_t, symbol_t> last_el_of_offsets_and_indices = offsets_and_indices[offsets_and_indices.size() - 1];

            // auto offset_iter = thrust::make_transform_iterator(offsets_and_indices.begin(), [num_symbols] __device__(const thrust::tuple<symbol_t, symbol_t> &item) -> symbol_t
            //                                                    { return num_symbols + thrust::get<0>(item); });
            // auto index_iter = thrust::make_transform_iterator(offsets_and_indices.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t> &item) -> symbol_t
            //                                                   { return thrust::get<1>(item); });
            symbol_t num_new_symbols = offsets.iter[offsets.size - 1] + 1;
            num_symbols += num_new_symbols;

            // std::cout << "sz: " << offsets_and_indices.size() << std::endl;
            // thrust::host_vector<thrust::tuple<symbol_t, symbol_t>> tmp7 = offsets_and_indices;
            // for (auto [e1, e2] : tmp7)
            // {
            //     std::cout << e1 << " " << e2 << ", ";
            // }
            // std::cout << std::endl;

            // {
            //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input_end);
            //     for (auto el : tmp5)
            //     {
            //         std::cout << el << ", ";
            //     }
            //     std::cout << "\n";
            // }

            timeF([&]()
                  { thrust::scatter(offsets.iter, offsets.iter + offsets.size, indices.iter, curr_input.iter); return 0; }, "scatter");
            // {
            //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input_end);
            //     for (auto el : tmp5)
            //     {
            //         std::cout << el << ", ";
            //     }
            //     std::cout << "\n";
            // }
            auto curr_input_and_idx = thrust::make_zip_iterator(curr_input.iter, thrust::make_counting_iterator(0));
            auto assigned_bit_raw_ptr = thrust::device_pointer_cast(&assigned_bit.iter[0]);
            auto new_end = timeF([&]()
                                 { return thrust::remove_if(curr_input_and_idx, curr_input_and_idx + curr_input.size, [assigned_bit_raw_ptr] __device__(const thrust::tuple<symbol_t, symbol_t> &item)
                                                            {
                auto idx = thrust::get<1>(item);
                return *(assigned_bit_raw_ptr + idx + 1) == 1; }); }, "remove_if");

            curr_input.shrink(new_end - curr_input_and_idx);

            // curr_input_end = curr_input.begin() + (new_end - curr_input_and_idx);

            // now shift one to the left, and do a remove_copy_if(?) to remove just the elements we don't care about :)
        }
        auto out = timeF([&]()
                         { return thrust::device_vector<symbol_t>(curr_input.iter, curr_input.iter + curr_input.size); }, "copy out");
        return {out, num_symbols};
    }

    void init()
    {
        cudaSetDevice(0);
        cudaFree(0);
        cudaDeviceSynchronize();
    }

    void recompression(symbol_t alphabet_size, std::vector<symbol_t> input)
    {
        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::nanoseconds;
        symbol_t num_symbols = alphabet_size;

        symbol_t unit_size = std::max(alphabet_size * 4, (symbol_t)input.size());

        Arena arena(unit_size * sizeof(symbol_t) * 5);
        auto curr_input = arena.view_start_at_items<symbol_t>(2 * unit_size, input.size());

        // thrust::device_vector<symbol_t> curr_input = input;
        thrust::device_vector<thrust::tuple<symbol_t, symbol_t>> rules(curr_input.size);

        thrust::copy(input.begin(), input.end(), curr_input.iter);

        // {
        //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input.end());
        //     for (auto el : tmp5)
        //     {
        //         std::cout << el << ", ";
        //     }
        //     std::cout << "\n";
        // }

        int num_layers = 0;
        while (curr_input.size > 1)
        {
            // {
            //     thrust::host_vector<symbol_t> tmp5(curr_input.iter, curr_input.iter + curr_input.size);
            //     for (auto el : tmp5)
            //     {
            //         std::cout << el << ", ";
            //     }
            //     std::cout << "\n";
            // }
            auto t1 = high_resolution_clock::now();
            auto new_num_symbols = bcomp(arena, curr_input, unit_size, num_symbols, rules);
            auto t2 = high_resolution_clock::now();
            duration<double, std::milli> ms_double = t2 - t1;
            std::cout << "bcomp " << (ms_double.count()) << " ms, " << num_symbols << " symbols" << std::endl;
            num_symbols = new_num_symbols;
            // {
            //     thrust::host_vector<symbol_t> tmp5(curr_input.iter, curr_input.iter + curr_input.size);
            //     for (auto el : tmp5)
            //     {
            //         std::cout << el << ", ";
            //     }
            //     std::cout << "\n";
            // }

            // exit(0);

            num_layers++;

            if (curr_input.size == 1)
                break;

            auto new_input = arena.view_start_at_items<symbol_t>(3 * unit_size, curr_input.size);

            // {
            //     thrust::host_vector<symbol_t> tmp5(new_input.iter, new_input.iter + new_input.size);
            //     for (auto el : tmp5)
            //     {
            //         std::cout << el << ", ";
            //     }
            //     std::cout << "\n";
            // }

            // auto curr_input_3 = thrust::device_vector<symbol_t>(new_input.iter, new_input.iter + new_input.size);

            auto t3 = high_resolution_clock::now();
            auto t4 = high_resolution_clock::now();
            int cnt = 0;
            symbol_t orig_symbols = num_symbols;
            while (true)
            {
                size_t prev_size = new_input.size;
                auto [curr_input_new2, new_num_symbols2] = pcomp(arena, new_input, unit_size, num_symbols, rules, cnt);
                // {
                //     thrust::host_vector<symbol_t> tmp5(curr_input_new2.begin(), curr_input_new2.end());
                //     for (auto el : tmp5)
                //     {
                //         std::cout << el << ", ";
                //     }
                //     std::cout << "\n";
                // }
                // exit(1);
                t4 = high_resolution_clock::now();
                if (curr_input_new2.size() != prev_size)
                {
                    // curr_input_3 = curr_input_new2;
                    thrust::copy(curr_input_new2.begin(), curr_input_new2.end(), curr_input.iter);
                    curr_input.shrink(curr_input_new2.size());
                    num_symbols = new_num_symbols2;
                    break;
                }
                cnt++;
            }
            duration<double, std::milli> ms_double2 = t4 - t3;
            std::cout << "pcomp " << (ms_double2.count()) << " ms, " << orig_symbols << " symbols before, " << num_symbols << " symbols after" << std::endl;

            num_layers++;

            exit(1);

            // {
            //     thrust::host_vector<symbol_t> tmp5(curr_input.iter, curr_input.iter + curr_input.size);
            //     for (auto el : tmp5)
            //     {
            //         std::cout << el << ", ";
            //     }
            //     std::cout << "\n";
            // }
        }

        std::cout << "layers: " << num_layers << std::endl;
    }
}