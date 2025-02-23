#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/remove.h>

#include <chrono>
#include <iostream>
#include "types.hpp"

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

    // 9
    // new input, new num symbols
    symbol_t bcomp(thrust::device_vector<symbol_t> &curr_input, symbol_t num_symbols)
    {
        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::nanoseconds;
        thrust::device_vector<symbol_t> keys(curr_input.size());
        thrust::device_vector<symbol_t> counts(curr_input.size());
        auto [keys_end, counts_end] = thrust::reduce_by_key(curr_input.begin(), curr_input.end(), thrust::make_constant_iterator(1), keys.begin(), counts.begin());
        symbol_t num_blocks = counts.size() - thrust::count(counts.begin(), counts.end(), 1);
        if (num_blocks == 0)
        {
            return num_symbols;
        }
        thrust::copy(keys.begin(), keys.end(), curr_input.begin());

        thrust::device_vector<symbol_t> indices(curr_input.size());
        thrust::sequence(indices.begin(), indices.end());
        auto zipped_iter = thrust::make_zip_iterator(keys.begin(), counts.begin(), indices.begin());
        auto zipped_iter_end = thrust::remove_if(zipped_iter, zipped_iter + (keys_end - keys.begin()), [] __device__(const thrust::tuple<symbol_t, symbol_t, symbol_t> &item)
                                                 { return thrust::get<1>(item) > 1; });

        // // symbol, count, index where count > 1
        // thrust::device_vector<thrust::tuple<symbol_t, symbol_t, symbol_t>> symbols_with_counts(keys_end - keys.begin());

        // auto zipped_iter = thrust::make_zip_iterator(keys.begin(), counts.begin(), thrust::make_counting_iterator(0));
        // auto copy_end = thrust::copy_if(zipped_iter, zipped_iter + symbols_with_counts.size(), symbols_with_counts.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t, symbol_t> &item)
        //                                 { return thrust::get<1>(item) > 1; });
        if (zipped_iter_end - zipped_iter > 0)
        {
            auto t1 = high_resolution_clock::now();
            thrust::stable_sort(zipped_iter, zipped_iter_end);
            auto t2 = high_resolution_clock::now();
            duration<double, std::milli> ms_double = t2 - t1;
            std::cout << "bcomp sort time " << (ms_double.count()) << " ms\n";

            // new offset, index to scatter back to!
            // thrust::device_vector<thrust::tuple<symbol_t, symbol_t>> offsets_and_indices(copy_end - symbols_with_counts.begin()); // this can be made to share space with keys/counts

            thrust::device_vector<symbol_t> offsets(zipped_iter_end - zipped_iter);

            // thrust::tuple<symbol_t, symbol_t, symbol_t>
            //     x = zipped_iter[0];
            offsets[0] = 0;
            auto prev_and_next = thrust::make_zip_iterator(zipped_iter, zipped_iter + 1);
            thrust::transform_inclusive_scan(prev_and_next, prev_and_next + offsets.size() - 1, offsets.begin() + 1, [] __device__(const thrust::tuple<thrust::tuple<symbol_t, symbol_t, symbol_t>, thrust::tuple<symbol_t, symbol_t, symbol_t>> &item) -> symbol_t
                                             {
            bool eq = thrust::get<0>(thrust::get<0>(item)) == thrust::get<0>(thrust::get<1>(item)) && thrust::get<1>(thrust::get<0>(item)) == thrust::get<1>(thrust::get<1>(item));
            return (symbol_t)(!eq); }, [] __device__(const symbol_t &a, const symbol_t &b) -> symbol_t
                                             { return a + b; });

            symbol_t last_offset = offsets[offsets.size() - 1];
            // thrust::tuple<symbol_t, symbol_t> last_offset = offsets_and_indices[offsets_and_indices.size() - 1];

            auto offset_iter = thrust::make_transform_iterator(offsets.begin(), [num_symbols] __device__(const symbol_t &offset) -> symbol_t
                                                               { return num_symbols + offset; });
            // auto index_iter = thrust::make_transform_iterator(offsets_and_indices.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t> &item) -> symbol_t
            //                                                   { return thrust::get<1>(item); })
            symbol_t num_new_symbols = last_offset + 1;
            num_symbols += num_new_symbols;

            // thrust::device_vector<symbol_t> next_input(symbols_with_counts.size());
            // thrust::transform(zipped_iter, zipped_iter + symbols_with_counts.size(), curr_input.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t, symbol_t> &item) -> symbol_t
            //                   { return thrust::get<0>(item); });
            thrust::scatter(offset_iter, offset_iter + offsets.size(), indices.begin(), curr_input.begin());
        }
        curr_input.resize(zipped_iter_end - zipped_iter);
        return num_symbols;
        // return {thrust::device_vector<symbol_t>(curr_input.begin(), curr_input_end), num_symbols};
    }

    // 6
    // new input, new num symbols
    std::tuple<thrust::device_vector<symbol_t>, symbol_t> pcomp(thrust::device_vector<symbol_t> curr_input, symbol_t num_symbols, int x)
    {
        auto curr_input_end = curr_input.end();
        thrust::device_vector<uint8_t> pseudo_rand_bits(num_symbols);
        auto counting_iter = thrust::make_counting_iterator(0);
        thrust::transform(counting_iter, counting_iter + pseudo_rand_bits.size(), pseudo_rand_bits.begin(), efficient_random_byte_functor(num_symbols + x));

        thrust::device_vector<uint8_t> assigned_bit(curr_input_end - curr_input.begin());
        thrust::gather(curr_input.begin(), curr_input_end, pseudo_rand_bits.begin(), assigned_bit.begin());

        thrust::adjacent_difference(assigned_bit.begin(), assigned_bit.end(), assigned_bit.begin(), [] __device__(uint8_t curr, uint8_t prev) -> uint8_t
                                    { return curr && !prev; });
        assigned_bit[0] = 0;

        // if (thrust::count(assigned_bit.begin(), assigned_bit.end(), 1) != 0) {

        // pair, index
        thrust::device_vector<thrust::tuple<symbol_t, symbol_t, symbol_t>> pairs_and_sources(assigned_bit.size() / 2);

        auto iter_adj = thrust::make_zip_iterator(curr_input.begin(), curr_input.begin() + 1, thrust::make_counting_iterator(1));

        // auto iter_adj = thrust::make_zip_iterator(assigned_bit.begin() + 1, curr_input.begin(), curr_input.begin() + 1, thrust::make_counting_iterator(1));
        // auto pairs_and_sources_end = thrust::transform_if(iter_adj, iter_adj + assigned_bit.size() - 1, pairs_and_sources.begin(), [] __device__(const thrust::tuple<uint8_t, symbol_t, symbol_t, symbol_t> &item) -> thrust::tuple<symbol_t, symbol_t, symbol_t>
        //                                                   { return thrust::make_tuple(thrust::get<1>(item), thrust::get<2>(item), thrust::get<3>(item)); }, [] __device__(const thrust::tuple<uint8_t, symbol_t, symbol_t, symbol_t> &item) -> bool
        //                                                   { return false; });
        auto pairs_and_sources_end = thrust::copy_if(iter_adj, iter_adj + assigned_bit.size() - 1, assigned_bit.begin() + 1, pairs_and_sources.begin(), thrust::identity<uint8_t>());

        // {
        //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input_end);
        //     for (auto el : tmp5)
        //     {
        //         std::cout << el << ", ";
        //     }
        //     std::cout << "\n";
        // }

        if (pairs_and_sources_end - pairs_and_sources.begin() > 0)
        {
            thrust::stable_sort(pairs_and_sources.begin(), pairs_and_sources_end);

            // new offset, index to scatter back to
            thrust::device_vector<thrust::tuple<symbol_t, symbol_t>> offsets_and_indices(pairs_and_sources_end - pairs_and_sources.begin()); // this can be made to share space with keys/counts
            thrust::tuple<symbol_t, symbol_t, symbol_t> x = pairs_and_sources[0];
            offsets_and_indices[0] = thrust::make_tuple(0, thrust::get<2>(x));
            auto prev_and_next = thrust::make_zip_iterator(pairs_and_sources.begin(), pairs_and_sources.begin() + 1);
            thrust::transform_inclusive_scan(prev_and_next, prev_and_next + offsets_and_indices.size() - 1, offsets_and_indices.begin() + 1, [] __device__(const thrust::tuple<thrust::tuple<symbol_t, symbol_t, symbol_t>, thrust::tuple<symbol_t, symbol_t, symbol_t>> &item) -> thrust::tuple<symbol_t, symbol_t>
                                             {
            bool eq = thrust::get<0>(thrust::get<0>(item)) == thrust::get<0>(thrust::get<1>(item)) && thrust::get<1>(thrust::get<0>(item)) == thrust::get<1>(thrust::get<1>(item));
            return thrust::make_tuple((symbol_t)(!eq), thrust::get<2>(thrust::get<1>(item))); }, [] __device__(const thrust::tuple<symbol_t, symbol_t> &a, const thrust::tuple<symbol_t, symbol_t> &b) -> thrust::tuple<symbol_t, symbol_t>
                                             { return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b), thrust::get<1>(b)); });

            thrust::tuple<symbol_t, symbol_t> last_el_of_offsets_and_indices = offsets_and_indices[offsets_and_indices.size() - 1];

            auto offset_iter = thrust::make_transform_iterator(offsets_and_indices.begin(), [num_symbols] __device__(const thrust::tuple<symbol_t, symbol_t> &item) -> symbol_t
                                                               { return num_symbols + thrust::get<0>(item); });
            auto index_iter = thrust::make_transform_iterator(offsets_and_indices.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t> &item) -> symbol_t
                                                              { return thrust::get<1>(item); });
            symbol_t num_new_symbols = thrust::get<0>(last_el_of_offsets_and_indices) + 1;
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

            thrust::scatter(offset_iter, offset_iter + offsets_and_indices.size(), index_iter, curr_input.begin());
            // {
            //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input_end);
            //     for (auto el : tmp5)
            //     {
            //         std::cout << el << ", ";
            //     }
            //     std::cout << "\n";
            // }
            auto curr_input_and_idx = thrust::make_zip_iterator(curr_input.begin(), thrust::make_counting_iterator(0));
            auto assigned_bit_raw_ptr = thrust::device_pointer_cast(&assigned_bit[0]);
            auto new_end = thrust::remove_if(curr_input_and_idx, curr_input_and_idx + (curr_input_end - curr_input.begin()), [assigned_bit_raw_ptr] __device__(const thrust::tuple<symbol_t, symbol_t> &item)
                                             {
                auto idx = thrust::get<1>(item);
                return *(assigned_bit_raw_ptr + idx + 1) == 1; });

            curr_input_end = curr_input.begin() + (new_end - curr_input_and_idx);

            // now shift one to the left, and do a remove_copy_if(?) to remove just the elements we don't care about :)
        }
        return {thrust::device_vector<symbol_t>(curr_input.begin(), curr_input_end), num_symbols};
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

        thrust::device_vector<symbol_t> curr_input = input;

        // {
        //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input.end());
        //     for (auto el : tmp5)
        //     {
        //         std::cout << el << ", ";
        //     }
        //     std::cout << "\n";
        // }

        int num_layers = 0;
        while (curr_input.size() > 1)
        {
            auto t1 = high_resolution_clock::now();
            auto new_num_symbols = bcomp(curr_input, num_symbols);
            auto t2 = high_resolution_clock::now();
            duration<double, std::milli> ms_double = t2 - t1;
            std::cout << "bcomp " << (ms_double.count()) << " ms\n";
            num_symbols = new_num_symbols;

            // {
            //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input.end());
            //     for (auto el : tmp5)
            //     {
            //         std::cout << el << ", ";
            //     }
            //     std::cout << "\n";
            // }

            num_layers++;

            if (curr_input.size() == 1)
                break;

            auto t3 = high_resolution_clock::now();

            int cnt = 0;
            while (true)
            {
                auto [curr_input_new2, new_num_symbols2] = pcomp(curr_input, num_symbols, cnt);
                if (curr_input_new2.size() != curr_input.size())
                {
                    curr_input = curr_input_new2;
                    num_symbols = new_num_symbols2;
                    break;
                }
                cnt++;
            }
            auto t4 = high_resolution_clock::now();
            duration<double, std::milli> ms_double2 = t4 - t3;
            std::cout << "pcomp " << (ms_double2.count()) << " ms\n";

            num_layers++;

            // {
            //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input.end());
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