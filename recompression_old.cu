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

    // new input, new num symbols
    std::tuple<thrust::device_vector<symbol_t>, symbol_t> bcomp(thrust::device_vector<symbol_t> curr_input, symbol_t num_symbols)
    {
        thrust::device_vector<symbol_t> keys(curr_input.size());
        thrust::device_vector<symbol_t> counts(curr_input.size());
        auto [keys_end, counts_end] = thrust::reduce_by_key(curr_input.begin(), curr_input.end(), thrust::make_constant_iterator(1), keys.begin(), counts.begin());

        // symbol, count, index where count > 1
        thrust::device_vector<thrust::tuple<symbol_t, symbol_t, symbol_t>> symbols_with_counts(keys_end - keys.begin());

        auto zipped_iter = thrust::make_zip_iterator(keys.begin(), counts.begin(), thrust::make_counting_iterator(0));
        auto copy_end = thrust::copy_if(zipped_iter, zipped_iter + symbols_with_counts.size(), symbols_with_counts.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t, symbol_t> &item)
                                        { return thrust::get<1>(item) > 1; });
        if (copy_end - symbols_with_counts.begin() > 0)
        {
            thrust::stable_sort(symbols_with_counts.begin(), copy_end);

            // new offset, index to scatter back to!
            thrust::device_vector<thrust::tuple<symbol_t, symbol_t>> offsets_and_indices(copy_end - symbols_with_counts.begin()); // this can be made to share space with keys/counts
            thrust::tuple<symbol_t, symbol_t, symbol_t> x = symbols_with_counts[0];
            offsets_and_indices[0] = thrust::make_tuple(0, thrust::get<2>(x));
            auto prev_and_next = thrust::make_zip_iterator(symbols_with_counts.begin(), symbols_with_counts.begin() + 1);
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

            // thrust::device_vector<symbol_t> next_input(symbols_with_counts.size());
            thrust::transform(zipped_iter, zipped_iter + symbols_with_counts.size(), curr_input.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t, symbol_t> &item) -> symbol_t
                              { return thrust::get<0>(item); });
            thrust::scatter(offset_iter, offset_iter + offsets_and_indices.size(), index_iter, curr_input.begin());
        }
        auto curr_input_end = curr_input.begin() + symbols_with_counts.size();
        return {thrust::device_vector<symbol_t>(curr_input.begin(), curr_input_end), num_symbols};
    }

    // new input, new num symbols
    std::tuple<thrust::device_vector<symbol_t>, symbol_t> pcomp(thrust::device_vector<symbol_t> curr_input, symbol_t num_symbols)
    {
        auto curr_input_end = curr_input.end();
        thrust::device_vector<uint8_t> pseudo_rand_bits(num_symbols);
        auto counting_iter = thrust::make_counting_iterator(0);
        thrust::transform(counting_iter, counting_iter + pseudo_rand_bits.size(), pseudo_rand_bits.begin(), efficient_random_byte_functor(1));

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

    void recompression(symbol_t alphabet_size, std::vector<symbol_t> input)
    {
        symbol_t num_symbols = alphabet_size;

        auto [curr_input, new_num_symbols] = bcomp(input, num_symbols);
        num_symbols = new_num_symbols;

        // thrust::device_vector<symbol_t> curr_input = input;

        // thrust::device_vector<symbol_t> keys(curr_input.size());
        // thrust::device_vector<symbol_t> counts(curr_input.size());
        // auto [keys_end, counts_end] = thrust::reduce_by_key(curr_input.begin(), curr_input.end(), thrust::make_constant_iterator(1), keys.begin(), counts.begin());

        // {
        //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input.end());
        //     for (auto el : tmp5)
        //     {
        //         std::cout << el << ", ";
        //     }
        //     std::cout << "\n";
        // }

        // // symbol, count, index where count > 1
        // thrust::device_vector<thrust::tuple<symbol_t, symbol_t, symbol_t>> symbols_with_counts(keys_end - keys.begin());
        // if (symbols_with_counts.size() > 0)
        // {

        //     auto zipped_iter = thrust::make_zip_iterator(keys.begin(), counts.begin(), thrust::make_counting_iterator(0));
        //     auto copy_end = thrust::copy_if(zipped_iter, zipped_iter + symbols_with_counts.size(), symbols_with_counts.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t, symbol_t> &item)
        //                                     { return thrust::get<1>(item) > 1; });
        //     thrust::stable_sort(symbols_with_counts.begin(), copy_end);

        //     // new offset, index to scatter back to
        //     thrust::device_vector<thrust::tuple<symbol_t, symbol_t>> offsets_and_indices(copy_end - symbols_with_counts.begin()); // this can be made to share space with keys/counts
        //     thrust::tuple<symbol_t, symbol_t, symbol_t> x = symbols_with_counts[0];
        //     offsets_and_indices[0] = thrust::make_tuple(0, thrust::get<2>(x));
        //     auto prev_and_next = thrust::make_zip_iterator(symbols_with_counts.begin(), symbols_with_counts.begin() + 1);
        //     thrust::transform_inclusive_scan(prev_and_next, prev_and_next + offsets_and_indices.size() - 1, offsets_and_indices.begin() + 1, [] __device__(const thrust::tuple<thrust::tuple<symbol_t, symbol_t, symbol_t>, thrust::tuple<symbol_t, symbol_t, symbol_t>> &item) -> thrust::tuple<symbol_t, symbol_t>
        //                                      {
        //     bool eq = thrust::get<0>(thrust::get<0>(item)) == thrust::get<0>(thrust::get<1>(item)) && thrust::get<1>(thrust::get<0>(item)) == thrust::get<1>(thrust::get<1>(item));
        //     return thrust::make_tuple((symbol_t)(!eq), thrust::get<2>(thrust::get<1>(item))); }, [] __device__(const thrust::tuple<symbol_t, symbol_t> &a, const thrust::tuple<symbol_t, symbol_t> &b) -> thrust::tuple<symbol_t, symbol_t>
        //                                      { return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b), thrust::get<1>(b)); });

        //     thrust::tuple<symbol_t, symbol_t> last_el_of_offsets_and_indices = offsets_and_indices[offsets_and_indices.size() - 1];

        //     auto offset_iter = thrust::make_transform_iterator(offsets_and_indices.begin(), [num_symbols] __device__(const thrust::tuple<symbol_t, symbol_t> &item) -> symbol_t
        //                                                        { return num_symbols + thrust::get<0>(item); });
        //     auto index_iter = thrust::make_transform_iterator(offsets_and_indices.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t> &item) -> symbol_t
        //                                                       { return thrust::get<1>(item); });
        //     symbol_t num_new_symbols = thrust::get<0>(last_el_of_offsets_and_indices) + 1;
        //     num_symbols += num_new_symbols;

        //     // thrust::device_vector<symbol_t> next_input(symbols_with_counts.size());
        //     thrust::transform(zipped_iter, zipped_iter + symbols_with_counts.size(), curr_input.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t, symbol_t> &item) -> symbol_t
        //                       { return thrust::get<0>(item); });
        //     thrust::scatter(offset_iter, offset_iter + offsets_and_indices.size(), index_iter, curr_input.begin());
        // }
        // auto curr_input_end = curr_input.begin() + symbols_with_counts.size();

        auto [curr_input2, new_num_symbols2] = pcomp(curr_input, num_symbols);
        curr_input = curr_input2;
        num_symbols = new_num_symbols2;

        // auto curr_input_end = curr_input.end();
        // thrust::device_vector<uint8_t> pseudo_rand_bits(num_symbols);
        // auto counting_iter = thrust::make_counting_iterator(0);
        // thrust::transform(counting_iter, counting_iter + pseudo_rand_bits.size(), pseudo_rand_bits.begin(), efficient_random_byte_functor(1));

        // thrust::device_vector<uint8_t> assigned_bit(curr_input_end - curr_input.begin());
        // thrust::gather(curr_input.begin(), curr_input_end, pseudo_rand_bits.begin(), assigned_bit.begin());

        // thrust::adjacent_difference(assigned_bit.begin(), assigned_bit.end(), assigned_bit.begin(), [] __device__(uint8_t curr, uint8_t prev) -> uint8_t
        //                             { return curr && !prev; });
        // assigned_bit[0] = 0;

        // // if (thrust::count(assigned_bit.begin(), assigned_bit.end(), 1) != 0) {

        // // pair, index
        // thrust::device_vector<thrust::tuple<symbol_t, symbol_t, symbol_t>> pairs_and_sources(assigned_bit.size() / 2);

        // auto iter_adj = thrust::make_zip_iterator(curr_input.begin(), curr_input.begin() + 1, thrust::make_counting_iterator(1));

        // // auto iter_adj = thrust::make_zip_iterator(assigned_bit.begin() + 1, curr_input.begin(), curr_input.begin() + 1, thrust::make_counting_iterator(1));
        // // auto pairs_and_sources_end = thrust::transform_if(iter_adj, iter_adj + assigned_bit.size() - 1, pairs_and_sources.begin(), [] __device__(const thrust::tuple<uint8_t, symbol_t, symbol_t, symbol_t> &item) -> thrust::tuple<symbol_t, symbol_t, symbol_t>
        // //                                                   { return thrust::make_tuple(thrust::get<1>(item), thrust::get<2>(item), thrust::get<3>(item)); }, [] __device__(const thrust::tuple<uint8_t, symbol_t, symbol_t, symbol_t> &item) -> bool
        // //                                                   { return false; });
        // auto pairs_and_sources_end = thrust::copy_if(iter_adj, iter_adj + assigned_bit.size() - 1, assigned_bit.begin() + 1, pairs_and_sources.begin(), thrust::identity<uint8_t>());

        // {
        //     thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input_end);
        //     for (auto el : tmp5)
        //     {
        //         std::cout << el << ", ";
        //     }
        //     std::cout << "\n";
        // }

        // {
        //     thrust::stable_sort(pairs_and_sources.begin(), pairs_and_sources_end);

        //     // new offset, index to scatter back to
        //     thrust::device_vector<thrust::tuple<symbol_t, symbol_t>> offsets_and_indices(pairs_and_sources_end - pairs_and_sources.begin()); // this can be made to share space with keys/counts
        //     thrust::tuple<symbol_t, symbol_t, symbol_t> x = pairs_and_sources[0];
        //     offsets_and_indices[0] = thrust::make_tuple(0, thrust::get<2>(x));
        //     auto prev_and_next = thrust::make_zip_iterator(pairs_and_sources.begin(), pairs_and_sources.begin() + 1);
        //     thrust::transform_inclusive_scan(prev_and_next, prev_and_next + offsets_and_indices.size() - 1, offsets_and_indices.begin() + 1, [] __device__(const thrust::tuple<thrust::tuple<symbol_t, symbol_t, symbol_t>, thrust::tuple<symbol_t, symbol_t, symbol_t>> &item) -> thrust::tuple<symbol_t, symbol_t>
        //                                      {
        //     bool eq = thrust::get<0>(thrust::get<0>(item)) == thrust::get<0>(thrust::get<1>(item)) && thrust::get<1>(thrust::get<0>(item)) == thrust::get<1>(thrust::get<1>(item));
        //     return thrust::make_tuple((symbol_t)(!eq), thrust::get<2>(thrust::get<1>(item))); }, [] __device__(const thrust::tuple<symbol_t, symbol_t> &a, const thrust::tuple<symbol_t, symbol_t> &b) -> thrust::tuple<symbol_t, symbol_t>
        //                                      { return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b), thrust::get<1>(b)); });

        //     thrust::tuple<symbol_t, symbol_t> last_el_of_offsets_and_indices = offsets_and_indices[offsets_and_indices.size() - 1];

        //     auto offset_iter = thrust::make_transform_iterator(offsets_and_indices.begin(), [num_symbols] __device__(const thrust::tuple<symbol_t, symbol_t> &item) -> symbol_t
        //                                                        { return num_symbols + thrust::get<0>(item); });
        //     auto index_iter = thrust::make_transform_iterator(offsets_and_indices.begin(), [] __device__(const thrust::tuple<symbol_t, symbol_t> &item) -> symbol_t
        //                                                       { return thrust::get<1>(item); });
        //     symbol_t num_new_symbols = thrust::get<0>(last_el_of_offsets_and_indices) + 1;
        //     num_symbols += num_new_symbols;

        //     // std::cout << "sz: " << offsets_and_indices.size() << std::endl;
        //     // thrust::host_vector<thrust::tuple<symbol_t, symbol_t>> tmp7 = offsets_and_indices;
        //     // for (auto [e1, e2] : tmp7)
        //     // {
        //     //     std::cout << e1 << " " << e2 << ", ";
        //     // }
        //     // std::cout << std::endl;

        //     {
        //         thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input_end);
        //         for (auto el : tmp5)
        //         {
        //             std::cout << el << ", ";
        //         }
        //         std::cout << "\n";
        //     }

        //     thrust::scatter(offset_iter, offset_iter + offsets_and_indices.size(), index_iter, curr_input.begin());
        //     {
        //         thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input_end);
        //         for (auto el : tmp5)
        //         {
        //             std::cout << el << ", ";
        //         }
        //         std::cout << "\n";
        //     }
        //     auto curr_input_and_idx = thrust::make_zip_iterator(curr_input.begin(), thrust::make_counting_iterator(0));
        //     auto assigned_bit_raw_ptr = thrust::device_pointer_cast(&assigned_bit[0]);
        //     auto new_end = thrust::remove_if(curr_input_and_idx, curr_input_and_idx + (curr_input_end - curr_input.begin()), [assigned_bit_raw_ptr] __device__(const thrust::tuple<symbol_t, symbol_t> &item)
        //                                      {
        //         auto idx = thrust::get<1>(item);
        //         return *(assigned_bit_raw_ptr + idx + 1) == 1; });

        //     curr_input_end = curr_input.begin() + (new_end - curr_input_and_idx);

        //     // now shift one to the left, and do a remove_copy_if(?) to remove just the elements we don't care about :)
        // }

        // }

        // thrust::host_vector<symbol_t>
        //     tmp(keys.begin(), keys_end);
        // for (auto el : tmp)
        // {
        //     std::cout << el << " ";
        // }
        // std::cout << "\n";

        // thrust::host_vector<symbol_t> tmp2(counts.begin(), counts_end);
        // for (auto el : tmp2)
        // {
        //     std::cout << el << " ";
        // }
        // std::cout << "\n";

        // thrust::host_vector<thrust::tuple<symbol_t, symbol_t, symbol_t>> tmp3(symbols_with_counts.begin(), copy_end);
        // for (auto [e1, e2, e3] : tmp3)
        // {
        //     std::cout << e1 << " " << e2 << " " << e3 << ", ";
        // }
        // std::cout << "\n";

        // thrust::host_vector<thrust::tuple<symbol_t, symbol_t>> tmp4(offsets_and_indices.begin(), offsets_and_indices.end());
        // for (auto [e1, e2] : tmp4)
        // {
        //     std::cout << e1 << " " << e2 << ", ";
        // }
        // std::cout << "\n";
        {
            thrust::host_vector<symbol_t> tmp5(curr_input.begin(), curr_input.end());
            for (auto el : tmp5)
            {
                std::cout << el << ", ";
            }
            std::cout << "\n";
        }

        // for (auto el : pseudo_rand_bits)
        // {
        //     std::cout << (int)el;
        // }
        // std::cout << "\n";

        // for (auto el : assigned_bit)
        // {
        //     std::cout << (int)el;
        // }
        // std::cout << "\n";

        // std::cout << (pairs_and_sources_end - pairs_and_sources.begin()) << std::endl;

        // thrust::host_vector<thrust::tuple<symbol_t, symbol_t, symbol_t>>
        //     tmp6(pairs_and_sources.begin(), pairs_and_sources_end);
        // for (auto [e1, e2, e3] : tmp6)
        // {
        //     std::cout << (int)e1 << " " << (int)e2 << " " << (int)e3 << ", ";
        // }
        // std::cout << "\n";
    }
}