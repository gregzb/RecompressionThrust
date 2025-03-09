#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include "types.hpp"

namespace Cu
{
    struct stride_functor
    {
        size_t stride;
        stride_functor() : stride(1) {}
        stride_functor(size_t stride) : stride(stride) {}
        __host__ __device__
            size_t
            operator()(size_t i) const
        {
            return i * stride;
        }
    };

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
};