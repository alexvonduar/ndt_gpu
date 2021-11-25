#include <algorithm>
#include <cassert>
#include <iostream>

#include "fast_pcl/ndt_gpu/CUDADeviceMemoryPool.hpp"
#include "fast_pcl/ndt_gpu/debug.h"

static gpu::CUDADeviceMemoryPool* foo = gpu::CUDADeviceMemoryPool::getInstance();

gpu::CUDADeviceMemoryPool::~CUDADeviceMemoryPool()
{
    for (auto& block : memory_pool_)
    {
        checkCudaErrors(cudaFree(block.ptr));
    }
    memory_pool_.clear();
}

gpu::CUDADeviceMemoryPool::CUDADeviceMemoryPool()
{
}

std::ptrdiff_t gpu::CUDADeviceMemoryPool::getIndexByTag(const int64_t& tag)
{
    //diff_t index = -1;
    MemoryBlock block{nullptr, 0, tag};
    const auto it = std::find(
        memory_pool_.begin(),
        memory_pool_.end(), block);
    if (it != memory_pool_.end())
    {
        return std::distance(memory_pool_.begin(), it);
    } else {
        return -1;
    }
}

void * gpu::CUDADeviceMemoryPool::allocate(const int64_t& tag, const size_t& size)
{
    const auto index = getIndexByTag(tag);

    if (index < 0) {
        memory_pool_.emplace_back(MemoryBlock{nullptr, size, tag});
        return memory_pool_.back().ptr;
    } else {
        const auto it = memory_pool_.begin() + index;
        if (it->size < size) {
            assert(it->ptr != nullptr);
            free(tag, it->ptr);
            it->ptr = nullptr;
            checkCudaErrors(cudaMalloc(&it->ptr, size));
            it->size = size;
        }
        return it->ptr;
    }
}

void gpu::CUDADeviceMemoryPool::free(const int64_t& tag, void * ptr)
{
    const auto index = getIndexByTag(tag);

    if (index < 0) {
        return;
    } else {
        const auto it = memory_pool_.begin() + index;
        assert(it->ptr == ptr);
        checkCudaErrors(cudaFree(it->ptr));
        it->ptr = nullptr;
        it->size = 0;
        return;
    }
}

void gpu::CUDADeviceMemoryPool::print()
{
    for(const auto& block : memory_pool_) {
        std::cout << "tag: " << block.tag << ", ptr: " << block.ptr << ", size: " << block.size << std::endl;
    }
}
