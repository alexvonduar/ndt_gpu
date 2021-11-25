#ifndef _CUDA_DEVICE_MEMORY_POOL_HPP_
#define _CUDA_DEVICE_MEMORY_POOL_HPP_

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

//#if defined(USE_GPU)
#if defined(USE_GPU_MEMORY_POOL)
#undef USE_GPU_MEMORY_POOL
#endif
#define USE_GPU_MEMORY_POOL 1
//#else
//#define USE_GPU_MEMORY_POOL 0
//#endif

namespace gpu
{
static const int64_t TAG(const char& a, const char& b, const char& c, const char& d, const int32_t& l)
{
    return (a << 56) | (b << 48) | (c << 40) | d <<32 | (l & 0xffffffff);
}

class CUDADeviceMemoryPool {
public:
    struct MemoryBlock {
        void *ptr;
        size_t size;
        int64_t tag;
        bool operator==(const MemoryBlock& rhs) const {
            return tag == rhs.tag;
        }
    };

    static CUDADeviceMemoryPool* getInstance() {
        static CUDADeviceMemoryPool instance;
        return &instance;
    };
    void* allocate(const int64_t& tag, const size_t& size);
    void* getPtr(const int64_t& tag);
    size_t getSize(const int64_t& tag);
    void free(const int64_t& tag, void* ptr);
    void print();
private:
    CUDADeviceMemoryPool();
    ~CUDADeviceMemoryPool();
    CUDADeviceMemoryPool(const CUDADeviceMemoryPool&) = delete;
    CUDADeviceMemoryPool& operator=(const CUDADeviceMemoryPool&) = delete;
    std::ptrdiff_t getIndexByTag(const int64_t& tag);
    //static CUDADeviceMemoryPool* instance_;
    //std::vector<void*> memory_pool_;
    //std::vector<size_t> memory_pool_size_;
    //std::vector<int64_t> memory_pool_tag_;
    std::vector<MemoryBlock> memory_pool_;
};
}

#endif//_CUDA_DEVICE_MEMORY_POOL_HPP_
