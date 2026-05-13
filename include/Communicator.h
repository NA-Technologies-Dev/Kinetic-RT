#pragma once

#include <vector>
#include <mutex>

#if defined(MOCK_HIP)
#include "../tests/mock_hip.h"
#include "../tests/mock_rccl.h"
#else
#include <hip/hip_runtime.h>
// #include <rccl/rccl.h> // Mocked for now
typedef void* ncclComm_t;
#endif

class Communicator {
public:
    Communicator(int rank, int world_size);
    ~Communicator();

    void all_reduce_async(void* sendbuff, void* recvbuff, size_t count, int datatype, int op, uintptr_t stream_ptr);

private:
    int rank_;
    int world_size_;
    ncclComm_t comm_;
};
