#include "../include/Communicator.h"
#include <iostream>

#if defined(MOCK_HIP)
// In mock mode, we use the mock_rccl implementation
#else
// Real RCCL calls would go here
#endif

Communicator::Communicator(int rank, int world_size) : rank_(rank), world_size_(world_size), comm_(nullptr) {
    // In a real implementation, we would call ncclCommInitRank
#if defined(MOCK_HIP)
    // Initialize mock communicator
#endif
}

Communicator::~Communicator() {
    // In a real implementation, we would call ncclCommDestroy
}

void Communicator::all_reduce_async(void* sendbuff, void* recvbuff, size_t count, int datatype, int op, uintptr_t stream_ptr) {
#if defined(MOCK_HIP)
    mock_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm_, reinterpret_cast<void*>(stream_ptr));
#else
    // ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm_, reinterpret_cast<hipStream_t>(stream_ptr));
#endif
}
