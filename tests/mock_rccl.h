#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <stdexcept>
#include <map>

typedef void* ncclComm_t;

// Minimal RCCL/NCCL constants
#define ncclFloat32 0
#define ncclSum 0

struct MockAllReduceOp {
    int expected_ranks;
    int arrived_ranks = 0;
    std::vector<float*> recv_buffers;
    std::mutex mtx;
    std::condition_variable cv;
};

extern std::map<ncclComm_t, MockAllReduceOp> global_mock_rccl_ops;

inline void mock_ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, int datatype, int op, ncclComm_t comm, void* stream) {
    if (datatype != ncclFloat32 || op != ncclSum) {
        throw std::runtime_error("mock_ncclAllReduce only supports Float32 Sum");
    }

    // Auto-register comm if not exists
    if (global_mock_rccl_ops.find(comm) == global_mock_rccl_ops.end()) {
        // Assume 2 expected ranks for our mock topology test
        global_mock_rccl_ops[comm].expected_ranks = 2;
    }

    MockAllReduceOp& mock_op = global_mock_rccl_ops[comm];

    std::unique_lock<std::mutex> lock(mock_op.mtx);
    mock_op.recv_buffers.push_back(static_cast<float*>(recvbuff));
    mock_op.arrived_ranks++;

    // Copy send to recv buffer
    const float* send_f32 = static_cast<const float*>(sendbuff);
    float* recv_f32 = static_cast<float*>(recvbuff);
    for (size_t i = 0; i < count; ++i) {
        recv_f32[i] = send_f32[i];
    }

    if (mock_op.arrived_ranks == mock_op.expected_ranks) {
        // Perform actual sum reduction across all recv buffers
        for (size_t i = 0; i < count; ++i) {
            float sum = 0.0f;
            for (auto* buf : mock_op.recv_buffers) {
                sum += buf[i];
            }
            // Write sum back to all recv buffers
            for (auto* buf : mock_op.recv_buffers) {
                buf[i] = sum;
            }
        }

        // Reset state for next iteration
        mock_op.arrived_ranks = 0;
        mock_op.recv_buffers.clear();

        // Wake up waiting threads
        mock_op.cv.notify_all();
    } else {
        // Wait for other ranks
        if (!mock_op.cv.wait_for(lock, std::chrono::seconds(5), [&mock_op]{ return mock_op.arrived_ranks == 0; })) {
            throw std::runtime_error("mock_ncclAllReduce timed out waiting for ranks");
        }
    }
}
