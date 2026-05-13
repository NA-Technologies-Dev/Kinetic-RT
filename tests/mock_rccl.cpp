#include "mock_rccl.h"

std::map<ncclComm_t, MockAllReduceOp> global_mock_rccl_ops;
