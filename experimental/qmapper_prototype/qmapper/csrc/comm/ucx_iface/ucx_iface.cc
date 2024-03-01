#include "ucx_iface/ucx_iface.h"
#include "comm_iface.h"
#include <cstdint>
#include <glog/logging.h>


namespace qmap {
namespace comm {

std::map<std::string, uint64_t> ucx_comm_iface_default_init() {
    std::map<std::string, uint64_t> params;
    // default comm init
    params["ALLTOALL_PAIRWISE_NUM_POSTS"] = 1ul;                    // Maximum number of outstanding send and receive messages in alltoall 
                                                                    // pairwise algorithm
    params["ALLTOALLV_PAIRWISE_NUM_POSTS"] = 1ul;                   // Maximum number of outstanding send and receive messages in alltoallv 
                                                                    // pairwise algorithm
    params["ALLTOALLV_HYBRID_NUM_SCRATCH_SENDS"] = 1ul;             // Number of send operations issued from scratch buffer per radix step
    params["ALLTOALLV_HYBRID_NUM_SCRATCH_RECVS"] = 3ul;             // Number of recv operations issued from scratch buffer per radix step
    params["ALLTOALLV_HYBRID_PAIRWISE_NUM_POSTS"] = 3ul;            // The maximum number of pairwise messages to send before waiting for completion
    params["ALLTOALLV_HYBRID_BUFF_SIZE"] = 256*1024ul;              // Total size of scratch buffer, used for sends and receives
    params["ALLTOALLV_HYBRID_CHUNK_BYTE_LIMIT"] = 12*1024ul;        // Max size of data send in pairwise step of hybrid alltoallv algorithm
    
    params["KN_RADIX"] = 0ul;                                       // Radix of all algorithms based on knomial pattern. When set to a positive value it is used as a convinience parameter to set all other KN_RADIX values
    params["BARRIER_KN_RADIX"] = 4ul;                               // Radix of the recursive-knomial barrier algorithm

    params["GATHERV_LINEAR_NUM_POSTS"] = 0ul;                       // Maximum number of outstanding send and receive messages in gatherv linear algorithm
    params["SCATTERV_LINEAR_NUM_POSTS"] = 16ul;                     // Maximum number of outstanding send and receive messages in scatterv linear algorithm
    params["REDUCE_AVG_PRE_OP"] = 1ul;                              // Reduce will perform division by team_size in early stages of the algorithm, ekse - in result 
    params["REDUCE_SCATTER_RING_BIDIRECTIONAL"] = 1ul;              // Launch 2 inverted rings concurrently during ReduceScatterV Ring algorithm
    params["USE_TOPO"] = 1ul;                                       // Allow usage of tl ucp topo
    params["RANKS_REORDERING"] = 1ul;                               // Use topology information in TL UCP to reorder ranks. Requires topo info
    return params;
}

std::map<std::string, uint64_t> ucx_comm_context_default_init() {
    std::map<std::string, uint64_t> params;
    // default context init
    params["PRECONNECT"] = 0ul;                                     // Threshold that defines the number of ranks in the team/context below which the team/context enpoints will be preconnected during corresponding team/context create call
    params["NPOLLS"] = 10ul;                                        // Number of ucp progress polling cycles for p2p requests testing
    params["OOB_NPOLLS"] = 20ul;                                    // Number of polling cycles for oob allgather and service coll request
    params["PRE_REG_MEM"] = 0ul;                                    // Pre Register collective memory region with UCX
    params["SERVICE_WORKER"] = 0ul;                                 // If set to 0, uses the same worker for collectives and  service. If not, creates a special worker for service collectives for which UCX_TL and UCX_NET_DEVICES are configured by the variables TL_UCP_SERVICE_TLS and TL_UCP_SERVICE_NET_DEVICES respectively  
    params["SERVICE_THROTTLING_THRESH"] = 100ul;                    // Number of call to ucc_context_progress function between two consecutive calls to service worker progress function
    return params;
}

status_t UcxCommIface::init(std::map<std::string, uint64_t> &params) {
    std::map<std::string, uint64_t> default_params = ucx_comm_iface_default_init();
    for (const auto& pair: default_params) {
        this->params.insert(pair);
    }
    for(const auto& pair: params) {
        this->params.insert(pair);
    }

    
    return OK;
}


status_t UcxContext::init(std::map<std::string, uint64_t> &params) {
    params[""];
    return OK;
}


}; // namespace comm
}; // namespace qmap




