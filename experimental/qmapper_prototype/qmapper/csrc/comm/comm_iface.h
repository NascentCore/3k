#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <deque>
#include <string>
#include <vector>
#include <ucs/type/status.h>

#ifndef QMAP_COMM_IFACE_H
#define QMAP_COMM_IFACE_H

#define GET_BIT(i) (1ul <<(i))
#define GET_MSK(i) (GET_BIT(i)-1)
#define PTR_OFFSET(_ptr, _offset) ((void *)((ptrdiff_t)(_ptr) + (size_t)(_offset)))

typedef enum {
    COMM_OK,
    COMM_INPROGRESS,
    COMM_INITIALIZING,
    COMM_ERROR,
} comm_status_t;

typedef enum comm_memory_type {
    COMM_MEMORY_TYPE_HOST,         /*!< Default system memory */
    COMM_MEMORY_TYPE_CUDA,         /*!< NVIDIA CUDA memory */
    COMM_MEMORY_TYPE_CUDA_MANAGED, /*!< NVIDIA CUDA managed memory */
    COMM_MEMORY_TYPE_ROCM,         /*!< AMD ROCM memory */
    COMM_MEMORY_TYPE_ROCM_MANAGED, /*!< AMD ROCM managed system memory */
    COMM_MEMORY_TYPE_UNKNOWN,
    COMM_MEMORY_TYPE_LAST,
} comm_memory_type_t;

typedef enum {
    COMM_DATATYPE_PREDEFINED = 0,
    COMM_DATATYPE_GENERIC    = GET_BIT(0),
    COMM_DATATYPE_SHIFT      = 3,
    COMM_DATATYPE_CLASS_MASK = GET_MSK(COMM_DATATYPE_SHIFT)
} comm_dt_type_t;

#define COMM_PREDEFINED_DT(_id) \
    (uint64_t)((((uint64_t)(_id)) << COMM_DATATYPE_SHIFT) | \
                     (COMM_DATATYPE_PREDEFINED))

typedef enum comm_datatype {
    COMM_DATATYPE_INT8 = COMM_PREDEFINED_DT(0),
    COMM_DATATYPE_INT16 = COMM_PREDEFINED_DT(1),
    COMM_DATATYPE_INT32 = COMM_PREDEFINED_DT(2),
    COMM_DATATYPE_INT64 = COMM_PREDEFINED_DT(3),
    COMM_DATATYPE_UINT8 = COMM_PREDEFINED_DT(4),
    COMM_DATATYPE_UINT16 = COMM_PREDEFINED_DT(5),
    COMM_DATATYPE_UINT32 = COMM_PREDEFINED_DT(6),
    COMM_DATATYPE_UINT64 = COMM_PREDEFINED_DT(7),
    COMM_DATATYPE_FLOAT16 = COMM_PREDEFINED_DT(8),
    COMM_DATATYPE_FLOAT32 = COMM_PREDEFINED_DT(9),
    COMM_DATATYPE_FLOAT64 = COMM_PREDEFINED_DT(10),
    COMM_DATATYPE_BFLOAT16 = COMM_PREDEFINED_DT(11),
    COMM_DATATYPE_PREDEFINED_LAST = COMM_PREDEFINED_DT(12),
} comm_datatype_t;

typedef enum {
    COMM_OP_SUM,
    COMM_OP_PROD,
    COMM_OP_MAX,
    COMM_OP_MIN,
    COMM_OP_LAND,
    COMM_OP_LOR,
    COMM_OP_LXOR,
    COMM_OP_BAND,
    COMM_OP_BOR,
    COMM_OP_BXOR,
    COMM_OP_MAXLOC,
    COMM_OP_MINLOC,
    COMM_OP_AVG,
    COMM_OP_LAST
} comm_reduction_op_t;

typedef enum {
    THREAD_SINGLE       = 0,
    THREAD_FUNNELED     = 1,
    THREAD_MULTIPLE     = 2
} thread_mode_t;

typedef struct comm_addr_storage {
    void     *storage;
    size_t   addr_len;
    uint32_t size;
    uint32_t rank;
    uint64_t flag;
} comm_addr_storage_t;

typedef struct comm_eps_storage {
    void   *storage;
    size_t ep_len;
} comm_eps_storage_t;

typedef struct comm_buffer_info {
    void     *buf;
    uint64_t count;
    comm_datatype_t datatype;
    comm_memory_type_t memory_type;
} comm_buffer_info_t;

namespace qmap {
namespace comm {

class CommIface;
class CommOOBIface;
class CommTeam;
class CommContext;
class CommTopo;
class CommTask;
class CommExecuteEngine;
class CommEvent;

namespace memory_pool {
    class CommMemoryPool;
};

comm_status_t ucs_status_to_qmap_status(ucs_status_t status);
ucs_status_t qmap_status_to_ucs_status(comm_status_t status);



class CommIface {
public:
    CommIface() {}
    ~CommIface() {}
    virtual comm_status_t init(std::map<std::string, uint64_t> &params);
    virtual comm_status_t finialize();
    virtual std::map<std::string, uint64_t> &get_params();
    virtual CommContext &create_context();
    virtual comm_status_t    destory_context(CommContext *ctx);

public:
    CommIface                       *super;
    std::string                     prefix;
    std::string                     level;
    std::vector<CommIface>          algo_level_ifaces;
    std::vector<CommIface>          transport_level_ifaces;
    std::map<std::string, uint64_t> params;
    memory_pool::CommMemoryPool     *mpool;
    std::vector<CommContext>        created_ctxs;
};

class CommOOBIface {
public:
    virtual comm_status_t allgather(void *src_buf, void *recv_buf, size_t size, void *allgather_info, void **reqs);
    virtual comm_status_t req_test(void *req);
    virtual comm_status_t req_free(void *req);
public:
    void                            *coll_info;
    uint64_t                        n_oob_eps;
    uint64_t                        oob_ep;
    comm_eps_storage_t                   eps_storage;
};

class CommTopo {
public:

public:

};

class CommContext {
public:
    CommContext() {}
    ~CommContext() {}
    virtual comm_status_t init(std::map<std::string, uint64_t> &params);
    virtual comm_status_t finialize();
    virtual std::map<std::string, uint64_t> &get_params();
    virtual comm_status_t progress();
    virtual CommTeam &create_team();
    virtual comm_status_t create_team_test(CommTeam &team);
    virtual comm_status_t destory_team(CommTeam *team);
public:
    CommIface                       *lib;
    CommContext                     *super;
    std::string                     prefix;
    std::string                     level;
    uint64_t                        context_id;
    std::vector<CommContext>        algo_ctxs;
    std::vector<CommContext>        transport_ctxs;
    CommContext                     *service_ctx;
    std::map<std::string, uint64_t> params;
    CommTopo                        context_topo;
    comm_addr_storage_t                  addr_storage;
};


class CommTeam {
public:
    CommTeam() {}
    ~CommTeam() {}
    virtual comm_status_t init(std::map<std::string, uint64_t> &params);
    virtual comm_status_t finialize();
    virtual std::map<std::string, uint64_t> &get_params();
    virtual CommTeam &create_from_this();
    virtual CommTask &create_task(std::map<std::string, uint64_t> &params);



public:
    std::vector<CommContext *>     ctxs;
    uint32_t                       team_id;
    std::string                    prefix;
    std::string                    level;
    std::vector<CommTeam>          algo_teams;
    std::vector<CommTeam>          transport_teams;
    CommTopo                       topo;
    std::vector<uint32_t>          ctx_ranks;
    comm_eps_storage_t                  ep_storage;
    CommOOBIface                   oob;
    CommTeam                       *service_team;
    uint64_t                       ep;
    uint64_t                       team_size;

    int scope;
    int scope_id;
    uint64_t rank;
    uint64_t size;
    uint16_t id;

};

typedef enum {
    COLL_TYPE_ALLGATHER          = GET_BIT(0),
    COLL_TYPE_ALLGATHERV         = GET_BIT(1),
    COLL_TYPE_ALLREDUCE          = GET_BIT(2),
    COLL_TYPE_ALLTOALL           = GET_BIT(3),
    COLL_TYPE_ALLTOALLV          = GET_BIT(4),
    COLL_TYPE_BARRIER            = GET_BIT(5),
    COLL_TYPE_BCAST              = GET_BIT(6),
    COLL_TYPE_FANIN              = GET_BIT(7),
    COLL_TYPE_FANOUT             = GET_BIT(8),
    COLL_TYPE_GATHER             = GET_BIT(9),
    COLL_TYPE_GATHERV            = GET_BIT(10),
    COLL_TYPE_REDUCE             = GET_BIT(11),
    COLL_TYPE_REDUCE_SCATTER     = GET_BIT(12),
    COLL_TYPE_REDUCE_SCATTERV    = GET_BIT(13),
    COLL_TYPE_SCATTER            = GET_BIT(14),
    COLL_TYPE_SCATTERV           = GET_BIT(15),
    COLL_TYPE_LAST
} coll_type_t;

typedef enum {
    ERROR_LOCAL = 0,
    ERROR_GLOBAL = 1,
} error_type_t;

typedef struct coll_callback {
    void (*cb)(void *data, comm_status_t status);
    void  *data;
} coll_callback_t;

typedef struct coll_args {
    uint64_t           mask;
    coll_type_t        coll_type;
    comm_buffer_info_t      src_buf_info;
    comm_buffer_info_t      dst_buf_info;
    comm_reduction_op_t     op; 
    uint64_t           flags; 
    uint64_t           root; 
    error_type_t       error_type;
    uint16_t           tag; 
    void               *global_work_buffer;
    coll_callback_t    cb;
    double             timeout;
    struct {
        uint64_t start;
        int64_t  stride;
        uint64_t size;
    } active_set;
} coll_args_t;

enum coll_args_field {
    COLL_ARGS_FIELD_FLAGS                           = GET_BIT(0),
    COLL_ARGS_FIELD_TAG                             = GET_BIT(1),
    COLL_ARGS_FIELD_CB                              = GET_BIT(2),
    COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER              = GET_BIT(3),
    COLL_ARGS_FIELD_ACTIVE_SET                      = GET_BIT(4)
};



class CommTask {
public:

public:
    CommTask() {}
    ~CommTask() {}
    virtual comm_status_t init(std::map<std::string, uint64_t> &params);
    virtual comm_status_t finialize();
    virtual comm_status_t test();
public:
    std::string                    name;
    std::string                    type;
    comm_status_t                       status;
    coll_args_t                    args;
    CommTeam                       *team;
    uint64_t                       root_ep;
    CommExecuteEngine              *ee;
    CommEvent                      *event;
    
};

}; // namespace comm
}; // namespace qmap

#endif