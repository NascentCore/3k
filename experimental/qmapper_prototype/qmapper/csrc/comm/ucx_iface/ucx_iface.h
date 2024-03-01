#pragma once

#include "comm_iface.h"
#include <cstdint>
#include <ucp/api/ucp_def.h>

#ifndef COMM_UCX_IFACE_H
#define COMM_UCX_IFACE_H

namespace qmap {
namespace comm {

class UcxCommIface : public CommIface {
public:
    UcxCommIface() {}
    ~UcxCommIface() {}
    status_t init(std::map<std::string, uint64_t> &params) final;
};

class UcxOOBIface : public CommOOBIface {
public:
    UcxOOBIface() {}
    ~UcxOOBIface() {}
public:
    status_t allgather(void *src_buf, void *recv_buf, size_t size, void *allgather_info, void **reqs);
};

class UcxContext : public CommContext {
public:
    UcxContext() {}
    ~UcxContext() {}
public:
    status_t init(std::map<std::string, uint64_t> &params) final;
    status_t service_init();
};

class UcxTeam : public CommTeam {
public:
    UcxTeam() {}
    ~UcxTeam() {}
    status_t ucx_get_ep(uint64_t dest_group_rank, ucp_ep_h *ep);
public:
    status_t     status;
    uint32_t     seq_num;
    
};

class UcxTask : public CommTask {
public:
    typedef union {
        struct {
            uint32_t send_posted;
            uint32_t send_completed;
            uint32_t recv_posted;
            uint32_t recv_completed;
            uint32_t tag;
        } tagged;
        struct {
            uint32_t put_posted;
            uint32_t put_completed;
            uint32_t get_posted;
            uint32_t get_completed;
        } onesided;
    } task_status_t;
public:
    UcxTask() {}
    ~UcxTask() {}
public:
    task_status_t          task_status;
};

};
};


#endif