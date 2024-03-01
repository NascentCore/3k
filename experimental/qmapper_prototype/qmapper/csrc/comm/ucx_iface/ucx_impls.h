#pragma once

#include "comm_iface.h"
#include "ucx_iface/ucx_iface.h"
#include <ucs/memory/memory_type.h>
#include <ucs/stats//stats_fwd.h>

#ifndef COMM_UCX_IMPLS_H
#define COMM_UCX_IMPLS_H

namespace qmap {
namespace comm {

/* TL/UCP endpoint address layout: (ucp_addrlen may vary per proc)

   [worker->ucp_addrlen][ucp_worker_address][onesided_info]
       8 bytes    ucp_addrlen bytes

    If a special service worker is set through UCP_SERVICE_TLS:
   [worker->ucp_addrlen][ucp_worker_address][service_worker->ucp_addrlen][ucp_service_worker_address][onesided_info]
       8 bytes    ucp_addrlen bytes      8 bytes        service.ucp_addrlen bytes
*/

#define UCP_RESERVED_BITS 2
#define UCP_SCOPE_BITS    3
#define UCP_SCOPE_ID_BITS 3
#define UCP_USER_TAG_BITS 1
#define UCP_TAG_BITS      15
#define UCP_SENDER_BITS   24
#define UCP_ID_BITS       16

#define UCP_RESERVED_BITS_OFFSET                                        \
    (UCP_ID_BITS + UCP_SENDER_BITS + UCP_SCOPE_ID_BITS +  \
     UCP_SCOPE_BITS + UCP_TAG_BITS + UCP_USER_TAG_BITS)

#define UCP_USER_TAG_BITS_OFFSET                                        \
    (UCP_ID_BITS + UCP_SENDER_BITS + UCP_SCOPE_ID_BITS +  \
     UCP_SCOPE_BITS + UCP_TAG_BITS)

#define UCP_TAG_BITS_OFFSET                                             \
    (UCP_ID_BITS + UCP_SENDER_BITS + UCP_SCOPE_ID_BITS +  \
     UCP_SCOPE_BITS)

#define UCP_SCOPE_BITS_OFFSET                                        \
    (UCP_ID_BITS + UCP_SENDER_BITS + UCP_SCOPE_ID_BITS)

#define UCP_SCOPE_ID_BITS_OFFSET (UCP_ID_BITS + UCP_SENDER_BITS)
#define UCP_SENDER_BITS_OFFSET   (UCP_ID_BITS)
#define UCP_ID_BITS_OFFSET       0

#define UCP_MAX_TAG         GET_MSK(UCP_TAG_BITS)
#define UCP_RESERVED_TAGS   8
#define UCP_MAX_COLL_TAG   (UCP_MAX_TAG - UCP_RESERVED_TAGS)
#define UCP_SERVICE_TAG    (UCP_MAX_COLL_TAG + 1)
#define UCP_ACTIVE_SET_TAG (UCP_MAX_COLL_TAG + 2)
#define UCP_MAX_SENDER      GET_MSK(UCP_SENDER_BITS)
#define UCP_MAX_ID          GET_MSK(UCP_ID_BITS)
#define UCP_TAG_SENDER_MASK                                             \
    GET_MSK(UCP_ID_BITS + UCP_SENDER_BITS + \
             UCP_SCOPE_ID_BITS + UCP_SCOPE_BITS)
#define UCP_GET_SENDER(_tag) ((uint32_t)(((_tag) >> UCP_SENDER_BITS_OFFSET) & \
                                                GET_MSK(UCP_SENDER_BITS)))


static ucs_memory_type_t qmap_memtype_to_ucs[MEMORY_TYPE_UNKNOWN+1] = {
    UCS_MEMORY_TYPE_HOST,
    UCS_MEMORY_TYPE_CUDA,
    UCS_MEMORY_TYPE_CUDA_MANAGED,
    UCS_MEMORY_TYPE_ROCM,
    UCS_MEMORY_TYPE_ROCM_MANAGED,
    UCS_MEMORY_TYPE_UNKNOWN,
};


class UcxEndPoint {
public:

};




// send and recv


#define UCP_MAKE_TAG(_user_tag, _tag, _rank, _id, _scope_id, _scope)    \
    ((((uint64_t) (_user_tag)) << UCP_USER_TAG_BITS_OFFSET) |           \
     (((uint64_t) (_tag))      << UCP_TAG_BITS_OFFSET)      |           \
     (((uint64_t) (_rank))     << UCP_SENDER_BITS_OFFSET)   |           \
     (((uint64_t) (_scope))    << UCP_SCOPE_BITS_OFFSET)    |           \
     (((uint64_t) (_scope_id)) << UCP_SCOPE_ID_BITS_OFFSET) |           \
     (((uint64_t) (_id))       << UCP_ID_BITS_OFFSET))

#define UCP_MAKE_SEND_TAG(_user_tag, _tag, _rank, _id, _scope_id, _scope)          \
    UCP_MAKE_TAG(_user_tag, _tag, _rank, _id, _scope_id, _scope)

#define UCP_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _user_tag, _tag,     \
                                 _src, _id, _scope_id, _scope)                 \
    do {                              \
        (_ucp_tag_mask) = (uint64_t)(-1);                                      \
        (_ucp_tag) =                                                           \
            UCP_MAKE_TAG((_user_tag), (_tag), (_src), (_id),            \
                                (_scope_id), (_scope)); \
    } while (0)

// #define UCP_CHECK_REQ_STATUS()                                          \
//     do {                                                                       \
//         if (ucc_unlikely(UCS_PTR_IS_ERR(ucp_status))) {                        \
//             tl_error(TEAM_LIB(team),                                    \
//                      "tag %u; dest %d; team_id %u; errmsg %s",                 \
//                      task->tagged.tag, dest_group_rank,                        \
//                      team->super.super.params.id,                              \
//                      ucs_status_string(UCS_PTR_STATUS(ucp_status)));           \
//             ucp_request_cancel(team->worker->ucp_worker, ucp_status);          \
//             ucp_request_free(ucp_status);                                      \
//             return ucs_status_to_ucc_status(UCS_PTR_STATUS(ucp_status));       \
//         }                                                                      \
//     } while (0)

status_t ucp_send_common(void *buffer, size_t msg_len, 
                         memory_type_t mem_type, 
                         uint64_t dest_group_rank, 
                         UcxTeam &team,
                         CommTask &task);



status_t ucp_service_allgather(UcxTeam &team, void *sbuf, void *rbuf, size_t msg_size);

};
};


#endif