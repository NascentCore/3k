#include "ucx_iface/ucx_impls.h"
#include "comm_iface.h"
#include "ucx_iface/ucx_iface.h"
#include <cstdint>
#include <ucp/api/ucp.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include <glog/logging.h>

using namespace qmap;
using namespace comm;

#define UCX_CHECK_REQ_STATUS() \
    do { \
        if (ucs_unlikely(UCS_PTR_IS_ERR(ucp_status))) { \
            LOG(ERROR) << ucs_status_string(UCS_PTR_STATUS(ucp_status)); \
            ucp_request_cancel(team.worker.\
        } \
    } while (0)


void ucp_send_completion_cb(void *req, ucs_status_t status, void *user_data) {
    UcxTask *task = (UcxTask *) user_data;
    if(ucs_unlikely(UCS_OK != status)) {
        LOG(ERROR) << "failure in send completion" << ucs_status_string(status);
        task->status = QMAP_ERROR;
    }
    task->task_status.tagged.send_completed ++;
    ucp_request_free(req);
}


static inline ucs_status_ptr_t 
ucp_send_common(void *buffer, size_t msg_len, 
                memory_type_t mem_type, 
                uint64_t dest_group_rank, 
                UcxTeam &team,
                UcxTask &task,
                void *user_data)
{
    coll_args_t         *args = &task.args;
    ucp_request_param_t req_param;
    status_t            status;
    ucp_ep_h            ep;
    ucp_tag_t           ucp_tag;

    status = team.ucx_get_ep(dest_group_rank, &ep);
    if (ucs_likely(status != qmap::comm::QMAP_OK)) {
        LOG(FATAL) << "error while get ep";
        return UCS_STATUS_PTR(UCS_ERR_NO_MESSAGE);
    }
    LOG(INFO) << "get ep succeeded";

    ucp_tag = UCP_MAKE_SEND_TAG(args->mask & COLL_ARGS_FIELD_TAG, task.task_status.tagged.tag, team.rank, team.id, team.scope_id, team.scope);
    req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE |
                             UCP_OP_ATTR_FIELD_USER_DATA | UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    req_param.datatype = ucp_dt_make_contig(msg_len);
    req_param.cb.send = ucp_send_completion_cb;
    req_param.memory_type = qmap_memtype_to_ucs[mem_type];
    req_param.user_data = user_data;
    task.task_status.tagged.send_posted ++;
    return ucp_tag_send_nbx(ep, buffer, 1, ucp_tag, &req_param);
}

static inline status_t
ucp_send_nb(void *buffer, size_t msg_len, memory_type_t mem_type,
            uint64_t dest_group_rank, UcxTeam &team, UcxTask &task)
{
    ucs_status_ptr_t ucp_status;
    ucp_status = ucp_send_common(buffer, msg_len, mem_type, 
                                 dest_group_rank, team, 
                                 task, (void *)&task);
    if (ucp_status != UCS_OK) {

    } else {
        task.task_status.tagged.send_completed ++;
    }
    return qmap::comm::QMAP_OK
}