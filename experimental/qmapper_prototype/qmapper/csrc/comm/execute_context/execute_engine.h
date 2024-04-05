#pragma once

#include "comm_iface.h"
#include <cstdint>

#ifndef QMAP_COMM_EXECUTE_ENGINE_H
#define QMAP_COMM_EXECUTE_ENGINE_H

#define EXECUTOR_NUM_BUFS 9
#define EXECUTOR_MULTI_OP_NUM_BUFS 7

typedef enum ee_type {
    EE_CUDA_STREAM,
    EE_CPU_THREAD,
    EE_ROCM_STREAM,
} ee_type_t;

typedef enum event_type {
    EVENT_COLLECTIVE_POST,
    EVENT_COLLECTIVE_COMPLETE,
    EVENT_COMPUTE_COMPLETE,
    EVENT_OVERFLOW,
} event_type_t;

typedef enum {
    EVENT_COMPLETED,
    EVENT_SCHEDULE_STARTED,
    EVENT_TASK_STARTED,
    EVENT_COMPLETED_SCHEDULE,
    EVENT_ERROR,
} event_t;

typedef struct task_reduce {
    void *dst;
    void *src[EXECUTOR_NUM_BUFS];
    size_t count;
    double alpha;
    uint64_t data_type;
    comm_reduction_op_t op;
    uint16_t n_srcs;
} task_reduce_t;

typedef struct task_reduce_multi_dst {
    void *dst[EXECUTOR_MULTI_OP_NUM_BUFS];
    void *src1[EXECUTOR_MULTI_OP_NUM_BUFS];
    void *src2[EXECUTOR_MULTI_OP_NUM_BUFS];
    size_t counts[EXECUTOR_MULTI_OP_NUM_BUFS];
    uint64_t data_type;
    comm_reduction_op_t op;
    uint16_t n_bufs;
} task_reduce_multi_dst_t;

typedef struct task_copy {
    void *src;
    void *dst;
    size_t len;
} task_copy_t;

typedef struct task_copy_multi {
    void *src[EXECUTOR_MULTI_OP_NUM_BUFS];
    void *dst[EXECUTOR_MULTI_OP_NUM_BUFS];
    size_t counts[EXECUTOR_MULTI_OP_NUM_BUFS];
    size_t num_vectors;
} task_copy_multi_t;

typedef struct executor_task_args {
    uint16_t task_type;
    uint16_t flags;
    union {
        task_reduce_t reduce;
        task_reduce_multi_dst_t reduce_multi_dst;
        task_copy_t copy;
        task_copy_multi_t copy_multi;
    };
} executor_task_args_t;

typedef enum executor_task_type {
    TASK_REDUCE = GET_BIT(0),
    TASK_REDUCE_MULTI_DST = GET_BIT(1),
    TASK_COPY = GET_BIT(2),
    TASK_COPY_MULTI = GET_BIT(3),
} executor_task_type_t;

namespace qmap {
namespace comm {
namespace execute_engine {

class CommEvent;
class CommExecuteEngine;
class CommExecuteEngineContext;
class CommExecuteEngineTask;

class CommEvent {
public:
    event_type_t                  type;
    void                          *ctx;
    size_t                        ctx_size;
    comm_status_t                      status;
};

class CommExecuteEngine {
public:
    CommExecuteEngine() {}
    ~CommExecuteEngine() {}
    virtual comm_status_t init(CommTeam *team, std::map<std::string, uint64_t> &params);
    virtual comm_status_t finialize();
    virtual comm_status_t test();
    virtual std::vector<CommEvent *> get_events();
    virtual comm_status_t ack_event(CommEvent *event);
    virtual comm_status_t set_event(CommEvent *event);
    virtual comm_status_t wait_event(CommEvent *event);

    virtual comm_status_t status();
    virtual comm_status_t start(CommExecuteEngineContext *context);
    virtual comm_status_t stop();
    virtual comm_status_t task_post(executor_task_args_t *task_args, CommExecuteEngineTask **task);
    virtual comm_status_t task_test(const CommExecuteEngineTask *task);
    virtual comm_status_t task_finalize(CommExecuteEngineTask *task);

public:
    std::string                   name;
    void                          *context;
    ee_type_t                     type;
    CommTeam                      *team;
    std::mutex                    lock;
    std::deque<CommEvent *>       events;
};

class CommExecuteEngineTask {
public:
    CommExecuteEngine      *engine;
    executor_task_args_t   args;
    comm_status_t               status;
};

class CommExecuteEngineContext {
    virtual comm_status_t create_event(void **event);
    virtual comm_status_t destroy_event(void *event);
    virtual comm_status_t event_post(void *event);
    virtual comm_status_t event_test(void *event);
    comm_status_t init(std::map<std::string, uint64_t> &params);
    comm_status_t finialize();
public:
    uint32_t               ref_count;
    std::string            name;
    CommExecuteEngine      *engine;
    ee_type_t              type;
};

}; // namespace execute_engine
}; // namespace comm
}; // namespace qmap

#endif