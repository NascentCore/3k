#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifndef QMAP_META_IR_H
#define QMAP_META_IR_H

namespace qmap {
namespace ir {

#define MAX_TENSOR_SHAPE_LENGTH 5

class MetaNode;
class MetaEdge;

typedef enum {
    FLOAT64,
    FLOAT32,
    FLOAT16,
    BOOL,
    INT32,
    INT64,
    UINT32,
    UINT8,
    COMPLEX64,
} tensor_dtype_t;

typedef struct tensor_info {
    uint32_t shape[MAX_TENSOR_SHAPE_LENGTH];
    tensor_dtype_t dtype;
} tensor_info_t;

typedef struct pyobj_info {

} pyobj_info_t;

class MetaEdge {

public:
    uint32_t uuid;
    std::string name;
    MetaNode *up_node;
    int index_in_up_node;
    std::vector<MetaNode *> down_nodes;
    std::vector<int> indice_in_down_nodes;
};

class MetaNode {

public:
    uint32_t uuid;
    std::string name;
    std::string op_name;
    std::vector<MetaEdge *> inputs;
    std::vector<MetaEdge *> outputs;
};





}; // namespace ir
}; // namespace qmap

#endif