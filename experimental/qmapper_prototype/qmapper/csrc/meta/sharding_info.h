#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "meta_ir.h"

#ifndef QMAP_SHARDING_INFO_H
#define QMAP_SHARDING_INFO_H

namespace qmap {
namespace ir {

class ShardFuncIface {
public:
    virtual int get_shardable_size() = 0;
    virtual std::vector<tensor_info_t> shard(tensor_info_t tensor) = 0;
    virtual tensor_info_t gather(std::vector<tensor_info_t> tensors) = 0;
public:
    std::string shard_method_name;
};

class ShardDim {
public:
    static ShardDim& get_empty_shard_dim();
    static ShardDim& get_new_shard_dim();
public:
    static int global_shard_dim_cnt;
    int global_shard_dim_id;
    ShardFuncIface *shard_iface;
};

class ShardAnnotation {
public:
    ShardAnnotation(int num_inputs, int ouptuts);
    void add_shard_dim(std::vector<ShardDim> input_shard_dims, std::vector<ShardDim> output_shard_dims);
public:
    std::vector<std::vector<ShardDim> > input_annotations;
    std::vector<std::vector<ShardDim> > output_annotations;
};

}; // namespace ir
}; // namespace qmap

#endif