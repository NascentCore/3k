import torch
import itertools
from copy import deepcopy, copy
from typing import List, Callable
from ..config import qmapper_logger
from .meta_ir.definitions import MetaGraph, MetaVariable, MetaNode
from .meta_ir.shard_annotation import SplitShardFunc, ReplicateShardFunc, CopyShardFunc, ReduceShardFunc, ShardFuncBase, ShardAnnotation

def node_strategy_search(node: MetaNode):
    def check_no_tensor(meta_var):
        return meta_var.shape == None or len(meta_var.shape) == 0
    
    def apply_pass(real_inputs: List[torch.Tensor], shard_passes: List[ShardFuncBase]) -> List[List[torch.Tensor]]:
        ret = [[],[]]
        for i in range(len(real_inputs)):
            if isinstance(real_inputs[i], torch.Tensor):
                x = real_inputs[i].clone().detach()
                t1, t2 = shard_passes[i].shard(x)
            elif isinstance(real_inputs[i], torch.nn.Parameter):
                x = real_inputs[i].clone().detach()
                t1, t2 = shard_passes[i].shard(x)
            ret[0].append(t1)
            ret[1].append(t2)
        return ret
    
    def check_combine(out0, out1, output, output_shards:List[ShardFuncBase]):
        ret = None
        for shard_pass in output_shards:
            try:
                combined = shard_pass.combine(out0, out1)
                if torch.allclose(combined, output, atol=1e-05):
                    ret = shard_pass
            finally:
                pass
            if not ret is None:
                return ret
        if out0.shape == out1.shape:
            if torch.allclose(out0+out1, output):
                return ReduceShardFunc(None, None, None, None)
        return None

    def traverse_pattern(real_inputs: List[torch.Tensor], input_pattern: List[List[ShardFuncBase]], 
                         output: torch.Tensor, output_shards: List[ShardFuncBase], func: Callable):
        split_passes = {}
        for i in range(len(input_pattern)):
            for j in range(len(input_pattern[i])):
                if isinstance(input_pattern[i][j], SplitShardFunc):
                    if i not in split_passes:
                        split_passes[i] = [input_pattern[i][j]]
                    else:
                        split_passes[i].append(input_pattern[i][j])
        
        copy_pass = CopyShardFunc(None, None, None, None)
        traversed_time = 0
        ret = []
        for combined_split_num in range(1, len(split_passes)+1):
            traversed_list = list(split_passes.keys())
            for index_list in itertools.combinations(traversed_list, combined_split_num):
                traversed_split_func = [split_passes[idx] for idx in index_list]
                tmp_helper = {}
                for i,idx in enumerate(index_list):
                    tmp_helper[idx] = i
                for splited_passes in itertools.product(*tuple(traversed_split_func)):
                    qmapper_logger.debug(f'traversing {traversed_time} times in node searching')
                    traversed_time += 1
                    shard_pass = []
                    for i in range(len(input_pattern)):
                        if i in index_list:
                            shard_pass.append(splited_passes[tmp_helper[i]])
                        else:
                            shard_pass.append(copy_pass)
                    splited_args = apply_pass(real_inputs, shard_pass)
                    try:
                        out0 = func(*tuple(splited_args[0]))
                        out1 = func(*tuple(splited_args[1]))
                        output_shard_pass = check_combine(out0, out1, output, output_shards)
                        if not output_shard_pass is None:
                            ret.append((shard_pass, output_shard_pass))
                    finally:
                        continue
        return ret
                    


    if node.is_placeholder:
        return []
    meta_inputs = [meta_var for meta_var in node.inputs]
    meta_outputs = [meta_var for meta_var in node.outputs]
    
    real_inputs = [meta_var.get_real() for meta_var in node.inputs]
    real_outputs = [meta_var.get_real() for meta_var in node.outputs]
    if (len(real_outputs) > 1):
        assert False

    shardpass_pattern = []
    for idx, meta_var in enumerate(meta_inputs):
        if check_no_tensor(meta_var):
            shardpass_pattern.append([CopyShardFunc(shard_factor=None, dim=None, 
                                                     tensor_shape=None, 
                                                     tensor_dtype=None)])
        else:
            shardpass_pattern.append([SplitShardFunc(shard_factor=1/meta_var.shape[dim], dim=dim, 
                                                     tensor_shape=meta_var.shape, 
                                                     tensor_dtype=meta_var.dtype) for dim in range(len(meta_var.shape))])
    for idx, meta_var in enumerate(meta_outputs):
        if check_no_tensor(meta_var):
            shardpass_pattern.append([CopyShardFunc(shard_factor=None, dim=None, 
                                                     tensor_shape=None, 
                                                     tensor_dtype=None)])
        else:
            shardpass_pattern.append([SplitShardFunc(shard_factor=1/meta_var.shape[dim], dim=dim, 
                                                     tensor_shape=meta_var.shape, 
                                                     tensor_dtype=meta_var.dtype) for dim in range(len(meta_var.shape))])
    
    input_pattern = deepcopy(shardpass_pattern[:len(meta_inputs)])
    output = real_outputs[0]
    output_shards = shardpass_pattern[-1]
    return traverse_pattern(real_inputs, input_pattern, output, output_shards, eval(f'torch.ops.{node.op_name}'))

