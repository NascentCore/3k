from ..internal_meta_ir import InternalMetaOperator, InternalMetaVariable, InternalDtype, InternalType, SplitPass, ReplicatePass
from ....api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable
from typing import List

import numpy as np
import tvm
from tvm import te, auto_scheduler

import os
import sys

class Internal_Transpose_Tensor(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'transpose_t'

    def get_parallel_options(self):
        batched_dims = len(self.input_signature[0].shape) - 2
        input_options = [[SplitPass(dim_id)] for dim_id in range(batched_dims)]
        output_options = [[SplitPass(dim_id)] for dim_id in range(batched_dims)]

        input_options.append([SplitPass(batched_dims)])
        output_options.append([SplitPass(batched_dims+1)])

        input_options.append([SplitPass(batched_dims+1)])
        output_options.append([SplitPass(batched_dims)])

        parallel_options = {
            'input_options': input_options,
            'output_options': output_options
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_transpose_t(shape,dtype,prefix):
            A = te.placeholder(tuple(shape), dtype=dtype, name=f'{prefix}_A')
            
            if len(shape) == 1:
                assert False, "Transpose a tensor which has only one dim"
            elif len(shape) == 2:
                transpose = te.compute(
                    tuple(shape),
                    lambda i,j: A[j,i],
                    name=f'{prefix}_transpose',
                )
            elif len(shape) == 3:
                transpose = te.compute(
                    tuple(shape),
                    lambda i,j,k: A[i,k,j],
                    name=f'{prefix}_transpose',
                )
            elif len(shape) == 4:
                transpose = te.compute(
                    tuple(shape),
                    lambda i,j,k,l: A[i,j,l,k],
                    name=f'{prefix}_transpose',
                )
            elif len(shape) == 5:
                transpose = te.compute(
                    tuple(shape),
                    lambda i,j,k,l,m: A[i,j,k,m,l],
                    name=f'{prefix}_transpose',
                )
            else:
                assert False
            return [A,transpose]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'

        shape = splited_input_signature[0].shape
        dtype = splited_input_signature[0].dtype
        task = auto_scheduler.SearchTask(_internal_transpose_t, args=(shape, dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
        tune_option = auto_scheduler.TuningOptions (
            num_measure_trials=20,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2
        )
        task.tune(tune_option)
        sch, args = task.apply_best(log_file)
        func = tvm.build(sch, args, target)
        return func
    
    def get_output_pattern(self):
        out_shape = self.input_signature[0].shape
        out_shape[-2], out_shape[-1] = out_shape[-1],out_shape[-2]
        output = InternalMetaVariable(None, self.input_signature[0].type, self.input_signature[0].dtype, out_shape, None, None)
        return output

class Internal_Transpose:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None)]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        return Internal_Transpose_Tensor(id, input_signatrue)
    
