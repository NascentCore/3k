from ..internal_meta_ir import InternalMetaOperator, InternalMetaVariable, InternalDtype, InternalType, SplitPass, ReplicatePass, ReducePass
from ....api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable
from typing import List
from copy import deepcopy

import numpy as np
import tvm
from tvm import te, auto_scheduler

import os
import sys

class Internal_Matmul_Tensor_Tensor(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'matmul_t_t'

    def get_parallel_options(self):
        batched_dims = len(self.input_signature[0].shape) - 1
        input_options = [[SplitPass(dim_id), ReplicatePass()] for dim_id in range(batched_dims)]
        output_options = [[SplitPass(dim_id)] for dim_id in range(batched_dims)]

        input_options.append([SplitPass(batched_dims), SplitPass(1)])
        output_options.append([ReducePass])

        parallel_options = {
            'input_options': input_options,
            'output_options': output_options
        }

        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_matmul_t_t(shape1, shape2,dtype,prefix):
            A = te.placeholder(tuple(shape1), dtype=dtype, name=f'{prefix}_A')
            B = te.placeholder(tuple(shape2), dtype=dtype, name=f'{prefix}_B')
            output_shape = deepcopy(shape1)
            output_shape[-1] = shape2[-1]
            
            if len(shape1) == 1:
                assert False, f"Matmul with shape {shape1} and {shape2}"
            elif len(shape1) == 2:
                reduce_axis_index = [0 for _ in range(len(shape1))]
                reduce_axis_index[-1] = shape1[-1]
                k = te.reduce_axis(tuple(reduce_axis_index), name=f'{prefix}_k')

                matmul = te.compute(
                    tuple(output_shape),
                    lambda i,j: te.sum(A[i,k] * B[j,k], axis=k),
                    name=f'{prefix}_matmul',
                )
            elif len(shape1) == 3:
                reduce_axis_index = [0 for _ in range(len(shape1))]
                reduce_axis_index[-1] = shape1[-1]
                k = te.reduce_axis(tuple(reduce_axis_index), name=f'{prefix}_k')
                matmul = te.compute(
                    tuple(output_shape),
                    lambda i,j,l: te.sum(A[i,j,k] * B[l,k], axis=k),
                    name=f'{prefix}_matmul',
                )
            elif len(shape1) == 4:
                reduce_axis_index = [0 for _ in range(len(shape1))]
                reduce_axis_index[-1] = shape1[-1]
                k = te.reduce_axis(tuple(reduce_axis_index), name=f'{prefix}_k')
                matmul = te.compute(
                    tuple(output_shape),
                    lambda i,j,l,m: te.sum(A[i,j,l,k] * B[m,k], axis=k),
                    name=f'{prefix}_matmul',
                )
            elif len(shape1) == 5:
                reduce_axis_index = [0 for _ in range(len(shape1))]
                reduce_axis_index[-1] = shape1[-1]
                k = te.reduce_axis(tuple(reduce_axis_index), name=f'{prefix}_k')
                matmul = te.compute(
                    tuple(output_shape),
                    lambda i,j,l,m,n: te.sum(A[i,j,l,m,k] * B[n,k], axis=k),
                    name=f'{prefix}_matmul',
                )
            else:
                assert False
            return [A,B,matmul]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'

        shape1 = splited_input_signature[0].shape
        shape2 = splited_input_signature[1].shape
        dtype = splited_input_signature[0].dtype
        task = auto_scheduler.SearchTask(_internal_matmul_t_t, args=(shape1, shape2, dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
        tune_option = auto_scheduler.TuningOptions (
            num_measure_trials=128,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2
        )
        task.tune(tune_option)
        sch, args = task.apply_best(log_file)
        func = tvm.build(sch, args, target)
        return func
    
    def get_output_pattern(self):
        out_shape = []
        out_shape.extend(self.input_signature[0].shape[:len(self.input_signature[0].shape)-1])
        out_shape.append(self.input_signature[1].shape[-1])
        output = InternalMetaVariable(None, self.input_signature[0].type, self.input_signature[0].dtype, out_shape, None, None)
        return output

class Internal_Matmul:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None),
                           InternalMetaVariable(None, InternalType.Tensor, meta_inputs[1].dtype, meta_inputs[1].shape, None, None)]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        return Internal_Matmul_Tensor_Tensor(id, input_signatrue)
    
