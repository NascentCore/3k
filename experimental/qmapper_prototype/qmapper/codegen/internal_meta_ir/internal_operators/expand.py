from ..internal_meta_ir import InternalMetaOperator, InternalMetaVariable, InternalDtype, InternalType, SplitPass, ReplicatePass
from ....api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable
from typing import List

import numpy as np
import tvm
from tvm import te, auto_scheduler

import os
import sys

class Internal_Expand_Tensor_Shape(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'expand_t_s'

    # Fix: 需要实现并行选项
    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[ReplicatePass(), ReplicatePass()]],
            'output_options': [[ReplicatePass()]]
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_expand_Tensor_Shape(shape,dtype,prefix):
            A = te.placeholder((1), dtype=dtype, name=f'{prefix}_A')
            if len(shape) == 1:
                expand = te.compute(
                    tuple(shape),
                    lambda i: A[0],
                    name=f'{prefix}_expand',
                )
            elif len(shape) == 2:
                expand = te.compute(
                    tuple(shape),
                    lambda i,j: A[0],
                    name=f'{prefix}_expand',
                )
            elif len(shape) == 3:
                expand = te.compute(
                    tuple(shape),
                    lambda i,j,k: A[0],
                    name=f'{prefix}_expand',
                )
            elif len(shape) == 4:
                expand = te.compute(
                    tuple(shape),
                    lambda i,j,k,l: A[0],
                    name=f'{prefix}_expand',
                )
            elif len(shape) == 5:
                expand = te.compute(
                    tuple(shape),
                    lambda i,j,k,l,m: A[0],
                    name=f'{prefix}_expand',
                )
            else:
                assert False
            return [A,expand]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'

        shape = splited_input_signature[1].shape
        dtype = splited_input_signature[0].dtype
        task = auto_scheduler.SearchTask(_internal_expand_Tensor_Shape, args=(shape, dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
        tune_option = auto_scheduler.TuningOptions (
            num_measure_trials=10,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2
        )
        task.tune(tune_option)
        sch, args = task.apply_best(log_file)
        func = tvm.build(sch, args, target)
        return func
    
    def get_output_pattern(self):
        out_shape = self.input_signature[1].shape
        output = InternalMetaVariable(None, InternalType.Tensor, self.input_signature[0].dtype, out_shape, self.input_signature[0].value, None)
        return output


class Internal_Expand:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None),
                           InternalMetaVariable(None, InternalType.Shape, None, None, meta_inputs[1].get_real(), None)]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        return Internal_Expand_Tensor_Shape(id, input_signatrue)
    