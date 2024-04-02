from ..internal_meta_ir import InternalMetaOperator, InternalMetaVariable, InternalDtype, InternalType, SplitPass, ReplicatePass
from ....api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable
from typing import List

import numpy as np
import tvm
from tvm import te, auto_scheduler

import os
import sys

class Internal_Reciprocal_Tensor(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'reciprocal_t'

    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[SplitPass(dim_id)] for dim_id in range(len(self.input_signature[0].shape))] ,
            'output_options':[[SplitPass(dim_id)] for dim_id in range(len(self.input_signature[0].shape))]
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_reciprocal_t(shape,dtype,prefix):
            A = te.placeholder(tuple(shape), dtype=dtype, name=f'{prefix}_A')
            
            if len(shape) == 1:
                reciprocal = te.compute(
                    tuple(shape),
                    lambda i: 1/A[i],
                    name=f'{prefix}_reciprocal',
                )
            elif len(shape) == 2:
                reciprocal = te.compute(
                    tuple(shape),
                    lambda i,j: 1/A[i,j],
                    name=f'{prefix}_reciprocal',
                )
            elif len(shape) == 3:
                reciprocal = te.compute(
                    tuple(shape),
                    lambda i,j,k: 1/A[i,j,k],
                    name=f'{prefix}_reciprocal',
                )
            elif len(shape) == 4:
                reciprocal = te.compute(
                    tuple(shape),
                    lambda i,j,k,l: 1/A[i,j,k,l],
                    name=f'{prefix}_reciprocal',
                )
            elif len(shape) == 5:
                reciprocal = te.compute(
                    tuple(shape),
                    lambda i,j,k,l,m: 1/A[i,j,k,l,m],
                    name=f'{prefix}_reciprocal',
                )
            else:
                assert False
            return [A,reciprocal]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'

        shape = splited_input_signature[0].shape
        dtype = splited_input_signature[0].dtype
        task = auto_scheduler.SearchTask(_internal_reciprocal_t, args=(shape, dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
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
        output = InternalMetaVariable(None, self.input_signature[0].type, self.input_signature[0].dtype, out_shape, None, None)
        return output
    
class Internal_Reciprocal_Scalar(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'reciprocal_s'

    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[ReplicatePass()]] ,
            'output_options':[[ReplicatePass()]]
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_reciprocal_s(dtype,prefix):
            A = te.placeholder([1], dtype=dtype, name=f'{prefix}_A')
            reciprocal = te.compute([1], lambda i: 1/A[i], dtype = dtype, name=f'{prefix}_reciprocal')
            return [A, reciprocal]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'
        dtype = splited_input_signature[0].dtype
        task = auto_scheduler.SearchTask(_internal_reciprocal_s, args=(dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
        tune_option = auto_scheduler.TuningOptions (
            num_measure_trials=1,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2
        )
        task.tune(tune_option)
        sch, args = task.apply_best(log_file)
        func = tvm.build(sch, args, target)
        return func
    
    def get_output_pattern(self):
        out_shape = self.input_signature[0].shape
        output = InternalMetaVariable(None, self.input_signature[0].type, self.input_signature[0].dtype, out_shape, None, None)
        return output
    
class Internal_Reciprocal:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        if meta_inputs[0].shape is None:
            input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None)]
        else:
            input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None)]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        if input_signatrue[0].shape is None:
            return Internal_Reciprocal_Scalar(id, input_signatrue)
        return Internal_Reciprocal_Tensor(id, input_signatrue)
