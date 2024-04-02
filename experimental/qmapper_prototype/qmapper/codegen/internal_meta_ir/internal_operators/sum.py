from ..internal_meta_ir import InternalMetaOperator, InternalMetaVariable, InternalDtype, InternalType, SplitPass, ReplicatePass, ReducePass
from ....api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable
from typing import List
from copy import deepcopy

import numpy as np
import tvm
from tvm import te, auto_scheduler

import os
import sys

class Internal_Sum_Dim_Tensor_Scalar(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'sum_t_s'

    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[SplitPass(dim_id), ReplicatePass()] for dim_id in range(len(self.input_signature[0].shape))] ,
            'output_options':[[SplitPass(dim_id)] for dim_id in range(len(self.input_signature[0].shape))]
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_sum_t_s(shape, dim, dtype,prefix):
            A = te.placeholder((shape), dtype=dtype, name=f'{prefix}_A')
            output_shape = deepcopy(shape)
            output_shape[dim] = 1
            reduce_axis_dim = deepcopy(shape)
            for i in range(len(shape)):
                if i == dim:
                    continue
                else:
                    reduce_axis_dim[i] = 0
            k = te.reduce_axis(tuple(reduce_axis_dim), name=f'{prefix}_k')
            sum = te.compute(
                (output_shape),
                lambda *args: te.sum(*tuple([args[i] if i != dim else args[i] + k for i in range(len(args))]), axis=[k]),
                name=f'{prefix}_sum_dim',
            )
            return [A,sum]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'
        dtype = splited_input_signature[0].dtype
        shape = splited_input_signature[0].shape
        dim = splited_input_signature[1].value
        task = auto_scheduler.SearchTask(_internal_sum_t_s, args=(shape, dim, dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
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
        out_shape[self.input_signature[1].value] = 1
        output = InternalMetaVariable(None, self.input_signature[0].type, self.input_signature[0].dtype, out_shape, None, None)
        return output

class Internal_Sum_Dim:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        
        input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None),
                           InternalMetaVariable(None, InternalType.Scalar, meta_inputs[0].dtype, None, meta_inputs[1].get_real()[0], None),]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        return  Internal_Sum_Dim_Tensor_Scalar(id, input_signatrue)
    
class Internal_Sum_Tensor(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'sum_t'

    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[SplitPass(dim_id)] for dim_id in range(len(self.input_signature[0].shape))] ,
            'output_options':[[ReducePass()] for dim_id in range(len(self.input_signature[0].shape))]
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_sum_t(shape, dim, dtype,prefix):
            A = te.placeholder((shape), dtype=dtype, name=f'{prefix}_A')
            sum = te.compute(
                ([1]),
                lambda i: te.sum(A[i], axis=None),
                name=f'{prefix}_sum',
            )
            return [A,sum]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'
        dtype = splited_input_signature[0].dtype
        shape = splited_input_signature[0].shape
        dim = splited_input_signature[1].value
        task = auto_scheduler.SearchTask(_internal_sum_t, args=(shape, dim, dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
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
        out_shape = [1]
        output = InternalMetaVariable(None, InternalType.Scalar, self.input_signature[0].dtype, out_shape, None, None)
        return output

class Internal_Sum:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None)]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        return Internal_Sum_Tensor(id, input_signatrue)
    