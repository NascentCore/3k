from ..internal_meta_ir import InternalMetaOperator, InternalMetaVariable, InternalDtype, InternalType, SplitPass, ReplicatePass
from ....api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable
from typing import List

import numpy as np
import tvm
from tvm import te, auto_scheduler

import os
import sys

class Internal_Mul_Tensor_Scalar(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'mul_t_s'

    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[SplitPass(dim_id), ReplicatePass()] for dim_id in range(len(self.input_signature[0].shape))] ,
            'output_options':[[SplitPass(dim_id)] for dim_id in range(len(self.input_signature[0].shape))]
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_mul_t_s(shape,dtype,prefix):
            A = te.placeholder(tuple(shape), dtype=dtype, name=f'{prefix}_A')
            B = te.placeholder((1),dtype=dtype,name=f'{prefix}_B')
            
            if len(shape) == 1:
                mul = te.compute(
                    tuple(shape),
                    lambda i: A[i] * B[0],
                    name=f'{prefix}_mul',
                )
            elif len(shape) == 2:
                mul = te.compute(
                    tuple(shape),
                    lambda i,j: A[i,j] * B[0],
                    name=f'{prefix}_mul',
                )
            elif len(shape) == 3:
                mul = te.compute(
                    tuple(shape),
                    lambda i,j,k: A[i,j,k] * B[0],
                    name=f'{prefix}_mul',
                )
            elif len(shape) == 4:
                mul = te.compute(
                    tuple(shape),
                    lambda i,j,k,l: A[i,j,k,l] * B[0],
                    name=f'{prefix}_mul',
                )
            elif len(shape) == 5:
                mul = te.compute(
                    tuple(shape),
                    lambda i,j,k,l,m: A[i,j,k,l,m] * B[0],
                    name=f'{prefix}_mul',
                )
            else:
                assert False
            return [A,B,mul]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'

        shape = splited_input_signature[0].shape
        dtype = splited_input_signature[0].dtype
        task = auto_scheduler.SearchTask(_internal_mul_t_s, args=(shape, dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
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

class Internal_Mul_Tensor_Tensor(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        assert input_signature[0].shape == input_signature[1].shape, "Two shape must be identical"
        self.op_name = 'mul_t_t'

    def get_parallel_options(self):
        if self.input_signature[0].shape == self.input_signature[1].shape:
            parallel_options = {
                'input_options': [[SplitPass(dim_id), SplitPass(dim_id)] for dim_id in range(len(self.input_signature[0].shape))] ,
                'output_options':[[SplitPass(dim_id)] for dim_id in range(len(self.input_signature[0].shape))]
            }
        else:
            parallel_options = {
                'input_options': [[SplitPass(dim_id), ReplicatePass()] for dim_id in range(len(self.input_signature[0].shape) - len(self.input_signature[1].shape))],
                'output_options': [[SplitPass(dim_id)] for dim_id in range(len(self.input_signature[0].shape)- len(self.input_signature[1].shape))]
            }
            for i in range(len(self.input_signature[1].shape)):
                dim_id0 = len(self.input_signature[0].shape) + i - 1
                dim_id1 = i
                parallel_options['input_options'].append([SplitPass(dim_id0), SplitPass(dim_id1)])
                parallel_options['output_options'].append([SplitPass(dim_id0)])
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_mul_t_s(shape0, shape1,dtype,prefix):
            A = te.placeholder(tuple(shape0), dtype=dtype, name=f'{prefix}_A')
            B = te.placeholder(tuple(shape1), dtype=dtype,name=f'{prefix}_B')
            mul = te.compute(
                    tuple(shape0),
                    lambda *args: A[*args] * B[*args[-len(shape1):]],
                    name=f'{prefix}_mul',
                )
            return [A,B,mul]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'

        shape0 = splited_input_signature[0].shape
        shape1 = splited_input_signature[1].shape
        dtype = splited_input_signature[0].dtype
        task = auto_scheduler.SearchTask(_internal_mul_t_s, args=(shape0, shape1, dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
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


class Internal_Mul_Scalar_Scalar(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        assert input_signature[0].shape == input_signature[1].shape, "Two shape must be identical"
        self.op_name = 'mul_s_s'

    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[ReplicatePass(), ReplicatePass()]],
            'output_options':[[ReplicatePass()]]
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_mul_t_s(dtype,prefix):
            A = te.placeholder((1), dtype=dtype, name=f'{prefix}_A')
            B = te.placeholder((1), dtype=dtype,name=f'{prefix}_B')

            mul = te.compute(
                (1),
                lambda i: A[i] * B[i],
                name=f'{prefix}_mul',
            )
            return [A,B,mul]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'
        dtype = splited_input_signature[0].dtype
        task = auto_scheduler.SearchTask(_internal_mul_t_s, args=(dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
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


class Internal_Mul:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        if len(meta_inputs[1].shape) == 0:
            if len(meta_inputs[0].shape) == 0:
                input_signature = [InternalMetaVariable(None, InternalType.Scalar, meta_inputs[0].dtype, None, None, None),
                                   InternalMetaVariable(None, InternalType.Scalar, meta_inputs[0].dtype, None, None, None)]
            else:
                input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None),
                                   InternalMetaVariable(None, InternalType.Scalar, meta_inputs[0].dtype, None, None, None)]
        else:
            input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None),
                               InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None),]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        if input_signatrue[0].type == InternalType.Scalar:
            return Internal_Mul_Scalar_Scalar(id, input_signatrue)
        elif input_signatrue[1].type == InternalType.Scalar:
            return Internal_Mul_Tensor_Scalar(id, input_signatrue)
        else:
            return Internal_Mul_Tensor_Tensor(id, input_signatrue)
    
