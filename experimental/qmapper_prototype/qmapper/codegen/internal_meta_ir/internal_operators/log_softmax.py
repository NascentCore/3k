from ..internal_meta_ir import InternalMetaOperator, InternalMetaVariable, InternalDtype, InternalType, SplitPass, ReplicatePass
from ....api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable
from typing import List

import numpy as np
import tvm
from tvm import te, auto_scheduler

import os
import sys

class Internal_Log_Softmax_Tensor_Scalar(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'log_softmax_t_s'

    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[ReplicatePass(), ReplicatePass()]] ,
            'output_options':[[ReplicatePass()]]
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_log_softmax_Tensor_Shape(shape,dtype,prefix):
            A = te.placeholder((shape), dtype=dtype, name=f'{prefix}_A')
            if len(shape) == 1:
                log_softmax = te.compute(
                    tuple(shape),
                    lambda i: tvm.tir.log(tvm.tir.exp(A[i])/tvm.te.sum(tvm.tir.exp(A[i]))),
                    name=f'{prefix}_log_softmax',
                )
            elif len(shape) == 2:
                log_softmax = te.compute(
                    tuple(shape),
                    lambda i,j: tvm.tir.log(tvm.tir.exp(A[i,j])/tvm.te.sum(tvm.tir.exp(A[i,j]))),
                    name=f'{prefix}_log_softmax',
                )
            elif len(shape) == 3:
                log_softmax = te.compute(
                    tuple(shape),
                    lambda i,j,k: tvm.tir.log(tvm.tir.exp(A[i,j,k])/tvm.te.sum(tvm.tir.exp(A[i,j,k]))),
                    name=f'{prefix}_log_softmax',
                )
            elif len(shape) == 4:
                log_softmax = te.compute(
                    tuple(shape),
                    lambda i,j,k,l: tvm.tir.log(tvm.tir.exp(A[i,j,k,l])/tvm.te.sum(tvm.tir.exp(A[i,j,k,l]))),
                    name=f'{prefix}_log_softmax',
                )
            elif len(shape) == 5:
                log_softmax = te.compute(
                    tuple(shape),
                    lambda i,j,k,l,m: tvm.tir.log(tvm.tir.exp(A[i,j,k,l,m])/tvm.te.sum(tvm.tir.exp(A[i,j,k,l,m]))),
                    name=f'{prefix}_log_softmax',
                )
            else:
                assert False
            return [A,log_softmax]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'

        shape = splited_input_signature[0].shape
        dtype = splited_input_signature[0].dtype
        task = auto_scheduler.SearchTask(_internal_log_softmax_Tensor_Shape, args=(shape, dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
        tune_option = auto_scheduler.TuningOptions (
            num_measure_trials=64,
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

class Internal_Log_Softmax:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None),
                           InternalMetaVariable(None, InternalType.Scalar, meta_inputs[1].dtype, None, meta_inputs[1].get_real(), None)]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        return Internal_Log_Softmax_Tensor_Scalar(id, input_signatrue)
    

# def _log_softmax_backward_data(
#     grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype
# ):
#     grad_input = grad_output - torch.exp(output) * torch.sum(
#         grad_output, dim=dim, keepdim=True
#     )
#     return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype)

    
class Internal_Log_Softmax_Backward_Tensor_Scalar(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'log_softmax_t_t_s'

    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[ReplicatePass(), ReplicatePass(), ReplicatePass()]] ,
            'output_options':[[ReplicatePass()]]
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):
        @auto_scheduler.register_workload
        def _internal_log_softmax_backward_Tensor_Shape(shape,dtype, dim, prefix):
            grad_out = te.placeholder((shape), dtype=dtype, name=f'{prefix}_A')
            output = te.placeholder((shape), dtype=dtype, name=f'{prefix}_B')
            exp_output = te.compute(shape, lambda *indices: te.exp(output(*indices)), name=f"{prefix}_exp_output")
            sum_grad_output = te.compute((shape), lambda *indices: te.sum(grad_out(*indices), axis=dim, keepdims=True), name=f"{prefix}_sum_grad_output")
            grad_input = te.compute((grad_out.shape), lambda *indices: grad_out(*indices) - exp_output(*indices) * sum_grad_output(*indices), name=f"{prefix}_grad_input")
            return [grad_out, output, grad_input]
        
        if not os.path.exists('tmp_operator_search_logs'):
            os.makedirs('tmp_operator_search_logs')
        else:
            if os.path.exists(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'):
                os.remove(f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log')
        log_file = f'tmp_operator_search_logs/{self.op_name}_{splited_input_signature[0].id}.log'

        shape = splited_input_signature[0].shape
        dtype = splited_input_signature[0].dtype
        task = auto_scheduler.SearchTask(_internal_log_softmax_backward_Tensor_Shape, args=(shape, dtype, f'{self.op_name}_{splited_input_signature[0].id}'), target=target)
        tune_option = auto_scheduler.TuningOptions (
            num_measure_trials=64,
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

class Internal_Log_Softmax_Backward:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None),
                           InternalMetaVariable(None, InternalType.Tensor, meta_inputs[1].dtype, meta_inputs[1].shape, None, None),
                           InternalMetaVariable(None, InternalType.Scalar, meta_inputs[2].dtype, None, meta_inputs[2].get_real(), None)]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        return Internal_Log_Softmax_Backward_Tensor_Scalar(id, input_signatrue)
    
