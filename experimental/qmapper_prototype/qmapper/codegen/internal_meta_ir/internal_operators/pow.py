from ..internal_meta_ir import InternalMetaOperator, InternalMetaVariable, InternalDtype, InternalType, SplitPass, ReplicatePass
from ....api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable
from typing import List

import numpy as np
import tvm
from tvm import te, auto_scheduler

import os
import sys

class Internal_Pow_Scalar_Tensor(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'pow_s_t'

    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[ReplicatePass(), SplitPass(dim_id)] for dim_id in range(len(self.input_signature[1].shape))] ,
            'output_options':[[SplitPass(dim_id)] for dim_id in range(len(self.input_signature[1].shape))]
        }
        return parallel_options
    
    def get_output_pattern(self):
        out_shape = self.input_signature[1].shape
        output = InternalMetaVariable(None, self.input_signature[1].type, self.input_signature[1].dtype, out_shape, None, None)
        return output
    
class Internal_Pow_Scalar_Scalar(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'pow_s_s'

    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[ReplicatePass(), ReplicatePass()]] ,
            'output_options':[[ReplicatePass()]]
        }
        return parallel_options
    
    def get_output_pattern(self):
        out_shape = self.input_signature[1].shape
        output = InternalMetaVariable(None, self.input_signature[1].type, self.input_signature[1].dtype, out_shape, None, None)
        return output
    
class Internal_Pow:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        if meta_inputs[1].shape is not None:
            input_signature = [InternalMetaVariable(None, InternalType.Scalar, meta_inputs[0].dtype, None, None, None),
                               InternalMetaVariable(None, InternalType.Tensor, meta_inputs[1].dtype, meta_inputs[1].shape, None, None)]
        else:
            input_signature = [InternalMetaVariable(None, InternalType.Scalar, meta_inputs[0].dtype, None, None, None),
                               InternalMetaVariable(None, InternalType.Scalar, meta_inputs[1].dtype, None, None, None)]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        if input_signatrue[1].shape is not None:
            return Internal_Pow_Scalar_Tensor(id, input_signatrue)
        else:
            return Internal_Pow_Scalar_Scalar(id, input_signatrue)
