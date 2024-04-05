from ..internal_meta_ir import InternalMetaOperator, InternalMetaVariable, InternalDtype, InternalType, SplitPass, ReplicatePass
from ....api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable
from typing import List

import numpy as np
import tvm
from tvm import te, auto_scheduler

import os
import sys

class Internal_View_Tensor_Shape(InternalMetaOperator):
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        super().__init__(op_id, input_signature)
        self.op_name = 'view_t_s'

    # Fix: 需要实现并行选项
    def get_parallel_options(self):
        parallel_options = {
            'input_options': [[ReplicatePass(), ReplicatePass()]],
            'output_options': [[ReplicatePass()]]
        }
        return parallel_options
    
    def generate_op(self, splited_input_signature: List[InternalMetaVariable], target: tvm.target.Target):    
        def func(input: tvm.nd.NDArray, out: tvm.nd.NDArray):
            out = input._create_view(splited_input_signature[1].value)
        return func
    
    def get_output_pattern(self):
        out_shape = self.input_signature[1].shape
        output = InternalMetaVariable(None, self.input_signature[0].type, self.input_signature[0].dtype, out_shape, None, None)
        return output


class Internal_View:
    @staticmethod
    def get_input_sig_from_meta_node(meta_node: MetaNode):
        meta_inputs = meta_node.inputs
        input_signature = [InternalMetaVariable(None, InternalType.Tensor, meta_inputs[0].dtype, meta_inputs[0].shape, None, None),
                           InternalMetaVariable(None, InternalType.Shape, None, None, meta_inputs[1].get_real(), None)]
        return input_signature
    
    @staticmethod
    def get_dispatched(id: int, input_signatrue: List[InternalMetaVariable]):
        return Internal_View_Tensor_Shape(id, input_signatrue)
    