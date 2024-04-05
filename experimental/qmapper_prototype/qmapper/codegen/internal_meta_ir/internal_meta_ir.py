from enum import Enum
from copy import deepcopy
from typing import List, Optional, Union, Tuple, Dict
import functools
import operator
from ...api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable


class InternalType(Enum):
    Scalar = 0
    Tensor = 1
    Shape  = 2

class InternalDtype(Enum):
    float32 = 0
    int64   = 1
    bool    = 2

class ParallelType:
    Split = 0
    Replicate = 1
    Reduce = 2

class SplitPass:
    def __init__(self, dim: int):
        self.dim = dim
        self.name = 'split'
        self.type = ParallelType.Split

    def __str__(self):
        return f'split({self.dim})'
    
    def __repr__(self):
        return self.__str__()

class ReplicatePass:
    def __init__(self):
        self.name = 'repliceate'
        self.type = ParallelType.Replicate

    def __str__(self):
        return f'replicate()'
    
    def __repr__(self):
        return self.__str__()

class ReducePass:
    def __init__(self):
        self.name = 'reduce'
        self.type = ParallelType.Reduce

    def __str__(self):
        return f'reduce()'
    
    def __repr__(self):
        return self.__str__()

class InternalMetaVariable:
    dtype2size = {
        InternalDtype.bool: 1,
        'bool': 1,
        InternalDtype.float32: 4,
        'float32': 4,
        InternalDtype.int64: 8,
        'int64':8,
    }
    def __init__(self, var_id: int, type: InternalType, dtype: InternalDtype, shape: Optional[List[int]] = None, 
                 value: Union[int, float, None] = None, split_info: Optional[List[Tuple[int,int]]] = None):
        self.id = var_id
        self.type = type
        self.dtype = dtype
        self.shape = shape
        self.value = value
        self.split_info = split_info
        self.gen_node_id = None
        self.index_in_gen_node = None

        self.consume_node_ids = []
        self.indice_in_consume_node = []

    def __str__(self):
        return f'{self.id}: {self.shape}({self.dtype}:{self.shape}): gen_node({self.gen_node_id}), consume_node({self.consume_node_ids})'
    
    def __repr__(self):
        return self.__str__()

    def get_original_size(self):
        return functools.reduce(operator.mul, self.shape, 1.0)
    
    def get_splited_size(self):
        if self.split_info is None:
            return self.get_original_size()
        return functools.reduce(lambda x,y: operator.mul(x, y[1]-y[0]), self.split_info, 1.0)
    
    def get_size(self):
        return self.get_splited_size() * self.dtype2size[self.dtype]
    
    def get_splited(self, split_factor: float, dim: int):
        if self.shape is None:
            self.shape = [1]
        if self.split_info is None:
            self.split_info = [(0, s) for s in self.shape]
        splited_info0 = deepcopy(self.split_info)
        splited_info1 = deepcopy(self.split_info)
        dim_size = self.split_info[dim][1] - self.split_info[dim][0]

        dim_size_0 = max(1, int(dim_size * split_factor))
        dim_size_1 = dim_size - dim_size_0
        splited_info0[dim] = (self.split_info[dim][0], self.split_info[dim][0] + dim_size_0)
        splited_info1[dim] = (self.split_info[dim][0] + dim_size_0, self.split_info[dim][1])
        splited_var0 = InternalMetaVariable(self.id, self.type, self.dtype, self.shape, self.value, splited_info0)
        splited_var1 = InternalMetaVariable(self.id, self.type, self.dtype, self.shape, self.value, splited_info1)
        return splited_var0, splited_var1
    
    def get_intersection(self, another):
        if self.shape is None:
            return (None, None, None)
        if self.split_info is None:
            self.split_info = [(0, s) for s in self.shape]
        this_split_info = deepcopy(self.split_info)
        if another.split_info is None:
            another.split_info = [(0, s) for s in another.shape]
        other_split_info = deepcopy(another.split_info)
        intersected = deepcopy(this_split_info)
        if self.type != InternalType.Tensor:
            return (None, None, None)
        for i in range(len(this_split_info)):
            le = max(this_split_info[i][0], other_split_info[i][0])
            ri = min(this_split_info[i][1], other_split_info[i][1])
            if le >= ri:
                return (None, None, None)
            else:
                intersected[i] = (le, ri)
                this_split_info[i] = (this_split_info[i][0], le)
                other_split_info[i] = (ri, other_split_info[i][1])
        inter0=InternalMetaVariable(self.id, self.type, self.dtype, self.shape, self.value, this_split_info)
        inter1=InternalMetaVariable(self.id, self.type, self.dtype, self.shape, self.value, intersected)
        inter2=InternalMetaVariable(self.id, self.type, self.dtype, self.shape, self.value, other_split_info)
        return (inter0,inter1,inter2)

        
class InternalMetaOperator:
    def __init__(self, op_id: int, input_signature: List[InternalMetaVariable]):
        self.id = op_id
        self.input_signature = input_signature
        self.output_signature = None

    def __call__(*args):
        raise NotImplementedError
    
    def get_parallel_options(self):
        raise NotImplementedError
    
    @staticmethod
    def from_meta_node(self, meta_node: MetaNode):
        raise NotImplementedError
    
    def __str__(self):
        return f"{self.op_name}_{self.id}: {[self.input_signature[i].id for i in range(len(self.input_signature))]}"
    
    def __repr__(self):
        return self.__str__()


class InternalMetaGraph:
    def __init__(self, nodes: List[InternalMetaOperator], meta_vars: Dict[int, InternalMetaVariable], remapping=Dict[int,int]):
        self.nodes = {node.id: node for node in nodes}
        self.variables = meta_vars
        self.remapping = remapping
        self.cluster_info = None


    
