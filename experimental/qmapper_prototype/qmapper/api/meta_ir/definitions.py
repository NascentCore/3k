import functools
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
import torch.utils._pytree as pytree
from typing import List, Any, Dict
from numpy import gcd
from copy import deepcopy
import networkx as nx
import operator
    
class MetaVariable:
    _id_count = 0
    _sizes = {
        'float32': 32,
        'float16': 16,
        'int32':   32,
        'int16':   16,
    }

    @staticmethod
    def generate_uuid() -> int:
        uuid = MetaVariable._id_count
        MetaVariable._id_count += 1
        return uuid
    
    @staticmethod
    def clear_id_count() -> None:
        MetaVariable._id_count = 0

    def __init__(self, t:torch.Tensor, name:str, shape:List[int],
                 dtype: str, no_torch: bool) -> None:
        self.uuid = self.generate_uuid()
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.up_node = None
        self.index_in_up_node = None
        self.down_nodes = []
        self.indice_in_down_nodes = []
        self.orig_tensor = t
        self.no_torch = no_torch

    def get_num_elems(self):
        return functools.reduce(self.shape, operator.mul)
    
    def get_size_in_bytes(self):
        return self.get_num_elems() * self._sizes[self.dtype] // 8

    def get_real(self):
        if self.no_torch:
            if self.shape is None:
                return self.orig_tensor
            if len(self.shape) == 0:
                return self.orig_tensor.item()
            return self.orig_tensor.tolist()
        return self.orig_tensor

    def __str__(self) -> str:
        return f'Variable({self.name}, shape({self.shape}), dtype({self.dtype}))'
    
    def __repr__(self) -> str:
        return self.__str__()
    
class MetaNode:
    @staticmethod
    def generate_uuid() -> int:
        uuid = MetaVariable._id_count
        MetaVariable._id_count += 1
        return uuid
    
    @staticmethod
    def clear_id_count() -> None:
        MetaVariable._id_count = 0

    def __init__(self, name:str, op_name: str, inputs: List[MetaVariable], 
                 outputs: List[MetaVariable], sharding_info: Any, is_placeholder: bool):
        self.uuid = MetaVariable.generate_uuid()
        self.name = name
        self.op_name = op_name
        self.inputs = inputs
        self.outputs = outputs
        self.sharding_info = sharding_info

        self.is_placeholder = is_placeholder

    def get_output(self):
        return 

    def __str__(self) -> str:
        return f'Node({self.name}/{self.op_name}, inputs({[t.name for t in self.inputs]})), outputs({[t.name for t in self.outputs]})'
    
    def __repr__(self) -> str:
        return self.__str__()
    
class MetaGraph:
    def __init__(self, name: str, nodes: Dict[str, MetaNode], re_mapping: Dict[str,str]):
        self.name = name
        self.edges = []
        self.nodes = nodes
        self.re_mapping = re_mapping

        self.dependency_connections = {}
        for k,node in self.nodes.items():
            up_nodes = [n.up_node.name for n in node.inputs]
            self.dependency_connections[k] = up_nodes

        self.stateful_variables = self.nodes['stateful_variables'].outputs

    def check_parameter(self, meta_var: MetaVariable):
        for stateful_var in self.stateful_variables:
            if stateful_var.name == meta_var.name:
                return True
        return False
    
    def to_networkx(self):
        G = nx.DiGraph()
        for k, neighbors in self.dependency_connections.items():
            for neighbor in neighbors:
                G.add_edge(neighbor,k)
        return G
    
    def to_networkx_clear(self):
        G = nx.DiGraph()
        for k, neighbors in self.dependency_connections.items():
            if k in ["stateful_variables", "static_variables", "input_variables"]:
                continue
            for neighbor in neighbors:
                G.add_edge(k, neighbor)
        return G
    
    def topology_sort(self) -> List[MetaNode]:
        G = self.to_networkx()
        return [self.nodes[node_name] for node_name in nx.topological_sort(G)]


