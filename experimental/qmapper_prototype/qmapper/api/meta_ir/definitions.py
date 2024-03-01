import functools
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
import torch.utils._pytree as pytree
from typing import List, Any
from numpy import gcd
from copy import deepcopy

class MetaStaticObj:
    def __init__(self, name:str, obj:Any) -> None:
        self.name = name
        self.is_module = False
        try:
            self.obj  = deepcopy(obj)
        except:
            self.is_module = True
            self.obj  = None
        self.up_node = None
        self.index_in_up_node = None
        self.down_nodes = []
        self.indice_in_down_nodes = []

    def __str__(self) -> str:
        return f'StaticObj({self.obj.__str__()})'
    
    def __repr__(self) -> str:
        return self.__str__
    
class MetaVariable:
    _id_count = 0

    @staticmethod
    def generate_uuid() -> int:
        uuid = MetaVariable._id_count
        MetaVariable._id_count += 1
        return uuid
    
    @staticmethod
    def clear_id_count() -> None:
        MetaVariable._id_count = 0

    def __init__(self, name:str, shape:List[int],
                 dtype: str) -> None:
        self.uuid = self.generate_uuid()
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.up_node = None
        self.index_in_up_node = None
        self.down_nodes = []
        self.indice_in_down_nodes = []

    def __str__(self) -> str:
        return f'Variable({self.name})'
    
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

        self.placeholder_sign = is_placeholder

    def __str__(self) -> str:
        return f'Node({self.op_name})'
    
    def __repr__(self) -> str:
        return self.__str__()