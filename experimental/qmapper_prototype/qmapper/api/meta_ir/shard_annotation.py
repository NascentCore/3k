from .cluster_info import ClusterInfo
from typing import List
from copy import deepcopy
import torch

class ShardFuncBase:

    def shardable_size(self):
        raise NotImplementedError
    
    def reset_shard_factor(self, shard_factor):
        raise NotImplementedError

    def shard(self, t):
        raise NotImplementedError
    
    def combine(self, t1, t2):
        raise NotImplementedError
    

class SplitShardFunc(ShardFuncBase):
    def __init__(self, shard_factor: float, dim: int, tensor_shape: List[int], tensor_dtype: str):
        self.name = 'split-shard'
        self.shard_factor = shard_factor
        self.dim = dim
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.is_split = True

    def reset_shard_factor(self, shard_factor):
        self.shard_factor = shard_factor

    def shardable_size(self):
        return self.tensor_shape[self.dim]
    
    def shard(self, t: torch.Tensor):
        splited_size1 = int(self.shardable_size() * self.shard_factor)
        splited_size2 = self.shardable_size() - splited_size1
        t1 = torch.narrow(t, self.dim, 0, splited_size1)
        t2 = torch.narrow(t, self.dim, splited_size1, splited_size2)
        return t1, t2
    
    def combine(self, t1: torch.Tensor, t2: torch.Tensor):
        splited_size1 = int(self.shardable_size() * self.shard_factor)
        if t1.shape[self.dim] != splited_size1:
            t1,t2 = t2,t1
        return torch.concatenate((t1,t2), dim=self.dim)
        

class ReplicateShardFunc(ShardFuncBase):
    def __init__(self, shard_factor: float, dim: int, tensor_shape: List[int], tensor_dtype: str):
        self.name = 'replicate-shard'
        self.shard_factor = shard_factor
        self.dim = dim
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.is_split = False

    def reset_shard_factor(self, shard_factor):
        assert False, 'replicate shard does not has shard factor'

    def shardable_size(self):
        assert False, 'replicate shard does not has shardable size'
    
    def shard(self, t):
        return torch.t_copy(t), torch.t_copy(t)
    
    def combine(self, t1, t2):
        return t1
        

class CopyShardFunc(ShardFuncBase):
    def __init__(self, shard_factor: float, dim: int, tensor_shape: List[int], tensor_dtype: str):
        self.name = 'copy-shard-for-python-obj'
        self.shard_factor = shard_factor
        self.dim = dim
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.is_split = False


    def reset_shard_factor(self, shard_factor):
        assert False, 'replicate shard does not has shard factor'

    def shardable_size(self):
        assert False, 'replicate shard does not has shardable size'
    
    def shard(self, t):
        return deepcopy(t), deepcopy(t)
    
    def combine(self, t1, t2):
        return t1
        
class ReduceShardFunc(ShardFuncBase):
    def __init__(self, shard_factor: float, dim: int, tensor_shape: List[int], tensor_dtype: str):
        self.name = 'reduce-shard'
        self.shard_factor = shard_factor
        self.dim = dim
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.is_split = False

    def reset_shard_factor(self, shard_factor):
        assert False, 'replicate shard does not has shard factor'

    def shardable_size(self):
        assert False, 'replicate shard does not has shardable size'
    
    def combine(self, t1, t2):
        return t1 + t2
    
class ShardOption:
    def __init__(self, shard_option: List[ShardFuncBase]):
        self.shard_option = shard_option
    
class ShardAnnotation:
    def __init__(self, cluster_info: ClusterInfo, shard_options: List[ShardOption]):
        self.cluster_info = cluster_info
        self.shard_options = shard_options