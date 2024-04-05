from ..codegen.internal_meta_ir.internal_meta_ir import InternalMetaGraph, InternalDtype, InternalMetaOperator, InternalMetaVariable, InternalType
from ..api.meta_ir.cluster_info import ClusterInfo
from typing import Dict, List

class Executable:
    def __init__(self,cluster_info: ClusterInfo, internal_graph: InternalMetaGraph, shard_options: Dict[int, List[int]]):
        self.internal_graph = internal_graph
        self.cluster_info = cluster_info
        self.shard_options = shard_options

    