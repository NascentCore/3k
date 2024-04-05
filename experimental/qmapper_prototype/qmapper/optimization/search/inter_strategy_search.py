from ..cost_model.cost_model import CostModel
from ...codegen.internal_meta_ir.internal_meta_ir import InternalDtype, InternalMetaGraph, InternalMetaOperator, InternalMetaVariable, InternalType, SplitPass, ReducePass, ReplicatePass
from ...api.meta_ir.cluster_info import ClusterInfo
from typing import List, Dict
from copy import deepcopy
import itertools

class InterStrategySearch:
    def __init__(self):
        pass