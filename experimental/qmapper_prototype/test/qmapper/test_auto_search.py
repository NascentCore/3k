import sys
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append("../..")
from qmapper.bridge.torch_bridge.comp_graph import GraphExtracter, _qmapper_fx_module_list_
from qmapper.api.shard_discovery import node_strategy_search
from qmapper.codegen.internal_meta_ir.ir_transform import meta_ir_to_internal_meta_ir
from qmapper.api.meta_ir.cluster_info import ClusterInfo, get_mocked_cluster
from qmapper.optimization.cost_model.cost_model import CostModel
from qmapper.optimization.search.intra_strategy_search import IntraStrategySearcher

class BasicMLP(nn.Module):
    def __init__(self, n,m):
        super(BasicMLP,self).__init__()
        self.ff = nn.Linear(n,n)
        self.layer = nn.Linear(n,m)

    def forward(self, x):
        return self.layer(self.ff(x))

model = BasicMLP(10240, 64)
x = torch.ones(32, 10240)
y = torch.zeros(32, 64)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

def train_step(model, optimizer, x, y):
    eval_y = model.forward(x)
    loss = nn.CrossEntropyLoss()(eval_y, y)
    loss.backward()
    optimizer.step()
    # optimizer.zero_grad()

train_step(model, optimizer, x, y)

gextracter = GraphExtracter(train_step)
gextracter(model, optimizer, x, y)
graph = gextracter.get_meta_graph()

internal_graph = meta_ir_to_internal_meta_ir(graph)

cluster_info = get_mocked_cluster(8)

cost_model = CostModel(internal_graph, cluster_info)

cost_model.traverse_all_node_strategies()

intra_searcher = IntraStrategySearcher(cost_model, internal_graph, cluster_info)

intra_searcher.search()

# for k in intra_searcher.cost_mem.keys():
#     for option in intra_searcher.cost_mem[k].keys():
#         print(f'node {k}-{option}: cost: {intra_searcher.cost_mem[k][option]}, resharding: {intra_searcher.resharding_cost_mem[k][option]}')