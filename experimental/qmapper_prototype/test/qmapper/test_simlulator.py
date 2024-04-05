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
from qmapper.simulator.simulator import Simulator

class BasicMLP(nn.Module):
    def __init__(self, n,m):
        super(BasicMLP,self).__init__()
        self.ff = nn.Linear(n,n)
        self.ff1 = nn.Linear(n,n)
        self.ff2 = nn.Linear(n,n)
        self.layer = nn.Linear(n,m)

    def forward(self, x):
        x = self.ff(x)
        x1 = self.ff1(x)
        x2 = self.ff2(x1)
        return self.layer(x2)

model = BasicMLP(4096, 64)
x = torch.ones(32, 4096)
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
intra_searcher.search_naive()
sim = Simulator(cluster_info, internal_graph, intra_searcher.options, cost_model)
sim.prepare()
print(f'simulate time after naive optimization {sim.simulate()}')

intra_searcher.search()

sim = Simulator(cluster_info, internal_graph, intra_searcher.options, cost_model)
sim.prepare()
print(f'simulate time after intra optimization {sim.simulate()}')