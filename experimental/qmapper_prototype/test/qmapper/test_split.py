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

class BasicMLP(nn.Module):
    def __init__(self, n,m):
        super(BasicMLP,self).__init__()
        self.ff = nn.Linear(n,n)
        self.layer = nn.Linear(n,m)

    def forward(self, x):
        return self.layer(self.ff(x))

model = BasicMLP(1024, 1024)
x = torch.ones(32, 1024)
y = torch.zeros(32, 1024)
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

cluster_info = get_mocked_cluster(4)

cost_model = CostModel(internal_graph, cluster_info)

cost_model.traverse_all_node_strategies()

for node in internal_graph.nodes:
    parallel_options = node.get_parallel_options()
    num_parallel_option = len(parallel_options['output_options'])
    options = cost_model.gen_option(num_parallel_option)
    print(node)
    for option in options:
        print(option)
        print(cost_model.single_node_cost(node.id, option))
