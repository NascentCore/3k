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

model = BasicMLP(4096, 4096)
x = torch.ones(32, 4096)
y = torch.zeros(32, 4096)
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

# for node in internal_graph.nodes:
#     parallel_options = node.get_parallel_options()
#     num_parallel_option = len(parallel_options['output_options'])
#     options = cost_model.gen_option(num_parallel_option)
#     print(node)
#     for option in options:
#         print(option)
#         print(cost_model.single_node_cost(node.id, option))


cost_mem = {}
resharding_mem = {}
past_funcs = {}
past_result = {}
global_options = {}


for node_id in internal_graph.nodes:
    min_cost = 99999999999.9

    internal_node = internal_graph.nodes[node_id]
    num_options = len(internal_node.get_parallel_options()['output_options'])
    options = cost_model.gen_option(num_options)

    output_options = internal_node.get_parallel_options()['output_options']

    for option in options:
        cur_cost = cost_model.single_node_cost(node_id, option)
        if node_id not in cost_mem:
            cost_mem[node_id] = {}
        cost_mem[node_id][option] = cur_cost
        if cur_cost < min_cost:
            global_options[node_id] = option
            past_funcs[internal_node.output_signature[0].id] = [output_options[i] for i in option]
            past_result[internal_node.output_signature[0].id] = cost_model.split_to_single_node(internal_node.output_signature[0], [output_options[i] for i in option])

for node_id in internal_graph.nodes:
    min_cost = 99999999999.9

    internal_node = internal_graph.nodes[node_id]
    num_options = len(internal_node.get_parallel_options()['output_options'])
    options = cost_model.gen_option(num_options)

    output_options = internal_node.get_parallel_options()['output_options']

    for option in options:
        resharding_cost = cost_model.append_node_cost(node_id, option, past_funcs, past_result)
        if node_id not in resharding_mem:
            resharding_mem[node_id] = {}
        resharding_mem[node_id][option] = resharding_cost

for k in cost_mem.keys():
    for option in cost_mem[k].keys():
        print(f'node {k}-{option}: cost: {cost_mem[k][option]}, resharding: {resharding_mem[k][option]}')