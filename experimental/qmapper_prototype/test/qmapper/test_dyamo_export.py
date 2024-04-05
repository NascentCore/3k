import sys
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append("../..")
from qmapper.bridge.torch_bridge.comp_graph import GraphExtracter, _qmapper_fx_module_list_
from qmapper.api.shard_discovery import node_strategy_search

class BasicMLP(nn.Module):
    def __init__(self, n,m):
        super(BasicMLP,self).__init__()
        self.ff = nn.Linear(n,n)
        self.layer = nn.Linear(n,m)

    def forward(self, x):
        return self.layer(self.ff(x))

model = BasicMLP(10, 4)
x = torch.ones(32, 10)
y = torch.zeros(32, 4)
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

# print(graph.dependency_connections)
# for k,v in graph.nodes.items():
#     # print(len(node_strategy_search(node)))
#     # # for i in range(len(node.inputs)):
#     # #     print(node.inputs[i].shape)
#     # print(v)
#     pass

# G = graph.to_networkx_clear()
# plt.figure(figsize=(50,50))
# nx.draw(G, pos=nx.spring_layout(G), with_labels = True)
# plt.show()
    

# print()
# for node in graph.topology_sort():
#     print(node)
#     for input in node.inputs:
#         print(input)

#     for output in node.outputs:
#         print(output)
#     print(node_strategy_search(node))
#     print()

# for module in _qmapper_fx_module_list_:
#     # print(module)
#     module[-1].graph.print_tabular()