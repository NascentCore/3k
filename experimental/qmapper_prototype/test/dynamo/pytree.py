import torch
import torch.nn as nn
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten

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
optimizer = torch.optim.Adam(model.parameters(),lr=torch.tensor(0.001))

x = tree_flatten((model, x, y, optimizer))

for t in x[0]:
    print(t)
    print()