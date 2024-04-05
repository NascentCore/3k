import torch
x = torch.tensor(0, dtype=torch.int64)
y = torch.tensor([1,2,3], dtype=torch.float32)
t = torch.ops.aten._log_softmax.default(y, x, False)
print(torch.ops.aten._log_softmax.default(y, x, False))
print(torch.ops.aten._log_softmax_backward_data.default(t,t,x,torch.float32))