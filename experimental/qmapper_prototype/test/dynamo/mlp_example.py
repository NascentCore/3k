import torch
import torch.nn as nn
import contextlib
from torch.nn.utils import stateless
import functools
import traceback
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten
from torch.fx.experimental.proxy_tensor import make_fx

def rsetattr(obj, attr, val):
    pre,_,post = attr.rpartition('.')
    return setattr(rgetattr(obj,pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


global_cnt = 0
GRAPH_INPUTS_CACHE = {}
from functorch.compile import make_boxed_func
def my_compiler(gm: torch.fx.GraphModule, example_inputs):
    class wraped_forward:
        def __init__(self, cnt,gm):
            self.cnt = cnt
            self.gm = gm

        def __call__(self, *args: PyTree, **kwds: PyTree) -> PyTree:
            global GRAPH_INPUTS_CACHE
            GRAPH_INPUTS_CACHE[self.cnt] = (args,kwds)
            # self.gm.graph.print_tabular()
            return self.gm.forward(*args, **kwds)
        
    global global_cnt
    # print(f">>> my_compiler() invoked {global_cnt}th time:")
    # print(">>> FX graph:")
    gm.graph.print_tabular()
    # print('>>> Inputs')
    # print(example_inputs)
    # for node in gm.graph.nodes:
    #     # print(node)
    #     # print(node.target)
    #     pass
    # print(f">>> Code:\n{gm.code}")
    global_cnt += 1
    # traceback.print_stack()
    # return gm.forward  # return a python callable
    return make_boxed_func(wraped_forward(global_cnt, gm))  # return a python callable


from torch._dynamo.backends.common import aot_autograd
my_compiler = aot_autograd(fw_compiler= my_compiler, bw_compiler=my_compiler)

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

# @torch.compile(backend=my_compiler,dynamic=True)
def train_step(model, optimizer, x, y):
    eval_y = model.forward(x)
    loss = nn.CrossEntropyLoss()(eval_y, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def export_graph(func,
                *args, **kwargs):
    module = None
    opt = None
    for idx,it in enumerate(args):
        if isinstance(it, torch.nn.Module):
            if module is None:
                module = it
            else:
                assert False, 'No support for multi-model'

        if isinstance(it, torch.optim.Optimizer):
            if opt is None:
                opt = it
            else:
                assert False, 'No support for multi-model'
    for idx,it in enumerate(kwargs.values()):
        if isinstance(it, torch.nn.Module):
            if module is None:
                module = it
            else:
                assert False, 'No support for multi-model'

        if isinstance(it, torch.optim.Optimizer):
            if opt is None:
                opt = it
            else:
                assert False, 'No support for multi-model'
            
    if not module is None:
        named_params = dict(module.named_parameters())
        named_buffers = dict(module.named_buffers())
    else:
        named_params = {}
        named_buffers = {}

    if module is not None and opt is not None:
        # training
        named_states = {}
        mode = contextlib.nullcontext()
        for name in named_params.keys():
            with torch.no_grad():
                rsetattr(module, name + '.grad', 
                        torch.ones_like(rgetattr(module, name).data).mul_(1e-8))
            with mode:
                opt.step()
                opt.zero_grad(True)
    else:
        named_states = {}
    
    for name, parameter in named_params.items():
        if parameter in opt.state:
            named_states[name] = opt.state[parameter]
            if 'step' in named_states[name]:
                named_states[name]['step'] -= 1

    @torch.compile(backend=my_compiler)
    def stateless_func(func, named_params, named_buffers, named_states, *args, **kwargs):
        with stateless._reparametrize_module(
            module, {**named_params, **named_buffers}, tie_weights=True
        ):
            ret = func(*args, **kwargs)
        named_grads = {k: v.grad for k,v in named_params.items()}
        return named_params, named_buffers, named_states, named_grads, ret
    state_tensor_num = len(named_params) + len(named_buffers) + len(named_states)
    named_params, named_buffers, named_states, named_grads, ret = stateless_func(func, named_params, named_buffers, named_states, *args, **kwargs)
    # for it in (named_params, named_buffers, named_states, named_grads, ret):
    #     if it is None:
    #         continue
    #     print(it)
    # print("\n\n\n\n\n\n")
    
    #print(tree_flatten((named_params, named_buffers, named_states, args, kwargs))[0])
    # traced_graph = make_fx(functools.partial(stateless_func, func),
    #                            tracing_mode='real', 
    #                            _allow_non_fake_inputs=False)(named_params, named_buffers, named_states, *args, **kwargs)

@torch.compile(backend=my_compiler,dynamic=True)
def train_step2(model, optimizer, x, y):
    eval_y = model.forward(x)
    loss = nn.CrossEntropyLoss()(eval_y, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


export_graph(train_step, model, optimizer, x, y)
# train_step2(model, optimizer, x, y)
# print(len(list(model.parameters())))
# from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten
# print(tree_unflatten(model, optimizer, x, y))

# print(train_step2)
# for k,v in GRAPH_INPUTS_CACHE.items():
#     if k == 2:
#         for i in v[0]:
#             print(i)
#         print(len(v[0]), len(v[1]))
#         print()