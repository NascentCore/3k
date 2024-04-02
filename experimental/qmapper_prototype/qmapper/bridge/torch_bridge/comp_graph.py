import torch 
import contextlib
import functools
from tabulate import tabulate
import torch.utils._pytree as pytree
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten
from torch.fx.graph_module import GraphModule
from torch.fx.interpreter import Interpreter
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch.fx import Node
from torch.nn.utils import stateless

from ...api.meta_ir.definitions import MetaNode, MetaVariable, MetaGraph
from ...config import qmapper_logger

def rsetattr(obj, attr, val):
    pre,_,post = attr.rpartition('.')
    return setattr(rgetattr(obj,pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def flatten_keys(dic):
    flattened_dict = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            for _k, _v in flatten_keys(v).items():
                flattened_dict[f'{k}.{_k}'] = _v
        else:
            flattened_dict[k] = v
    return flattened_dict

def get_stateless_func(func, *args, **kwargs):
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
    train_extracter = aot_autograd(fw_compiler = dynamo_extracter, bw_compiler = dynamo_extracter)
    @torch.compile(backend=train_extracter, dynamic=True)
    def stateless_func(named_params, named_buffers, named_states, *_args, **_kwargs):
        with stateless._reparametrize_module(
            module, {**named_params, **named_buffers}, tie_weights=True
        ):
            ret = func(*_args, **_kwargs)
        named_grads = {f'{k}.grad': v.grad for k,v in named_params.items()}
        return named_params, named_buffers, named_states, named_grads, ret
    return stateless_func, named_params, named_buffers, named_states



_qmapper_fx_module_list_ = []
_qmapper_fx_module_cnt_  = 0

def dynamo_extracter(gm: torch.fx.GraphModule, example_inputs):
    class wrapped_forward:
        def __init__(self, cnt, gm):
            self.cnt = cnt
            self.gm = gm

        def __call__(self, *args: PyTree, **kwds: PyTree) -> PyTree:
            qmapper_logger.info(f'graph module {self.cnt}: arglen({len(args)}, kwargslen({len(kwds)}))')
            out = self.gm.forward(*args, **kwds)
            global _qmapper_fx_module_list_
            _qmapper_fx_module_list_.append((args, kwds, out, self.gm))
            return out
    global _qmapper_fx_module_cnt_
    _qmapper_fx_module_cnt_ += 1
    return make_boxed_func(wrapped_forward(_qmapper_fx_module_cnt_, gm))  # return a python callable

class GraphExtracter:
    def __init__(self, func):
        self.func = func             
        self.stateless_func = None    
        self.graph_modules = []        # cache graph modules
        self.input_cache = {}          # cache input of one turn
        self.output_cache = {}         # cache output of one tuen
        self.state_variables = {}      # params, grad, state, buffer
        self.params_variables = {}     # params, buffer
        self.input_name_mapping = []   # mapping graph inputs to the args already known
        # self.output_name_mapping = {}  # mapping graph outputs to the args already known
        self.output_pos = None
        self.re_mapping = {}

        self.meta_arg_pool = {}
        self.real_arg_pool = {}

        self.stateful_variable_pool = MetaNode(name='stateful_variables', 
                                               op_name='stateful_variables',
                                               inputs=[], outputs = [],
                                               sharding_info=None, 
                                               is_placeholder=True)
        self.input_variable_pool = MetaNode(name='input_variables', 
                                            op_name='input_variables',
                                            inputs=[], outputs=[],
                                            sharding_info=None, 
                                            is_placeholder=True)
        self.static_variable_pool = MetaNode(name='static_variables', 
                                             op_name='static_variables',
                                             inputs=[], outputs=[],
                                             sharding_info=None, 
                                             is_placeholder=True)
        
    def make_t(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, list):
            return torch.tensor(obj)
        elif isinstance(obj, int):
            return torch.tensor(obj)
        elif isinstance(obj, float):
            return torch.tensor(obj)
        else:
            return obj

    def make_meta_var(self, up_node: MetaNode, name:str, t:torch.Tensor):
        if isinstance(t, torch.Tensor):
            no_torch = False
        else:
            no_torch = True
        if not isinstance(t, torch.dtype) and t is not None:
            t = self.make_t(t)
            meta_var = MetaVariable(t=t, name = name, shape=list(t.shape), 
                                    dtype = str(t.dtype).split('.')[-1], no_torch=no_torch)
        else:
            meta_var = MetaVariable(t=t, name=str(t).split('.')[-1], shape=None,
                                    dtype=None, no_torch=True)
        up_node.outputs.append(meta_var)
        meta_var.index_in_up_node = len(up_node.outputs)-1
        meta_var.up_node = up_node
        return meta_var

    def get_signatured_input(self, prefix, *args, **kwargs):
        signatures = {}
        for idx, arg in enumerate(args):
            signatures[f'{prefix}_arg_{idx}'] = arg
        for k in kwargs.keys():
            signatures[f'{prefix}_arg_{k}'] = kwargs[k]
        return signatures

    def get_mapping(self, i):
        args, kwds, out, gm = self.graph_modules[i]
        module_input_signature = self.get_signatured_input(f'graph_{i}_input', *args, **kwds)
        module_output_signature = self.get_signatured_input(f'graph_{i}_output', *out)

        local2global_mapping = {}
        # input must find something to match (from real pool)
        for k,v in module_input_signature.items():
            if not isinstance(v, torch.nn.parameter.Parameter) and not isinstance(v, torch.Tensor):
                continue
            check_matched = False
            for _k, _v in self.real_arg_pool.items():
                if isinstance(_v, torch.nn.parameter.Parameter) or isinstance(_v, torch.Tensor):
                    if(v.data_ptr() == _v.data_ptr()):
                        local2global_mapping[k] = _k
                        check_matched = True
                        break
            if not check_matched:
                qmapper_logger.error(f"arg cannot find matched item: {k}")
                # assert False
        
        self.input_name_mapping.append(local2global_mapping)

    def get_meta_graph(self):
        nodes = {
            self.stateful_variable_pool.name: self.stateful_variable_pool,
            self.input_variable_pool.name: self.input_variable_pool,
            self.static_variable_pool.name: self.static_variable_pool
        }
        for graph_module_idx in range(len(self.graph_modules)):
            qmapper_logger.info(f"process graph module {graph_module_idx}")
            self.get_mapping(graph_module_idx)
            _,_,gout,gm = self.graph_modules[graph_module_idx]
            gm: GraphModule
            input_map = self.input_name_mapping[graph_module_idx]
            input_map_list = list(input_map.keys())
            for node_idx, node in enumerate(gm.graph.nodes):
                node: Node
                if node.op == 'placeholder':
                    if node.name in nodes:
                        continue
                    if node.name.startswith("tangents"):
                        meta_var = self.make_meta_var(self.static_variable_pool, node.name, float(node.name.split('_')[-1]))
                        nodes[node.name] = MetaNode(name=node.name,
                                                   op_name='placeholder',
                                                   inputs=[meta_var],
                                                   outputs=[meta_var],
                                                   sharding_info=None,
                                                   is_placeholder=True)
                        meta_var.down_nodes.append(node)
                        meta_var.indice_in_down_nodes.append(0)
                        self.meta_arg_pool[node.name] = meta_var
                        continue
                    real_input = input_map[input_map_list[node_idx]]
                    nodes[node.name] = MetaNode(name=node.name,
                                                op_name='placeholder',
                                                inputs=[self.meta_arg_pool[real_input]],
                                                outputs=[self.meta_arg_pool[real_input]],
                                                sharding_info=None,
                                                is_placeholder=True)
                    self.meta_arg_pool[real_input].down_nodes.append(nodes[node.name])
                    self.meta_arg_pool[real_input].indice_in_down_nodes.append(len(self.meta_arg_pool[real_input].down_nodes)-1)
                elif node.op == 'call_function':
                    def make_arg_kwarg(input_meta_vars):
                        arg = []
                        kwarg = {}
                        for input_meta_var in input_meta_vars:
                            if 'kwarg' in input_meta_var.name:
                                kwarg[input_meta_var.name.split('_')[-1]] = input_meta_var.get_real()
                            else:
                                arg.append(input_meta_var.get_real())
                        return arg, kwarg
                    input_node_list = list(node.args)
                    # print(node.op,node.name, node.args, node.kwargs)
                    input_meta_vars = [nodes[tnode.name].outputs[0] if isinstance(tnode, Node) else tnode for tnode in input_node_list]
                    for i in range(len(input_meta_vars)):
                        if not isinstance(input_meta_vars[i], MetaVariable):
                            input_meta_vars[i] = self.make_meta_var(
                                                 self.static_variable_pool, 
                                                 f'{node.name}_arg_{i}', 
                                                 input_meta_vars[i])
                            

                    input_kwargs = node.kwargs
                    for k,v in input_kwargs.items():
                        self.meta_arg_pool[f'{node.name}_kwargs_{k}'] = \
                            self.make_meta_var(self.static_variable_pool, 
                                               f'{node.name}_kwargs_{k}',
                                               v)
                        input_meta_vars.append(self.meta_arg_pool[f'{node.name}_kwargs_{k}'])
                    # print(input_meta_vars)
                    tmp_arg, tmp_kwarg = make_arg_kwarg(input_meta_vars)

                    real_output = node.target(*tmp_arg, **tmp_kwarg)
                    self.real_arg_pool[node.name] = real_output

                    nodes[node.name] = MetaNode(name=node.name,
                                                op_name=str(node.target),
                                                inputs=input_meta_vars,
                                                outputs=[],
                                                sharding_info=None,
                                                is_placeholder=False)
                    self.meta_arg_pool[node.name] = self.make_meta_var(nodes[node.name], node.name, real_output)
                elif node.op == 'output':
                    node_args = node.args[0]
                    local2global_mapping = {}
                    for targ in node_args:
                        if targ is None:
                            continue
                        if isinstance(targ, Node):
                            # if graph_module_idx == len(self.graph_modules) - 1:
                            #     print(targ)
                            #     output_tensor = nodes[targ.name].outputs[0].get_real()
                            #     check_matched = False
                            #     for _k, _v in self.real_arg_pool.items():
                            #         if _k not in self.state_variables:
                            #             continue
                            #         if isinstance(_v, torch.nn.parameter.Parameter) or isinstance(_v, torch.Tensor):
                            #             if(output_tensor.data_ptr() == _v.data_ptr()):
                            #                 local2global_mapping[targ.name] = _k
                            #                 check_matched = True
                            #                 break
                            #     if check_matched == False:
                            #         assert False
                            #     continue
                            if not nodes[targ.name].is_placeholder:
                                self.real_arg_pool[targ.name] = nodes[targ.name].outputs[0].get_real()
                            else:
                                continue
                        else:
                            qmapper_logger.debug(f"unexpected input for torch fx output node: {targ}")
                            assert False
                    if graph_module_idx == len(self.graph_modules) - 1:
                        # gm.graph.print_tabular()
                        output_name = [out_node.name for out_node in node.args[0]]
                        global_output_idx = 0
                        for it in self.output_pos:
                            if isinstance(it, dict):
                                for k in it.keys():
                                    self.re_mapping[output_name[global_output_idx]] = k
                                    global_output_idx += 1
                            elif isinstance(it, tuple):
                                for k in it:
                                    self.re_mapping[output_name[global_output_idx]] = f'output_{global_output_idx}'
                                    global_output_idx += 1
                            elif it is None:
                                continue
                            else:
                                qmapper_logger.error(f"unexpected output sign: {it}")
                                assert False

                else:
                    qmapper_logger.error(f"unexpected node: {node.name} {node.op}")
                    assert False
        
        return MetaGraph('train_graph', nodes=nodes, re_mapping=self.re_mapping)
        
    def __call__(self, *args, **kwargs):
        self.stateless_func, named_params, named_buffers, named_states = get_stateless_func(self.func, *args, **kwargs)
        named_params, named_buffers, named_states, named_grads, ret = self.stateless_func(named_params, named_buffers, named_states, *args, **kwargs)
        global _qmapper_fx_module_list_
        for i in range(len(_qmapper_fx_module_list_)):
            self.graph_modules.append(_qmapper_fx_module_list_[i])
        self.state_variables =  {**flatten_keys(named_params), 
                                 **flatten_keys(named_buffers),
                                 **flatten_keys(named_states),
                                 **flatten_keys(named_grads)}
        self.params_variables = {**flatten_keys(named_params),
                                 **flatten_keys(named_buffers)}

        global_inputs  = self.get_signatured_input('global_input', *args, **kwargs)
        if not ret is None:
            global_outputs = self.get_signatured_input('global_output', *ret)
        self.real_arg_pool = {**global_inputs, **self.state_variables}
        for k,v in self.real_arg_pool.items():
            if isinstance(v, torch.nn.Module) or isinstance(v, torch.optim.Optimizer):
                continue
            if not isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.parameter.Parameter):
                tmp_v = self.make_t(v)
            else:
                tmp_v = v
            if k in self.state_variables:
                self.meta_arg_pool[k] = self.make_meta_var(self.stateful_variable_pool, k, tmp_v)
            else:
                self.meta_arg_pool[k] = self.make_meta_var(self.input_variable_pool, k, tmp_v)

        self.output_pos = (named_params, flatten_keys(named_buffers), flatten_keys(named_states), ret)


