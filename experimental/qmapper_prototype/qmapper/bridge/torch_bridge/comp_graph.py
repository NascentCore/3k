import torch 
import contextlib
import functools
from tqdm import tqdm
import traceback
from typing import Callable, Any, Tuple, List, Optional, Dict, Set, Iterator, cast
from functools import partial
import torch.utils._pytree as pytree
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten
from torch.nn.utils import stateless
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import Argument, Node, Target, _get_qualified_name
from torch.utils._mode_utils import no_dispatch
from torch.fx.graph_module import GraphModule
from torch.fx.interpreter import Interpreter
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from .third_party_utils import EASYDIST_DECOMP_TABLE, adam_traceable_context
from ...api.meta_ir.definitions import MetaNode, MetaVariable, MetaStaticObj

def rsetattr(obj, attr, val):
    pre,_,post = attr.rpartition('.')
    return setattr(rgetattr(obj,pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

@contextlib.contextmanager
def _enable_compile():
    with adam_traceable_context():
        def f_true():
            return True
        def myaddcdiv_inplace_traceable(params, exp_avg, denom, value):
            return params.exp_avg
        orig_is_compiling_code = torch._utils.is_compiling.__code__
        torch._utils.is_compiling.__code__ = f_true.__code__
        try:
            yield
        finally:
            torch._utils.is_compiling.__code__ = orig_is_compiling_code

def get_shape_info(node_output):
    if isinstance(node_output, torch.Tensor) or isinstance(node_output, torch.nn.Parameter):
        return {'shape': node_output.shape,'dtype': node_output.dtype}
    return {}

def to_meta(node_output):
    if isinstance(node_output, FakeTensor):
        with no_dispatch():
            return torch.zeros_like(node_output, device="meta")
    if type(node_output) is torch.Tensor:
        return node_output.detach().to(device="meta").contiguous()
    elif type(node_output) is torch.nn.parameter.Parameter:
        return node_output.data.detach().to(device="meta").contiguous()
    else:
        return node_output
    
def to_real(tensor: torch.Tensor, size=None):
    device = 'cpu'
    if size is None:
        size = tensor.size()
    if isinstance(tensor, torch.Tensor) and tensor.is_meta:
        if tensor.dtype == torch.bool:
            return torch.rand(size, dtype=torch.float, device=device) > 0.5
        elif torch.is_floating_point(tensor):
            return torch.rand(size, dtype=tensor.dtype, device=device)
        else:
            return torch.randint(high=1, size=size, dtype=tensor.dtype, device=device)
    return tensor

def inplace_to_no_inplace(op):
    if op is torch.ops.aten.add_.Tensor:
        return torch.ops.aten.add.Tensor
    
def preprocess_traced_graph(fx_module: torch.fx.GraphModule):
    # fx_module = fix_embedding(fx_module)
    # fx_module = fix_addmm_bias(fx_module)
    # fx_module = fix_convoluation_bias(fx_module)
    # fx_module = eliminate_detach(fx_module)

    # fx_module.recompile()

    return fx_module

def input_signature(prefix, idx):
    return f'{prefix}-{idx}'

def export_graph(func: Callable,
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

    if not module is None and not opt is None:
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

    def stateless_func(func, named_params, named_buffers, named_states, *args, **kwargs):
        with stateless._reparametrize_module(
            module, {**named_params, **named_buffers}, tie_weights=True
        ):
            ret = func(*args, **kwargs)
        named_grads = {k: v.grad for k,v in named_params.items()}
        return named_params, named_buffers, named_states, named_grads, ret
    state_tensor_num = len(named_params) + len(named_buffers) + len(named_states)
    with _enable_compile():
        traced_graph = make_fx(partial(stateless_func, func),
                               tracing_mode='real', 
                               decomposition_table=EASYDIST_DECOMP_TABLE,
                               _allow_non_fake_inputs=False)(named_params, named_buffers, named_states, *args, **kwargs)
    return traced_graph, state_tensor_num, named_params, named_buffers, named_states, args, kwargs

class MetaTracker(Interpreter):
    @compatibility(is_backward_compatible=True)
    def __init__(self, module: GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)
        self.meta_nodes = []
        self.meta_vars = {}

    def make_metas(self, t: Any, var_name: str, unique_name: str):
        if unique_name in self.meta_vars:
            input_arg = self.meta_vars[unique_name]
        else:
            if isinstance(t, torch.Tensor):
                input_arg = MetaVariable(var_name, list(t.shape), str(t.dtype))
            else:
                input_arg = MetaStaticObj(var_name, t)
        if var_name not in self.meta_vars:
            self.meta_vars[unique_name] = input_arg
        return input_arg
                
    @compatibility(is_backward_compatible=True)
    def run(self, *args, initial_env: Optional[Dict[Node, Any]] = None,
            enable_io_processing: bool = True):
        self.env = initial_env if initial_env is not None else {}
        if enable_io_processing:
            args = self.module.graph.process_inputs(*args)
        self.args_iter: Iterator[Any] = iter(args)
        pbar = tqdm(total = len(self.module.graph.nodes),
                    desc = f"{self.name}: {str(list(self.module.graph.nodes))}",
                    initial=0, position=0, leave=True, disable=False, delay=0)
        for node in self.module.graph.nodes:
            pbar.update(1)
            if node in self.env:
                continue
            try:
                self.env[node] = self.run_node(node)
            except Exception as e:
                msg = f'While executing {node.format_node()}'
                msg = '{}\n\n{}'.format(e.args[0], msg) if e.args else str(msg)
                msg += f"\nOriginal traceback:\n{node.stack_trace}"
                e.args = (msg,) + e.args[1:]
                if isinstance(e, KeyError):
                    raise RuntimeError(*e.args) from e
                raise

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == 'output':
                output_val = self.env[node]
                inputs: List[MetaVariable] = []
                for idx, t in enumerate(output_val):
                    input_arg = self.make_metas(t, f'output-input-{idx}', str(node.args[0][idx]))
                    inputs.append(input_arg)
                for idx, arg in enumerate(inputs):
                    if arg.up_node is None:
                        static_id = MetaNode.generate_uuid()
                        arg.up_node = MetaNode(name=f'static_element-{static_id}', 
                                                op_name='static',
                                                inputs=[arg],
                                                outputs=[arg],
                                                sharding_info=None,
                                                is_placeholder=True)
                arg.idx_in_up_node = 0
                output_node = MetaNode('output', 'output', inputs, [], [], is_placeholder=True)
                for idx, arg in enumerate(inputs):
                    arg.down_nodes.append(output_node)
                    arg.indice_in_down_nodes.append(idx)
                self.meta_nodes.append(output_node)
                return self.module.graph.process_outputs(output_val)
            
    @compatibility(is_backward_compatible=True)
    def run_node(self, n : Node) -> Any:
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            new_kwargs = {k:v for k,v in kwargs.items()}
            new_kwargs['__fmapper_only_orig_input'] = n.args
            return getattr(self, n.op)(n.target, args, new_kwargs)

    @compatibility(is_backward_compatible=True)
    def placeholder(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        assert isinstance(target, str)
        ori_input_signature = None
        if '__fmapper_only_orig_input' in kwargs:
            ori_input_signature = kwargs['__fmapper_only_orig_input']
            del kwargs['__fmapper_only_orig_input']
        if target.startswith('*'):
            arg_list = list(self.args_iter)
            inputs = []
            for idx, arg in enumerate(arg_list):
                meta_arg = self.make_metas(arg, f'{target}-input-{idx}', str(ori_input_signature[idx]))
                inputs.append(meta_arg)
            placeholder_node = MetaNode(name=f'placeholder-{target}', 
                                        op_name='placeholder',
                                        inputs=inputs,
                                        outputs=inputs,
                                        is_placeholder=True)
            self.meta_nodes.append(placeholder_node)
            for idx, arg in enumerate(inputs):
                arg:MetaStaticObj
                if arg.up_node is not None:
                    arg.up_node = placeholder_node
                    arg.idx_in_up_node = idx
                else:
                    pass
                    # logging.warning(f"creating a placeholder node for {arg}, which has already an up node {arg.up_node}")
            return list(self.args_iter)
        else:
            try:
                t = next(self.args_iter)
                meta_arg = self.make_metas(t, f'{target}-input', str(target))
                placeholder_node = MetaNode(name=f'placeholder-{target}', 
                                            op_name='placeholder', 
                                            inputs=[meta_arg], 
                                            outputs=[meta_arg], 
                                            sharding_info=None, 
                                            is_placeholder=True)
                self.meta_nodes.append(placeholder_node)
                if meta_arg.up_node is None:
                    meta_arg.up_node = placeholder_node
                    meta_arg.idx_in_up_node = 0
                else:
                    pass
                    # logging.warning(f"creating a placeholder node for {meta_arg}, which has already an up node {meta_arg.up_node}")
                return t
            except StopIteration as si:
                if len(args) > 0:
                    t = args[0]
                    meta_arg = self.make_metas(t, f'{target}-input', str(target))
                    placeholder_node = MetaNode(name=f'placeholder-{target}', 
                                                op_name='placeholder', 
                                                inputs=[meta_arg], 
                                                outputs=[meta_arg], 
                                                sharding_info=None, 
                                                is_placeholder=True)
                    self.meta_nodes.append(placeholder_node)
                    if meta_arg.up_node is None:
                        meta_arg.up_node = placeholder_node
                        meta_arg.idx_in_up_node = 0
                    else:
                        pass 
                        # logging.warning(f"creating a placeholder node for {meta_arg}, which has already an up node {meta_arg.up_node}")
                    return args[0]
                else:
                    raise RuntimeError(f'Expected positional argument for parameter {target}, but one was not passed in!') from si
    
    @compatibility(is_backward_compatible=True)
    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        assert not isinstance(target, str)
        ori_input_signature = None
        if '__fmapper_only_orig_input' in kwargs:
            ori_input_signature = kwargs['__fmapper_only_orig_input']
            del kwargs['__fmapper_only_orig_input']
        inputs = []
        outputs = []
        for idx, arg in enumerate(args):
            meta_arg = self.make_metas(arg, f'{target}-input-{idx}', str(ori_input_signature[idx]))
            inputs.append(meta_arg)
        # print(target)
        # for arg in args:
        #     if isinstance(arg, torch.Tensor):
        #         print(arg.shape)
        #     else:
        #         print(arg)
        try:
            output = target(*args, **kwargs)
        except RuntimeError as e:
            # logging.warning(e)
            op = inplace_to_no_inplace(target)
            output = op(*args, **kwargs)
        ori_output = output
        if not isinstance(output, list):
            output = [output]
        for idx, out_arg in enumerate(output):
            meta_arg = self.make_metas(out_arg, f'{target}-output-{idx}', str(target))
            outputs.append(meta_arg)

        # if 'log_softmax_backward_data' in f'{target}':
        #     for arg in args:
        #         if isinstance(arg, torch.Tensor):
        #             print(arg.shape)
        #         else:
        #             print(arg)
        #     for meta_arg in inputs:
        #         if isinstance(meta_arg, MetaVar):
        #             print(meta_arg.shape)
        #         else:
        #             print(meta_arg.obj)
        #     print(ori_input_signature)
        #     print(type(ori_input_signature[0]))
        #     print(str(ori_input_signature[0]))
        #     assert False

        meta_node = MetaNode(name=f'callfunction-{target}',
                             op_name=f'{target}',
                             inputs=inputs,
                             outputs=outputs,
                             sharding_info=None,
                             is_placeholder=False)
        self.meta_nodes.append(meta_node)
        for idx, arg in enumerate(inputs):
            if arg.up_node is None:
                static_id = MetaNode.generate_uuid()
                arg.up_node = MetaNode(name=f'static_element-{static_id}', 
                                       op_name='static',
                                       inputs=[arg],
                                       outputs=[arg],
                                       sharding_info=None,
                                       is_placeholder=True)
                arg.idx_in_up_node = 0
        for idx, arg in enumerate(inputs):
            arg.down_nodes.append(meta_node)
            arg.indice_in_down_nodes.append(idx)
        for idx, arg in enumerate(outputs):
            arg.up_node = meta_node
            arg.idx_in_up_node = idx
        return ori_output

def trace_meta_graph(traced_graph, named_params, named_buffers, named_states, args, kwargs):
    interpreter = MetaTracker(traced_graph)
    flattened_args = tree_flatten((named_params, named_buffers, named_states, args, kwargs))
    interpreter.run(flattened_args)
    print(f'nodes number: {len(interpreter.meta_nodes)}')
    print(f'variables number: {len(interpreter.meta_vars)}')
    return interpreter.meta_nodes, interpreter.meta_vars

