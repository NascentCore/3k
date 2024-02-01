from typing import Callable
from functools import update_wrapper
from ..bridge.torch_bridge.comp_graph import export_graph, trace_meta_graph
from .meta_ir.visual import visual_meta

def to_meta(func, *args, **kwargs):
    traced_graph, state_tensor_num, named_params, named_buffers, named_states, args, kwargs = export_graph(func, *args, **kwargs)
    meta_nodes, meta_vars = trace_meta_graph(traced_graph, named_params, named_buffers, named_states, args, kwargs)
    visual_meta(meta_nodes)


def compileFunc(func, *args, **kwargs):
    return None

class JitCompiledFunc:
    def __init__(self, func:Callable):
        update_wrapper(self, func)
        self.func = func
        self.compiled_func = None

    def __call__(self, *args, **kwargs):
        if self.compiled_func is None:
            self.compiled_func = compileFunc(self.func, *args, **kwargs)
        ## test api
        if True:
            to_meta(self.func, *args, **kwargs)
            assert False

        if self.compiled_func is None:
            self.func(*args, **kwargs)
        else:
            self.compiled_func(*args, **kwargs)
    
def qmapper_compile(func: Callable):
    return JitCompiledFunc(func)