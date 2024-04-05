from typing import List
import torch
global_cnt = 0
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    global global_cnt
    print(f">>> my_compiler() invoked {global_cnt}th time:")
    print(">>> FX graph:")
    gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    global_cnt += 1
    return gm.forward  # return a python callable

@torch.compile(backend=my_compiler)
def foo(x, y):
    if torch.any(x > 1):
        return (x + y) * x
    else:
        return x + y

if __name__ == "__main__":
    a, b = torch.randn(10), torch.ones(10)
    foo(a, b)
    a, b = torch.randn(12), torch.ones(12)
    foo(a, b)
    a, b = torch.randn(14), torch.ones(14)
    foo(a, b)