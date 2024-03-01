import tvm
from tvm import te
import numpy as np
from tvm.script import tir as T
from tvm import te, auto_scheduler
from tvm.contrib import cc, utils
from typing import Callable
import os

from .operator_impls import mm

def operator_dispatcher(op_name):
    # Short this to test following functions
    return mm.matmul_add

def operator_export(op_name: str, args, target: tvm.target.Target):
    operator_func = operator_dispatcher(op_name)
    task = auto_scheduler.SearchTask(operator_func, args, target=target)
    log_file = "./operator_trails.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    ) 
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    f = tvm.build(sch, args, target=target, name=op_name)
    os.mkdir("./tmp_operator_dir")
    f.save(f"./tmp_operator_dir/{op_name}.o")
    if target.kind.name == "cuda":
        f.imported_modules[0].save(f"./tmp_operator_dir/{op_name}.cubin")
    if target.kind.name == "rocm":
        f.imported_modules[0].save(f"./tmp_operator_dir/{op_name}.hsaco")
    cc.create_shared("./tmp_operator_dir/{op_name}.so", [f"./tmp_operator_dir/{op_name}.o"])

def operator_load(op_name: str, target: tvm.target.Target):
    operator_func = tvm.runtime.load_module(f"./tmp_operator_dir/{op_name}.so")
    if target.kind.name == "cuda":
        operator_func_dev = tvm.runtime.load_module(f"./tmp_operator_dir/{op_name}.cubin")
        operator_func.import_module(operator_func_dev)
    elif target.kind.name == "rocm":
        operator_func_dev = tvm.runtime.load_module(f"./tmp_operator_dir/{op_name}.hsaco")
        operator_func.import_module(operator_func_dev)
    return operator_func