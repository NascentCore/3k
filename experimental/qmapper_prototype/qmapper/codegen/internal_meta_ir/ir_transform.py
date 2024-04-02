from enum import Enum
from copy import deepcopy
from typing import List, Optional, Union, Tuple
import functools
import operator
from ...api.meta_ir.definitions import MetaGraph, MetaNode, MetaVariable
from .internal_meta_ir import InternalDtype, InternalMetaGraph, InternalMetaOperator, InternalMetaVariable, InternalType
from .internal_operators.internal_operators import Internal_Add, Internal_Div, \
    Internal_Expand, Internal_Log_Softmax, Internal_Log_Softmax_Backward, \
    Internal_Matmul,Internal_Mul,Internal_Neg,Internal_Pow,Internal_Reciprocal,\
    Internal_Sqrt,Internal_Sub,Internal_Transpose,Internal_View, dispatch_internal_operator

def get_matching():
    match_rules = {}
    with open("config", "r") as f:
        while True:
            s = f.readline()
            if len(s) == 0:
                break
            torch_operator_section = s[:s.find('(')]
            args_section = s[s.find('(') + 1: s.find(')')]
            args_signatures = args_section.split(',')
            if "addcdiv" in torch_operator_section or "addcmul" in torch_operator_section:
                torch_operator_section = f'{torch_operator_section}_{len(args_signatures)}'
            original_args = {args_signatures[i]:i for i in range(len(args_signatures))}
            my_impl_section = s.split()[-1]
            my_op_impl_sections = my_impl_section.split('/')
            match_rules[torch_operator_section] = [[len(original_args)]]
            output_idx = len(original_args)
            for op_impl_section in my_op_impl_sections:
                my_op_section = op_impl_section[:op_impl_section.find('(')]
                my_op_args_section = op_impl_section[op_impl_section.find('(')+1: op_impl_section.find(')')]
                my_op_args_list = my_op_args_section.split(',')
                
                match_rules[f'{torch_operator_section}'].append([my_op_section])
                for idx, arg in enumerate(my_op_args_list):
                    if arg == '_':
                        match_rules[torch_operator_section][-1].append(output_idx)
                        output_idx += 1
                    else:
                        match_rules[torch_operator_section][-1].append(original_args[arg])
    return match_rules             
        

def meta_ir_to_internal_meta_ir(meta_graph: MetaGraph):
    stateful_node   = meta_graph.nodes['stateful_variables']
    input_node      = meta_graph.nodes['input_variables']
    static_node     = meta_graph.nodes['static_variables']

    registered_parameters   = {meta_var.name: meta_var for meta_var in stateful_node.outputs}
    registered_buffers      = {meta_var.name: meta_var for meta_var in static_node.outputs}
    registered_inputs       = {meta_var.name: meta_var for meta_var in input_node.outputs}
    registered_activations  = {}

    nodes = meta_graph.topology_sort()
    arg_matching_rules = get_matching()
    # for k,v in arg_matching_rules.items():
    #     print(k)
    #     print(v)
    #     print()

    internal_meta_nodes = []
    meta_vars = {}
    node_id = 0
    output_cur_id = 1000000

    for meta_node in nodes:
        if meta_node.is_placeholder:
            continue
        if meta_node.op_name == "aten.detach.default":
            internal_meta_var = InternalMetaVariable(meta_node.inputs[0].uuid, InternalType.Tensor, meta_node.inputs[0].dtype, meta_node.inputs[0].shape, None, None)
            meta_vars[meta_node.inputs[0].uuid] = internal_meta_var
            meta_vars[meta_node.outputs[0].uuid] = meta_vars[meta_node.inputs[0].uuid]
            continue
        meta_inputs = meta_node.inputs
        if "addcdiv" in meta_node.op_name or "addcmul" in meta_node.op_name:
            num_meta_input = arg_matching_rules[f"{meta_node.op_name}_{len(meta_node.inputs)}"][0][0]
            internal_instances = arg_matching_rules[f"{meta_node.op_name}_{len(meta_node.inputs)}"][1:]
        else:
            num_meta_input = arg_matching_rules[meta_node.op_name][0][0]
            internal_instances = arg_matching_rules[meta_node.op_name][1:]
        outputs = []

        meta_input_names = [meta_var.name for meta_var in meta_inputs]

        for internal_instance in internal_instances:
            input_idx = internal_instance[1:]
            tmp_meta_node = MetaNode(None, None, 
                                     inputs=[meta_inputs[i] if i<num_meta_input else outputs[i-num_meta_input] for i in input_idx], 
                                     outputs=None, sharding_info=None, is_placeholder=False)
            internal_op_instance = dispatch_internal_operator(internal_instance[0])
            internal_input_pattern: List[InternalMetaVariable]
            internal_input_pattern = internal_op_instance.get_input_sig_from_meta_node(tmp_meta_node)
            internal_meta_inputs = []
            for i in range(len(internal_input_pattern)):
                idx = int(input_idx[i])
                if idx < num_meta_input:
                    meta_var_id = meta_inputs[idx].uuid
                    if meta_var_id in meta_vars:
                        meta_vars[meta_var_id].consume_node_ids.append(node_id)
                        meta_vars[meta_var_id].indice_in_consume_node.append(i)
                        internal_meta_inputs.append(meta_vars[meta_var_id])
                    else:
                        meta_vars[meta_var_id] = internal_input_pattern[i]
                        meta_vars[meta_var_id].id = meta_var_id
                        meta_vars[meta_var_id].consume_node_ids.append(node_id)
                        meta_vars[meta_var_id].indice_in_consume_node.append(i)
                        internal_meta_inputs.append(meta_vars[meta_var_id])
                else:
                    outputs[idx-num_meta_input].consume_node_ids.append(node_id)
                    outputs[idx-num_meta_input].indice_in_consume_node.append(i)
                    internal_meta_inputs.append(outputs[idx-num_meta_input])
            internal_op_name = f'{meta_node.name}.{internal_instance[0]}'
            internal_op = internal_op_instance.get_dispatched(node_id, internal_meta_inputs)
            internal_meta_nodes.append(internal_op)
            output_internal_meta_var = internal_op.get_output_pattern()
            if internal_instance == internal_instances[-1]:
                # print(meta_node.name)
                # print(meta_node.outputs[0].uuid)
                output_internal_meta_var.id = meta_node.outputs[0].uuid
                meta_vars[output_internal_meta_var.id] = output_internal_meta_var
            else:
                output_internal_meta_var.id = output_cur_id
                output_cur_id += 1
                outputs.append(output_internal_meta_var)
                meta_vars[output_internal_meta_var.id] = output_internal_meta_var
            output_internal_meta_var.index_in_gen_node = 0
            output_internal_meta_var.gen_node_id = node_id
            internal_op.output_signature = [output_internal_meta_var]
            node_id += 1
    # for node in internal_meta_nodes:
    #     print()
    #     print(node)
    #     for input in node.input_signature:
    #         print(input)

    # print(meta_graph.re_mapping)
    remapping = {meta_graph.nodes[k].outputs[0].uuid:[meta_var.uuid for meta_var in meta_graph.nodes['stateful_variables'].outputs if meta_var.name == v][0] for k, v in meta_graph.re_mapping.items()}
    # print(remapping)

    # print(len(meta_graph.re_mapping))
    # print(len(remapping))

    # for k,v in meta_vars.items():
    #     print(v)

    # print(meta_vars[101].value)
    return InternalMetaGraph(internal_meta_nodes, meta_vars, remapping)        
    
