from ..cost_model.cost_model import CostModel
from ...codegen.internal_meta_ir.internal_meta_ir import InternalDtype, InternalMetaGraph, InternalMetaOperator, InternalMetaVariable, InternalType, SplitPass, ReducePass, ReplicatePass
from ...api.meta_ir.cluster_info import ClusterInfo
from typing import List, Dict
from copy import deepcopy
import itertools

class IntraStrategySearcher:
    def __init__(self, cost_model: CostModel, internal_graph: InternalMetaGraph, cluster_info: ClusterInfo):
        self.cost_model = cost_model
        self.internal_graph = internal_graph
        self.cluster_info = cluster_info


        self.options = {}
        self.past_funcs = {}
        self.past_results = {}

        self.cost_mem = {}
        self.resharding_cost_mem = {}

    def get_resharding_optimal(self, node_id):
        costs = self.cost_mem[node_id]
        resharding = self.resharding_cost_mem[node_id]

        options = list(costs.keys())

        ret_option = None
        val        = None

        resharding_min_cost = 999999999.99
        for option in options:
            resharding_cost = resharding[option]
            comm_cost       = costs[option]
            if resharding_cost < resharding_min_cost:
                ret_option = option
                val = resharding_cost + comm_cost
                resharding_min_cost = resharding_cost
        return ret_option, val
            

            

    def get_no_resharding_optimal(self, node_id):
        costs = self.cost_mem[node_id]
        resharding = self.resharding_cost_mem[node_id]

        options = list(costs.keys())

        ret_option = None
        val        = None

        min_cost = 999999999.99
        for option in options:
            resharding_cost = resharding[option]
            comm_cost       = costs[option]
            if comm_cost < min_cost:
                ret_option = option
                val = resharding_cost + comm_cost
                min_cost = comm_cost
        return ret_option, val

    def change_parallel_option(self, node_id, option):
        internal_node = self.internal_graph.nodes[node_id]
        concerned_node_ids = self.internal_graph.nodes[node_id].output_signature[0].consume_node_ids
        output_options = self.internal_graph.nodes[node_id].get_parallel_options()['output_options']
        self.past_funcs[internal_node.output_signature[0].id] = [output_options[i] for i in option]
        self.past_results[internal_node.output_signature[0].id] = self.cost_model.split_to_single_node(internal_node.output_signature[0], [output_options[i] for i in option])
        self.options[node_id] = option
        for node_id in concerned_node_ids:
            internal_node = self.internal_graph.nodes[node_id]
            num_options = len(internal_node.get_parallel_options()['output_options'])
            options = self.cost_model.gen_option(num_options)

            output_options = internal_node.get_parallel_options()['output_options']

            for option in options:
                resharding_cost = self.cost_model.append_node_cost(node_id, option, self.past_funcs, self.past_results)
                if node_id not in self.resharding_cost_mem:
                    self.resharding_cost_mem[node_id] = {}
                self.resharding_cost_mem[node_id][option] = resharding_cost

    def calculate_total_cost(self):
        total_cost = 0.0
        for node_id in self.internal_graph.nodes:
            total_cost += self.cost_mem[node_id][self.options[node_id]]
            total_cost += self.resharding_cost_mem[node_id][self.options[node_id]]
        return total_cost
    
    def search_naive(self):
        ## get optimal

        for node_id in self.internal_graph.nodes:
            min_cost = 99999999999.9

            internal_node = self.internal_graph.nodes[node_id]
            num_options = len(internal_node.get_parallel_options()['output_options'])
            options = self.cost_model.gen_option(num_options)

            output_options = internal_node.get_parallel_options()['output_options']

            for option in options:
                cur_cost = self.cost_model.single_node_cost(node_id, option)
                if node_id not in self.cost_mem:
                    self.cost_mem[node_id] = {}
                self.cost_mem[node_id][option] = cur_cost
                if cur_cost < min_cost:
                    self.options[node_id] = option
                    self.past_funcs[internal_node.output_signature[0].id] = [output_options[i] for i in option]
                    self.past_results[internal_node.output_signature[0].id] = self.cost_model.split_to_single_node(internal_node.output_signature[0], [output_options[i] for i in option])

        ## get resharding info
                    
        for node_id in self.internal_graph.nodes:
            min_cost = 99999999999.9

            internal_node = self.internal_graph.nodes[node_id]
            num_options = len(internal_node.get_parallel_options()['output_options'])
            options = self.cost_model.gen_option(num_options)

            output_options = internal_node.get_parallel_options()['output_options']

            for option in options:
                resharding_cost = self.cost_model.append_node_cost(node_id, option, self.past_funcs, self.past_results)
                if node_id not in self.resharding_cost_mem:
                    self.resharding_cost_mem[node_id] = {}
                self.resharding_cost_mem[node_id][option] = resharding_cost

        original_total_cost = self.calculate_total_cost()



    def search(self):
        ## get optimal

        for node_id in self.internal_graph.nodes:
            min_cost = 99999999999.9

            internal_node = self.internal_graph.nodes[node_id]
            num_options = len(internal_node.get_parallel_options()['output_options'])
            options = self.cost_model.gen_option(num_options)

            output_options = internal_node.get_parallel_options()['output_options']

            for option in options:
                cur_cost = self.cost_model.single_node_cost(node_id, option)
                if node_id not in self.cost_mem:
                    self.cost_mem[node_id] = {}
                self.cost_mem[node_id][option] = cur_cost
                if cur_cost < min_cost:
                    self.options[node_id] = option
                    self.past_funcs[internal_node.output_signature[0].id] = [output_options[i] for i in option]
                    self.past_results[internal_node.output_signature[0].id] = self.cost_model.split_to_single_node(internal_node.output_signature[0], [output_options[i] for i in option])

        ## get resharding info
                    
        for node_id in self.internal_graph.nodes:
            min_cost = 99999999999.9

            internal_node = self.internal_graph.nodes[node_id]
            num_options = len(internal_node.get_parallel_options()['output_options'])
            options = self.cost_model.gen_option(num_options)

            output_options = internal_node.get_parallel_options()['output_options']

            for option in options:
                resharding_cost = self.cost_model.append_node_cost(node_id, option, self.past_funcs, self.past_results)
                if node_id not in self.resharding_cost_mem:
                    self.resharding_cost_mem[node_id] = {}
                self.resharding_cost_mem[node_id][option] = resharding_cost

        original_total_cost = self.calculate_total_cost()


        to_optimization_seq = []
        for node_id in self.internal_graph.nodes:
            resharding_option, resharding_val = self.get_resharding_optimal(node_id)
            parallel_option, parallel_val = self.get_no_resharding_optimal(node_id)
            # print(node_id, resharding_val, parallel_val)
            to_optimization_seq.append((node_id, resharding_val - parallel_val))
        
        to_optimization_seq.sort(key=lambda x:x[1], reverse=True)
        
        for to_optimize_node_id, _ in to_optimization_seq:
            costs = self.cost_mem[to_optimize_node_id]
            resharding = self.resharding_cost_mem[to_optimize_node_id]

            options = list(costs.keys())

            ret_option = None
            val        = None

            min_cost = 999999999.99
            for option in options:
                resharding_cost = resharding[option]
                comm_cost       = costs[option]
                if resharding_cost + comm_cost < min_cost:
                    self.change_parallel_option(to_optimize_node_id, option)
                    min_cost = resharding_cost + comm_cost

        optimized_cost = self.calculate_total_cost()


        print(f'before: {original_total_cost}, after: {optimized_cost}')
                

        