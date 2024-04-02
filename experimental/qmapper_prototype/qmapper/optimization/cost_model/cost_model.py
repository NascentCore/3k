from ...codegen.internal_meta_ir.internal_meta_ir import InternalDtype, InternalMetaGraph, InternalMetaOperator, InternalMetaVariable, InternalType, SplitPass, ReducePass, ReplicatePass
from ...api.meta_ir.cluster_info import ClusterInfo
from typing import List, Dict
from copy import deepcopy
import itertools

class CostModel:
    def __init__(self, internal_meta_ir: InternalMetaGraph, cluster_info:ClusterInfo) -> None:
        self.internal_graph=internal_meta_ir
        self.cluster_info=cluster_info
        self.intermediate_result = {}
        self.intermediate_shard_function = {}
        self.shard_results = {}
        self.shard_scoreboard = {}

    def split_to_single_node(self,meta_variable: InternalMetaVariable, shard_options: List[SplitPass]):
        splited_info = {}
        split_info = self.cluster_info.splited_cluster
        def recursive_split(meta_variable: InternalMetaVariable, shard_options, cur_idx):
            if len(split_info[cur_idx]) == 1:
                splited_info[list(split_info[cur_idx])[0]] = meta_variable
            else:
                sub_0 = cur_idx * 2 + 1
                sub_1 = cur_idx * 2 + 2
                sub0_comp = self.cluster_info.get_sum_compcapacity(split_info[sub_0])
                sub1_comp = self.cluster_info.get_sum_compcapacity(split_info[sub_1])
                min_comm = self.cluster_info.get_min_bandwidth_between_group(split_info[sub_0], split_info[sub_1])
                split_factor = sub0_comp/(sub0_comp + sub1_comp)
                layer_num = 0
                while cur_idx != 0:
                    layer_num += 1
                    cur_idx = (cur_idx-1) >> 1
                split_func = shard_options[layer_num]
                if isinstance(split_func, SplitPass):
                    split0, split1 = meta_variable.get_splited(split_factor, split_func.dim)
                    recursive_split(split0, shard_options, sub_0)
                    recursive_split(split1, shard_options, sub_1)
                else:
                    recursive_split(deepcopy(meta_variable), shard_options, sub_0)
                    recursive_split(deepcopy(meta_variable), shard_options, sub_1)
        
        recursive_split(meta_variable, shard_options, 0)
        return splited_info

    def traverse_all_node_strategies(self):
        def get_depth(tree,i=0):
            if i >= len(tree):
                return 0
            max_depth = 0
            for ni in [i*2+1, i*2+2]:
                depth = get_depth(tree,ni)
                if depth > max_depth:
                    max_depth = depth
            return max_depth + 1
            
        split_device_tree = self.cluster_info.get_split_cluster_tree()[1]
        num_layers = get_depth(split_device_tree)-1

        splited_result = {}
        self.intermediate_shard_function = {}

        for node_id in self.internal_graph.nodes.keys():
            node = self.internal_graph.nodes[node_id]
            parallel_options = node.get_parallel_options()
            input_parallel_options = parallel_options['input_options']
            output_parallel_options= parallel_options['output_options']
            num_parallel_options = len(parallel_options['input_options'])
            splited_result[node.id] = {}
            self.intermediate_shard_function[node.id] = {}
            for option_list in itertools.product(*tuple([list(range(num_parallel_options)) for _ in range(num_layers)])):
                split_f = [[] for _ in range(len(node.input_signature))]
                splited_result[node.id][option_list] = {}
                self.intermediate_shard_function[node.id][option_list] = {}
                for i in range(len(node.input_signature)):
                    for t in option_list:
                        split_f[i].append(input_parallel_options[t][i])
                for i in range(len(node.input_signature)):
                    self.intermediate_shard_function[node.id][option_list][node.input_signature[i].id] = split_f[i]
                    splited_info = self.split_to_single_node(node.input_signature[i], split_f[i])
                    splited_result[node.id][option_list][node.input_signature[i].id] = splited_info
                split_f = [output_parallel_options[option_list[i]][0] for i in range(len(option_list))]
                splited_info = self.split_to_single_node(node.output_signature[0], split_f)
                splited_result[node.id][option_list][node.output_signature[0].id] = splited_info
        self.intermediate_result = splited_result

    def gen_option(self, num_options):
        def get_depth(tree,i=0):
            if i >= len(tree):
                return 0
            max_depth = 0
            for ni in [i*2+1, i*2+2]:
                depth = get_depth(tree,ni)
                if depth > max_depth:
                    max_depth = depth
            return max_depth + 1
        split_device_tree = self.cluster_info.get_split_cluster_tree()[1]
        num_layers = get_depth(split_device_tree)-1
        return [x for x in itertools.product(*tuple([list(range(num_options)) for _ in range(num_layers)]))]

    def single_node_cost(self, node_id: int, shard_options: List[int]):
        shard_funcs = self.intermediate_shard_function[node_id][shard_options]
        shard_results = self.intermediate_result[node_id][shard_options]
        
        communication_cost = 0.0

        send_recvs_count = {}
        for meta_var_id in shard_funcs.keys():
            if shard_results[meta_var_id][0].type != InternalType.Tensor:
                continue
            sending_groups = []
            sharded_result = shard_results[meta_var_id]
            for i in range(len(shard_funcs[meta_var_id])):
                if isinstance(shard_funcs[meta_var_id][i], ReplicatePass) or isinstance(shard_funcs[meta_var_id][i], ReducePass):
                    splited_cluster = self.cluster_info.get_split_cluster_tree()[1]
                    lemost = 1
                    rimost = 2
                    ii = i
                    while ii > 0:
                        ii -= 1
                        lemost = lemost * 2 + 1
                        rimost = rimost * 2 + 2
                    for tmp_idx in range(lemost, rimost+1, 2):
                        sending_groups.append([splited_cluster[tmp_idx], splited_cluster[tmp_idx + 1]])
            for sending_group in sending_groups:
                rank_group0 = sending_group[0]
                rank_group1 = sending_group[1]
                for rank0 in rank_group0:
                    for rank1 in rank_group1:
                        if (rank0, rank1) not in send_recvs_count:
                            intersected = sharded_result[rank0].get_intersection(sharded_result[rank1])[1]
                            if intersected is None:
                                continue
                            else:
                                send_recvs_count[(rank0, rank1)] = intersected.get_size()
                        else:
                            intersected = sharded_result[rank0].get_intersection(sharded_result[rank1])[1]
                            if intersected is None:
                                continue
                            else:
                                send_recvs_count[(rank0, rank1)] += intersected.get_size()
        
        for k,v in send_recvs_count.items():
            bandwidth = self.cluster_info.comm_bandwidth[k[0]][k[1]]
            latency = self.cluster_info.comm_latencies[k[0]][k[1]]
            communication_cost = communication_cost + v/1024/bandwidth + latency # us
        return communication_cost

    def append_node_cost(self, node_id: int, shard_options: List[int], past_info: Dict[int, List[SplitPass]], past_result: Dict[int, Dict[int, InternalMetaVariable]]):
        def get_ith_layer(layer_idx):
            splited_cluster = self.cluster_info.get_split_cluster_tree()[1]
            lemost = 1
            rimost = 2
            ii = layer_idx
            while ii > 0:
                ii -= 1
                lemost = lemost * 2 + 1
                rimost = rimost * 2 + 2
            return splited_cluster[lemost: rimost + 1]

        def get_replicas(_info):
            replicas_group = {0:[self.cluster_info.splited_cluster[0]]}
            assembling_replicas_group = {}
            for i in range(len(_info)):
                assembling_replicas_group = {}
                if isinstance(_info[i], ReplicatePass) or isinstance(_info[i], ReducePass):
                    layer = get_ith_layer(i)
                    num_current = len(replicas_group)
                    for i in range(num_current * 2):
                        assembling_replicas_group[i] = []
                    for i in range(0, len(layer)-1, 2):
                        sub_rank0: set[int]
                        sub_rank0 = layer[i]
                        sub_rank1 = layer[i+1]
                        for k in replicas_group.keys():
                            for t in range(len(replicas_group[k])):
                                if sub_rank0.issubset(replicas_group[k][t]):
                                    assembling_replicas_group[k].append(sub_rank0)
                                    assembling_replicas_group[k+num_current].append(sub_rank1)
                else:
                    layer = get_ith_layer(i)
                    num_current = len(replicas_group)
                    for i in range(num_current):
                        assembling_replicas_group[i] = []
                    for i in range(0, len(layer)-1, 2):
                        sub_rank0: set[int]
                        sub_rank0 = layer[i]
                        sub_rank1 = layer[i+1]
                        for k in replicas_group.keys():
                            for t in range(len(replicas_group[k])):
                                if sub_rank0.issubset(replicas_group[k][t]):
                                    assembling_replicas_group[k].append(sub_rank0)
                                    assembling_replicas_group[k].append(sub_rank1)
                replicas_group = deepcopy(assembling_replicas_group)
            return replicas_group
        
        shard_funcs = self.intermediate_shard_function[node_id][shard_options]
        shard_results = self.intermediate_result[node_id][shard_options]
        
        communication_cost = 0.0
        send_recvs_count = {}
        
        for meta_var_id in shard_funcs.keys():
            if meta_var_id not in past_info:
                continue
            _past_info = past_info[meta_var_id]
            _shard_funcs = shard_funcs[meta_var_id]
            _past_result = past_result[meta_var_id]
            _shard_result = shard_results[meta_var_id]

            replica_past = get_replicas(_past_info)
            replica_curr = get_replicas(_shard_funcs)

            for k in replica_past.keys():
                ranks = []
                for s in replica_past[k]:
                    ranks.extend(s)
                replica_past[k] = ranks             

            for k in replica_curr.keys():
                ranks = []
                for s in replica_curr[k]:
                    ranks.extend(s)
                replica_curr[k] = ranks

            if len(replica_past) >= len(replica_curr):
                gap_past = len(replica_past) // len(replica_curr)
                for i in range(0, len(replica_past), gap_past):
                    past_idx = i
                    curr_idx = i // gap_past

                    for rank0 in replica_past[past_idx]:
                        for rank1 in replica_curr[curr_idx]:
                            if rank0 == rank1:
                                continue
                            intersected = _past_result[rank0].get_intersection(_shard_result[rank1])[1]
                            if intersected is None:
                                continue
                            if (rank0, rank1) not in send_recvs_count:
                                send_recvs_count[(rank0, rank1)] = intersected.get_size()
                            else:
                                send_recvs_count[(rank0, rank1)] += intersected.get_size()
            else:
                gap_curr = len(replica_curr) // len(replica_past)
                for i in range(0, len(replica_curr), gap_curr):
                    past_idx = i // gap_curr
                    curr_idx = i
                    for rank0 in replica_past[past_idx]:
                        for rank1 in replica_curr[curr_idx]:
                            if rank0 == rank1:
                                continue
                            intersected = _past_result[rank0].get_intersection(_shard_result[rank1])[1]
                            if intersected is None:
                                continue
                            if (rank0, rank1) not in send_recvs_count:
                                send_recvs_count[(rank0, rank1)] = intersected.get_size()
                            else:
                                send_recvs_count[(rank0, rank1)] += intersected.get_size()

        for k,v in send_recvs_count.items():
            bandwidth = self.cluster_info.comm_bandwidth[k[0]][k[1]]
            latency = self.cluster_info.comm_latencies[k[0]][k[1]]
            communication_cost = communication_cost + v/1024/bandwidth + latency # us
        return communication_cost
