from ..codegen.internal_meta_ir.internal_meta_ir import InternalDtype, InternalMetaGraph, InternalMetaOperator, InternalMetaVariable, InternalType
from ..api.meta_ir.cluster_info import ClusterInfo
from ..optimization.cost_model.cost_model import CostModel
from ..codegen.internal_meta_ir.internal_meta_ir import ReducePass, ReplicatePass, SplitPass
from collections import deque
from enum import Enum
from typing import Dict, List
from copy import deepcopy

class TaskType(Enum):
    CommTask = 0
    CompTask = 1

class Task:
    def __init__(self, duration: float, sim_dev_id: int, task_type: TaskType):
        self.duration = duration
        self.sim_dev_id  = sim_dev_id
        self.task_type = task_type
        self.dependecies_cnt = 0
        self.release_dependencies = []
        self.dependecies_ids = []
        self.ready_time = 0.0
        self.id = None

    def add_up_task(self, task_id):
        self.dependecies_cnt += 1
        self.dependecies_ids.append(task_id)

    def add_down_task(self, task_id):
        self.release_dependencies.append(task_id)

    def __str__(self):
        return f'task({self.id}): duration {self.duration}, up({self.dependecies_ids}), down({self.release_dependencies})'
    
    def __repr__(self):
        return self.__str__()

class SimulateCompDevice:
    def __init__(self, id: int, computability: float):
        self.tasks = []
        self.ready_time = 0.0
        self.id = id
        self.vars = {}
        self.computability = computability

class SimulatedCommDevice:
    def __init__(self, id: int, bw: float, latency: float):
        self.tasks = []
        self.ready_time = 0.0
        self.id = id
        self.bw = bw
        self.latency = latency

class Simulator:
    def __init__(self, cluster_info: ClusterInfo, internal_meta_graph: InternalMetaGraph, shard_options: Dict[int, List[int]], cost_model: CostModel):
        self.simulate_time = 0.0
        self.cluster_info = cluster_info
        self.internal_meta_graph = internal_meta_graph
        self.shard_option = shard_options
        self.cost_model = cost_model
        self.call_cost = 10.0

        self.comp_devices = []
        self.comm_devices = []
        self.comm_device_match_rule = {}

        for i in range(len(self.cluster_info.comp_capability)):
            self.comp_devices.append(SimulateCompDevice(len(self.comp_devices), self.cluster_info.comp_capability[len(self.comp_devices)]))
            for j in range(len(self.cluster_info.comm_bandwidth)):
                if i == j:
                    continue
                comm_device_id = len(self.comm_devices)
                self.comm_devices.append(SimulatedCommDevice(comm_device_id, self.cluster_info.comm_bandwidth[i][j], self.cluster_info.comm_latencies[i][j]))
                self.comm_device_match_rule[(i,j)] = comm_device_id

        self.past_results = {}
        self.past_funcs   = {}        
        self.task_list = []

    def get_resharding_tasks(self, meta_var: InternalMetaVariable, cur_shard_f: List[SplitPass]):
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
        
        send_recv_tasks = {}
        ret = []
        var_id = meta_var.id
        if var_id not in self.past_funcs:
            return ret
        
        past_info = self.past_funcs[var_id]
        past_result = self.past_results[var_id]
        cur_funcs = cur_shard_f
        cur_result = self.cost_model.split_to_single_node(meta_var, cur_funcs)

        replica_past = get_replicas(past_info)
        replica_curr = get_replicas(cur_funcs)

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
                        intersected = past_result[rank0].get_intersection(cur_result[rank1])[1]
                        if intersected is None:
                            continue
                        if (rank0, rank1) not in send_recv_tasks:
                            send_recv_tasks[(rank0, rank1)] = [intersected.get_size()]
                        else:
                            send_recv_tasks[(rank0, rank1)].append(intersected.get_size())
        else:
            gap_curr = len(replica_curr) // len(replica_past)
            for i in range(0, len(replica_curr), gap_curr):
                past_idx = i // gap_curr
                curr_idx = i
                for rank0 in replica_past[past_idx]:
                    for rank1 in replica_curr[curr_idx]:
                        if rank0 == rank1:
                            continue
                        intersected = past_result[rank0].get_intersection(cur_result[rank1])[1]
                        if intersected is None:
                            continue
                        if (rank0, rank1) not in send_recv_tasks:
                            send_recv_tasks[(rank0, rank1)] = [intersected.get_size()]
                        else:
                            send_recv_tasks[(rank0, rank1)].append(intersected.get_size())
        
        for k in send_recv_tasks.keys():
            comm_dev_id = self.comm_device_match_rule[k]
            comm_dev = self.comm_devices[comm_dev_id]
            bw = comm_dev.bw
            latency = comm_dev.latency
            for i in range(len(send_recv_tasks[k])):
                duration = self.call_cost + send_recv_tasks[k][i]/1024/bw + latency
                task = Task(duration, comm_dev_id, TaskType.CommTask)
                task.id = len(self.task_list)
                self.task_list.append(task)
                ret.append(task.id)
        return ret

    def prepare(self):
        comp_tasks = {}
        comm_tasks = {} # outputs_only
        for node_id in self.internal_meta_graph.nodes:
            internal_node = self.internal_meta_graph.nodes[node_id]
            shards = internal_node.get_parallel_options()
            inputs_annotation = shards['input_options']
            output_annotation = shards['output_options']

            option = self.shard_option[node_id]

            preforward_task_ids = {}

            shard_infos = []
            for idx, input_meta_var in enumerate(internal_node.input_signature):
                var_id = input_meta_var.id
                shards = [inputs_annotation[i][idx] for i in range(len(inputs_annotation))]
                shard_f = [shards[i] for i in option]
                shard_info = self.cost_model.split_to_single_node(input_meta_var, shard_f)
                shard_infos.append(shard_info)
                this_preforward_task_ids = self.get_resharding_tasks(input_meta_var, shard_f)
                for task_id in this_preforward_task_ids:
                    comm_task: Task
                    comm_task = self.task_list[task_id]
                    comm_dev_id = comm_task.sim_dev_id
                    src_comp_dev_id = comm_dev_id // (len(self.comp_devices) - 1)
                    dst_comp_dev_id = comm_dev_id % (len(self.comp_devices) - 1)
                    if dst_comp_dev_id >= src_comp_dev_id:
                        dst_comp_dev_id += 1
                    correspond_comp_task = comp_tasks[input_meta_var.gen_node_id][src_comp_dev_id]
                    self.task_list[task_id].add_up_task(correspond_comp_task)
                    self.task_list[correspond_comp_task].add_down_task(task_id)
                    preforward_task_ids[dst_comp_dev_id] = task_id
            
            for rank in range(len(shard_infos[0])):
                inputs = [shard_infos[i][rank] for i in range(len(shard_infos))]
                # self.internal_meta_graph.nodes[node_id].generate_op()
                duration = self.call_cost
                comp_task = Task(duration, rank, TaskType.CompTask)
                comp_task.id = len(self.task_list)
                if node_id not in comp_tasks:
                    comp_tasks[node_id] = {}
                comp_tasks[node_id][rank] = comp_task.id
                if rank in preforward_task_ids:
                    comp_task.add_up_task(preforward_task_ids[rank])
                    self.task_list[preforward_task_ids[rank]].add_down_task(comp_task.id)
                if node_id - 1 in comp_tasks:
                    comp_task.add_up_task(comp_tasks[node_id-1][rank])
                    self.task_list[node_id-1].add_down_task(comp_task.id)
                comp_tasks[node_id][rank] = comp_task.id
                self.task_list.append(comp_task)

            output_shard_funcs = [output_annotation[i] for i in option]
            self.past_funcs[internal_node.output_signature[0].id] = output_shard_funcs
            self.past_results[internal_node.output_signature[0].id] = self.cost_model.split_to_single_node(internal_node.output_signature[0], output_shard_funcs)


    def simulate(self):
        def get_device(task: Task):
            if task.task_type == TaskType.CommTask:
                dev = self.comm_devices[task.sim_dev_id]
            else:
                dev = self.comp_devices[task.sim_dev_id]
            return dev

        sim_time = 0.0
        # dependencies = {node.name:1 for node in self.internal_meta_graph.nodes}
        tasks = deepcopy(self.task_list)
        ready_tasks = []
        for i in range(len(tasks)):
            if tasks[i].dependecies_cnt == 0:
                ready_tasks.append(tasks[i])
                tasks[i].dependecies_cnt = -1
        while len(ready_tasks) > 0:
            ready_tasks = sorted(ready_tasks, key=lambda x: max(get_device(x).ready_time, x.ready_time), reverse=True)
            task = ready_tasks.pop()
            task: Task
            dev = get_device(task)
            dev.ready_time = max(dev.ready_time, task.ready_time) + task.duration
            sim_time = max(sim_time, dev.ready_time)
            for follower_id in task.release_dependencies:
                tasks[follower_id].dependecies_cnt -= 1
                tasks[follower_id].ready_time = dev.ready_time
            
            for i in range(len(tasks)):
                if tasks[i].dependecies_cnt == 0:
                    ready_tasks.append(tasks[i])
                    tasks[i].dependecies_cnt = -1
        return sim_time
