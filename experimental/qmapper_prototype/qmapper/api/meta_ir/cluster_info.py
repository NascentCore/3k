from typing import List
from collections import deque
import random
import networkx as nx
import logging
import numpy as np
import math

class SplitedClusterTree:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class ClusterInfo:
    def __init__(self, global_group_size: int, comp_capability: List[float],
                 comm_latencies: List[List[float]], comm_bandwidth: List[List[float]]):
        self.global_group_size = global_group_size
        self.comp_capability = comp_capability # TFlops
        self.comm_latencies = comm_latencies   # us
        self.comm_bandwidth = comm_bandwidth   # KB/us (GB/s)
        self.splited_cluster = []
        self.G = None
        self.splited_nodes = None


    def split_metric(self, rank_i, rank_j):
        return self.comm_bandwidth[rank_i][rank_j]
    
    def get_min_bandwidth_between_group(self, g0, g1):
        min_bw = 99999999
        for i in g0:
            for j in g1:
                min_bw = min(min_bw, self.comm_bandwidth[i][j])
                min_bw = min(min_bw, self.comm_bandwidth[j][i])
        return min_bw

    def get_sum_compcapacity(self, g0):
        return sum([self.comp_capability[i] for i in g0 if self.comp_capability[i] is not None])
    
    def get_max_layer(self):
        if self.G is None:
            self.get_split_cluster_tree()
        return int(math.log2(len(self.splited_cluster)))

    def get_split_cluster_tree(self):
        def digraph(G: nx.Graph):
            if len(G.nodes) == 1:
                return
            splited_cluster = nx.community.kernighan_lin_bisection(G=G, weight = 'weight', max_iter=int(np.sqrt(len(G.nodes))))
            return splited_cluster
        G = nx.Graph()
        for i in range(len(self.comm_bandwidth)):
            for j in range(i+1,len(self.comm_bandwidth)):
                G.add_edge(i, j, weight=self.split_metric(i, j)+self.split_metric(j,i))
        
        splited_graph = []
        splited_graph.append(set([node for node in G.nodes]))
        begin_index = 0
        meaningful = 1
        while begin_index<meaningful:
            logging.error(f"spliting layer {int(math.log2(begin_index+1))}")
            if splited_graph[begin_index]!=None and len(splited_graph[begin_index]) > 1:
                subG = G.subgraph(splited_graph[begin_index])
                sub0, sub1 = digraph(subG)
                splited_graph.append(sub0)
                splited_graph.append(sub1)
                meaningful = len(splited_graph)
            else:
                splited_graph.append(None)
                splited_graph.append(None)
            begin_index += 1
        self.G = G
        self.splited_cluster = splited_graph
        return G,splited_graph[:meaningful]

    def get_ith_layer_informations(self, i):
        if self.G is None:
            self.get_split_cluster_tree()
        left,right = 0,0
        ret = []
        while i > 0:
            left = 2*left + 1
            right = 2*right + 2
            i -= 1
        splited_layer = self.splited_cluster[left:right + 1]
        for i in range(len(splited_layer)//2):
            idx0 = i*2
            idx1 = i*2+1
            sum_compcapacity0 = self.get_sum_compcapacity(splited_layer[idx0])
            sum_compcapacity1 = self.get_sum_compcapacity(splited_layer[idx1])
            min_bw = self.get_min_bandwidth_between_group(splited_layer[idx0], splited_layer[idx1])
            ret.append((sum_compcapacity0, sum_compcapacity1, min_bw))
        return ret
        

        

def get_mocked_cluster(global_size):
    comp_capability = [random.random()*10+5 for i in range(global_size)]
    comm_latencies = [[random.random()*300+100 for i in range(global_size)] for i in range(global_size)]
    comm_bandwidth = [[random.random()*300+100 for i in range(global_size)] for i in range(global_size)]

    return ClusterInfo(global_size, comp_capability, comm_latencies, comm_bandwidth)

if __name__ == '__main__':
    mocked_cluster = get_mocked_cluster(64)
    # print(mocked_cluster.comm_bandwidth)
    print(mocked_cluster.splited_cluster)
    print(mocked_cluster.get_split_cluster_tree())
    for i in range(mocked_cluster.get_max_layer()):
        print(mocked_cluster.get_ith_layer_informations(i))

