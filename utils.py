import os
import random
import numpy as np
import pandas as pd
import networkx as nx


# Check the existence of directory(file) path, if not, create one
def check_and_make_path(file_path):
    if file_path == '':
        return
    if not os.path.exists(file_path):
        os.makedirs(file_path)


# Get networkx graph object from file path.
def get_nx_graph(file_path, nodes_set_path, sep='\t'):
    # df = pd.read_csv(file_path, sep=sep)
    # graph = nx.from_pandas_edgelist(df, "from_id", "to_id", edge_attr='time', create_using=nx.MultiGraph)
    # graph.remove_edges_from(nx.selfloop_edges(graph))
    
    # 读取全图所有节点，将节点映射为编号
    nodes_set = pd.read_csv(nodes_set_path, names=['node'])
    node2id = {node: i for i, node in enumerate(nodes_set['node'])}

    # 加载整张图
    df = pd.read_csv(file_path, sep=sep)

    def trans_id(nid):
        return node2id[nid]

    df[['from_id', 'to_id']] = df[['from_id', 'to_id']].applymap(trans_id)

    graph = nx.from_pandas_edgelist(df, "from_id", "to_id", edge_attr='time', create_using=nx.MultiGraph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def add_attr(graph, min_t=0, max_t=float('inf')):
    # 节点-影响因子-计算
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 0:
            act = 0
        else:
            deg = len(neighbors)
            avg_nei_deg = sum(len(list(graph.neighbors(neighbor))) for neighbor in neighbors) / len(neighbors)
            act = deg + avg_nei_deg
        # 为节点添加属性
        graph.nodes[node]['act'] = act
    
    # 边-结构链接强度-计算
    for u, v, key in graph.edges(keys=True):
        L_uv = get_links(graph, u, v, min_t, max_t)
        L_sum_u = get_all_links(graph, u, min_t, max_t)
        L_sum_v = get_all_links(graph, v, min_t, max_t)
        
        CN_uv = len(list(nx.common_neighbors(graph, u, v)))
        D_u = len(list(graph.neighbors(u)))
        D_v = len(list(graph.neighbors(v)))
        
        sci_uv = (L_uv / L_sum_u) * (L_uv / L_sum_v) + (CN_uv / D_u) * (CN_uv / D_v)
        graph[u][v][key]['sci'] = sci_uv
    return graph
            
def get_links(graph, u, v, min_t=0, max_t=float('inf')):
    count = 0
    if graph.has_edge(u, v):
        for key in graph[u][v]:  # 遍历节点 u 和 v 之间的所有多重边
            edge_data = graph[u][v][key]
            if 'time' in edge_data:
                timestamp = edge_data['time']
                if min_t <= timestamp <= max_t:
                    count += 1
    return count

def get_all_links(graph, node, min_t=0, max_t=float('inf')):
    count = 0
    for neighbor in graph.neighbors(node):
        for key in graph[node][neighbor]:  # 遍历节点 node 和其邻居之间的所有多重边
            edge_data = graph[node][neighbor][key]
            if 'time' in edge_data:
                timestamp = edge_data['time']
                if min_t <= timestamp <= max_t:
                    count += 1
    return count

def negative_sampling(graph, size):
    negative_edges = []
    nodes = list(graph.nodes())

    while len(negative_edges) < size:
        # 随机选择两个节点
        u, v = random.choice(nodes), random.choice(nodes)

        # 检查是否存在边，若不存在，则添加到负样本
        if u != v and not graph.has_edge(u, v):
            negative_edges.append((u, v))

    return negative_edges


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round