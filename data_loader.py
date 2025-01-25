import random
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from tqdm import tqdm
from utils import add_attr


class DataLoader:
    def __init__(self, graph):
        self.graph = graph
        self.full_node_list = list(graph.nodes())    # 获取原始图的节点集合
        self.node_num = len(self.full_node_list)
    
    def data_split(self, train_ratio = 0.5, val_ratio = 0.3):
        # 划分数据集
        # edges_with_time = [(u, v, self.graph[u][v]['time']) for u, v in self.graph.edges()]
        edges_with_time = [(u, v, d['time']) for u, v, d in self.graph.edges(data=True)]

        sorted_edges = sorted(edges_with_time, key=lambda x: x[2])
        total_edges = len(sorted_edges)
        train_end = int(total_edges * train_ratio)
        val_end = int(total_edges * (train_ratio + val_ratio))
        train_edges = sorted_edges[:train_end]
        val_edges = sorted_edges[train_end:val_end]
        test_edges = sorted_edges[val_end:]
        
        # 创建新的图并添加边
        train_graph = nx.MultiGraph()
        val_graph = nx.MultiGraph()
        test_graph = nx.MultiGraph()

        train_graph.add_weighted_edges_from(train_edges, weight="time")
        # 获取 train_graph 中出现的节点
        train_nodes = list(train_graph.nodes())
        
        # 仅保留 val_graph 中在 train_graph 中出现的节点的边
        val_edges_filtered = [(u, v, time) for u, v, time in val_edges if u in train_nodes and v in train_nodes]
        val_graph.add_weighted_edges_from(val_edges_filtered, weight="time")
        
        # 仅保留 test_graph 中在 train_graph 中出现的节点的边
        test_edges_filtered = [(u, v, time) for u, v, time in test_edges if u in train_nodes and v in train_nodes]
        test_graph.add_weighted_edges_from(test_edges_filtered, weight="time")

        # 保持节点集合不变
        train_graph.add_nodes_from(train_nodes)
        val_graph.add_nodes_from(train_nodes)
        test_graph.add_nodes_from(train_nodes)
        
        final_train_graph = add_attr(train_graph)
        return final_train_graph, val_graph, test_graph
    
    
    def filter(self, train_graph, test_graph):
        # 随机挑选 10% 的节点进行 mask
        train_nodes_num = train_graph.number_of_nodes()
        sample_num = int(0.1 * train_nodes_num)
        sampled_nodes = random.sample(train_graph.nodes(), sample_num)
        train_graph.remove_nodes_from(sampled_nodes)
        
        # 仅保留 test_graph 中在 train_graph 中出现的节点的边 transductive learning
        test_edges_with_time = [(u, v, d['time']) for u, v, d in test_graph.edges(data=True)]
        test_edges_filtered = [(u, v, time) for u, v, time in test_edges_with_time if u in sampled_nodes or v in sampled_nodes]
        test_graph = nx.MultiGraph()
        test_graph.add_weighted_edges_from(test_edges_filtered, weight="time")
        
        
    def extract_node_interaction_sequences(self, graph):
        node_interaction_seq = {}
        
        for node in graph.nodes():
            neighbors_with_time = []
            
            for neighbor, data in graph[node].items():
                for edge_attr in data.values():
                    neighbors_with_time.append((neighbor, edge_attr['time']))

            # 按照时间倒序排列
            neighbors_with_time.sort(key=lambda x: x[1], reverse=True)
            
            unique_neighbors_with_time = []
            for i in range(len(neighbors_with_time)):
                if i == 0 or neighbors_with_time[i][0] != neighbors_with_time[i-1][0]:
                    unique_neighbors_with_time.append((neighbors_with_time[i]))
                    
            node_interaction_seq[node] = unique_neighbors_with_time
            # print(f"Node {node} neighbors: {neighbors_with_time}")
            
        return node_interaction_seq


    def extract_walks(self, graph, walk_len, num_walks, p=1, q=1, workers=1, quiet=False):
        FIRST_TRAVEL_KEY = 'first_travel_key'
        PROBABILITIES_KEY = 'probabilities'
        NEIGHBORS_KEY = 'neighbors'
        NEIGHBORS_TIME_KEY = 'neighbors_time'
        WEIGHT_KEY = 'sci'
        # Precomputes transition probabilities for each node.
        d_graph, max_time = self._precompute_probabilities(graph, p, q, WEIGHT_KEY, NEIGHBORS_KEY, NEIGHBORS_TIME_KEY, 
                                                           PROBABILITIES_KEY, FIRST_TRAVEL_KEY, quiet)
        
        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(num_walks), workers)
        walks = Parallel(n_jobs=workers)(delayed(self._parallel_generate_walks)(d_graph, 
                                                                                walk_len, 
                                                                                len(num_walks), 
                                                                                max_time, 
                                                                                idx, 
                                                                                NEIGHBORS_KEY,
                                                                                NEIGHBORS_TIME_KEY, 
                                                                                PROBABILITIES_KEY, 
                                                                                FIRST_TRAVEL_KEY, 
                                                                                quiet=False, 
                                                                                use_linear=True, 
                                                                                half_life=1) for idx, num_walks 
                                         in enumerate(num_walks_lists, 1))
        walks_list = []
        walks_dict = defaultdict(list)
        # 遍历列表中的每个字典
        for d in walks:
            # 遍历当前字典的键值对
            for key, value in d.items():
                # 将键值对添加到合并后的字典中
                walks_dict[key].extend(value)
                walks_list.append([node_time[0] for val in value for node_time in val]) # 只取节点，不带时间
        # walks = self._generate_walks(d_graph, max_time, walk_len, num_walks, NEIGHBORS_KEY,NEIGHBORS_TIME_KEY, PROBABILITIES_KEY, 
        #                              FIRST_TRAVEL_KEY, use_linear=True, half_life=1)
        # my_walks = [item for sublist in walks.values() for item in sublist]
        return walks_dict, walks_list
    
    def extract_context(self, graph, walk_len, num_walks, p=1, q=1, workers=1, quiet=False):
        
        FIRST_TRAVEL_KEY = 'first_travel_key'
        PROBABILITIES_KEY = 'probabilities'
        NEIGHBORS_KEY = 'neighbors'
        NEIGHBORS_TIME_KEY = 'neighbors_time'
        WEIGHT_KEY = 'sci'
        # Precomputes transition probabilities for each node.
        d_graph, max_time = self._precompute_probabilities(graph, p, q, WEIGHT_KEY, NEIGHBORS_KEY, NEIGHBORS_TIME_KEY, 
                                                           PROBABILITIES_KEY, FIRST_TRAVEL_KEY, quiet)
        
        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(num_walks), workers)
        context = Parallel(n_jobs=workers)(delayed(self._parallel_generate_context)(d_graph, 
                                                                                walk_len, 
                                                                                len(num_walks), 
                                                                                max_time, 
                                                                                idx, 
                                                                                NEIGHBORS_KEY,
                                                                                NEIGHBORS_TIME_KEY, 
                                                                                PROBABILITIES_KEY, 
                                                                                FIRST_TRAVEL_KEY, 
                                                                                quiet=False, 
                                                                                use_linear=True, 
                                                                                half_life=1) for idx, num_walks 
                                         in enumerate(num_walks_lists, 1))
        context_list = []
        context_dict = defaultdict(list)
        # 遍历列表中的每个字典
        for d in context:
            # 遍历当前字典的键值对
            for key, value in d.items():
                # 将键值对添加到合并后的字典中
                context_dict[key].extend(value)
                context_list.append([node_time[0] for val in value for node_time in val]) # 只取节点，不带时间
        # walks = self._generate_walks(d_graph, max_time, walk_len, num_walks, NEIGHBORS_KEY,NEIGHBORS_TIME_KEY, PROBABILITIES_KEY, 
        #                              FIRST_TRAVEL_KEY, use_linear=True, half_life=1)
        # my_walks = [item for sublist in walks.values() for item in sublist]
        return context_dict, context_list
    
    def _precompute_probabilities(self, graph, p=1, q=1, WEIGHT_KEY='weight', NEIGHBORS_KEY=None, NEIGHBORS_TIME_KEY=None, 
                                  PROBABILITIES_KEY=None, FIRST_TRAVEL_KEY=None, quiet=False):
        """
        Precomputes transition probabilities for each node.
        """
        d_graph = defaultdict(dict)
        
        # 添加最大时间戳（认为是当前时间）
        max_time = max([attr['time'] for _, _, attr in graph.edges(data=True)])
         
        first_travel_done = set()
        
        nodes_generator = graph.nodes() if quiet else tqdm(graph.nodes(), desc='Computing transition probabilities')
        
        act = nx.get_node_attributes(graph, 'act')
        for source in nodes_generator:

            # Init probabilities dict for first travel
            if PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][PROBABILITIES_KEY] = dict()
            if FIRST_TRAVEL_KEY not in d_graph[source]:
                d_graph[source][FIRST_TRAVEL_KEY] = dict()
            if NEIGHBORS_KEY not in d_graph[source]:
                d_graph[source][NEIGHBORS_KEY] = dict()
            if NEIGHBORS_TIME_KEY not in d_graph[source]:
                d_graph[source][NEIGHBORS_TIME_KEY] = dict()

            for current_node in graph.neighbors(source):
                # Init probabilities dict
                if PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in graph.neighbors(current_node):
                    
                    edge_weight = graph[current_node][destination][0].get(WEIGHT_KEY, 1)
                    
                    if destination == source:  # Backwards probability
                        ss_weight = edge_weight * act[destination] * 1 / p
                    elif destination in graph[source]:  # If the neighbor is connected to the source
                        ss_weight = edge_weight * act[destination]
                    else:
                        ss_weight = edge_weight * act[destination] * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(edge_weight)
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][PROBABILITIES_KEY][source] = unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][FIRST_TRAVEL_KEY] = unnormalized_weights / unnormalized_weights.sum()
                    first_travel_done.add(current_node)

                # Save neighbors
                d_graph[current_node][NEIGHBORS_KEY] = d_neighbors

                # Save neighbors time_edges
                neighbor2times = {}
                for neighbor in d_neighbors:
                    neighbor2times[neighbor] = []
                    if 'time' in graph[current_node][neighbor]:
                        neighbor2times[neighbor].append(graph[current_node][neighbor]['time'])
                    else:
                        for att in list(graph[current_node][neighbor].values()):
                            if 'time' not in att:
                                raise ('no time attribute')
                            neighbor2times[neighbor].append(att['time'])
                d_graph[current_node][NEIGHBORS_TIME_KEY] = neighbor2times

        return d_graph, max_time
    
    
    def _parallel_generate_walks(self, d_graph, walk_length, num_walks, max_time, cpu_num, neighbors_key=None, neighbors_time_key=None,
                            probabilities_key=None, first_travel_key=None, quiet=False, use_linear=True, half_life=1):
        
        walks = defaultdict(list)
        
        if not quiet:
            pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

        for n_walk in range(num_walks):

            # Update progress bar
            if not quiet:
                pbar.update(1)

            # Shuffle the nodes
            shuffled_nodes = list(d_graph.keys())
            random.shuffle(shuffled_nodes)

            # Start a random walk from every node
            for source in shuffled_nodes:

                # Start walk
                walk = [(source, 0)]
                last_time = 0

                # Perform walk
                while len(walk) < walk_length:
                    # For the first step
                    try:
                        if len(walk) == 1:
                            probabilities = d_graph[walk[-1][0]][first_travel_key]
                        else:
                            probabilities = d_graph[walk[-1][0]][probabilities_key][walk[-2][0]]
                        #probabilities = [1] * len(d_graph[walk[-1]].get(neighbors_key, []))
                    except KeyError:
                        print(walk[-1], "\t", d_graph[walk[-1]])
                        raise KeyError
                    walk_options = []
                    for neighbor, p in zip(d_graph[walk[-1][0]].get(neighbors_key, []), probabilities):
                        times = d_graph[walk[-1][0]][neighbors_time_key][neighbor]
                        walk_options += [(neighbor, p, time) for time in times if time > last_time]

                    # Skip dead end nodes
                    if len(walk_options) == 0:
                        break

                    if len(walk) == 1:
                        last_time = max(map(lambda x: x[2], walk_options))

                    if use_linear:
                        time_probabilities = np.array(np.argsort(np.argsort(list(map(lambda x: x[2], walk_options)))[::-1])+1, dtype=float)
                        final_probabilities = time_probabilities * np.array(list(map(lambda x: x[1], walk_options)))
                        final_probabilities /= sum(final_probabilities)
                    else:
                        final_probabilities = np.array(list(map(lambda x: np.exp(x[1]*(last_time - x[2])/half_life), walk_options)))
                        final_probabilities /= sum(final_probabilities)

                    walk_to_idx = np.random.choice(range(len(walk_options)), size=1, p=final_probabilities)[0]
                    walk_to = walk_options[walk_to_idx]

                    last_time = walk_to[2]
                    walk.append((walk_to[0], last_time))

                # walk = list(map(str, walk))  # Convert all to strings
                walks[source].append(walk)

        if not quiet:
            pbar.close()

        return walks
    
    
    def _parallel_generate_context(self, d_graph, walk_length, num_walks, max_time, cpu_num, neighbors_key=None, neighbors_time_key=None,
                            probabilities_key=None, first_travel_key=None, quiet=False, use_linear=True, half_life=1):
        
        walks = defaultdict(list)
        half_len = (walk_length - 1) // 2 + 1
        
        if not quiet:
            pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

        for n_walk in range(num_walks):

            # Update progress bar
            if not quiet:
                pbar.update(1)

            # Shuffle the nodes
            shuffled_nodes = list(d_graph.keys())
            random.shuffle(shuffled_nodes)

            # Start a random walk from every node
            for source in shuffled_nodes:

                # Start walk
                walk = [(source, 0)]
                last_time = 0
                # Perform walk
                while len(walk) < half_len:
                    # For the first step
                    try:
                        if len(walk) == 1:
                            probabilities = d_graph[walk[-1][0]][first_travel_key]
                        else:
                            probabilities = d_graph[walk[-1][0]][probabilities_key][walk[-2][0]]
                        #probabilities = [1] * len(d_graph[walk[-1]].get(neighbors_key, []))
                    except KeyError:
                        print(walk[-1], "\t", d_graph[walk[-1]])
                        raise KeyError
                    walk_options = []
                    for neighbor, p in zip(d_graph[walk[-1][0]].get(neighbors_key, []), probabilities):
                        times = d_graph[walk[-1][0]][neighbors_time_key][neighbor]
                        walk_options += [(neighbor, p, time) for time in times if time > last_time]

                    # Skip dead end nodes
                    if len(walk_options) == 0:
                        break

                    if len(walk) == 1:
                        last_time = max(map(lambda x: x[2], walk_options))

                    if use_linear:
                        time_probabilities = np.array(np.argsort(np.argsort(list(map(lambda x: x[2], walk_options)))[::-1])+1, dtype=float)
                        final_probabilities = time_probabilities * np.array(list(map(lambda x: x[1], walk_options)))
                        final_probabilities /= sum(final_probabilities)
                    else:
                        final_probabilities = np.array(list(map(lambda x: np.exp(x[1]*(last_time - x[2])/half_life), walk_options)))
                        final_probabilities /= sum(final_probabilities)

                    walk_to_idx = np.random.choice(range(len(walk_options)), size=1, p=final_probabilities)[0]
                    walk_to = walk_options[walk_to_idx]

                    last_time = walk_to[2]
                    walk.append((walk_to[0], last_time))
                # Start walk
                last_time = walk[1][1]
                walk.pop(0)
                walk.insert(0, (source, last_time))

                while len(walk) < walk_length:
                # For the first step
                    try:
                        if len(walk) == 1:
                            probabilities = d_graph[walk[0][0]][first_travel_key]
                        else:
                            probabilities = d_graph[walk[0][0]][probabilities_key][walk[1][0]]
                        #probabilities = [1] * len(d_graph[walk[-1]].get(neighbors_key, []))
                    except KeyError:
                        print(walk[0], "\t", d_graph[walk[0]])
                        raise KeyError
                    walk_options = []
                    for neighbor, p in zip(d_graph[walk[0][0]].get(neighbors_key, []), probabilities):
                        times = d_graph[walk[0][0]][neighbors_time_key][neighbor]
                        walk_options += [(neighbor, p, time) for time in times if time < last_time]

                    # Skip dead end nodes
                    if len(walk_options) == 0:
                        break

                    if len(walk) == 1:
                        last_time = max(map(lambda x: x[2], walk_options))

                    if use_linear:
                        time_probabilities = np.array(np.argsort(np.argsort(list(map(lambda x: x[2], walk_options)))[::-1])+1, dtype=float)
                        final_probabilities = time_probabilities * np.array(list(map(lambda x: x[1], walk_options)))
                        final_probabilities /= sum(final_probabilities)
                    else:
                        final_probabilities = np.array(list(map(lambda x: np.exp(x[1]*(last_time - x[2])/half_life), walk_options)))
                        final_probabilities /= sum(final_probabilities)

                    walk_to_idx = np.random.choice(range(len(walk_options)), size=1, p=final_probabilities)[0]
                    walk_to = walk_options[walk_to_idx]

                    last_time = walk_to[2]
                    walk.insert(0, (walk_to[0], last_time))

                # walk = list(map(str, walk))  # Convert all to strings
                # padding
                # while len(walk) < walk_length:
                #     walk.insert(0, (0, 0))
                walks[source].append(walk)
                
        if not quiet:
            pbar.close()

        return walks