import os
import numpy as np
import torch
import torch.nn.functional as F


class TimeEncode(torch.nn.Module):
    def __init__(self, layers, enc_dim):
        super(TimeEncode, self).__init__()
        self.layers = layers
        self.time_dim = enc_dim
        
        self.node_time_map = {}
        
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
    
    def init_node_time_map(self, node_interaction_seq):
        # initialize internal data structure to index node positions
        if self.time_dim == 0:
            return
        for node in node_interaction_seq:
            self.node_time_map[node] = np.zeros(self.layers, dtype=np.float32)
        
        for node, neighbors_with_time in node_interaction_seq.items():
            for index, (_, timestamp) in enumerate(neighbors_with_time):
                if index >= self.layers:
                    break
                self.node_time_map[node][index] = timestamp


    def forward(self, times_tensor):
        # ts: [N, L]
        batch_size = times_tensor.size(0)
        seq_len = times_tensor.size(1)

        times_tensor = times_tensor.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = times_tensor * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)
    
class PositionEncoder(torch.nn.Module):
    def __init__(self, layers, enc_dim):
        super(PositionEncoder, self).__init__()
        self.layers = layers
        self.enc_dim = enc_dim
        
        self.node_pos_enc = {}
        
        # landing prob at [0, 1, ... num_layers]
        self.trainable_embedding = torch.nn.Sequential(torch.nn.Linear(in_features=self.layers, out_features=self.enc_dim), 
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim))
        
    
    def init_node_pos_map(self, node_interaction_seq, node_feat):
        # initialize internal data structure to index node positions
        if self.enc_dim == 0:
            return
        node_pos_map = {}
        for node in node_interaction_seq:
            node_pos_map[node] = np.zeros(self.layers, dtype=np.float32)
            # self.node_pos_map[node][0] = 1   # 将0号位置设为1
            self.node_pos_enc[node] = np.zeros((self.layers, self.layers), dtype=np.float32)
        for _, neighbors_with_time in node_interaction_seq.items():
            for index, (neighbor, _) in enumerate(neighbors_with_time):
                if index >= self.layers:
                    break
                node_pos_map[neighbor][index] += 1
                
        for node, neighbors_with_time in node_interaction_seq.items():
            for index, (neighbor, _) in enumerate(neighbors_with_time):
                if index >= self.layers:
                    break
                self.node_pos_enc[node][index] = node_pos_map[neighbor]
        
        # 去除匿名化策略
        # for node in node_interaction_seq:
        #     self.node_pos_enc[node] = np.zeros((self.layers, self.enc_dim), dtype=np.float32)
            
        # for node, neighbors_with_time in node_interaction_seq.items():
        #     for index, (neighbor, _) in enumerate(neighbors_with_time):
        #         if index >= self.layers:
        #             break
        #         self.node_pos_enc[node][index] = node_feat[neighbor]
                    
    def forward(self, pos_tensor):  
        return self.trainable_embedding(pos_tensor)
    
class CausalityTimeEncode(torch.nn.Module):
    def __init__(self, seq_num, layers, enc_dim):
        super(CausalityTimeEncode, self).__init__()
        self.seq_num = seq_num
        self.layers = layers
        self.time_dim = enc_dim
        
        self.node_time_map = {}
        
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
    
    def init_node_time_map(self, node_causality_seq_list):
        # initialize internal data structure to index node positions
        for node in node_causality_seq_list:
            self.node_time_map[node] =np.zeros((self.seq_num, self.layers), dtype=np.float32)
            
        for node, node_causality_seq in node_causality_seq_list.items():
            for i in range(min(len(node_causality_seq), self.seq_num)):
                max_timestamp = node_causality_seq[i][-1][1]
                for index, (_, timestamp) in enumerate(node_causality_seq[i]):
                    if index >= self.layers:
                        break
                    if index == 0:
                        self.node_time_map[node][i][index] = 0
                    else:
                        self.node_time_map[node][i][index] = timestamp - self.node_time_map[node][i][index - 1]


    def forward(self, times_tensor):
        # ts: [N, S, L]
        batch_size = times_tensor.size(0)
        times_tensor = times_tensor.view(batch_size, self.seq_num, self.layers, 1)  # [N, S, L, 1]
        map_ts = times_tensor * self.basis_freq.view(1, 1, 1, -1)  # [N, S, L, time_dim]
        map_ts += self.phase.view(1, 1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)
    

class CausalityPositionEncoder(torch.nn.Module):
    def __init__(self, seq_num, layers, enc_dim):
        super(CausalityPositionEncoder, self).__init__()
        self.seq_num = seq_num
        self.layers = layers
        self.enc_dim = enc_dim
        
        self.node_pos_enc = {}
        
        # landing prob at [0, 1, ... num_layers]
        self.trainable_embedding = torch.nn.Sequential(torch.nn.Linear(in_features=self.layers, out_features=self.enc_dim), 
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim))
        
    
    def init_node_pos_map(self, node_causality_seq_list, node_feat):
        # initialize internal data structure to index node positions
        node_pos_map = {}
        for node in node_causality_seq_list:
            node_pos_map[node] = np.zeros(self.layers, dtype=np.float32)
            # self.node_pos_map[node][0] = 1   # 将0号位置设为1
            self.node_pos_enc[node] = np.zeros((self.seq_num, self.layers, self.layers), dtype=np.float32)

        for _, node_causality_seq in node_causality_seq_list.items():
            for neighbors_with_time in node_causality_seq:
                for index, (neighbor, _) in enumerate(neighbors_with_time):
                    if index >= self.layers:
                        break
                    node_pos_map[neighbor][index] += 1
                
        for node, node_causality_seq in node_causality_seq_list.items():
            for i in range(min(len(node_causality_seq), self.seq_num)):
                for index, (neighbor, _) in enumerate(node_causality_seq[i]):
                    if index >= self.layers:
                        break
                    self.node_pos_enc[node][i][index] = node_pos_map[neighbor]

        # 去除匿名化策略
        # for node in node_causality_seq_list:
        #     self.node_pos_enc[node] = np.zeros((self.seq_num, self.layers, self.enc_dim), dtype=np.float32)
            
        # for node, node_causality_seq in node_causality_seq_list.items():
        #     for i in range(min(len(node_causality_seq), self.seq_num)):
        #         for index, (neighbor, _) in enumerate(node_causality_seq[i], 1):
        #             if index >= self.layers:
        #                 break
        #             self.node_pos_enc[node][i][index] = node_feat[neighbor]
                    
    def forward(self, pos_tensor):  
        return self.trainable_embedding(pos_tensor)
    
class FeatureEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='GRU', dropout_p=0.1):
        super(FeatureEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_one_direction = self.hidden_dim // 2
        self.rnn_type = rnn_type
        self.model_dim = self.hidden_dim_one_direction * 2  # notice that we are using bi-lstm
        if self.model_dim == 0:  # meaning that this encoder will be useless
            return
        
        assert self.rnn_type in ['LSTM', 'GRU']
        if self.rnn_type == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim_one_direction,
                                    batch_first=True, bidirectional=True)
        else:
            self.rnn = torch.nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim_one_direction,
                                    batch_first=True, bidirectional=True)
            
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, X):
        encoded_features = self.rnn(X)[0]
        encoded_features = encoded_features.select(dim=1, index=0)
        encoded_features = self.dropout(encoded_features)
        return encoded_features

class CausalityFeatureEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, version='mean', rnn_type='GRU', dropout_p=0.5):
        super(CausalityFeatureEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_one_direction = self.hidden_dim // 2
        self.rnn_type = rnn_type
        self.version = version
        self.model_dim = self.hidden_dim_one_direction * 2  # notice that we are using bi-lstm

        assert self.rnn_type in ['LSTM', 'GRU']
        if self.rnn_type == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim_one_direction,
                                    batch_first=True, bidirectional=True)
        else:
            self.rnn = torch.nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim_one_direction,
                                    batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout_p)
        
        self.attAgg = torch.nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=8, dim_feedforward=2 * self.model_dim, 
                                                       dropout=dropout_p, activation='relu')
        
        self.norm = torch.nn.LayerNorm(self.model_dim)
        
    def forward(self, X):
        # [batch, seq_num, cs_len, pos_dim]
        f = []
        for i in range(X.shape[1]):
            encoded_features = self.rnn(X[:,i,:,:])[0]
            # encoded_features = encoded_features.select(dim=1, index=0)
            encoded_features = encoded_features.mean(dim=1)
            encoded_features = self.dropout(encoded_features)
            f.append(encoded_features)
        ft = torch.stack(f)
        
        if self.version == 'mean':
            output = torch.mean(ft, dim=0)
        else:
            # Transformer聚合
            output = self.attAgg(ft)
            output = torch.mean(output, dim=0)
        return output
        
class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        # x = (x1 + x2) / 2
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        z = self.fc2(h)
        return z

            
class IPNet(torch.nn.Module):
    def __init__(self, node_feature, node_interaction_seq, is_len, node_causality_seq, seq_num, cs_len, device, checkpoint_dir, 
                 n_head=8, dropout_p=0.1, bias=True, rnn_type='GRU', version='mean'):
        super(IPNet, self).__init__()
        self.rnn_type = rnn_type
        self.bias = bias
        
        self.node_feat = torch.nn.Parameter(torch.from_numpy(node_feature.astype(np.float32)))
        self.version = version
        self.feat_dim = self.node_feat.shape[1] # 节点特征维度
        self.time_dim = self.feat_dim // 2  # 时序特征
        self.pos_dim = self.feat_dim // 2   # 位置编码特征
        self.model_dim = self.time_dim + self.pos_dim
        self.attn_dim = self.model_dim
        self.out_dim = self.feat_dim
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.device = device
        self.seq_num = seq_num
        self.cs_len = cs_len
        self.node_embed = torch.nn.Embedding.from_pretrained(self.node_feat, padding_idx=0, freeze=True)
        
        # node interaction pattern learning
        self.time_encoder = TimeEncode(layers=is_len, enc_dim=self.time_dim)
        self.time_encoder.init_node_time_map(node_interaction_seq)
        
        self.position_encoder = PositionEncoder(layers=is_len, enc_dim=self.pos_dim)
        self.position_encoder.init_node_pos_map(node_interaction_seq, node_feature)
        
        # node causality learning
        self.causality_time_encoder = CausalityTimeEncode(seq_num=seq_num, layers=cs_len, enc_dim=self.time_dim)
        self.causality_time_encoder.init_node_time_map(node_causality_seq)
        
        self.causality_position_encoder = CausalityPositionEncoder(seq_num=seq_num, layers=cs_len, enc_dim=self.pos_dim)
        self.causality_position_encoder.init_node_pos_map(node_causality_seq, node_feature)
        
        # encode all types of features along each temporal walk
        self.feature_encoder = FeatureEncoder(self.model_dim, self.model_dim, self.rnn_type)
        self.causality_feature_encoder = CausalityFeatureEncoder(self.model_dim, self.model_dim, self.version, self.rnn_type)
        
        # not use
        self.projector = torch.nn.Sequential(torch.nn.Linear(self.out_dim , self.out_dim), 
                                             torch.nn.ReLU(), 
                                             torch.nn.Dropout(self.dropout_p))
        # not use
        self.self_attention = torch.nn.TransformerEncoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                      dim_feedforward=2 * self.attn_dim, dropout=self.dropout_p,
                                                      activation='relu')
        # final projection layer
        self.affinity_score = MergeLayer(self.out_dim * 2, self.out_dim * 2, self.feat_dim, 1)
        
        # set checkpoint path
        self.get_checkpoint_path = lambda epoch: os.path.join(checkpoint_dir, 'checkpoint-epoch-{}.pth'.format(epoch))
        
    
    def forward(self, edges):
        src_nodes = edges[:, 0]
        tgt_nodes = edges[:, 1]
        src_embed = self.forward_msg(src_nodes)
        tgt_embed = self.forward_msg(tgt_nodes)
        score = self.affinity_score(src_embed, tgt_embed)
        score.squeeze_(dim=-1)
        return score.sigmoid()

    def forward_msg(self, nodes):
        batch = len(nodes)
        
        nodes_th = torch.from_numpy(nodes).to(self.device)
        
    
        node_time_map = self.time_encoder.node_time_map
        time_enc = np.array([node_time_map[node] for node in nodes if node in node_time_map])
        assert len(time_enc) == batch
        times_tensor = torch.tensor(time_enc).to(self.device)
        times_tensor = times_tensor.select(dim=-1, index=0).unsqueeze(dim=-1) - times_tensor
        time_features = self.time_encoder(times_tensor)     # 节点交互的时序特征
        
        node_pos_enc = self.position_encoder.node_pos_enc   
        pos_enc = np.array([node_pos_enc[node] for node in nodes if node in node_pos_enc])
        assert len(pos_enc) == batch
        pos_tensor = torch.tensor(pos_enc).to(self.device)       
        pos_features = self.position_encoder(pos_tensor)    # 节点交互的位置编码特征
    
        combined_features = torch.cat([time_features, pos_features], dim=-1)
        combined_features = self.feature_encoder(combined_features)
        
        if self.version == 'w2v':
            causality_features = self.node_embed(nodes_th)
            return torch.cat([combined_features, causality_features], dim=-1)

        # 节点的路径 or 因果关联信息，捕获高阶信息
        node_causality_time_map = self.causality_time_encoder.node_time_map
        causality_time_enc = np.array([node_causality_time_map[node] for node in nodes if node in node_causality_time_map])
        assert len(causality_time_enc) == batch
        causality_times_tensor = torch.tensor(causality_time_enc).to(self.device)
        causality_time_features = self.causality_time_encoder(causality_times_tensor)     # [batch, is_len, cs_len, time_dim]
        
        node_causality_pos_enc = self.causality_position_encoder.node_pos_enc
        causality_pos_enc = np.array([node_causality_pos_enc[node] for node in nodes if node in node_causality_pos_enc])
        assert len(causality_pos_enc) == batch
        causality_pos_tensor = torch.tensor(causality_pos_enc).to(self.device)
        causality_pos_features = self.causality_position_encoder(causality_pos_tensor)  # [batch, is_len, cs_len, pos_dim]
        
        # concat
        causality_combined_features = torch.cat([causality_time_features, causality_pos_features], dim=-1)
        causality_combined_features = self.causality_feature_encoder(causality_combined_features)
        X = torch.cat([combined_features, causality_combined_features], dim=-1)
        return X