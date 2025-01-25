import csv
import math
import os
import time
import torch
import numpy as np
import warnings
import argparse
import gensim
from IPNet import IPNet
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import EarlyStopMonitor, check_and_make_path, get_nx_graph, negative_sampling
from tqdm import tqdm
from data_loader import DataLoader
from datetime import datetime
# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


def train(model, train_graph, val_graph, batch_size=64, epochs=50, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.BCELoss()
    early_stopper = EarlyStopMonitor(tolerance=0)
    device = model.device
    
    num_instance = train_graph.number_of_edges()
    num_batch = math.ceil(num_instance / batch_size)
    
    # 获取训练集图中的边列表
    train_edges_list = np.array(list(train_graph.edges()))   # 若在train_graph.edges()中加 data=True, 则包含边的属性信息
    idx_list = np.arange(num_instance)
    
    for epoch in range(epochs):
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)  # shuffle the training samples for every epoch
        for k in tqdm(range(num_batch)):
            s_idx = k * batch_size
            e_idx = min(num_instance - 1, s_idx + batch_size)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            pos_edges = train_edges_list[batch_idx]
            size = len(batch_idx)
            
            # 负采样
            neg_edges = np.array(negative_sampling(train_graph, size))
            
            optimizer.zero_grad()
            model.train()
            
            pos_prob = model(pos_edges)
            neg_prob = model(neg_edges)
            pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
            neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False)
            loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
            loss.backward()
            optimizer.step()
            
            # collect training results
            with torch.no_grad():
                model.eval()
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                # f1.append(f1_score(true_label, pred_label))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))
        # validation phase use all information
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch(model, val_graph, batch_size)
        print('epoch: {}:'.format(epoch))
        print('epoch mean loss: {}'.format(np.mean(m_loss)))
        print('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
        print('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
        print('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))

        # early stop check and checkpoint saving
        if early_stopper.early_stop_check(val_auc):
            print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            print(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), model.get_checkpoint_path(epoch))

def eval_one_epoch(model, graph, test_batch_size=32):  
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    
    # 获取验证集中的边列表
    val_edges_list = np.array(list(graph.edges()))   # 若在train_graph.edges()中加 data=True, 则包含边的属性信息
    
    with torch.no_grad():
        model = model.eval()
        num_test_instance = graph.number_of_edges()
        num_test_batch = math.ceil(num_test_instance / test_batch_size)
        for k in range(num_test_batch):
            s_idx = k * test_batch_size
            e_idx = min(num_test_instance - 1, s_idx + test_batch_size)
            if s_idx == e_idx:
                continue
            
            pos_edges = val_edges_list[s_idx:e_idx]
            size = len(pos_edges)
            # 负采样
            neg_edges = np.array(negative_sampling(graph, size))

            pos_prob = model(pos_edges)
            neg_prob = model(neg_edges)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
            
    return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", dest="DATASET", default="UCI")
    parser.add_argument("--fd", help="time_dim and pos_dim", dest="FEAT_DIM", type=int, default="128")
    parser.add_argument("--ty", help="Transductive task (T) or Inductive task (I)", dest="TASK_TYPE", default="T")
    parser.add_argument("--bs", help="batch size", dest="BATCH_SIZE", type=int, default="64")
    parser.add_argument("--thread", help="multi-thread", dest="THREAD_NUM", type=int, default="10")
    parser.add_argument("--c", help="gpu id to run on, -1 for cpu", dest="GPU", type=int, default="0")
    parser.add_argument("--rnn", help="rnn type: LSTM/GRU", dest="RNN_TYPE", default="GRU")
    parser.add_argument("--is", "--IS", help="interaction sequence length", dest="IS_LEN", type=int, default="5")
    parser.add_argument("--n", "--N", help="walk num", dest="WALK_NUM", type=int, default="10")
    parser.add_argument("--l", "--L", help="walk length", dest="WALK_LEN", type=int, default="9")
    parser.add_argument("--v", "--V", help="IPNet version: mean, att, or w2v", dest="VERSION", default="w2v")
    args = parser.parse_args()

    DATASET = args.DATASET
    ORIGIN_GRAPH_PATH = 'data/{}/0.origin/graph.csv'.format(DATASET)
    NODES_SET_PATH = 'data/{}/1.nodes_set/nodes.csv'.format(DATASET)
    MODEL_PATH_DIR = 'model/{}'.format(DATASET)
    check_and_make_path(MODEL_PATH_DIR)
    LP_RESULT_DIR = 'results/{}'.format(DATASET)
    
    # 超参
    BATCH_SIZE = args.BATCH_SIZE
    RNN_TYPE = args.RNN_TYPE
    FEAT_DIM = args.FEAT_DIM  
    THREAD_NUM = args.THREAD_NUM 
    WALK_NUM = args.WALK_NUM
    WALK_LEN = args.WALK_LEN
    IS_LEN = args.IS_LEN
    VERSION = args.VERSION
    TASK_TYPE = args.TASK_TYPE
    
    # 模型保存路径
    formatted_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H_%M_%S")
    checkpoint_dir = os.path.join(MODEL_PATH_DIR, 'saved_checkpoints/{}'.format(formatted_time))
    check_and_make_path(checkpoint_dir)
    best_model_dir = os.path.join(MODEL_PATH_DIR, 'best_models/{}'.format(formatted_time))
    check_and_make_path(best_model_dir)
    best_model_path = os.path.join(best_model_dir, 'best-model.pth')
    
    # Device
    if args.GPU == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")

    # 读取全图所有节点
    # nodes_set = pd.read_csv(NODES_SET_PATH, names=['node'])
    # full_node_list = nodes_set['node'].tolist() # 暂时不需要
    
    # 记录程序运行时间
    start_time = time.time()
    print(f'Dynamic Network {DATASET} is processing...')
    
    # 加载整张图
    graph = get_nx_graph(ORIGIN_GRAPH_PATH, NODES_SET_PATH, sep='\t')
        
    # 创建DataLoader，分割训练集、验证集、测试集
    data_loader = DataLoader(graph)
    train_graph, val_graph, test_graph = data_loader.data_split()   # 默认: 0.5-0.3-0.2

    node_interaction_seq = data_loader.extract_node_interaction_sequences(train_graph)  # 交互信息，主要的编码对象
    walks_dict, walks_list = data_loader.extract_context(train_graph, walk_len=WALK_LEN, num_walks=WALK_NUM, workers=min(THREAD_NUM, os.cpu_count()))
    
    node_feature = np.eye(len(graph.nodes()), FEAT_DIM * 2)
    if VERSION == 'w2v':
        word2vec = gensim.models.Word2Vec(walks_list, vector_size=FEAT_DIM * 2, workers=min(THREAD_NUM, os.cpu_count()), 
                                        window=5, min_count=1, batch_words=64, negative=10)
        for key in word2vec.wv.index_to_key:
            node_feature[key] = word2vec.wv.get_vector(key)
        
    model = IPNet(node_feature, node_interaction_seq, IS_LEN, walks_dict, WALK_NUM, WALK_LEN, device, checkpoint_dir, rnn_type=RNN_TYPE, version=VERSION)
    model.to(device)
    
    if TASK_TYPE == 'I':
        # Inductive Learning
        data_loader.filter(train_graph, test_graph)
    
    # Train and val
    train(model, train_graph, val_graph, BATCH_SIZE)
    # final testing
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch(model, test_graph, BATCH_SIZE)
    print('Test statistics: acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
    
    # save model
    # print('Saving Model ...')
    # torch.save(model.state_dict(), best_model_path)
    # print('Model saved')
    
    # 记录结束时间
    end_time = time.time()
    execution_time = end_time - start_time
    task = "transductive link prediction task" if TASK_TYPE == 'T' else "Inductive link prediction task"
    print(f"finish {task} on {DATASET}! cost time: {execution_time} seconds!")
    desc = f"IS_LEN-{IS_LEN} WALK_NUM-{WALK_NUM} WALK_LEN-{WALK_LEN} Version-{VERSION}"
    
    # Test auc result of the graph
    test_auc_results = [test_auc * 100, execution_time]
    print(desc, test_auc_results)
    
    # save lp auc results
    LP_RESULT_DIR = os.path.join(LP_RESULT_DIR, desc)
    needHeader = False
    if not os.path.exists(LP_RESULT_DIR):
        needHeader = True
    check_and_make_path(LP_RESULT_DIR)
    
    with open(os.path.join(LP_RESULT_DIR, f'IPNet-{VERSION}.csv'), "a+", newline='') as f:
        writer = csv.writer(f)
        if needHeader:
            writer.writerow(["AUC", "Time"])  
        writer.writerow(test_auc_results)
                