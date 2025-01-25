from datetime import datetime
import os
import numpy as np
import pandas as pd


# 按照月份切割
def snapshots_month(df):
    df['date'] = df['time'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m'))
    for time, df_time in df.groupby(by='date'):
        df_output = df_time.drop(['date'], axis=1)
        datestamp = str(time)
        output_file_path = os.path.join(output_snap_dir, datestamp + '.csv')
        df_output.to_csv(output_file_path, sep='\t', index=False)

# 均匀切割
def snapshots_uniform(df):
    # 由于只有百分之50的数据用于训练，所以将前百分之50的数据，均匀切割成快照
    df = df.iloc[0:df.index.values.max()//2,:]
    snapshots_num = 5
    spl_df = np.array_split(df, snapshots_num)
    for i in range(snapshots_num):
        sub_df = spl_df[i]
        output_file_path = os.path.join(output_snap_dir, 'train_snap_{}.csv'.format(i + 1))
        sub_df.to_csv(output_file_path, sep='\t', index=False)
    
if __name__ == '__main__':
    node_dict = {}

    def trans_id(nid):
        node_dict['U' + str(nid)] = 1
        return 'U' + str(nid)

    input_file_path = 'data/UCI/0.origin/graph.txt'
    output_graph_path = 'data/UCI/0.origin/graph.csv'
    output_snap_dir = 'data/UCI/1.snapshots'
    output_node_dir = 'data/UCI/1.nodes_set'

    if not os.path.exists(output_snap_dir):
        os.makedirs(output_snap_dir)
    if not os.path.exists(output_node_dir):
        os.makedirs(output_node_dir)

    # 整个网络
    df = pd.read_csv(input_file_path, sep=' ', header=None, skiprows=2, names=['from_id', 'to_id', 'weight', 'time'])
    df[['from_id', 'to_id']] = df[['from_id', 'to_id']].applymap(trans_id)
    df.to_csv(output_graph_path, sep='\t', index=False)

    # 快照, 用于DTDG baselines训练
    snapshots_uniform(df)

    # 节点集
    node_list = list(node_dict.keys())
    node_list = sorted(node_list)
    df_node = pd.DataFrame(node_list, columns=['node'])
    node_file_path = os.path.join(output_node_dir, 'nodes.csv')
    df_node.to_csv(node_file_path, sep='\t', index=False, header=False)