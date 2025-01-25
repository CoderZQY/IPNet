from datetime import datetime
import os
import pandas as pd


node_dict = {}

def trans_id(nid):
    node_dict['U' + str(nid)] = 1
    return 'U' + str(nid)

input_file_path = 'data/Enron/0.origin/graph.txt'
output_graph_path = 'data/Enron/0.origin/graph.csv'
output_snap_dir = 'data/Enron/1.snapshots'
output_node_dir = 'data/Enron/1.nodes_set'

if not os.path.exists(output_snap_dir):
    os.makedirs(output_snap_dir)
if not os.path.exists(output_node_dir):
    os.makedirs(output_node_dir)

df = pd.read_csv(input_file_path, sep=' ', header=None, skiprows=0, names=['from_id', 'to_id', 'type', 'time'])
df[['from_id', 'to_id']] = df[['from_id', 'to_id']].applymap(trans_id)
df.to_csv(output_graph_path, sep='\t', index=False)

df['date'] = df['time'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m'))

for time, df_time in df.groupby(by='date'):
    df_output = df_time.drop(['date'], axis=1)

    datestamp = str(time)
    output_file_path = os.path.join(output_snap_dir, datestamp + '.csv')
    df_output.to_csv(output_file_path, sep='\t', index=False)

node_list = list(node_dict.keys())
node_list = sorted(node_list)
df_node = pd.DataFrame(node_list, columns=['node'])
node_file_path = os.path.join(output_node_dir, 'nodes.csv')
df_node.to_csv(node_file_path, sep='\t', index=False, header=False)