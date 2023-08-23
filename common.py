import json
import os
import datetime
import numpy as np
import graphviz 
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

SUPER_PARAM = 1.5
GRAPH_RELATIVE_FOLDER = "graphs"

def read_one_data(raw_data_file_path, raw_time_splits_path, select_app):
    df = pd.read_csv(raw_data_file_path)
    df.set_index('Time', inplace=True)
    df = map_total(df)
    
    with open(raw_time_splits_path, 'r') as f:
        time_splits = json.load(f)
        
    X = []
    Y = []
    apps = []
    class_names = list(set(time_splits.keys()))
    
    for (app_name, time_pairs) in time_splits.items():
        if app_name == select_app:
            for time_pair in time_pairs:
                filtered_df = map_for_each_class(df.loc[(df.index >= time_pair["start"]) & (df.index <= time_pair["end"])])
                if filtered_df.empty:
                    print(time_pair["start"], time_pair["end"])
                    continue
                processed_data = pre_process_data(filtered_df)
    
                X.append(processed_data)
                Y.append((len(apps), class_names.index(app_name)))
            apps.append(app_name)
    return (np.array(X) , np.array(Y), df.columns, class_names, apps)

def read_all_in_one_data(raw_data_file_path, raw_time_splits_path, app_map):
    df = pd.read_csv(raw_data_file_path)
    df.set_index('Time', inplace=True)
    df = map_total(df)
    
    with open(raw_time_splits_path, 'r') as f:
        time_splits = json.load(f)
        
    X = []
    Y = []
    apps = []
    class_names = list(set([e if e not in app_map else app_map[e] for e in time_splits.keys()]))
    for (app_name, time_pairs) in time_splits.items():
        for time_pair in time_pairs:
            filtered_df = map_for_each_class(df.loc[(df.index >= time_pair["start"]) & (df.index <= time_pair["end"])])
            if filtered_df.empty:
                print(time_pair["start"], time_pair["end"])
                continue
            processed_data = pre_process_data(filtered_df)

            X.append(processed_data)
            Y.append((len(apps), class_names.index(app_name if app_name not in app_map else app_map[app_name])))
        apps.append(app_name)
    return (np.array(X) , np.array(Y), df.columns, class_names, apps)


def pre_process_data(df): 
    return df.mean().values

# reduce noise by IQR
def filter_noise(df):
    
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    outliers = ((df < (q1 - SUPER_PARAM * iqr)) | (df > (q3 + SUPER_PARAM * iqr))).any(axis=1)
    
    return df.drop(df[outliers].index)

# filter for total data
def map_total(df):
    
    # keep only vm metric
    start_column = "vm_cpu_time_sys" 
    end_column = df.columns[-1]
    df = df.loc[:, start_column:end_column]
    
    # filter std or mean zero
    cols_to_drop = set()
    std = df.std()
    mean = df.mean()

    cols_to_drop.update(std[std < 1e-10].index)
    cols_to_drop.update(mean[mean == 0].index)

    df.drop(columns = cols_to_drop, inplace=True)
        
    # normalize 
#     df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
#     df.rename(columns={"vm_cpu_ips": "vm_cpu_instructions"}, inplace=True)
    
    # Network throughput
    net_bytes_transmit_df = df["vm_network_io_bytes_transmit"] + 1
    net_bytes_receive_df = df["vm_network_io_bytes_receive"] + 1
    net_packets_transmit_df = df["vm_network_io_packets_transmit"] + 1
    net_packets_receive_df =  df["vm_network_io_packets_receive"] + 1
    
    df["vm_network_transmit_receive_packet_rate"] = net_packets_transmit_df / net_packets_receive_df
    df["vm_network_avg_receive_packet_size"] = net_bytes_receive_df / net_packets_receive_df
    df["vm_network_avg_transmit_packet_size"] = net_bytes_transmit_df / net_packets_transmit_df
    
    # calculate IPS, CPU Time  per Network throughput
    temp = ["vm_cpu_ips", "vm_cpu_vcpu_time", "vm_cpu_time_user", "vm_cpu_time_sys", "vm_cpu_branch_ips"]
    filtered_col = []
    for metric in temp:
        if metric in df.columns:
            # df[f"{metric}_per_transmit_network_byte"] = df[metric] / net_bytes_transmit_df
            # df[f"{metric}_per_receive_network_byte"] = df[metric] / net_bytes_receive_df
            # df[f"{metric}_per_transmit_network_packet"] = df[metric] / net_packets_transmit_df
            df[f"{metric}_per_receive_network_packet"] = df[metric] / net_packets_receive_df
            filtered_col.append(metric)
        
    # calculate Block IO throupt per Network throughput
    df["vm_block_bytes_write_per_receive_network_packet"] = df["vm_block_io_bytes_write"]  / net_packets_receive_df
    df["vm_block_bytes_read_per_receive_network_packet"] = df["vm_block_io_bytes_read"] / net_packets_receive_df
    
    df["vm_block_requests_write_per_receive_network_packet"] = df["vm_block_io_requests_write"]/ net_packets_receive_df
    df["vm_block_requests_read_per_receive_network_packet"] =  df["vm_block_io_requests_read"] / net_packets_receive_df

    df = df.drop(filtered_col + ["vm_block_io_bytes_write", "vm_block_io_bytes_read", "vm_block_io_requests_write", "vm_block_io_requests_read",
                  "vm_network_io_bytes_transmit", "vm_network_io_bytes_receive", "vm_network_io_packets_transmit", "vm_network_io_packets_receive"],
                 axis=1)       

    return df

# filter for each class data
def map_for_each_class(df):    
    df = filter_noise(df)
    return df

# save decision tree to png
def save_dt_to_png(name, clf, feature_names, class_names):
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names, 
                                    class_names=class_names, 
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render(filename=os.path.join(GRAPH_RELATIVE_FOLDER, name))
    return graph

# SVM need normalization
def normalize(matrix):
    matrix_std = np.std(matrix, axis=0)
    matrix_mean = np.mean(matrix, axis=0)

    zero_std_cols = np.where(matrix_std == 0)[0]

    matrix_mean[zero_std_cols] = 0
    matrix_std[zero_std_cols] = 1

    return (matrix - matrix_mean) / matrix_std

def model_score(model, x_test, y_test):
    y_pred = model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    
    print(f"accuracy: {accuracy}, recall_micro: {recall}, f1_micro: {f1}")
    
def spilt_for_train_test(X, Y, mode="random", **kwargs):
    
    match mode:
        case "random":
            return train_test_split(X, Y[:, 1], random_state=42, test_size=0.3)
        case "class":
            indices = Y[:, 0] == kwargs["apps"].index(kwargs["app"])
            Y = Y[:, 1]
        
            x_test = X[indices]
            y_test = Y[indices]
            x_train = X[~indices]
            y_train = Y[~indices]
            
            return x_train, x_test, y_train, y_test