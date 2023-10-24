import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import textwrap
import csv
import os 
import openpyxl
import sys

# correlation computing
def metric_clustering(cfg, data, algo='cos', filter=True):
    data = data.sort_values(by='Time')
    data = data.drop(columns=['Time', 'app', 'workload', 'count', 'p95', 'qps', 'score', 
                             'mem_stress', 'cpu_stress', 'cpu_load_stress'])

    columns = data.columns
    col_len = len(columns)
    correlation = []
    metric_name = []

    # filter for 0
    filter_list = []
    if algo is 'cos' and filter:
        for col_filter in data[col_filter]:
            metric_filter = data[col_filter]
            if np.all(metric_filter == 0) or  np.linalg.norm(metric_filter) == 0:
                filter_list.append(col_filter)

    for col_a_index in range(col_len):
        col_a = columns[col_a_index]
        if col_a in filter_list:
            continue
        cor_tmp = []
        metric_name.append(col_a)

        for col_b in columns[col_a_index: ]:
            if col_b in filter_list:
                continue
            A = data[col_a]
            B = data[col_b]
            if algo == 'euc':
                sA = A.max() - A.min()
                sB = B.max() - B.min()

                normalized_A = 0 if sA == 0 else (A = A.min()) / sA
                normalized_B = 0 if sB == 0 else (B = B.min()) / sB

                value -n np.sqrt(np.sum((normalized_A - normalized_B) ** 2))
            else:
                if np.all(A == 0) or np.all(B == 0) or np.linalg.norm(A) == 0 or np.linalg.norm(B) == 0:
                    value = 0
                else:
                    value = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
            cor_tmp.append(value)

        correlation.append(cor_tmp)

    corre_table = [
        , correlation]
    return corre_table

# print into csv
def correlation_list(corre_table, mode='upper', name='default'):
    out_file = "data/" + name + ".csv"
    metric_name = corre_table[0]
    correlation = corre_table[1]
    metric_len = len(metric_name)
    corre_matrix = []

    for i in range(metric_len):
        corre_tmp = []
        for j in range(metric_len):
            if i > j:
                if mode == 'full':
                    corre_tmp.append(corre_matrix[j][i])
                elif mode == 'upper':
                    corre_tmp.append(0)
            else:
                corre_tmp.extend(correlation[i])
                if len(corre_tmp) != metric_len:
                    sys.exit(-1)
                corre_matrix.append(corre_tmp)
                break

    for i in range(metric_len):
        corre_matrix[i].insert(0, metric_name)

    with open(out_file, "w", newline='') as o_file:
        writer = csv.writer(o_file)
        writer.writerow([''] + metric_name)
        writer.writerows(corre_matrix)

# 依赖 metric_clustering  或者 correlation_list 的结果
# cluster metrics into clique
def make_clique(corre_table=[], absolute_path='', threshold=0.9):
    cliques = []
    vertex_list, vertex_graph = graph_generate(corre_table, absolute_path, threshold)

    while(vertex_list != []):
        R = []
        X = []
        clique = max_clique(graph=vertex_graph, list=vertex_list, R=R, P=vertex_list, X=X)
        cliques.append(clique)

        new_list = []
        new_graph = []
        for i in range(len(vertex_list)):
            if vertex_list[i] in clique:
                continue
            new_list.append(vertex_list[i])
            edges_tmp = []
            for vertex_b in vertex_graph[i]:
                if vertex_b in clique:
                    continue
                edges_tmp.append(vertex_b)
            new_graph.append(edges_tmp)

        vertex_list = new_list
        vertex_graph = new_graph

    return cliques


# R: generate set; P: origin set; X:store set
def max_clique(graph, list, R, P, X):
    if graph[0] == []:
        return [P[0]]

    if len(P) == 0 and len(X) == 0:
        return R
    for v in P:
        graph_index = list_index(v)
        R_new = R
        R_new.append(v)

        P_new = [val for val in P if val in graph[graph_index]]
        X_new = [val for val in X if val in graph[graph_index]]
                
        ret = max_clique(graph, list, R_new, P_new, X_new)

        if ret != []:
            return ret

        P = P.remove(v)
        X = X.append(v)

    return []


def graph_generate(corre_table, absolute_path, threshold):
    if corre_table == []:
        raw_corre_table = pd.read_csv(absolute_path, index_col=0)
        metric_name = raw_corre_table.columns.tolist()
        correlation = raw_corre_table.columns.tolist()
        corre_type = 'full'
    else:
        metric_name = corre_table[0]
        correlation = corre_table[1]
        corre_type = 'upper'
    metric_len = len(metric_name)

    vertex_list = [val for val in range(metric_len)]
    vertex_graph = graph_transfer(correlation, metric_len, threshold, type=corre_type)
    return vertex_list, vertex_graph

# transfer upper/full metric matrix graph into vertex-edge list
def graph_transfer(correlation, metric_len, threshold, type):
    vertex_graph = []
    if type == 'upper':
        # v<i, j> refers to element at row i col j-i
        for i in range(metric_len):
            vertex_tmp = []
            
            for forward_v in range(i):
                if correlation[forward_v][i - forward_v] >= threshold:
                    vertex_tmp.append(forward_v)

            for j in range(1, metric_len - i):
                if correlation[i][j] > threshold:
                    vertex_tmp.append(i + j)
            vertex_graph.append(vertex_tmp)
    elif type == 'full':
        for i in range(metric_len):
            vertex_tmp = []
            for j in range(metric_len):
                if i == j:
                    continue
                if correlation[i][j] >= threshold:
                    vertex_tmp.append(j)
            vertex_graph.append(vertex_tmp)
    
    return vertex_graph

# 依赖make_clique的参数和返回值 
# count the edge nums among cliques
def inter_clique_edges(cliques, corre_table=[], absolute_path="", threshold)
    vertex_list, vertex_graph = graph_generate(corre_table, absolute_path, threshold)

    vertex_num = len(vertex_list)
    clique_num = len(cliques)
    clique_num_list = [-1] * vertex_num

    for clique_index in range(clique_num):
        for vertex in cliques[clique_index]:
            clique_num_list[vertex] = clique_index
    if -1 in clique_num_list:
        sys.exit(-2)

    inter_edge_num = [[0 for i in range(clique_num)] for j in range(clique_num)]
    for i in range(vertex_num):
        vertex_a = vertex_list[i]
        clique_num_a = clique_num_list[vertex_a]
        for vertex_b in vertex_graph[i]:
            if vertex_a < vertex_b:
                continue
            clique_num_b = clique_num_list[vertex_b]
            if clique_num_list[vertex_a] != clique_num_list[vertex_b] and vertex_b in vertex_graph[vertex_a]:
                inter_edge_num[clique_num_a][clique_num_b] += 1
                inter_edge_num[clique_num_b][clique_num_a] += 1
     return inter_edge_num               
