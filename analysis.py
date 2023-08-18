import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import json

SUPER_PARAM = 1.5
GRAPH_RELATIVE_FOLDER = "graphs"

def read_all_in_one_data(raw_data_file_path, raw_time_splits_path, app_map):
    df = pd.read_csv(raw_data_file_path)
    with open(raw_time_splits_path, 'r') as f:
        time_splits = json.load(f)
        
    dfs = []
    dataCount = 0
    class_names = list(set([e if e not in app_map else app_map[e] for e in time_splits.keys()]))
    for (app_name, time_pairs) in time_splits.items():
        for time_pair in time_pairs:
            filtered_df = filter_noise(df.loc[(df['Time'] >= time_pair["start"]) & (df['Time'] <= time_pair["end"])])
            if filtered_df.empty:
                print(time_pair["start"], time_pair["end"])
                continue
            filtered_df['data_count'] = dataCount
            filtered_df['app_remap'] = app_name if app_name not in app_map else app_map[app_name]
            filtered_df['app'] = app_name
            dataCount = dataCount + 1
            dfs.append(filtered_df)
    result = pd.concat(dfs, ignore_index=True)
    return result


def filter_noise(df):
    q1 = df.quantile(0.15)
    q3 = df.quantile(0.90)
    iqr = q3 - q1
    outliers = ((df < (q1 - SUPER_PARAM * iqr)) | (df > (q3 + SUPER_PARAM * iqr))).any(axis=1)
    return df.drop(df[outliers].index)

def normalize(matrix):
    matrix_std = np.std(matrix, axis=0)
    matrix_mean = np.mean(matrix, axis=0)

    zero_std_cols = np.where(matrix_std == 0)[0]

    matrix_mean[zero_std_cols] = 0
    matrix_std[zero_std_cols] = 1

    return (matrix - matrix_mean) / matrix_std


def jobs_prefer_resource(data,x_column,y_column,xmax,file,bucket=100,yLabel='Utilization'):
    fontsize_set = 18

    x = data[x_column].to_numpy()
    y = data[y_column].to_numpy()
    left, width = 0.17, 0.5
    bottom, height = 0.17, 0.65
    spacing = 0.05
    
    rect_scatter = [left, bottom, width, height]
    # rect_histx = [left, bottom + height + spacing, width, 0.25]
    rect_histy = [left + width + spacing+0.01, bottom, 0.25, height]
    
    fig = plt.figure(figsize=(8, 5))
    
    # 添加图形
    ax = fig.add_axes(rect_scatter)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # 中间曲线图
    ax.set_title("Temporal Distribution", fontsize=fontsize_set)
    ax.plot(y, x, color="#717d7e",  marker='o', markersize=5, linestyle='-', linewidth=2)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, xmax)
    # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    # ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0)) 
    ax.tick_params(labelsize=fontsize_set-4, width=2, length=4, grid_linestyle=':')
    ax.grid()
    ax.set_xlabel('Normalized Relative Time', fontsize=fontsize_set, labelpad=15)
    ax.set_ylabel(yLabel, fontsize=fontsize_set, labelpad=15)
    
    # 右侧直方图图
    ax_histy.set_title("Histogram", fontsize=fontsize_set)
    ax_histy.tick_params(axis="y", labelleft=False)
    ht,_,_=ax_histy.hist(x, bins=bucket,range=(0, xmax), orientation='horizontal', color="#641e16")
    # ax_histy.hist(x, bins=100,range=(0, 1), density=True, orientation='horizontal', color="#641e16")
    # print(ht)
    ax_histy.tick_params(labelsize=fontsize_set-5, width=2, length=2, grid_linestyle=':')
    ax_histy.grid()
    ax_histy.set_xlim(0, max(ht)+5)
    # ax_histy.set_xticks([0,0.25,0.5,0.75,1])
    # ax_histy.set_xticklabels(["0%","25%","50%","75%","100%"])
    ax_histy.set_xlabel("Data Points", fontsize=fontsize_set-2)

    plt.suptitle(f"Metirc {x_column}", fontsize=fontsize_set+2)
    fig.savefig(file, dpi=200)
    plt.close(fig)
