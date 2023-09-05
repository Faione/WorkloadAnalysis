import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import textwrap
import csv
import os
import openpyxl
from openpyxl.utils import get_column_letter
from matplotlib import ticker
from scipy.stats import entropy
from sklearn import tree
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

SUPER_PARAM = 1.5
GRAPH_RELATIVE_FOLDER = "graphs"

# 数据读取
def read_all_in_one_data(cfg):
    result = load_data(cfg)

    metric_rename = {}
    drop_list = []

    remap = {}
    for v in cfg["metric_remap"]:
        remap[v["metric"]]=v["rename"]
    
    for v in cfg["metric"]:
        if v in result.columns[1:]:
            if v in remap:
                metric_rename[v]=remap[v]
            else:
                print(f'{v} need an describe in cfg.app_remap')
                exit(1)
    
    for v in result.columns[1:-3]:
        if v not in metric_rename:
            drop_list.append(v)

    if cfg["config"]["drop_un_remap"] == "true":
        result = result.drop(columns=drop_list) 
    result.rename(columns=metric_rename, inplace=True)
    
    return result,drop_list

def load_data(cfg):
    raw_data_file_path = cfg["config"]["input_file_csv"]
    raw_time_splits_path = cfg["config"]["input_file_json"]
    apps_remap = cfg["apps_remap"]
    filter_threshold_min = cfg["config"]["filter"]["min_threshold"]
    filter_threshold_max = cfg["config"]["filter"]["max_threshold"]

    df = pd.read_csv(raw_data_file_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    with open(raw_time_splits_path, 'r') as f:
        time_splits = json.load(f)

    app_label = {}
    for i,v in enumerate(cfg["apps"]):
        app_label[v] = i + 1
    
    dfs = []
    dataCount = 0
    class_names = list(set([e if e not in apps_remap else apps_remap[e] for e in time_splits.keys()]))
    for (app_name, time_pairs) in time_splits.items():
        for time_pair in time_pairs:
            filtered_df = filter_noise(df.loc[(df['Time'] >= time_pair["start"]) & (df['Time'] <= time_pair["end"])],filter_threshold_min,filter_threshold_max)
            if filtered_df.empty:
                continue
            filtered_df['data_count'] = dataCount
            filtered_df['app_remap'] = app_name if app_name not in apps_remap else apps_remap[app_name]
            filtered_df['app'] = app_label[app_name]
            dataCount = dataCount + 1
            dfs.append(filtered_df)
    result = pd.concat(dfs, ignore_index=True)
    return result

# 查看所有质变
def show_all_metric(cfg):
    data = load_data(cfg)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    for column in data.columns[1:-3]:
        print(f'"{column}",')

# 决策树预选指标
def decision_tree_select_k_metric(cfg,per_times=10,k=1):
    data = load_data(cfg)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    data = data.fillna(0)
    topk = []
    for i in range(1,k+1):
        _,_,feature_top1 = decision_tree_select_one_metric(
            cfg,
            data.copy(),
            times=per_times,
            drop_list=topk
        )
        topk.append(feature_top1)

    for v in topk:
        print(f'"{v}",')
    return topk

def decision_tree_select_one_metric(cfg,data,times=10,drop_list=[]):
    data = data.drop(columns=drop_list)
    feature_names = data.drop(columns=['Time','data_count','app','app_remap']).columns
    X = np.vstack(data.drop(columns=['Time','data_count','app','app_remap']).values)
    data = data.reset_index()
    Y = data['app'].values

    parameters = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [1, 2,3,4],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': [0.90]
    }

    feature_topk = {}
    result = []
    for i in range(1,times+1):
        x_train, x_test, y_train, y_test = spilt_for_train_test(X, Y, mode="random")
        clf = tree.DecisionTreeClassifier()
        grid_search = GridSearchCV(clf, parameters, cv=3, scoring='f1_macro')
        grid_search.fit(x_train, y_train)
        accuracy,recall,f1 = model_score(grid_search.best_estimator_, x_test, y_test)
        
        used_feature = []
        for v in grid_search.best_estimator_.tree_.feature:
            if v > 0:
                used_feature.append(feature_names[v])
                if feature_names[v] not in feature_topk:
                    feature_topk[feature_names[v]]=1
                else:
                    feature_topk[feature_names[v]]=feature_topk[feature_names[v]]+1
        result.append({"features":used_feature,"score": {"accuracy":accuracy,"recall":recall,"f1":f1}})

    max_name = ""
    max_value = 0
    for k,v in feature_topk.items():
        if v > max_value:
            max_value = v
            max_name  = k
    return feature_topk,result,max_name


def filter_noise(df,min,max):
    q1 = df.quantile(min)
    q3 = df.quantile(max)
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

def model_score(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    
    # print(f"accuracy: {accuracy}, recall_micro: {recall}, f1_micro: {f1}")
    return accuracy,recall,f1


def spilt_for_train_test(X, Y, mode="random", **kwargs):
    match mode:
        case "random":
            return train_test_split(X, Y, random_state=42, test_size=0.3)

def svm_check(data):
    data = data.fillna(0)
    X = np.vstack(data.drop(columns=['app','label']).values)
    data = data.reset_index()
    Y = data['label'].values
    
    x_train, x_test, y_train, y_test = spilt_for_train_test(X, Y, mode="random")
    
    clf = svm.SVC()
    clf = clf.fit(normalize(x_train), y_train)
    
    accuracy,recall,f1 = model_score(clf, normalize(x_test), y_test)
    print(f"accuracy: {accuracy}, recall_micro: {recall}, f1_micro: {f1}")

def decision_tree(data):
    data = data.fillna(0)
    X = np.vstack(data.drop(columns=['app','label']).values)
    data = data.reset_index()
    Y = data['label'].values
    
    x_train, x_test, y_train, y_test = spilt_for_train_test(X, Y, mode="random")
    cv = 3
    clf = tree.DecisionTreeClassifier()
    parameters = {'criterion': ['gini', 'entropy'],
                  'max_depth': [1, 2,3,4],
                  'min_samples_split': [2],
                  'min_samples_leaf': [1],
                  'max_features': [0.90]}
    
    grid_search = GridSearchCV(clf, parameters, cv=cv, scoring='f1_macro')
    grid_search.fit(x_train, y_train)
    
    accuracy,recall,f1 = model_score(grid_search.best_estimator_, x_test, y_test)
    print(f"accuracy: {accuracy}, recall_micro: {recall}, f1_micro: {f1}")
    

# k mean 聚类算法监测
def kmeans_check(cfg,data,random_state = 160,filename='sklearn_check'):
    data = data.fillna(0)
    X = np.vstack(data.drop(columns=['app','label']).values)
    data = data.reset_index()
    Y = data['label'].values

    common_params = {
        "n_init": "auto",
        "random_state": random_state,
    }

    k_means = KMeans(**common_params)
    k_means.fit(normalize(X))

    reslut=[]
    for i, label in enumerate(k_means.labels_):
        reslut.append([data.loc[i]['app'],label])

    with open(f'{cfg["config"]["output_dir"]}/{filename}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["app","class"])  # Write header
        writer.writerows(reslut) 

    silhouette_avg = silhouette_score(X, Y)

    print(silhouette_avg)
    return silhouette_avg


#  平滑负载影响，算法可以改变，目前是除以:
#  Receive Packet +  Receive Packet mean / 10
#  + Receive Packet mean / 10  主要原因是 Receive Packet 经常从0开始， 除0会导结果变很大

def dev_net_packet(cfg,df):
    need_dev_net_packet_columns=cfg["metric_need_dev_workload"]

    mean = df[cfg['workload_metric']].mean()
    for column in need_dev_net_packet_columns:
        df[column] = df[column] / ( df[cfg['workload_metric']] + mean / 10 )
        df[column] = df[column].fillna(0)
    return df

# 聚合指标，主要是包大小
def metric_aggregate(cfg,df):
    infos = {}
    for v in cfg['metric_aggregate']:
        if v['deal']['mode'] == 'div':
            df[v['rename']] = df[v['deal']['num']] / df[v['deal']['den']]
            df[v['rename']] = df[v['rename']].fillna(0)
        infos[v['rename']] = v
    return df,infos

# 计算指标的基本项
def metric_compute(cfg,data,column,ymax,kind):
    if kind == 'san':
        num_bins = 20
        if ymax == 0:
            return 0
        bins = np.linspace(0, ymax, num_bins + 1)   
        ds = pd.DataFrame(pd.cut(data[column], bins=bins,labels=False).value_counts()).reindex(range(0, num_bins), fill_value=0)
        ds['count'] = ds['count'] / ds['count'].sum()
        entropy_value = entropy(ds['count'], base=2)
        value = entropy_value    
    elif kind == 'avg':
        value = data[column].mean()
    elif kind == 'std':
        value = data[column].std()
    elif kind == 'p99':
        value = data[column].quantile(0.99)
    else:
        value = 0
    return value
    
# 提取指标到excel，改指标可以用来做聚类
def extral_metric(cfg,data,need_dev_net_packet=False,filename='metric_analysis',is_merge=True,is_save=True):
    metric_extra = cfg['config']['metric_extra']

    # 新增指标处理
    data,infos = metric_aggregate(cfg,data)

    # 平滑处理
    if need_dev_net_packet:
        data = dev_net_packet(cfg,data)

    ymax = {}
    for column in data.columns:
        ymax[column] = data[column].max()

    grouped = data.groupby(['app','app_remap'])
    if is_merge == False:
        grouped = data.groupby(['data_count','app','app_remap'])
    
    chName = {}
    stName = {}
    className = {}
    for v in cfg['metric_remap']:
        chName[v['rename']] = v['rename_ch']
        stName[v['rename']] = v['short_name']
        if v['belong'] not in className:
            className[v['belong']] = []
        className[v['belong']].append(v['rename'])
    for m,v in infos.items():
        chName[m]=v['rename_ch']
        stName[m] = v['short_name']
        if v['belong'] not in className:
            className[v['belong']] = []
        className[v['belong']].append(m)
    

    workbook = openpyxl.Workbook()
    worksheet = workbook.active

    # 首行
    title = "metric analysis"
    titleCols = (len(stName) + 1)
    worksheet[f'{get_column_letter(1)}1']='application\metric'
    worksheet.merge_cells(f'{get_column_letter(2)}1:{get_column_letter(titleCols)}1')
    worksheet[f'{get_column_letter(2)}1'] = title

    # 2 行
    columns_index = 1
    worksheet[f'{get_column_letter(1)}2']='metric_class'
    for class_name in className.keys():
        start = columns_index + 1
        end = columns_index + len(className[class_name]) * len(cfg['config']['metric_extra'])
        worksheet.merge_cells(f'{get_column_letter(start)}2:{get_column_letter(end)}2')
        worksheet[f'{get_column_letter(start)}2'] = class_name
        columns_index = end

    columns_index = 1
    worksheet[f'{get_column_letter(1)}3']='short_name'
    worksheet[f'{get_column_letter(1)}4']='ch_name'
    for k,ms in className.items():
        for m in ms:
            st_name = stName[m]
            ch_name = chName[m]
            
            start = columns_index + 1
            end = columns_index + len(cfg['config']['metric_extra'])
            
            worksheet.merge_cells(f'{get_column_letter(start)}3:{get_column_letter(end)}3')
            worksheet[f'{get_column_letter(start)}3'] = st_name
            
            worksheet.merge_cells(f'{get_column_letter(start)}4:{get_column_letter(end)}4')
            worksheet[f'{get_column_letter(start)}4'] = ch_name
            columns_index = end


    columns_index = 1
    worksheet[f'{get_column_letter(1)}5']='class'
    for v in stName:
        for cl in cfg['config']['metric_extra']:
            idx = columns_index + 1
            worksheet[f'{get_column_letter(idx)}5'] = cl
            columns_index = columns_index +1

    data_columns=[]
    for k,ms in className.items():
        for m in ms:
            for cl in cfg['config']['metric_extra']:
                data_columns.append(f'{m} {cl}')
    data_columns.append('app')
    data_columns.append('label')


    data_index=[]
    data_array=[]
    row_index = 6
    for app,gp in grouped:
        if is_merge == True:
            worksheet[f'{get_column_letter(1)}{row_index}']= app[1]
            data_index.append(app[1])
        else:
            worksheet[f'{get_column_letter(1)}{row_index}']= f'{app[2]} {app[0]}'
            data_index.append(f'{app[2]}_{app[0]}')
        columns_index = 1
        data_row = []
        for k,ms in className.items():
            for column in ms:
                for cl in cfg['config']['metric_extra']:
                    idx = columns_index + 1
                    
                    val = metric_compute(cfg,gp,column,ymax[column],cl)
                    worksheet[f'{get_column_letter(idx)}{row_index}'] = val
                    data_row.append(val)
                    
                    columns_index = columns_index + 1
        row_index = row_index + 1
        data_row.append(app[len(app)-1])
        data_row.append(app[len(app)-2])
        data_array.append(data_row)

    if is_save:
        workbook.save(f'{cfg["config"]["output_dir"]}/{filename}.xlsx')
    return pd.DataFrame(data_array, columns=data_columns, index=data_index)


# 这个代码主要是看一下指标长啥样，指标在data_deal.json里面配置
def draw_preview_chart(cfg,data,need_dev_net_packet=False,fig_size_width=24,fig_size_higth = 16,fontsize_set = 18,subtitle_width = 25,suby_label_width=25,subhist_buckets = 20,title_space_rate=0.03,bottom_space_rate=0.05,left_space_rate=0.05,is_merge=True,select_columns=[],sub_dir='data'):
    pic_dir=cfg['config']['output_dir']

    # 聚合处理部分数据
    data,infos = metric_aggregate(cfg,data)

    # 处理 y label
    units = {}
    for v in cfg['metric_remap']:
        units[v['rename']] = v['unit']
        if need_dev_net_packet and v['rename'] in cfg["metric_need_dev_workload"]:
            units[v['rename']] = v['dev_packet_unit']
    for m,v in infos.items():
        units[m]=v['unit']
        if need_dev_net_packet and m in cfg["metric_need_dev_workload"]:
            units[m] = v['dev_packet_unit']

    # 部分列消除负载影响
    if need_dev_net_packet:
        data = dev_net_packet(cfg,data)
        
    # 相同指标，不同应用的y 轴的值是否统一在一个范围
    normalize = cfg['config']['normalize_y']
    ymax = {}
    for column in data.columns:
        ymax[column] = data[column].max()

    if is_merge == True:
        grouped = data.groupby(['app','app_remap'])
    else:
        grouped = data.groupby(['data_count','app','app_remap'])
    
    for app_info, group in grouped: 
        data = group.drop(columns=['data_count','app','app_remap']) 
        data = data.sort_values(by='Time')
        data = data.reset_index()
        data  = data.drop(columns=['Time','index']) 
        data = data.reset_index()
        data.rename(columns={'index': 'Time'}, inplace=True)
        data['Time'] = (data['Time'] - data['Time'].min() ) * 100 / (data['Time'].max() - data['Time'].min()) / 100

        if len(select_columns) == 0:
            metric_count =  len(data.columns) - 1
        else:
            metric_count = len(select_columns)
        sq = int(math.sqrt(metric_count))
        higth_count = sq if sq * sq >= metric_count else sq + 1
        width_count = sq if (sq + 1) * sq > metric_count else sq + 1
    
        title_space = fig_size_higth * title_space_rate
        bottom_space = fig_size_higth * bottom_space_rate
        left_space = fig_size_width * left_space_rate
        higth = ( fig_size_higth - title_space - bottom_space) / higth_count
        width = ( fig_size_width - left_space ) / width_count
        
        higth_spacing = 0.25 * higth
        width_spacing = 0.2 * width
        
        rects_local = []
        for j in range(1,higth_count+1):
            for i in range(1,width_count+1):
                x = ( ( i - 1 ) % width_count ) * width
                y = fig_size_higth - (( ( j - 1 ) % higth_count ) + 1 ) * higth
                rects_local.append(
                    ( (x + left_space) / fig_size_width,
                      (y - title_space) / fig_size_higth, 
                     (width -  width_spacing) / fig_size_width,
                     (higth - higth_spacing)  / fig_size_higth
                ))

        if len(select_columns) == 0:
            columns = data.columns[1:]
        else:
            columns = select_columns
        
        fig = plt.figure(figsize=(fig_size_width,fig_size_higth))
        for k,column in enumerate(columns): 
            r_x,r_y,r_width,r_higth = rects_local[k]
            time_series = (r_x,r_y, r_width *0.65,r_higth * 0.9)
            hist = (r_x + r_width *0.70,r_y, r_width * 0.30,r_higth * 0.9 )
            
            ax_series =  fig.add_axes(time_series)
            ax_hist = fig.add_axes(hist, sharey=ax_series)
    
            # 时序图
            wrapped_title = textwrap.fill(column, width=subtitle_width)
            ax_series.set_title(wrapped_title, fontsize=fontsize_set)
            if column == cfg['workload_metric']:
                linewidth = 5
                markersize = 8
                color = "red" 
            else:
                linewidth = 2
                markersize = 5
                color = "#717d7e"
            ax_series.plot(data['Time'], data[column], color=color,  marker='o', markersize=5, linestyle='-', linewidth=linewidth)
            ax_series.set_xlim(0, 1.0)
            maxY = 0
            if normalize == "true":
                maxY = ymax[column]
            else:
                maxY = data[column].max()

            if maxY == 0:
                maxY = 1
            ax_series.set_ylim(0, maxY)
            ax_series.tick_params(labelsize=fontsize_set-4, width=2, length=4, grid_linestyle=':')
            ax_series.grid()
            if ((k+1) / width_count - higth_count + 1) > 0:
                ax_series.set_xlabel('Normalized Relative Time', fontsize=fontsize_set-2, labelpad=15)
            # ax_series.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax_series.ticklabel_format(axis='y', style='sci', scilimits=(0,4))

            wrapped_y_label = textwrap.fill(units[column], width=suby_label_width)
            ax_series.set_ylabel(wrapped_y_label, fontsize=fontsize_set-2, labelpad=15)
            
            
            # 直方图
            x = data[column].to_numpy()
            ax_hist.set_title("Histogram", fontsize=fontsize_set)
            ax_hist.tick_params(axis="y", labelleft=False)
            ht,_,_ = ax_hist.hist(x, bins=subhist_buckets,range=(0, maxY), orientation='horizontal', color="#641e16")
            ax_hist.tick_params(labelsize=fontsize_set-5, width=2, length=2, grid_linestyle=':')
            ax_hist.grid()
            ax_hist.set_xlim(0, max(ht)+5)
            ax_hist.set_xlabel("Data Points", fontsize=fontsize_set-2)

        if is_merge == True:
            app = app_info[1]
        else:
            app = f'{app_info[2]}_{app_info[0]}'
            
        plt.suptitle(f"{app} metirc", fontsize=fontsize_set+4)

        dir = f'{pic_dir}/{sub_dir}'
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        if need_dev_net_packet:
            fig.savefig(f'{dir}/{app}_dev_net_packet.png', dpi=200)
        else:
            fig.savefig(f'{dir}/{app}.png', dpi=200)
        plt.close(fig)


def all_in_one():
    # 读取配置文件
    with open("data_deal.json", "r") as file:
        cfg = json.load(file)
    
    # 读取数据
    df  = analysis.read_all_in_one_data(cfg)

    # 每个应用，每批数据，都画一个图，不消除负载
    dfcopy = df.copy()
    draw_preview_chart(
        cfg,                        # 配置文件
        data=dfcopy,                # 数据
        need_dev_net_packet=False,  # 是否消除负载影响， 消除方式可以更改
        fig_size_width=24,          # 整体图片宽度
        fig_size_higth=16,          # 整体图片高度 
        fontsize_set=18,            # 字体大小，所有字体都是相对这个变化的
        subtitle_width=25,          # 子图 title 每行字数
        suby_label_width=25,        # 子图 y label 每行字数
        subhist_buckets=20,         # 子图左侧分布图的柱子个数
        title_space_rate=0.03,      # 总图 title 占用空间大小
        bottom_space_rate=0.05,     # 总图 最底部 x label 占用空间比例
        is_merge=False,             # 数据是否合并分析
        sub_dir=''
    )
        
    # 每个应用，每批数据，都画一个图，消除负载
    dfcopy = df.copy()
    draw_preview_chart(
        cfg,                        # 配置文件
        data=dfcopy,                  # 数据
        need_dev_net_packet=True,  # 是否消除负载影响， 消除方式可以更改
        fig_size_width=24,          # 整体图片宽度
        fig_size_higth=16,          # 整体图片高度 
        fontsize_set=18,            # 字体大小，所有字体都是相对这个变化的
        subtitle_width=25,          # 子图 title 每行字数
        suby_label_width=25,        # 子图 y label 每行字数
        subhist_buckets=20,         # 子图左侧分布图的柱子个数
        title_space_rate=0.03,      # 总图 title 占用空间大小
        bottom_space_rate=0.05,     # 总图 最底部 x label 占用空间比例
        is_merge=False               # 数据是否合并分析
    )

    # 每个应用所有数据画一个图，不消除负载
    dfcopy = df.copy()
    draw_preview_chart(
        cfg,                        # 配置文件
        data=dfcopy,                  # 数据
        need_dev_net_packet=False,  # 是否消除负载影响， 消除方式可以更改
        fig_size_width=24,          # 整体图片宽度
        fig_size_higth=16,          # 整体图片高度 
        fontsize_set=18,            # 字体大小，所有字体都是相对这个变化的
        subtitle_width=25,          # 子图 title 每行字数
        suby_label_width=25,        # 子图 y label 每行字数
        subhist_buckets=20,         # 子图左侧分布图的柱子个数
        title_space_rate=0.03,      # 总图 title 占用空间大小
        bottom_space_rate=0.05,     # 总图 最底部 x label 占用空间比例
        is_merge=True               # 数据是否合并分析
    )
        
    # 每个应用所有数据画一个图，消除负载
    dfcopy = df.copy()
    draw_preview_chart(
        cfg,                        # 配置文件
        data=dfcopy,                  # 数据
        need_dev_net_packet=True,  # 是否消除负载影响， 消除方式可以更改
        fig_size_width=24,          # 整体图片宽度
        fig_size_higth=16,          # 整体图片高度 
        fontsize_set=18,            # 字体大小，所有字体都是相对这个变化的
        subtitle_width=25,          # 子图 title 每行字数
        suby_label_width=25,        # 子图 y label 每行字数
        subhist_buckets=20,         # 子图左侧分布图的柱子个数
        title_space_rate=0.03,      # 总图 title 占用空间大小
        bottom_space_rate=0.05,     # 总图 最底部 x label 占用空间比例
        is_merge=True               # 数据是否合并分析
    )

    # 每个应用，每批数据，都计算画像指标，不消除负载影响
    dfcopy = df.copy()
    extral_metric(
        cfg,
        dfcopy,
        need_dev_net_packet=False,
        is_merge=False,
        filename='metric_analysis'
    )

    # 每个应用所有数据聚合计算画像指标，除负载影响
    dfcopy = df.copy()
    extral_metric(
        cfg,
        dfcopy,
        need_dev_net_packet=False,
        is_merge=True,
        filename='metric_analysis_dev_net_packet'
    )

    # 聚合结果，轮廓系数评估，
    silhouette_avg = kmeans_check(
        cfg,
        data.copy(),
        random_state=120,
        filename='sklearn_check'
    )
    
    print(silhouette_avg)

    # 决策树评估结果
    decision_tree(
        data.copy()
    )

    # svm 评估结果
    svm_check(
        data.copy()
    )
