import numpy as np
import pandas as pd

def cosine_similarity(df):
    # 计算余弦相似度
    similarity_matrix = np.dot(df.T, df) / (np.linalg.norm(df, axis=0)[:, None] * np.linalg.norm(df, axis=0))
    
    # 将相似度矩阵转换为 DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=df.columns, columns=df.columns)
    return similarity_df

def pearson_correlation(df):
    similarity_df = df.corr()
    return similarity_df 

def single_corr(corr_df, col, similarity_filter=[]):
    df = corr_df[[col]].rename(columns={col: "Correlation"})
    df['AbsCorrelation'] = df['Correlation'].abs()
    df = df.sort_values(by='AbsCorrelation', ascending=False)
    df = df[["Correlation"]].iloc[1:].dropna(axis=0, how='all')
    for filter in similarity_filter:
        df = filter(df)
    return df

def flatten_corr(corr_matrix, similarity_filter=[]):
    pair_name = "Metric_A <-> Metric_B"
    # 将矩阵转换为长格式 DataFrame
    corr_df = corr_matrix.unstack().reset_index()
    corr_df.columns = ['Column1', 'Column2', 'Correlation']
    # 移除相同列的相关性
    corr_df = corr_df[corr_df['Column1'] != corr_df['Column2']]
    
    corr_df[pair_name] = corr_df.apply(lambda row: ' <-> '.join(sorted([row['Column1'], row['Column2']])), axis=1)
    
    # 添加一个绝对值列
    corr_df['AbsCorrelation'] = corr_df['Correlation'].abs()
    
    # 移除重复的列对
    corr_df = corr_df.drop_duplicates(subset=[pair_name]).sort_values(by='AbsCorrelation', ascending=False)

    # 重设索引
    corr_df = corr_df.set_index(pair_name)[['Correlation']]  

    for filter in similarity_filter:
        corr_df = filter(corr_df)
    return corr_df