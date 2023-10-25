import json
import os
import pandas as pd


defualt_workload_preprocess_funcs = [
    filter_column_startswith(col_prefix=("vm", "app")),
    filter_column_useless(std_min=1e-10),
    filter_row_noise(col_prefix=("app")),
]

defualt_workload_agg_funcs = [
    lambda x : x.mean().to_frame().T,
]

def format_to_13_timestamp(ts):
    assert ts + 1 < 10000000000000
    return int(format(ts, '0<13d'))

def filter_row_timerange(start, end):
    return lambda x : x.loc[(x.index >= start) & (x.index <=end)]

def filter_column_useless(std_min, mean_min=0):
    def __filter_column_useless(df):
        cols_to_drop = set()
        std = df.std()
        mean = df.mean()

        cols_to_drop.update(std[std <= std_min].index)
        cols_to_drop.update(mean[mean <= mean_min].index)
        return df.drop(columns = cols_to_drop)
    return __filter_column_useless

# same col may not change at all
# so it is important to set col prefix and choose those essential
def filter_row_noise(col_prefix, sparam=1.5, l=0.25, h=0.75):
    def __filter_row_noise(df):
        temp_df = filter_column_startswith(col_prefix)(df)
        q1 = temp_df.quantile(l)
        q3 = temp_df.quantile(h)
        iqr = q3 - q1
        outliers = ((temp_df < (temp_df - sparam * iqr)) | (temp_df > (q3 + sparam * iqr))).any(axis=1)
        return df.drop(temp_df[outliers].index)
    
    return __filter_row_noise

def filter_column_startswith(col_prefix):
    return lambda x : x[[col for col in x.columns if col.startswith(col_prefix)]]

def filter_workload(workload_info):
    assert "start_time" in workload_info and "end_time" in workload_info
    start = format_to_13_timestamp(workload_info["start_time"])
    end = format_to_13_timestamp(workload_info["end_time"]) 
    return filter_row_timerange(start, end)

def apply_df_funcs(df, df_funcs=[]):
    for df_func in df_funcs:
        df = df_func(df)
    return df

class ExpData:
    def __init__(self, df, exp):
        self.df = df
        self.exp = exp
        self.workload_preprocess_funcs = defualt_workload_preprocess_funcs
        self.workload_agg_funcs = defualt_workload_agg_funcs

    def set_workload_preprocess_funcs(self, df_funcs):
        self.workload_preprocess_funcs = df_funcs
        return self

    def set_workload_agg_funcs(self, df_funcs):
        self.workload_agg_funcs = df_funcs
        return self

    def workload_df(self, workload):
        df_funcs = [filter_workload(workload)] + self.workload_preprocess_funcs
        return apply_df_funcs(self.df, df_funcs)

    def agg_one_workload(self, workload):
        df_funcs =  [filter_workload(workload)] + self.workload_preprocess_funcs + self.workload_agg_funcs
        return apply_df_funcs(self.df, df_funcs)

    def __agg_one_epoch(self, n_epoch):
        epoch_info = self.exp["info_per_epoch"][n_epoch]
        
        dfs = []
        for k,v in epoch_info["workloads"].items():
            df = self.agg_one_workload(v)
            dfs.append(df)
        
        df = pd.concat(dfs).fillna(0).reset_index(drop=True)

        # assume stress is the same in one epoch
        if "stress" in epoch_info:
            stress_data = {}
            for k, v in epoch_info["stress"].items():
                for _k, _v in v.items():
                    stress_data[_k] = [_v for i in range(df.shape[0])]
                    
            stress_df = pd.DataFrame(stress_data)
            assert stress_df.shape[0] == df.shape[0], f"miss match length {stress_df.shape[0]} and {df.shape[0]}"
            df = pd.concat([stress_df, df], axis=1)

        keys = list(epoch_info["workloads"].keys())
        assert df.shape[0] == len(keys), f"miss match length {len(dfs)} and {len(keys)}"
        
        return df.rename(index=lambda x : keys[x])
        
    def agg_epoch(self, n_epoch=-1):
        assert n_epoch <= self.exp["n_epoch"], f"max len: {len(self.exp['info_per_epoch'])}"
        
        if n_epoch != -1:
            return self.__agg_one_epoch(n_epoch)
        else: 
            return pd.concat([self.__agg_one_epoch(n_epoch) for n_epoch in range(self.exp["n_epoch"])]).fillna(0)
        

def __check_exp(exp):
    essential_fields = [
        "n_epoch",
        "info_per_workload",
        "info_per_epoch",
    ]
    return all(field in exp for field in essential_fields)
    
def read_from_dir(dir):
    exp_json = os.path.join(dir, "exp.json")
    exp_data = os.path.join(dir, "merged.csv")

    assert os.path.exists(exp_json) and os.path.exists(exp_data)
    df_total = pd.read_csv(exp_data, index_col=0)
    df_total.set_index('Time', inplace=True)
    
    with open(exp_json, 'r') as f:
        exp = json.load(f)

    assert __check_exp(exp)
    return ExpData(df_total, exp)

