import json
import os
import pandas as pd

    
def format_to_13_timestamp(ts):
    assert ts + 1 < 10000000000000
    return int(format(ts, '0<13d'))

def filter_row_timerange(start, end):
    return lambda x : x.loc[(x.index >= start) & (x.index <=end)]

def filter_column_useless(excol_prefix=(), std_min=1e-10, mean_min=0):
    def __filter_column_useless(df):
        cols_to_drop = set()
        std = df.std()
        mean = df.mean()

        cols_to_drop.update(std[std <= std_min].index)
        cols_to_drop.update(mean[mean <= mean_min].index)
        return df.drop(columns = [col for col in cols_to_drop if not col.startswith(excol_prefix)])
    return __filter_column_useless

# same columu may not change at all
# so it is important to set col prefix and choose those essential
def filter_row_noise(col_prefix, sparam=1.5, l=0.25, h=0.75):
    def __filter_row_noise(df):
        temp_df = filter_column_startswith(col_prefix)(df)
        q1 = temp_df.quantile(l)
        q3 = temp_df.quantile(h)
        iqr = q3 - q1
  
        outliers = ((temp_df < (q1 - sparam * iqr)) | (temp_df > (q3 + sparam * iqr))).any(axis=1)
        return df.drop(temp_df[outliers].index)
    
    return __filter_row_noise

def filter_column_startswith(col_prefix):
    return lambda x : x[[col for col in x.columns if col.startswith(col_prefix)]]

def filter_workload(workload_info, with_stress=False):
    assert "start_time" in workload_info or "start" in workload_info
    assert "end_time" in workload_info or "end" in workload_info
  
    start = format_to_13_timestamp(workload_info["start_time"] if "start_time" in workload_info else workload_info["start"])
    end = format_to_13_timestamp(workload_info["end_time"] if "end_time" in workload_info else workload_info["end"]) 
  
    if not with_stress:
        return filter_row_timerange(start, end)
        
    def __filter_workload(df):
        df = filter_row_timerange(start, end)(df)
        if "stress" in workload_info and workload_info["stress"] != {}:
            stress_data = {}
            for k, v in workload_info["stress"].items():
                for _k, _v in v.items():
                    stress_data[f"stress_{_k}"] = [int(_v) for i in range(df.shape[0])]
                    
            stress_df = pd.DataFrame(stress_data)
            stress_df.set_index(df.index, inplace=True)
            df = pd.concat([stress_df, df], axis=1)
        return df
    return __filter_workload

def stress_to_str(workload_info):
    stress = ""
    if "stress" in workload_info and workload_info["stress"] != {}:
        for k, v in workload_info["stress"].items():
            for _k, _v in v.items():
                stress = f"{k}_{_v}"
    return stress    

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

    def workload_keys(self):
        return list(self.exp["info_per_workload"].keys())
      
    def workloads_of(self, workload_name):
        return self.exp["info_per_workload"][workload_name]["info_per_epoch"]

    def workload_df(self, workload, with_stress=False, custom_df_funcs=None):
      df_funcs = [filter_workload(workload, with_stress)]
      if custom_df_funcs == None:
        df_funcs = df_funcs + self.workload_preprocess_funcs
      else:
        df_funcs = df_funcs + custom_df_funcs
      return apply_df_funcs(self.df, df_funcs)
      
    def set_workload_preprocess_funcs(self, df_funcs):
        self.workload_preprocess_funcs = df_funcs
        return self

    def set_workload_agg_funcs(self, df_funcs):
        self.workload_agg_funcs = df_funcs
        return self

    def agg_one_workload(self, workload, with_stress=False):
        df_funcs =  [filter_workload(workload, with_stress)] + self.workload_preprocess_funcs + self.workload_agg_funcs
        return apply_df_funcs(self.df, df_funcs)

    def __agg_one_epoch(self, n_epoch):
        epoch_info = self.exp["info_per_epoch"][n_epoch]
        
        dfs = []
        for k,v in epoch_info["workloads"].items():
            df = self.agg_one_workload(v)
            dfs.append(df)
        
        df = pd.concat(dfs).fillna(0).reset_index(drop=True)

        # assume stress is the same in one epoch
        if "stress" in epoch_info and epoch_info["stress"] != {}:
            stress_data = {}
            for k, v in epoch_info["stress"].items():
                for _k, _v in v.items():
                    stress_data[f"stress_{_k}"] = [int(_v) for i in range(df.shape[0])]
                    
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

    def one_column_on_stresses(self, column, df_key):
        df_dict = {}
        for workload_info in self.workloads_of(df_key):
            workload_df = self.workload_df(workload_info)
            workload_df = workload_df[column].reset_index(drop=True)
            df_dict[stress_to_str(workload_info)] = workload_df 
        
        return pd.DataFrame(df_dict)
      
    def one_column_on_workloads(self, column, n_epoch=0):
        df_dict = {}
        for workload_key in self.workload_keys():
            workload_info = self.workloads_of(workload_key)[n_epoch]
            df_dict[workload_key] =  self.workload_df(workload_info)[column].reset_index(drop=True)
        return pd.DataFrame(df_dict)

def agg_per_workload_stress(exp_data, no_stress_df_epoch, qos_column, stress=""):
    exp_data.set_workload_preprocess_funcs([
        filter_column_startswith(col_prefix=("stress", "vm", "app")),
        filter_column_useless(excol_prefix=("stress")),
        filter_row_noise(col_prefix=("app")),
    ])
    df_epoch = exp_data.agg_epoch()
    df_epoch_group = df_epoch.groupby(df_epoch.index)
    
    qos_percentage_dfs = []
    for group in df_epoch_group.groups:
        df_workload = df_epoch_group.get_group(group)
        if stress == "" or stress not in df_workload.columns:
            stress = df_workload.columns[0]
        
        no_stress = no_stress_df_epoch.loc[[group]]
        delta_df = df_workload[qos_column] - no_stress[qos_column]
        qos_percentage_df = delta_df / no_stress[qos_column]
        qos_percentage_df
        qos_percentage_df.index = [f"{stress.split('_', 1)[1]}_{i}"for i in df_workload[stress]]
        qos_percentage_dfs.append(qos_percentage_df)
    
    df = pd.concat(qos_percentage_dfs, axis=1)
    df.columns = list(df_epoch_group.groups.keys())
    return df

def __check_exp(exp):
    essential_fields = [
        "n_epoch",
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

defualt_workload_preprocess_funcs = [
    filter_column_startswith(col_prefix=("stress", "host", "vm", "app")),
    filter_column_useless(excol_prefix=("stress")),
    filter_row_noise(col_prefix=("app")),
]

defualt_workload_agg_funcs = [
    lambda x : x.mean().to_frame().T,
]
