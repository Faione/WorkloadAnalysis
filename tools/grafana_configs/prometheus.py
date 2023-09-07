import requests
import json
import re
import pandas as pd


query_range_api = "/api/v1/query_range"
time_col_name = "Time"

class client:
    server = ""
    
    def __init__(self, server):
        self.server = server
    
    def query_range(self, query, start, end, step):
        params = {
        "query": query,
        "start": start, 
        "end": end,
        "step": step,
        }

        query_range_uri = f"http://{self.server}{query_range_api}"
        
        text = requests.get(query_range_uri, params=params).text
        data = json.loads(text)
    
        prom_rlts = data["data"]["result"]
        
        assert len(prom_rlts) > 0 , f"empty result: {query}"
        
        return prom_rlts
        
    def target_to_df(self, target, start, end, step):
        query = gen_prom_query(target["expr"], step)
        raw_legend = target["legendFormat"]
        prom_rlts = self.query_range(query, start, end, step)
    
        return results_concat_to_df(prom_rlts, raw_legend)
    
    def targets_to_df(self, targets, start, end, step):
        dfs = [self.target_to_df(target, start, end, step) for target in targets]
        aio_df = pd.concat(dfs, axis=1)
        return aio_df

def gen_prom_query(expr, step):
  replace_funcs = [
    lambda x : x.replace("$__interval", step),
  ]

  for f in replace_funcs:
      expr = f(expr)
  return expr

def gen_legend(raw_legend, labels):
    pattern = r"\{\{([^{}]+)\}\}"
    matches = re.findall(pattern, raw_legend)

    for match in matches:
        if match in labels:
            raw_legend = raw_legend.replace(f"{{{{{match}}}}}", str(labels[match]))
    return raw_legend

def result_to_df(prom_rlt, raw_legend):
    values = prom_rlt["values"]
    legend = gen_legend(raw_legend, prom_rlt["metric"])

    df = pd.DataFrame(values)
    df.rename(columns={0: time_col_name, 1: legend}, inplace=True)
    df.set_index(time_col_name, inplace=True)
    return df

def results_concat_to_df(prom_rlts, raw_legend):
    return pd.concat([result_to_df(prom_rlt, raw_legend) for prom_rlt in prom_rlts], axis=1)
