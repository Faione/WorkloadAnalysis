import os
import json
import pandas as pd

import sys
sys.path.append('./tools/experiment')
sys.path.append('./tools/client')

import experiment
import executor
import generator

import prometheus
import grafana
import config_parser

import yaml
import logging
import argparse


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# yaml range to range
def yrange_to_range(yrange):
    range_list = []
    if isinstance(yrange, list):
        range_list = yrange
    elif isinstance(yrange, dict) and "end" in yrange: 
        end = yrange["end"]
        start = 0
        step = 1
        f = None
        if "start" in yrange:
            start = yrange["start"]
        if "step" in yrange:
            step = yrange["step"]
        if "map" in yrange:
            f = eval(yrange["map"])
        range_list = list(range(start, end, step)) if f is None else [f(i) for i in range(start, end, step)]

    if len(range_list) == 0:
        logging.warning("yrange format invalid or empty")
    return range_list

def to_flag_val(val):
    if not isinstance(val, list) and not isinstance(val, dict):
        return [val]
    else:
        return yrange_to_range(val)

def exp_from_yaml(data):
    return experiment.Experiment(**data["raw"])
    
def workload_exec_from_yaml(data):
    wexe = executor.WorkloadExecutor(**data["raw"])

    if "flags" in data:
        for k,v in data["flags"].items():
            wexe.with_flag(k, to_flag_val(v))
            
    return wexe

def stress_exec_from_yaml(data):
    sexe = executor.StressExecutor(**data["raw"])

    if "flags" in data:
        for k,v in data["flags"].items():
            sexe.with_flag(k, to_flag_val(v))
            
    return sexe

def run_exp(exp_yaml):
    
    DEFAULT_OPT_INTERVAL = 60
    
    with open(exp_yaml, 'r') as f:
        file_data = f.read()    
        # cfg = yaml.load(file_data, yaml.FullLoader)
        cfg = yaml.load(file_data)
        
    if "default_opt_interval" in cfg:
        DEFAULT_OPT_INTERVAL = cfg["default_opt_interval"]
    
    if "opt_interval" not in cfg["workload_exec"]:
        cfg["workload_exec"]["raw"]["opt_interval"] = DEFAULT_OPT_INTERVAL
    
    if "stress_exec" in cfg and "opt_interval" not in cfg["stress_exec"]:
        cfg["stress_exec"]["raw"]["opt_interval"] = DEFAULT_OPT_INTERVAL

    exp = exp_from_yaml(cfg["experiment"])
    workload_exec = workload_exec_from_yaml(cfg["workload_exec"])
    stress_exec = None
    if "stress_exec" in cfg:
        stress_exec = stress_exec_from_yaml(cfg["stress_exec"])
        
    
    exp.run(stress_exec=stress_exec, workload_exec=workload_exec, interval=DEFAULT_OPT_INTERVAL)
    data_root = cfg["data_root"]
    
    dir_path = os.path.join(data_root, exp.dir_name())
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    exp_dict = exp.__dict__
    exp_dict["conf"] = exp_yaml
    with open(os.path.join(dir_path, cfg["experiment"]["save_file"]), 'w') as f:
        json.dump(exp_dict, f)

    ## Collet Data
    grafana_auth = cfg["collect"]["grafana"]["auth"]
    grafana_server = cfg["collect"]["grafana"]["server"]
    prom_server = cfg["collect"]["prometheus"]["server"]
    aio_db = cfg["collect"]["query"]["dashboard"]
    step = cfg["collect"]["query"]["step"]
    dash_boards = cfg["collect"]["query"]["dashboard"]
    
    pclient = prometheus.client(prom_server)
    gclient = grafana.client(grafana_server, grafana_auth)
    
    db_datas = [gclient.get_db(db) for db in dash_boards]
    assert len(db_datas) > 0, "no prometheus data collect"
    
    if "aio" in dash_boards:
        logging.warning("while aio dashboard is existing, only fetching aio data")
        dash_boards = ["aio"]
    
    targets = []
    if len(dash_boards) == 1 and dash_boards[0] == "aio":
        # assert it is aio
        targets = db_datas[0]["panels"][0]["targets"]
    else:
        for db_data in db_datas:
            targets = targets + config_parser.read_targets_from_json(db_data)
    
    df = pclient.targets_to_df(targets, exp.start_time, exp.end_time, step)
    print(df.info())
    df.to_csv(os.path.join(dir_path, cfg["collect"]["save_file"]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config', type=str, help='experiment config path')
    args = parser.parse_args()
    run_exp(args.config)