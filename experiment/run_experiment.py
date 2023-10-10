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

import yaml
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# yaml range to range
def yrange_to_range(yrange):
    if "start" in yrange and "end" in yrange and "step" in yrange:
        return range(yrange["start"], yrange["end"], yrange["step"])
    return yrange

def exp_from_yaml(data):
    exp = experiment.Experiment(**data["raw"])

    if "settings" in data:
        settings = data["settings"]
        if "stress" in settings:
            stress = settings["stress"]
            if "cpu" in stress:
                cpu = stress["cpu"]
                exp.with_cpu_stress(
                    cpu_range = yrange_to_range(cpu["cpu_range"]),
                    cpuload_range = yrange_to_range(cpu["cpuload_range"]),
                )
            if "mem" in stress:
                mem = stress["mem"]
                exp.with_mem_stress(
                    memrate_range = yrange_to_range(mem["memrate_range"]),
                    memrate_byte = mem["memrate_byte"],
                )
    
    return exp

def workload_from_yaml(data):
    workload = generator.Workload(**data["raw"])
    for flag, value in data["flags"].items():
        if flag == "-t":
            workload.with_flag(
                flag, yrange_to_range(value)
            )
        else:
            workload.with_flag(flag, [value])
            
    return workload
    
def workload_exec_from_yaml(data):
    return executor.WorkloadExecutor(**data["raw"])

def stress_exec_from_yaml(data):
    return executor.StressExecutor(**data["raw"])



if __name__ == '__main__':    
    DEFAULT_OPT_INTERVAL = 60
    with open("experiment.yaml", 'r') as f:
        file_data = f.read()    
        # cfg = yaml.load(file_data, yaml.FullLoader)
        cfg = yaml.load(file_data)
        
    if "default_opt_interval" in cfg:
        DEFAULT_OPT_INTERVAL = cfg["default_opt_interval"]
    
    workload = workload_from_yaml(cfg["workload"])
    cfg["workload_exec"]["raw"]["run_cmd"] = workload.build()
    
    if "opt_interval" not in cfg["workload_exec"]:
        cfg["workload_exec"]["raw"]["opt_interval"] = DEFAULT_OPT_INTERVAL
    
    if "opt_interval" not in cfg["stress_exec"]:
        cfg["stress_exec"]["raw"]["opt_interval"] = DEFAULT_OPT_INTERVAL

    exp = exp_from_yaml(cfg["experiment"])
    workload_exec = workload_exec_from_yaml(cfg["workload_exec"])
    stress_exec = stress_exec_from_yaml(cfg["stress_exec"])
    exp.run(stress_exec=stress_exec, workload_exec=workload_exec, interval=DEFAULT_OPT_INTERVAL)
    
    data_root = cfg["data_root"]
    
    dir_path = os.path.join(data_root, exp.dir_name())
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    with open(os.path.join(dir_path, cfg["experiment"]["save_file"]), 'w') as f:
        json.dump(exp.__dict__, f)
    
    with open(os.path.join(dir_path, cfg["workload_exec"]["save_file"]), 'w') as f:
        json.dump(workload_exec.info_per_workload, f)

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