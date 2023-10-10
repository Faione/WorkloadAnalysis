import os
import sys
import time
import experiment
import logging

DEFAUTL_OPT_INTERVAL=60

def run_shell(cmd):
    if cmd == "":
        return
    logging.info(f"run: {cmd}")
    if os.system(cmd) != 0:
        sys.exit(1)

class DummyExecutor(experiment.Executor):
    def __init__(self, name="dummy", opt_interval=DEFAUTL_OPT_INTERVAL):
        super().__init__(name, opt_interval)
        self.opt_interval = opt_interval
        self.name = name

class StressExecutor(experiment.Executor):
    
    def __init__(self, workerdo="", run_cmd={}, stop_cmd="", name="stress_executor", opt_interval=DEFAUTL_OPT_INTERVAL):
        super().__init__(name, opt_interval)
        self.workerdo = workerdo
        self.run_cmd = run_cmd
        self.stop_cmd = stop_cmd
        self.stress_containers = []
    
    def do_exec(self, **kwargs):
        stress_containers = []
        for tp, val in kwargs.items():
            app_flag_str = " ".join([" ".join(["--" + str(k), str(v)]) for k, v in val.items()])
            cmd = " ".join([self.workerdo, self.run_cmd[tp], app_flag_str])
            run_shell(cmd)
            stress_containers.append(f"stress_{tp}")
        self.stress_containers = stress_containers
    
    def do_epilogue(self):
        for container in self.stress_containers:
            cmd = " ".join([self.workerdo, self.stop_cmd, container])
            run_shell(cmd)
            
class WorkloadExecutor(experiment.Executor):
    
    def __init__(self, run_cmd, warmup_cmd, name="workload_executor", opt_interval=DEFAUTL_OPT_INTERVAL):
        super().__init__(name, opt_interval)
        self.info_per_workload = {}
        for k,v in run_cmd.items():
            info  = {
                "run_cmd": v,
                "info_per_epoch": []
            }
            self.info_per_workload[k] = info
        self.warmup_cmd= warmup_cmd
        
    def workloads(self):
        assert len(self.info_per_workload) > 0, "no workload setted"
        return list(self.info_per_workload.keys())
    
    def do_exec(self, **kwargs):
        assert "workload" in kwargs, "no workload select"
        workload = kwargs["workload"]
        assert workload in self.info_per_workload, "invalid workload"
        
        info = {"addition":{}}
        if "stress" in kwargs:
            info["addition"]["stress"] = kwargs["stress"]
        
        run_shell(self.warmup_cmd)
        info["start_time"] = int(time.time()) 
        run_shell(self.info_per_workload[workload]["run_cmd"])
        info["end_time"] = int(time.time())
        
        
        self.info_per_workload[workload]["info_per_epoch"].append(info)
        

        
    
        
        
