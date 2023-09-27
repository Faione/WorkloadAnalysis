import os
import sys
import time
import experiment

def run_shell(cmd):
    log(f"run: {cmd}")
    if os.system(cmd) != 0:
        sys.exit(1)

class StressExecutor(experiment.Executor):    
    def __init__(self, opt_interval=0):
        self.name = "stress_executor"
        self.opt_interval = opt_interval

class WorkloadExecutor(experiment.Executor):    
    def __init__(self, opt_interval=0):
        self.name = "workload_executor"
        self.opt_interval = opt_interval
