import os
import sys
import time
import datetime

def log(info):
    print(f"[INFO] {info}")

date_format = "%Y%m%d%H%M%S"

# An Experiment is of three parts
# 1. init client and server (scripts needed)
# 2. run inference app as setting
# 3. run client workload as setting
# 4. save result and clear env(if needed)
class Experiment:
    # mode define the way to parse result from a client result
    mode = ""
    
    # start_time is the start time of the whole experiment
    start_time = 0
    
    # end_time is the end time of the whole experiment
    end_time = 0

    # stress_per_epoch save the stress setting of each epoch
    # epoch is always the same length of each stress
    # but in each epoch there can be many workload and apps
    # and int different epoch, workload and apps must be the same, the only change is stress
    stress_per_epoch = []
    workloads = []
    
    def __init__(self, mode, start_time=0, end_time=0, stress_per_epoch=[]):
        self.mode = mode
        self.start_time = start_time
        self.end_time = end_time
        self.stress_per_epoch = stress_per_epoch

    def from_dict(self, exp_dict):
        self.mode = exp_dict[""]

    def dir_name(self):
        stress = "no"
        if len(self.stress_per_epoch) != 0:
            stress = '_'.join(self.stress_per_epoch[0].keys())

        return f"{self.mode}_stress_{stress}_{str(self.start_time)}"

    def to_dict(self):
        return {
            "mode": self.mode,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "stress_per_epoch": self.stress_per_epoch,
            "workloads": self.workloads,
        }
            
    def __check_and_set_stress_infos(self, max_epoch):
        current_len = len(self.stress_per_epoch)
        assert current_len == 0 or current_len== max_epoch, f"miss match max_epoch: cur={current_len}, target={max_epoch}"
        if current_len == 0:
            self.stress_per_epoch = [{} for i in range(max_epoch)]
            


    def __enter__(self):
        self.start_time = datetime.datetime.fromtimestamp(int(time.time())).strftime(date_format)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.datetime.fromtimestamp(int(time.time())).strftime(date_format)

    def run(self, stress_exec, workload_exec, interval=60):
        assert len(self.workloads) > 0, "no workloads"

        log(f"{self.start_time} experiment start")

        for stress in self.stress_per_epoch:
            with stress_exec as se:
                se.exec(**stress)
                with workload_exec as we:
                    for workload in self.workloads:
                        we.exec(**{"mode": self.mode, "workload": workload})
            time.sleep(interval)
        
            
    def with_workload(self, workloads=[]):
        self.workloads = workloads
        return self
        
    def with_mem_stress(self, memrate_range, memrate_byte="1G"):
        self.__check_and_set_stress_infos(len(memrate_range))
        idx = 0
        for rate in memrate_range:
            self.stress_per_epoch[idx]["mem"] = {
                "memrate": rate,
                "memrate-bytes": memrate_byte,
            }
            idx = idx + 1
        return self

    def with_cpu_stress(self, cpu_range, cpuload_range):
        self.__check_and_set_stress_infos(len(cpu_range) * len(cpuload_range))

        idx = 0
        for cpu in cpu_range:
            for cpuload in cpuload_range:
                self.stress_per_epoch[idx]["cpu"] = {
                    "cpu": cpu,
                    "cpu-load": cpuload,
                }
                idx = idx + 1
        return self

class Executor:
    name = "executor"
    opt_interval = 0

    def __wait_opt_interval(self):
        if self.opt_interval != 0:
            time.sleep(self.opt_interval)    
            
    def __enter__(self):
        self.__wait_opt_interval()
        log(f"{self.name}: prologue")
        self.do_prologue()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__wait_opt_interval()
        log(f"{self.name}: epilogue")
        self.do_epilogue()

    def exec(self, **kwargs):
        self.__wait_opt_interval()    
        args = "" if not kwargs else kwargs
        log(f"{self.name}: exec {args}")
        return self.do_exec(**kwargs)

    def do_prologue(self):
        pass

    def do_epilogue(self):
        pass

    def do_exec(self, **kwargs):
        pass