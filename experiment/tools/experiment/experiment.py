import os
import sys
import time
import datetime
import logging

DATE_FORMAT_TIMESTAMP = "timestamp"
DEFAULT_DATE_FORMAT = "%Y%m%d%H%M%S" 

# An Experiment is of three parts
# 1. init client and server (scripts needed)
# 2. run inference app as setting
# 3. run client workload as setting
# 4. save result and clear env(if needed)
class Experiment:
    def __init__(self, mode="", start_time=0, end_time=0, n_epoch=0, setting_per_epoch=[], date_format=DEFAULT_DATE_FORMAT):
        # mode define the way to parse result from a client result
        self.mode = mode

        self.date_format = date_format
        
        # start_time is the start time of the whole experiment
        self.start_time = start_time

        # end_time is the end time of the whole experiment
        self.end_time = end_time

        self.total_time = 0

        # stress_per_epoch save the stress setting of each epoch
        # epoch is always the same length of each stress
        # but in each epoch there can be many workload and apps
        # and int different epoch, workload and apps must be the same, the only change is stress
        self.setting_per_epoch = setting_per_epoch
        
        self.n_epoch = n_epoch
        self.__check_and_settings_epoch(n_epoch)

    def __stress_per_epoch(self):
        return [settings["stress"] for settings in self.setting_per_epoch]
        
    def __stress_setting(self):
        stress_per_epoch = self.__stress_per_epoch()

        stress = "no"
        if len(stress_per_epoch) != 0 and stress_per_epoch[0] != {}:
            stress = '_'.join(stress_per_epoch[0].keys())

        return stress

    # multi-stress must has the same epoch setting
    def __check_and_settings_epoch(self, max_epoch):
        n_epoch = len(self.setting_per_epoch)
        assert n_epoch == 0 or n_epoch == max_epoch, f"miss match max_epoch: cur={n_epoch}, target={max_epoch}"
        
        if n_epoch == 0:
            self.setting_per_epoch = [{"stress": {}} for i in range(max_epoch)]
    
    def dir_name(self):
        # stress_setting
        stress = self.__stress_setting()

        return f"{self.mode}_stress_{stress}_{str(self.start(DEFAULT_DATE_FORMAT))}"

            
    def run(self, stress_exec, workload_exec, interval=60):
        self.workloads = workload_exec.workloads()
        assert len(self.workloads) > 0, "no workloads"

        self.start_time = int(time.time())
        logging.info(f"{self.start_time} experiment start")
        for setting in self.setting_per_epoch:
            with stress_exec as se: 
                se.exec(**setting["stress"])
                
                with workload_exec as we:
                    for workload in self.workloads:
                        we.exec(**{"mode": self.mode, "workload": workload, "stress": setting["stress"]})
                        
            time.sleep(interval)
        self.end_time = int(time.time())
        self.total_time = self.end_time - self.start_time
        self.n_epoch = len(self.setting_per_epoch)
        
        logging.info(f"{self.start()} experiment end")
            
    def with_workload(self, workloads=[]):
        self.workloads = workloads
        return self

    def with_epoch(self, max_epoch):
        self.__check_and_settings_epoch(max_epoch)
        return self
    
    def with_mem_stress(self, memrate_range, memrate_byte="1G"):
        self.__check_and_settings_epoch(len(memrate_range))
        
        idx = 0
        for rate in memrate_range:
            self.setting_per_epoch[idx]["stress"]["mem"] = {
                "memrate": rate,
                "memrate-bytes": memrate_byte,
            }
            idx = idx + 1
        return self

    def start(self, date_format=""):
        if date_format == "":
            date_format = self.date_format

        if date_format == DATE_FORMAT_TIMESTAMP:
            return self.start_time

        return datetime.datetime.fromtimestamp(self.start_time).strftime(date_format)
        
    def end(self, date_format=""):
        if date_format == "":
            date_format = self.date_format

        if date_format == DATE_FORMAT_TIMESTAMP:
            return self.start_time

        return datetime.datetime.fromtimestamp(self.end_time).strftime(date_format)
        
    def with_cpu_stress(self, cpu_range, cpuload_range):
        self.__check_and_settings_epoch(len(cpu_range) * len(cpuload_range))

        idx = 0
        for cpu in cpu_range:
            for cpuload in cpuload_range:
                self.setting_per_epoch[idx]["stress"]["cpu"] = {
                    "cpu": cpu,
                    "cpu-load": cpuload,
                }
                idx = idx + 1
        return self

class Executor:
    
    def __init__(self, name="executor", opt_interval=0):
        self.name = name
        self.opt_interval = opt_interval

    def __wait_opt_interval(self):
        if self.opt_interval != 0:
            time.sleep(self.opt_interval)    
            
    def __enter__(self):
        self.__wait_opt_interval()
        logging.debug(f"{self.name}: prologue")
        self.do_prologue()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__wait_opt_interval()
        logging.debug(f"{self.name}: epilogue")
        self.do_epilogue()

    def exec(self, **kwargs):
        self.__wait_opt_interval()    
        args = "" if not kwargs else kwargs
        logging.debug(f"{self.name}: exec {args}")
        return self.do_exec(**kwargs)

    def do_prologue(self):
        pass

    def do_epilogue(self):
        pass

    def do_exec(self, **kwargs):
        pass