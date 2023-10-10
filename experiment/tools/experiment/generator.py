class Workload:
    def __init__(self, base_name = "", base_cmd=""):
        self.base_name = base_name
        self.base_cmd = base_cmd
        self.flags = {}
        
    def with_flag(self, flag, val_range):
        assert flag not in self.flags, "flag redefined"
        
        self.flags[flag] = val_range
        return self
    
    def build(self):
        run_cmd = [self.base_cmd]
        for k, v in self.flags.items():
            
            tmp_cmd = []
            for arg in v:
                flag_str = f"{k} {arg}"
                for cmd in run_cmd:
                    tmp_cmd.append(f"{cmd} {flag_str}")
                    
            run_cmd = tmp_cmd
        
        run_cmd_dict = {}
        for i, cmd in enumerate(run_cmd):
            run_cmd_dict[f"{self.base_name}_{i}"] = cmd
            
        return run_cmd_dict