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

class Flags:
    def __init__(self, flag_base=""):
        self.flag_ranges = {}
        self.flag_base = flag_base

    def with_flag(self, flag, vranges):
        assert flag not in self.flag_ranges, "flag redefined"

        self.flag_ranges[flag] = vranges
        return self

    def flag_list(self):
        flags = [self.flag_base]
        for k, v in self.flag_ranges.items():
            temp_flags = []
            for val in v:
                sflag = f"{k} {val}"
                for flag in flags:
                    temp_flags.append(f"{flag} {sflag}")
            flags = temp_flags

        if flags[0] == self.flag_base:
            return []
        else:
            return flags

    def iter(self):
        flags = self.flag_list()
        for flag in flags:
            yield flag
            

    