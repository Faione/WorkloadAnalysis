import sys
sys.path.append('./tools/client')

import os
import config_parser
import grafana
import json


config_root = "configs"
grafana_auth = "admin:admin@"
grafana_server = "10.208.130.243:3001"
grafana_dbs = {"host" : "host", "vm" : "vm", "app": "app"}

if __name__ == '__main__':
    gclient = grafana.client(grafana_server, grafana_auth)
    db_datas = [gclient.get_db(db) for db in grafana_dbs.keys()]
    
    # combine all targets from different grafana dashboard
    targets = []
    for db_data in db_datas:
        targets = targets + config_parser.read_targets_from_json(db_data)
    
    # using template to build a grafana config
    aio_tmp_json = "template.json"
    aio_tmp_path = os.path.join(config_root, aio_tmp_json)
    with open(aio_tmp_path, 'r') as f:
        data = json.load(f)
        try:
            data["panels"][0]["targets"] = targets
        except:
            print("invalid grafana format")
            sys.exit(1)
    
    # save all targets into all_in_one dashboard
    aio_json = "all_in_one.json"
    aio_path = os.path.join(config_root, aio_json)
    with open(aio_path, 'w') as f2:
        f2.truncate(0)
        json.dump(data, f2)
    
    # backup grafana configs
    i = 0
    for dbs in grafana_dbs.keys():
        db_name = grafana_dbs[dbs]
        db_data = db_datas[i]
        with open(os.path.join(config_root, db_name + "_backup.json"), 'w') as f:
            f.truncate(0)
            json.dump(db_data, f)
        i = i + 1
        
    # worknode
    # resctrl port 9001 -> 9900
    
    # cache size
    # cpufreq target
    # vm name
    # vm id    