
for y in experiment_no.yaml experiment_cache_same_cpu.yaml experiment_cache_same_numa.yaml experiment_io_diff_numa.yaml experiment_io_same_numa.yaml experiment_vm_same_cpu.yaml experiment_vm_same_numa.yaml experiment_net_same_numa.yaml experiment_cpu_same_cpu.yaml experiment_cpu_same_numa.yaml
do
python3 run_experiment.py -f conf/$y
sleep 120

# stat conf/$y
done