config_dir=stress_test
for y in `ls conf/$config_dir`
do
python3 run_experiment.py -f conf/$config_dir/$y
sleep 120

# stat conf/$config_dir/$y
done