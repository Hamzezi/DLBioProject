import itertools
import subprocess

num_layers_values = [1, 2, 3]
nhead_values = [1, 2, 3]
dropout_values = [0, 0.1, 0.2]

param_combinations = list(itertools.product(num_layers_values, nhead_values, dropout_values))

for params in param_combinations:
    num_layers, nhead, dropout = params
    exp_name = f"transformer_decoder_{num_layers}_{nhead}_{dropout}"
    cmd = f"python run.py exp.name={exp_name} method=transformer method.transformer_type=transformer_decoder\
          transformer_args.num_layers={num_layers} transformer_args.nhead={nhead} transformer_args.dropout={dropout}"
    subprocess.run(cmd, shell=True)
