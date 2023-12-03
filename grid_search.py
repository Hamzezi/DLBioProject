import itertools
import subprocess

transformer_type_values = ["transformer_encoder", "transformer_decoder", "transformer"]
num_layers_values = [1, 2, 3]
nhead_values = [1, 2, 3]
dropout_values = [0, 0.1, 0.2]
dataset = "tabula_muris"

param_combinations = list(itertools.product(transformer_type_values,\
                                            num_layers_values,\
                                            nhead_values,\
                                            dropout_values))

for params in param_combinations:
    transformer, num_layers, nhead, dropout = params
    exp_name = f"{transformer}_{num_layers}_{nhead}_{dropout}"
    cmd = f"python run.py exp.name={exp_name} method=transformer method.transformer_type={transformer} {transformer}_args.num_layers={num_layers} {transformer}_args.nhead={nhead} {transformer}_args.dropout={dropout} dataset={dataset}"
    print(cmd)
    subprocess.call(cmd, shell=True)
    break
