import argparse
import itertools
import multiprocessing
import os
import subprocess
import sys

def run_command(cmd):
    os.system("echo " + cmd)
    # os.system("conda activate fewshotbench") # not sure if this is necessary
    os.system(cmd)

def main(multiprocessing_enabled, num_processes):

    transformer_type_values = ["transformer_encoder", "transformer_decoder", "transformer"]
    num_layers_values = [1, 2, 3]
    nhead_values = [2, 4, 8]
    dropout_values = [0, 0.25, 0.5]
    num_GOs_values = [10, 20, 40]
    dataset = "tabula_muris"

    param_combinations = list(itertools.product(transformer_type_values,\
                                                num_GOs_values,\
                                                num_layers_values,\
                                                nhead_values,\
                                                dropout_values))
    cmds = []
    for params in param_combinations:
        transformer, num_GOs, num_layers, nhead, dropout = params
        exp_name = f"{transformer}_c{num_GOs}_l{num_layers}_h{nhead}_p{dropout}"
        # make sure to leave a space at the end of each line for the next line
        cmd = (
            f"python run.py exp.name={exp_name} method=transformer "
            f"method.transformer_type={transformer} method.{transformer}_args.num_layers={num_layers} "
            f"method.{transformer}_args.num_GOs={num_GOs} method.{transformer}_args.nhead={nhead} "
            f"method.{transformer}_args.dropout={dropout} dataset={dataset} "
            f"method.stop_epoch=10"
        )
        cmds.append(cmd)
    
    if multiprocessing_enabled:
        num_processes = num_processes or (multiprocessing.cpu_count()-1)
        num_processes = min(num_processes, multiprocessing.cpu_count()-1)

        print(f"Using {num_processes} processes")

        with multiprocessing.Pool(num_processes) as pool:
            pool.map(run_command, cmds)
        pool.close()
        return
    
    for cmd in cmds:
        subprocess.call(cmd, shell=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run commands with or without multiprocessing.")
    parser.add_argument('--multiprocess', action='store_true', help='Enable multiprocessing')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use')
    args = parser.parse_args()

    main(args.multiprocess, args.num_processes)