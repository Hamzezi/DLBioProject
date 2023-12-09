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


def build_command(*, transformer_type, num_GOs, num_layers, nhead, ffw_dim, dropout, given_mask, dataset, n_shot, stop_epoch=20):
    exp_name = f"{transformer_type}_c{num_GOs}_l{num_layers}_h{nhead}_ffw{ffw_dim}_p{dropout}_m{given_mask}_d{dataset}_k{n_shot}"
    # make sure to leave a space at the end of each line for the next line
    cmd = (
        f"python run.py exp.name={exp_name} method=transformer "
        f"dataset={dataset} n_shot={n_shot} "
        f"method.{transformer_type}_args.given_masks={given_mask} "
        f"method.{transformer_type}_args.num_GOs={num_GOs} "
        f"method.{transformer_type}_args.num_layers={num_layers} "
        f"method.{transformer_type}_args.ffw_dim={ffw_dim} "
        f"method.{transformer_type}_args.nhead={nhead} method.transformer_type={transformer_type} method.{transformer_type}_args.dropout={dropout} "
        f"method.stop_epoch={stop_epoch}"
    )
    return cmd

def build_command_mp(*, transformer_type, num_GOs, num_layers, nhead, ffw_dim, dropout, given_mask, dataset, n_shot, stop_epoch=20):
    exp_name = f"{transformer_type}_c{num_GOs}_l{num_layers}_h{nhead}_ffw{ffw_dim}_p{dropout}_m{given_mask}_d{dataset}_k{n_shot}"
    # make sure to leave a space at the end of each line for the next line
    cmd = (
        f"python model_params.py exp.name={exp_name} method=transformer "
        f"dataset={dataset} n_shot={n_shot} "
        f"method.{transformer_type}_args.given_masks={given_mask} "
        f"method.{transformer_type}_args.num_GOs={num_GOs} "
        f"method.{transformer_type}_args.num_layers={num_layers} "
        f"method.{transformer_type}_args.ffw_dim={ffw_dim} "
        f"method.{transformer_type}_args.nhead={nhead} method.transformer_type={transformer_type} method.{transformer_type}_args.dropout={dropout} "
        f"method.stop_epoch={stop_epoch}"
    )
    return cmd


def main(multiprocessing_enabled, num_processes):

    transformer_type_values = ["transformer_encoder"] # briefly
    num_GOs_values = [1, 10, 20, 40] # important
    # num_layers_values = [1] # briefly
    # nhead_values = [2] # briefly if possible
    ffw_dim_values = [64, 128, 256] # important
    # dropout_values = [0.1] # default
    given_masks_values = [False, True] # important
    datasets_values = ["tabula_muris", "swissprot"] # important
    n_shot_values = [1, 5] # important

    param_combinations = list(itertools.product(transformer_type_values,\
                                                num_GOs_values,\
                                                num_layers_values,\
                                                nhead_values,\
                                                ffw_dim_values,\
                                                dropout_values,\
                                                given_masks_values,\
                                                datasets_values,\
                                                n_shot_values))
    cmds = []
    for params in param_combinations:
        transformer, num_GOs, num_layers, nhead, ffw_dim, dropout, given_masks, dataset, n_shot = params
        if num_GOs == 1 and not given_masks:
            continue
        if num_GOs > 1 and given_masks:
            continue
        if given_masks is True and dataset == "swissprot":
            continue
        kwargs = {
            "transformer_type": transformer,
            "num_GOs": num_GOs,
            "num_layers": num_layers,
            "nhead": nhead,
            "ffw_dim": ffw_dim,
            "dropout": dropout,
            "given_mask": given_masks,
            "dataset": dataset,
            "n_shot": n_shot,
            "stop_epoch": 60
        }
        cmd = build_command(**kwargs)
        cmds.append(cmd)
    print("Number of experiments:", len(cmds))
    
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