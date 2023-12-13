import re, os
import json
import argparse


def parse_results_file(file_path):
    assert file_path.endswith("results.txt"), f"Unrecognized file: {file_path}"
    with open(file_path, 'r') as file:
        lines = file.readlines()

    parsed_data = []

    for line in lines:
        match = re.match(r'Time: (.*), Setting: (.*), Acc: (.*), Model: (.*)', line)
        assert match is not None, f"invalid results file {file_path}"
        time = match.group(1)
        setting = match.group(2)
        accuracy = match.group(3)
        model = match.group(4)

        parsed_data.append({
            'time': time,
            'setting': setting,
            'accuracy': accuracy,
            'model': model
        })
    
    assert len(parsed_data) == 3, f"wrong results.txt structure for {file_path}" # train, val (ablation), test (baseline comparison)

    return parsed_data


def get_all_exp_dirs(ckpt_dir):
    subfolders = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, f))]
    return subfolders


def get_json_obj(parsed_data):
    return {
        "setting": parsed_data[0]['setting'],
        "model": parsed_data[0]['model'],
        "train_acc": parsed_data[0]['accuracy'],
        "val_acc": parsed_data[1]['accuracy'],
        "test_acc": parsed_data[2]['accuracy'],
    }


def main(ckpt_dir, save_filename):
    # get all the results.txt paths
    exp_dirs = get_all_exp_dirs(ckpt_dir)
    exp_dirs = [p for p in exp_dirs if "comet" in p]
    file_paths = [os.path.join(p, "results.txt") for p in exp_dirs]
    file_paths = [f for f in file_paths if os.path.exists(f)]

    parsed_data_dict = {}

    # save results as dictionaries
    for f in file_paths:
        parsed_data = parse_results_file(f)
        f = os.path.normpath(f)
        f = f.split(os.sep)[-2]
        parsed_data_dict[f] = get_json_obj(parsed_data)

    # save parsed data to a json file
    with open(save_filename, "w") as f:
        json.dump(parsed_data_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default="./checkpoints", help='root checkpoint directory containing each exp.name folder')
    parser.add_argument('--save_filename', type=str, default="parsed_results.json", help='filename to save parsed results to')
    args = parser.parse_args()
    main(args.ckpt_dir)
