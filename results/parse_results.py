import re, os
import numpy as np
import json


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
    
    assert len(parsed_data) == 3, f"{file_path}" # train, val (ablation), test (baseline comparison)

    return parsed_data


def get_dev_acc(parsed_data):
    return float(parsed_data[2]['accuracy'][:5])


def get_all_exp_dirs(ckpt_dir):
    subfolders = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, f))]
    return subfolders


def get_json_obj(parsed_data):
    return {
        "setting": parsed_data[0]['setting'],
        "model": parsed_data[0]['model'],
        "train_acc": float(parsed_data[0]['accuracy'][:5]),
        "val_acc": float(parsed_data[1]['accuracy'][:5]),
        "test_acc": float(parsed_data[2]['accuracy'][:5]),
    }


exp_dirs = get_all_exp_dirs("./checkpoints")
exp_dirs = [p for p in exp_dirs if "transformer" in p]
file_paths = [os.path.join(p, "results.txt") for p in exp_dirs]
# print(len(file_paths))
file_paths = [f for f in file_paths if os.path.exists(f)]
# print(len(file_paths))

dev_accs = {}

parsed_data_dict = {}

for f in file_paths:
    try:
        parsed_data = parse_results_file(f)
    # except AssertionError:
    #     continue
    except FileNotFoundError:
        continue
    f = os.path.normpath(f)
    f = f.split(os.sep)[1]
    parsed_data_dict[f] = get_json_obj(parsed_data)
    # if "tabula_muris" not in f: continue
    dev_accs[f] = get_dev_acc(parsed_data)


# save parsed data to a json file
with open("parsed_results.json", "w") as f:
    json.dump(parsed_data_dict, f, indent=4)

dev_accs = dict(sorted(dev_accs.items(), key=lambda item: item[1]))



def get_avg_agg(dev_accs, which):
    dev_accs = [v for k, v in dev_accs.items() if which(k)]
    return np.mean(dev_accs)


for e, a in dev_accs.items():
    print(e, a)
print("#"*10)

for c in [10, 20, 40]:
    hyp = f"c{c}"
    print(hyp, get_avg_agg(dev_accs, lambda k: hyp in k))
print("#"*10)

# # for l in [1, 2]:
# #     hyp = f"l{l}"
# #     print(hyp, get_avg_agg(dev_accs, lambda k: hyp in k))
# # print("#"*10)

# for h in [1, 2, 4, 8]:
#     hyp = f"h{h}"
#     print(hyp, get_avg_agg(dev_accs, lambda k: hyp in k))
# print("#"*10)

# print("p0", get_avg_agg(dev_accs, lambda k: "p0." not in k))
# for p in ["0.25", "0.5"]:
#     hyp = f"p{p}"
#     print(hyp, get_avg_agg(dev_accs, lambda k: hyp in k))
# print("#"*10)

# print("ffw64", get_avg_agg(dev_accs, lambda k: "ffw" not in k or "ffw64" in k))
for f in [64, 128, 256]:
    hyp = f"ffw{f}"
    print(hyp, get_avg_agg(dev_accs, lambda k: hyp in k))
