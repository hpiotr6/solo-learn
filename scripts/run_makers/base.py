import itertools
import os
from pathlib import Path


def flatten_dict(d, parent_key="", separator="."):
    items = {}
    for k, v in d.items():
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, separator))
        else:
            items[new_key] = v
    return items


def generate_config_combinations(config):
    list_keys = [key for key, value in config.items() if isinstance(value, list)]
    combinations = [
        dict(zip(list_keys, values))
        for values in itertools.product(*[config[key] for key in list_keys])
    ]

    result_configs = []
    for combination in combinations:
        updated_config = config.copy()
        updated_config.update(combination)
        result_configs.append(updated_config)

    return result_configs


def add_pref_suf(input_dict, prefix="", sufix=""):
    output_dict = {}
    for key, value in input_dict.items():
        output_dict[f"{prefix}{key}{sufix}"] = value
    return output_dict


def make(root, name, kwargs):
    full_path = Path(root, name)
    os.makedirs(full_path.parent, exist_ok=True)

    with open(f"{full_path}.sh", "w") as rsh:
        rsh.write(
            f"""\
#!/bin/bash
#SBATCH -A plggenerativepw-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 48:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1

#!/bin/bash
export WANDB_API_KEY="47f70459596d37c22af356c72cbe0e8467c66e45"
module load Miniconda3/4.9.2
source activate /net/tscratch/people/plghpiotr/.conda/mgr_env

# python3.10 main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name barlow.yaml name=test_name
python3.10 main_pretrain.py {' '.join(f"{key}{value}" for key, value in kwargs.items())}
    """
        )


# def get_all_experiments_paths():
#     root = "trained_models_19_09"
#     dirs = ["barlow_twins", "simclr"]
#     all_experiments = []
#     for directory in dirs:
#         experiments_path = [
#             os.path.join(root, directory, exp)
#             for exp in os.listdir(os.path.join(root, directory))
#             if "noskips" in exp
#         ]
#         all_experiments.extend(experiments_path)
#     return all_experiments


# setups = {
#     "resnet34": {
#         "name": "resnet_custom",
#         "modify_to_cifat": [True, False],
#         "kwargs.model_config": {
#             "backbone_type": "resnet34",
#             "batchnorm_layers": True,
#             "width_scale": 1,
#             "skips": [False, True],
#         },
#     }
# }

# flattened_setups = flatten_dict(setups)
# combs = generate_config_combinations(flattened_setups)

# print(len(combs))
