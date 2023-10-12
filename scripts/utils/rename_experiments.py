import os
import json

from omegaconf import DictConfig

add_no = lambda x: "no" if  x else ""
root = "trained_models/10.03/barlow_twins"

for dir in os.listdir(root):
    experiment_dir_path = os.path.join(root, dir)

    for file in os.listdir(experiment_dir_path):
        if file.endswith(".ckpt"):
            file_path = os.path.join(experiment_dir_path, file)
        elif file.endswith(".json"):
            json_file = os.path.join(experiment_dir_path, file)

    with open(json_file) as file:
        kwargs = json.load(file)

    cfg = DictConfig(kwargs)
    is_modified = cfg.backbone.modify_to_cifar
    are_skips = cfg.backbone.kwargs.model_config.skips
    name = f"resnet34_{add_no(not is_modified)}modify_{add_no(not are_skips)}skips"
    os.rename(experiment_dir_path, os.path.join(root, name))


    