from pathlib import Path

from omegaconf import DictConfig
import base


root = "runs/10.11"
# methods = ["barlow_twins", "simclr"]
methods = ["vicreg"]
# methods_inverse = {"barlow_twins": "simclr", "simclr": "barlow_twins"}


setups = {
    "resnet34": {
        "backbone": {
            "name": "resnet_custom",
            "modify_to_cifar": [True, False],
            "kwargs.model_config": {
                "backbone_type": "resnet34",
                "batchnorm_layers": True,
                "width_scale": 1,
                "skips": [False, True],
            },
        },
    },
    "vgg19bn": {
        "backbone": {
            "name": "vgg19_bn",
        },
    },
}

date = Path(root).name
for method in methods:
    dashed_kw = {
        "config-path": "scripts/pretrain/cifar/",
        "config-name": f"{method}.yaml",
    }
    info = {
        "wandb.project": f"{date}_{method}",
        # "method": f"{method}",
        "checkpoint.dir": f"trained_models/{date}",
    }
    for name, config in setups.items():
        flattened_setups = base.flatten_dict(config)
        for i, experiment in enumerate(
            base.generate_config_combinations(flattened_setups)
        ):
            if "resnet" in name:
                add_no = lambda x: "no" if x else ""
                is_modified = experiment["backbone.modify_to_cifar"]
                are_skips = experiment["backbone.kwargs.model_config.skips"]
                exp_name = f"{method}_resnet34_{add_no(not is_modified)}modify_{add_no(not are_skips)}skips"
            else:
                exp_name = f"{method}_{name}"
            eq_kw = {**experiment, **info, **{"name": exp_name}}
            base.make(
                root,
                exp_name,
                {
                    **base.add_pref_suf(dashed_kw, "--", " "),
                    **base.add_pref_suf(eq_kw, prefix="+", sufix="="),
                },
            )
