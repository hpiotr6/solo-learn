from pathlib import Path
import base


root = "runs/09.30"
methods = ["barlow_twins", "simclr"]


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
        }
    },
    # "vgg19_bn": {
    #     "backbone": {
    #         "name": "vgg19_bn",
    #     },
    # },
}

date = Path(root).name
for method in methods:
    dashed_kw = {
        "config-path": "scripts/pretrain/cifar/",
        "config-name": f"{method}.yaml",
    }
    info = {
        "wandb.project": f"{date}_{method}",
        "method": f"{method}",
        "checkpoint.dir": f"trained_models/{date}",
    }
    for name, config in setups.items():
        flattened_setups = base.flatten_dict(config)
        for i, experiment in enumerate(
            base.generate_config_combinations(flattened_setups)
        ):
            exp_name = f"{method}_{name}_{i}"
            eq_kw = {**experiment, **info, **{"name": exp_name}}
            base.make(
                root,
                exp_name,
                {
                    **base.add_pref_suf(dashed_kw, "--", " "),
                    **base.add_pref_suf(eq_kw, prefix="+", sufix="="),
                },
            )
