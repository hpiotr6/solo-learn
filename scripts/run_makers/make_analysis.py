import os
from pathlib import Path


def make(root, path):
    full_path = Path(root, *Path(path).parts[1:])
    os.makedirs(full_path.parent, exist_ok=True)

    with open(f"{full_path}.sh", "w") as rsh:
        rsh.write(
            f"""\
#!/bin/bash
#SBATCH -A plggenerativepw-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 5:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1

#!/bin/bash
export WANDB_API_KEY="47f70459596d37c22af356c72cbe0e8467c66e45"
module load Miniconda3/4.9.2
source activate /net/tscratch/people/plghpiotr/.conda/mgr_env

python3.10 -m solo.tunnel_project.tunnel_analysis --path {path}
    """
        )


def get_all_experiments_paths(path):
    dirs = ["barlow_twins", "simclr"]
    all_experiments = []
    for directory in dirs:
        experiments_path = [
            os.path.join(path, directory, exp)
            for exp in os.listdir(os.path.join(path, directory))
            # if "noskips" in exp
        ]
        all_experiments.extend(experiments_path)
    return all_experiments


if __name__ == "__main__":
    path = "trained_models/10.03"
    experiments_paths = get_all_experiments_paths(path)
    date = Path(path).name
    root = f"runs/{date}_analysis"
    for exp_p in experiments_paths:
        make(root, exp_p)
