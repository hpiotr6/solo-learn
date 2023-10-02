import subprocess
from pathlib import Path

directory_path = Path("runs/09.30_analysis")
sh_files = list(directory_path.rglob("*.sh"))
for script_path in sh_files:
    subprocess.run(["sbatch", script_path])
