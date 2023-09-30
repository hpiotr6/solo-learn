import os
import subprocess

# Define the directories containing your scripts
# script_directories = ["runs/09.23_analysis/simclr", "runs/09.23_analysis/barlow_twins"]
script_directories = ["runs/09.30"]

# Loop through each directory
for directory in script_directories:
    if os.path.exists(directory):
        # Get a list of all .sh files in the directory
        script_files = [f for f in os.listdir(directory) if f.endswith(".sh")]

        # Submit each script
        for script in script_files:
            script_path = os.path.join(directory, script)
            subprocess.run(["sbatch", script_path])
    else:
        print(f"Directory {directory} does not exist.")
