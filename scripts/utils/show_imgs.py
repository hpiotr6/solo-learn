import os

from matplotlib import pyplot as plt


method_path = "results/barlow_1000_epochs"
experiments_paths = os.listdir(method_path)
result_paths = [
    os.path.join(method_path, e_path, "test_results") for e_path in experiments_paths
]
print(result_paths)
row_count = len(experiments_paths)
column_count = 3
# {experiment_name:result_p for experiment_name in experiments_paths}
experimetns_results = {}
for experiment_name in experiments_paths:
    experimetns_results[experiment_name] = [
        f
        for f in os.listdir(os.path.join(method_path, experiment_name, "test_results"))
        if f.endswith(".png")
    ]
    #  = os.listdir(
    #     os.path.join(method_path, experiments_paths)
    # )
print(experimetns_results)
fig, axs = plt.subplots(row_count, column_count, figsize=(20, 30), layout="constrained")

cols = ["representations_spectra.png", "early_exits.png", "early_exits_OOD.png"]
rows = experimetns_results.keys()

for ax, col in zip(axs[0], cols):
    ax.set_title(col)

for ax, row in zip(axs[:, 0], rows):
    ax.set_ylabel(row, rotation=0, size="large")
axs = axs.flatten()
for i, experiment_name in enumerate(experimetns_results.keys()):
    for j, result in enumerate(experimetns_results[experiment_name]):
        print(i, j)
        # print(i * column_count + j)
        image = plt.imread(
            os.path.join(method_path, experiment_name, "test_results", result)
        )
        axs[i * column_count + j].imshow(image)


fig.suptitle(method_path)
plt.savefig(f"{method_path}-grid.png")
# for ax, markevery in zip(axs.flat, cases):
#     ax.set_title(f"markevery={markevery}")
#     ax.plot(x, y, "o", ls="-", ms=4, markevery=markevery)
