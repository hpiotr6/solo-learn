import os
from pathlib import Path
import re


def get_slide(path):
    pattern = r".*/(.*?)\-.*"

    title = re.match(pattern, path).group(1).replace("_", " ").capitalize()
    return f"""
---
# {title}
![center w:900]({path})
"""


if __name__ == "__main__":
    plots_path = "results/10.03/plots"
    slides = "".join(
        [
            get_slide(os.path.join("plots", filename))
            for filename in os.listdir(plots_path)
        ]
    )
    print(slides)
