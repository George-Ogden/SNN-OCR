import functools
import os
import random
import shlex
import subprocess
import tempfile
from typing import List

from tqdm import trange

data_root = "training_data"
base_fonts = [
    "Arial",
    "Courier New",
    "Times New Roman",
    "Georgia",
    "Trebuchet MS",
    "Verdana",
    "Tex Gyre Schola Regular",
    "Noto Sans",
    "Noto Serif",
    "Lato",
    "DejaVu Sans",
    "DejaVu Serif",
    "FreeSans",
    "Fira Code",
    "Nimbus Sans",
    "Quicksand",
    "Liberation Sans",
    "Courier",
    "URW Gothic",
    "Comic Neue",
    "Cantarell",
]


@functools.lru_cache
def get_fonts() -> List[str]:
    fonts = set()
    for font in base_fonts:
        result = subprocess.run(
            f"fc-match '{font}'", shell=True, check=True, capture_output=True, text=True
        )
        alternative_font = shlex.split(result.stdout)
        if len(alternative_font) > 1:
            fonts.add(alternative_font[-2])
    return list(fonts)


def generate_images(character: str, number: int):
    fonts = get_fonts()
    directory = os.path.join(data_root, character)
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w") as temp:
        temp.write(character)
        temp.flush()

        for i in trange(number, leave=False):
            command = f"text2image --text '{temp.name}' --outputbase '{directory}/{i:06}' --font '{random.choice(fonts)}' --rotate_image --distort_image --white_noise --blur --xsize 128 --ysize 128 --margin 10 --leading 0 --degrade_image --exposure {random.choice((-1, 0, 1))}"
            subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
            )
        subprocess.run(f"rm {directory}/*.box", shell=True, check=True)


if __name__ == "__main__":
    generate_images("a", 1000)
