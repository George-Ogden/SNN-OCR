import functools
import itertools
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from glob import glob
from typing import List

from tqdm import tqdm, trange

from config import data_root, num_samples

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import image_size
from src.image import CharacterSegment, Image

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
    directory = os.path.join(data_path, f"{ord(character):03}")
    os.makedirs(directory, exist_ok=True)

    if re.match(r"[a-zA-Z0-9]", character):
        # Copy across downloaded files.
        downloaded_images = list(
            itertools.chain(
                *(
                    glob(os.path.join(download_dir, character, "*.png"))
                    for download_dir in download_dirs
                )
            )
        )
    else:
        downloaded_images = []

    with tempfile.NamedTemporaryFile(mode="w") as temp:
        temp.write(character)
        temp.flush()

        for i in trange(max(number, len(downloaded_images)), leave=False):
            if i < len(downloaded_images):
                image = Image.load(downloaded_images[i], invert=True)
            else:
                command = f"text2image --text '{temp.name}' --outputbase '{directory}/{i:06}' --font '{random.choice(fonts)}' --rotate_image --distort_image --white_noise --blur --xsize 128 --ysize 128 --margin 10 --leading 0 --degrade_image --exposure {random.choice((-1, 0, 1))}"
                subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    capture_output=True,
                )
                image = Image.load(f"{directory}/{i:06}.tif")

            # Trim image.
            char = CharacterSegment(image.image, image, (0, 0))
            cropped_char = char.trim()
            padded_char = cropped_char.resize_pad(image_size)
            padded_char.save(f"{directory}/{i:06}.png")
        subprocess.run(f"rm {directory}/*.box {directory}/*.tif", shell=True, check=True)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), data_root)
    download_dirs = [os.path.join(data_path, f"{split}ing_data") for split in ("train", "test")]

    for character in tqdm(
        r"\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    ):
        generate_images(character, num_samples)

    for directory in download_dirs:
        shutil.rmtree(directory)
