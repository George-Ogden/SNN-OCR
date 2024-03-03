import functools
import os
import random
import shlex
import subprocess
import sys
import tempfile
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
    directory = os.path.join(data_root, f"{ord(character):03}")
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

            # Trim image.
            image = Image.load(f"{directory}/{i:06}.tif")
            char = CharacterSegment(image.image, image, (0, 0))
            cropped_char = char.trim()
            padded_char = cropped_char.resize_pad(image_size)
            padded_char.save(f"{directory}/{i:06}.png")
        subprocess.run(f"rm {directory}/*.box {directory}/*.tif", shell=True, check=True)


if __name__ == "__main__":
    for character in tqdm(
        r"\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    ):
        generate_images(character, num_samples)
