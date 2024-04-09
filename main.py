import argparse
import os

import torch as th

from src.config import classes, image_size, num_characters, save_directory
from src.image import Image
from src.model import LSTM, SNN
from src.text import Block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize text in an image.")
    parser.add_argument("filename", type=str, help="The filename of the image.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Load the image.
    filename = args.filename
    image = Image.load(filename)
    image = image.preprocess()
    image = image.binarize()

    # Detect the lines.
    lines = image.detect_lines()

    # Group the lines into blocks.
    blocks = Block.from_lines(lines)
    blocks.sort(key=lambda block: block.y1)

    # Load the models.
    language_model = LSTM(num_characters)
    image_model = SNN(image_size, len(classes))
    language_model.load_state_dict(
        th.load(os.path.join(save_directory, "lstm.pth"), map_location=th.device("cpu"))
    )
    image_model.load_state_dict(
        th.load(os.path.join(save_directory, "snn.pth"), map_location=th.device("cpu")),
        strict=False,
    )

    # Predict the text.
    for block in blocks:
        text = block.to_str(language_model=language_model, image_model=image_model)
        print(text)


if __name__ == "__main__":
    args = parse_args()
    main(args)
