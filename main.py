import os

import torch as th

from src.config import classes, image_size, num_characters, save_directory
from src.image import Image
from src.model import LSTM, SNN
from src.text import Block

if __name__ == "__main__":
    # Load the image.
    image = Image.load("images/test.jpg")
    image = Image(image.image < 128)

    # Detect the lines.
    lines = image.detect_lines()

    # Group the lines into blocks.
    blocks = Block.from_lines(lines)

    # Load the models.
    language_model = LSTM(num_characters)
    image_model = SNN(image_size, len(classes))
    language_model.load_state_dict(th.load(os.path.join(save_directory, "lstm.pth")))
    image_model.load_state_dict(th.load(os.path.join(save_directory, "snn.pth")))

    # Predict the text.
    for block in blocks:
        text = block.to_str(language_model=language_model, image_model=image_model)
        print(text)
