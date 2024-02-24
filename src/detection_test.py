import numpy as np

from .detection import detect_characters, detect_lines
from .utils import find_in_image


def test_detect_lines(test_bw_image: np.ndarray):
    lines = detect_lines(test_bw_image)

    # Check correct number of lines are detected.
    assert len(lines) == 8

    # Check that each line appears in the image.
    for line in lines:
        assert find_in_image(line, test_bw_image)


def test_characters(test_bw_image: np.ndarray):
    lines = detect_lines(test_bw_image)
    characters = detect_characters(lines[0])

    # Check correct number of characters are detected.
    assert len(characters) == 32

    # Check that each character appears in the line and image.
    for character in characters:
        assert find_in_image(character, lines[0])
        assert find_in_image(character, test_bw_image)
