import numpy as np

from .detection import detect_lines
from .utils import find_in_image


def test_detect_lines(test_bw_image: np.ndarray):
    lines = detect_lines(test_bw_image)
    # Check correct number of lines are detected.
    assert len(lines) == 8
    # Check that each line appears in the image.
    test_gray_image = test_bw_image.astype(np.uint8) * 255
    for line in lines:
        line = line.astype(np.uint8) * 255
        assert find_in_image(line, test_gray_image)
