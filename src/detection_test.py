import numpy as np

from .detection import detect_lines
from .utils import find_in_image


def test_detect_lines(test_bw_image: np.ndarray):
    lines = detect_lines(test_bw_image)
    # Check correct number of lines are detected.
    assert len(lines) == 8
    # Check that each line appears in the image.
    for line in lines:
        assert find_in_image(line, test_bw_image)
