from typing import List

import numpy as np


def detect_lines(image: np.ndarray) -> List[np.ndarray]:
    """Takes in a BW image with True text on a False background and returns a list of cropped images of the same format that contain lines."""
    used_vertical_slices = np.any(image, axis=1)
    used_vertical_slices = np.pad(used_vertical_slices, (1, 1), mode="constant")
    diff = np.diff(used_vertical_slices.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [image[start : end + 1] for start, end in zip(starts, ends, strict=True)]
