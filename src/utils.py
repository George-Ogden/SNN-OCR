import cv2
import numpy as np


def find_in_image(
    needle: np.ndarray, haystack: np.ndarray, method: int = cv2.TM_SQDIFF_NORMED
) -> bool:
    """Check whether the needle image is contained in the haystack image.
    Modified from https://stackoverflow.com/a/15147009/1524913
    """
    if needle.dtype == bool:
        needle = needle.astype(np.uint8) * 255
    if haystack.dtype == bool:
        haystack = haystack.astype(np.uint8) * 255
    method = cv2.TM_SQDIFF_NORMED

    result = cv2.matchTemplate(needle, haystack, method)

    # Compute the minimum squared difference
    mn, _, mnLoc, _ = cv2.minMaxLoc(result)

    # Extract the coordinates of our best match
    MPx, MPy = mnLoc

    trows, tcols = needle.shape[:2]
    return np.allclose(mn, 0)
