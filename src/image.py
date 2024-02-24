from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np


class Image:
    def __init__(self, image: np.ndarray, parent: Optional[Image] = None):
        if isinstance(image, Image):
            image = image.image
        self._image = image
        if parent is None:
            self._parent = None
            self._x = self._y = 0
        else:
            if isinstance(parent, np.ndarray):
                parent = Image(parent)
            location = parent.locate(self.image)
            if location is None:
                self._parent = None
                self._x = self._y = 0
            else:
                self._parent = parent
                self._x, self._y = location
        self._h, self._w, *_ = self.image.shape

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    @property
    def w(self) -> int:
        return self._w

    @property
    def h(self) -> int:
        return self._h

    @property
    def xywh(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def x1y1x2y2(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def __repr__(self) -> str:
        return f"Image({self.image!r})"

    def locate(
        self, image: Union[np.ndarray, Image], method: int = cv2.TM_SQDIFF_NORMED
    ) -> Optional[Tuple[int, int]]:
        """Determine the location of the image in the current image.
        Returns x, y coordinates of the top-left corner of the image in the current image.
        Modified from https://stackoverflow.com/a/15147009/1524913
        """
        if isinstance(image, Image):
            image = image.image
        needle = image
        haystack = self.image
        if needle.dtype == bool:
            needle = needle.astype(np.uint8) * 255
        if haystack.dtype == bool:
            haystack = haystack.astype(np.uint8) * 255

        result = cv2.matchTemplate(needle, haystack, method)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)
        if min_val < 1e-5:
            return min_loc

    def __contains__(
        self, image: Union[np.ndarray, Image], method: int = cv2.TM_SQDIFF_NORMED
    ) -> bool:
        """Check whether the image is contained in the current image."""
        return self.locate(image, method) is not None

    def __eq__(self, other: Image) -> bool:
        return np.array_equal(self.image, other.image)

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.image[key]

    @property
    def T(self) -> Image:
        return Image(self.image.T)

    def detect_lines(self) -> List[Image]:
        """Takes in a BW image with True text on a False background and returns a list of cropped images of the same format that contain lines."""
        used_vertical_slices = np.any(self.image, axis=1)
        used_vertical_slices = np.pad(used_vertical_slices, (1, 1), mode="constant")
        diff = np.diff(used_vertical_slices.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return [Image(self[start : end + 1], self) for start, end in zip(starts, ends, strict=True)]

    def detect_characters(self) -> List[Image]:
        """Takes in a BW image with True text on a False background and returns a list of cropped images of the same format that contain characters."""
        transposed_characters = self.T.detect_lines()
        return [Image(character.T, self) for character in transposed_characters]
