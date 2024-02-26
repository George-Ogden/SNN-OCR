from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np


class Image:
    def __init__(self, image: np.ndarray):
        if isinstance(image, Image):
            image = image.image
        self._image = image
        self._x = self._y = 0
        self._h, self._w, *_ = self._image.shape

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
    def x1(self) -> int:
        return self._x

    @property
    def y1(self) -> int:
        return self._y

    @property
    def x2(self) -> int:
        return self._x + self._w

    @property
    def y2(self) -> int:
        return self._y + self._h

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
        return self.x1, self.y1, self.x2, self.y2

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

    def split_vertically(self) -> List[Segment]:
        """Takes in a BW image with True data on a False background and returns a list of segments of the same format that are separated by a full horizontal white space in the original image."""
        used_vertical_slices = np.any(self.image, axis=1)
        used_vertical_slices = np.pad(
            used_vertical_slices, (1, 1), mode="constant", constant_values=False
        )
        diff = np.diff(used_vertical_slices.astype(int))
        starts = sorted(np.where(diff == 1)[0])
        ends = sorted(np.where(diff == -1)[0])
        return [
            Segment(self[start:end], self, (0, start))
            for start, end in zip(starts, ends, strict=True)
        ]

    def split_horizontally(self) -> List[Segment]:
        """Takes in a BW image with True data on a False background and returns a list of segments of the same format that are separated by a full vertical white space in the original image."""
        return [
            Segment(segment.T.image, self, (segment.y, segment.x))
            for segment in self.T.split_vertically()
        ]

    def detect_lines(self) -> List[LineSegment]:
        """Takes in a BW image with True text on a False background and returns a list of cropped images of the same format that contain lines."""
        lines = [line.trim() for line in self.split_vertically()]
        return [LineSegment(line.image, self, (line.x, line.y)) for line in lines]


class Segment(Image):
    def __init__(self, image: np.ndarray, parent: Image, location: Tuple[int, int]):
        if isinstance(image, Image):
            image = image.image
        self._image = image

        if isinstance(parent, np.ndarray):
            parent = Image(parent)

        self._parent = parent
        self._x, self._y = location
        self._h, self._w, *_ = self._image.shape

    def trim(self) -> Segment:
        """Trim the segment to remove any whitespace."""
        vertical_slices = self.split_vertically()
        horizontal_slices = self.split_horizontally()
        top_segment = vertical_slices[0]
        bottom_segment = vertical_slices[-1]
        left_segment = horizontal_slices[0]
        right_segment = horizontal_slices[-1]
        x1, y1, x2, y2 = left_segment.x1, top_segment.y1, right_segment.x2, bottom_segment.y2
        return Segment(self[y1:y2, x1:x2], self._parent, (x1, y1))


class LineSegment(Segment):
    def detect_characters(self) -> List[CharacterSegment]:
        """Takes in a BW image with True text on a False background and returns a list of cropped images of the same format that contain characters."""
        characters = [character.trim() for character in self.split_horizontally()]
        return [
            CharacterSegment(character.image, self, (character.x, character.y))
            for character in characters
        ]


class CharacterSegment(Segment):
    ...
