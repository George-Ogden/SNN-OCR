from __future__ import annotations

import math
import subprocess
import tempfile
from typing import Any, List, Optional, Tuple, Union

from .position import Positionable

Self = "Self"

import cv2
import numpy as np


class Image(Positionable):
    def __init__(self, image: np.ndarray):
        if isinstance(image, Image):
            image = image.image
        self._image = image
        self._x = self._y = 0
        self._h, self._w, *_ = self._image.shape

    @property
    def image(self) -> np.ndarray:
        return self._image

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
        return [
            LineSegment(line.image, self, (line.x, line.y))
            for line in lines
            if line.h > 0 and line.w > 0
        ]

    @staticmethod
    def load(path: str, invert: bool = False) -> Image:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if invert:
            image = 255 - image
        return Image(image)

    def preprocess(self) -> Image:
        """Preprocess the image to make it easier to detect text."""
        # Use imagemagick's textcleaner.
        with tempfile.NamedTemporaryFile(suffix=".png") as input_image, tempfile.NamedTemporaryFile(
            suffix=".png"
        ) as output_image:
            cv2.imwrite(input_image.name, self.image)
            command = f"./scripts/textcleaner -g -u -e stretch -T -s 1 {input_image.name} {output_image.name}"
            subprocess.run(command, shell=True, check=True)
            image = cv2.imread(output_image.name, cv2.IMREAD_GRAYSCALE)

        # Increase resolution.
        image = cv2.resize(image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # Sharpen image.
        blur = cv2.GaussianBlur(image, (0, 0), 3)
        image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)

        return Image(image)

    def binarize(self) -> Image:
        """Convert the image to black and white."""
        _, image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = image == 0
        if np.count_nonzero(image) > np.count_nonzero(~image):
            image = ~image
        return Image(image)

    def save(self, path: str):
        cv2.imwrite(path, self.image)


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

    def trim(self) -> Self:
        """Trim the segment to remove any whitespace."""
        vertical_slices = self.split_vertically()
        horizontal_slices = self.split_horizontally()
        top_segment = vertical_slices[0]
        bottom_segment = vertical_slices[-1]
        left_segment = horizontal_slices[0]
        right_segment = horizontal_slices[-1]
        x1, y1, x2, y2 = left_segment.x1, top_segment.y1, right_segment.x2, bottom_segment.y2
        x, y = x1 + self.x1, y1 + self.y1
        return type(self)(self[y1:y2, x1:x2], self._parent, (x, y))


class LineSegment(Segment):
    def detect_characters(self) -> List[CharacterSegment]:
        """Takes in a BW image with True text on a False background and returns a list of cropped images of the same format that contain characters."""
        characters = [character.trim() for character in self.split_horizontally()]
        return [
            CharacterSegment(character.image, self, (character.x, character.y))
            for character in characters
            if character.h > 0 and character.w > 0
        ]


class CharacterSegment(Segment):
    def resize_pad(self, target_size: Tuple[int, int]) -> Self:
        """Resize the longest edge and then pad the character with zeros."""
        h, w = target_size
        if self.h > self.w:
            new_h = h
            new_w = math.ceil(self.w * h / self.h)
        else:
            new_w = w
            new_h = math.ceil(self.h * w / self.w)
        dtype = self.image.dtype
        resized_image = cv2.resize(self.image.astype(np.uint8), (new_w, new_h))

        left_padding = (w - new_w) // 2
        right_padding = w - new_w - left_padding
        top_padding = (h - new_h) // 2
        bottom_padding = h - new_h - top_padding
        return type(self)(
            np.pad(
                resized_image, ((top_padding, bottom_padding), (left_padding, right_padding))
            ).astype(dtype),
            self._parent,
            (self.x - left_padding, self.y - top_padding),
        )
