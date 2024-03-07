from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import numpy as np

from .image import CharacterSegment, Image, LineSegment
from .position import Positionable


class Block:
    def __init__(self, text: List[LineText], position: ...):
        ...


    @classmethod
    def group_by_left_position(cls, line_groups: List[List[LineText]]) -> List[List[LineText]]:
        groups: List[List[LineText]] = []
        for line_group in line_groups:
            line_group.sort(key=lambda line: line.y1)
            groups.append([line_group[0]])
            min_x = line_group[0].x1
            # Keep the lines in the indent is small.
            for line in line_group[1:]:
                if line.expected_spaces(line.x1 - min_x) < 10:
                    min_x = min(min_x, line.x1)
                    groups[-1].append(line)
        return groups


class LineText(Positionable):
    def __init__(self, chars: List[CharacterSegment], position: Tuple[int, int]):
        self._h = self.aggregate_height([char.h for char in chars])
        self._w = self.aggregate_width([char.w for char in chars])
        spaces = [right.x1 - left.x2 for left, right in itertools.pairwise(chars)]
        self._spacing = self.aggregate_spacing(spaces)
        self._x, self._y = position
        self._chars = [CharacterText(char.image) for char in chars]
        for char, spacing in zip(self._chars[1:], spaces, strict=True):
            char._spacing = self.expected_spaces(spacing)

    @classmethod
    def aggregate_height(cls, heights: List[int]) -> int:
        return np.quantile(heights, 0.25)

    @classmethod
    def aggregate_width(cls, widths: List[int]) -> int:
        return np.median(widths)

    @classmethod
    def aggregate_spacing(cls, spacings: List[int]) -> int:
        return np.quantile(spacings, 0.25)

    @classmethod
    def from_line(cls, line: LineSegment) -> LineText:
        return cls(
            chars=line.detect_characters(),
            position=(line.x1, line.y1),
        )

    @property
    def spacing(self) -> float:
        return self._spacing

    @property
    def stream(self) -> List[CharacterText]:
        return self._chars

    def expected_spaces(self, gap_width: float) -> float:
        if gap_width < self._spacing:
            return 0.0
        return gap_width / self._w


class CharacterText:
    def __init__(self, image: Image, spacing: Optional[float] = None):
        self._image = image
        self._spacing = spacing

    @property
    def spacing(self) -> float:
        return self._spacing

    @property
    def image(self) -> Image:
        return self._image
