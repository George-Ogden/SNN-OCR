from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import numpy as np

from .image import CharacterSegment, Image, LineSegment
from .position import Positionable


class Block(Positionable):
    def __init__(self, lines: List[LineText]):
        self._h = self.aggregate_height([line.h for line in lines])
        self._w = self.aggregate_width([char.image.w for line in lines for char in line.stream])
        self._x, self._y = self.aggregate_position([(line.x1, line.y1) for line in lines])

        spaces = [top.y1 - bottom.y2 for bottom, top in itertools.pairwise(lines)]
        self._line_spacing = self.aggregate_spacing(spaces)
        spaces = [self.expected_spacing(space) for space in spaces]
        for line, space in zip(lines, [None] + spaces):
            line.stream[0]._spacing = Spacing(
                vertical=space, horizontal=line.expected_spaces(line.x1 - self.x)
            )

        self._chars = [char for line in lines for char in line.stream]

    @classmethod
    def from_lines(cls, lines: List[LineSegment]) -> List[Block]:
        lines = [LineText.from_line(line) for line in lines]
        groups = cls.group_lines(lines)
        return [cls(group) for group in groups]

    @classmethod
    def group_lines(cls, lines: List[LineText]) -> List[List[LineText]]:
        size_groups = cls.group_by_size(lines)
        position_groups = cls.group_by_position(size_groups)
        return position_groups

    @classmethod
    def group_by_size(cls, lines: List[LineText]) -> List[List[LineText]]:
        lines.sort(key=lambda line: line.h)
        groups = [[lines[0]]]
        for line in lines[1:]:
            if groups[-1][0].h * 0.8 < line.h:
                groups[-1].append(line)
            else:
                groups.append([line])
        return groups

    @classmethod
    def group_by_position(cls, line_groups: List[List[LineText]]) -> List[List[LineText]]:
        vertical_groups = cls.group_by_vertical_position(line_groups)
        groups = cls.group_by_left_position(vertical_groups)
        return groups

    @classmethod
    def group_by_vertical_position(cls, line_groups: List[List[LineText]]) -> List[List[LineText]]:
        groups: List[List[LineText]] = []
        for line_group in line_groups:
            line_group.sort(key=lambda line: line.y1)
            groups.append([line_group[0]])
            mean_height = line_group[0].h
            # Keep the lines in the same group if they are close together.
            for line in line_group[1:]:
                if line.y1 - groups[-1][-1].y2 < mean_height * 3:
                    mean_height = (mean_height * len(groups[-1]) + line.h) / (len(groups[-1]) + 1)
                    groups[-1].append(line)
                else:
                    mean_height = line.h
                    groups.append([line])
        return groups

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

    @classmethod
    def aggregate_height(cls, heights: List[int]) -> int:
        return np.median(heights)

    @classmethod
    def aggregate_width(cls, widths: List[int]) -> int:
        return np.median(widths)

    @classmethod
    def aggregate_spacing(cls, spacings: List[int]) -> int:
        return np.quantile(spacings, 0.25)

    @classmethod
    def aggregate_position(cls, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        return np.min(positions, axis=0)

    def expected_spacing(self, gap_height: float) -> float:
        gap_height -= self._line_spacing
        return np.maximum(gap_height / (self._h + self._line_spacing), 0.0)

    @property
    def stream(self) -> List[CharacterText]:
        return self._chars

    @property
    def spacing(self) -> float:
        return self._line_spacing


class LineText(Positionable):
    def __init__(self, chars: List[CharacterSegment], position: Tuple[int, int]):
        self._h = self.aggregate_height([char.h for char in chars])
        self._w = self.aggregate_width([char.w for char in chars])
        spaces = [right.x1 - left.x2 for left, right in itertools.pairwise(chars)]
        self._spacing = self.aggregate_spacing(spaces)
        self._x, self._y = position
        spaces = [Spacing()] + [Spacing(self.expected_spaces(space)) for space in spaces]
        self._chars = [CharacterText(char.image, space) for char, space in zip(chars, spaces)]

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
    def __init__(self, image: np.ndarray, spacing: Spacing):
        self._image = image
        self._spacing = spacing

    @property
    def spacing(self) -> Spacing:
        return self._spacing

    @property
    def image(self) -> Image:
        return Image(self._image)


class Spacing:
    def __init__(self, horizontal: Optional[float] = None, vertical: Optional[float] = None):
        self._vertical = vertical
        self._horizontal = horizontal

    @property
    def v(self) -> float:
        return self._vertical

    @property
    def h(self) -> float:
        return self._horizontal
