from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import numpy as np

from .image import CharacterSegment, Image, LineSegment


class Block:
    def __init__(self, text: List[LineText], position: ...):
        ...


class LineText:
    def __init__(self, chars: List[CharacterSegment], position: Tuple[int, int]):
        self.height = self.aggregate_height([char.h for char in chars])
        self.width = self.aggregate_width([char.w for char in chars])
        spaces = [right.x1 - left.x2 for left, right in itertools.pairwise(chars)]
        self.spacing = self.aggregate_spacing(spaces)
        self.position = position
        self.left, self.top = position
        self.chars = [CharacterText(char.image) for char in chars]
        for char, spacing in zip(self.chars[1:], spaces, strict=True):
            char.spacing = np.maximum(
                (spacing - self.spacing) / self.width,
                0,
            )

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
    def stream(self) -> List[CharacterText]:
        return self.chars


class CharacterText:
    def __init__(self, image: Image, spacing: Optional[float] = None):
        self.image = image
        self.spacing = spacing
