from typing import Tuple


class Positionable:
    _x: int
    _y: int
    _w: int
    _h: int

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
