import numpy as np

from .image import Image
from .text import LineText


def test_resize_pad(test_bw_image: Image):
    line = test_bw_image.detect_lines()[0]

    # Convert to text.
    text = LineText.from_line(line)

    # Check position is the same.
    assert text.position == line.xywh[:2]

    # Check character sizes are approximately correct.
    assert np.allclose(text.spacing, 3, atol=1)
    assert np.allclose(text.width, 13, atol=2)
    assert np.allclose(text.height, 18, atol=3)

    # Check character spacing is approximately correct.
    for char in text.chars[1:]:
        assert 0 <= char.spacing < 1.2
