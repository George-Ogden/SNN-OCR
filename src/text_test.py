import numpy as np

from .image import Image
from .text import Block, LineText


def test_linetext(test_bw_image: Image):
    line = test_bw_image.detect_lines()[0]

    # Convert to text.
    text = LineText.from_line(line)

    # Check position is the same.
    assert text.xywh[:2] == line.xywh[:2]

    # Check character sizes are approximately correct.
    assert np.allclose(text.spacing, 3, atol=1)
    assert np.allclose(text.w, 13, atol=2)
    assert np.allclose(text.h, 18, atol=3)

    # Check character spacing is approximately correct.
    for char in text.stream[1:]:
        assert 0 <= char.spacing.h < 1.2


def test_single_block(test_bw_image: Image):
    lines = test_bw_image.detect_lines()
    blocks = Block.from_lines(lines)

    # Check that the block contains all the lines.
    assert len(blocks) == 1
    (block,) = blocks
    assert len(block.stream) == np.sum([len(line.detect_characters()) for line in lines])
    assert np.count_nonzero([char.spacing.v is not None for char in block.stream]) == len(lines) - 1

    # Check other spacing properties.
    assert np.allclose(block.spacing, 15, atol=3)
    assert np.all([char.spacing.h is not None for char in block.stream])

    # Check that the block is positioned correctly.
    assert lines[0].x1 - 5 <= block.x1 <= lines[0].x1
    assert block.y1 == lines[0].y1
