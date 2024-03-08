import numpy as np

from .config import image_size
from .image import Image
from .text import Block, LineText


def test_linetext(test_bw_image: Image):
    line = test_bw_image.detect_lines()[0]

    # Convert to text.
    text = LineText.from_line(line)

    # Check position is the same.
    assert text.xywh[:2] == line.xywh[:2]

    # Check character sizes are approximately correct.
    assert np.isclose(text.spacing, 3, atol=1)
    assert np.isclose(text.w, 13, atol=2)
    assert np.isclose(text.h, 18, atol=3)

    # Check character spacing is approximately correct.
    for char in text.stream[1:]:
        assert 0 <= char.spacing.h < 1.2
        assert char.image.w == image_size[0]
        assert char.image.h == image_size[1]


def test_single_block(test_bw_image: Image):
    lines = test_bw_image.detect_lines()
    blocks = Block.from_lines(lines)

    # Check that the block contains all the lines.
    assert len(blocks) == 1
    (block,) = blocks
    assert len(block.stream) == np.sum([len(line.detect_characters()) for line in lines])
    assert np.count_nonzero([char.spacing.v is not None for char in block.stream]) == len(lines) - 1

    # Check other spacing properties.
    assert np.isclose(block.spacing, 15, atol=3)
    assert np.all([char.spacing.h is not None for char in block.stream])

    # Check that the block is positioned correctly.
    assert lines[0].x1 - 5 <= block.x1 <= lines[0].x1
    assert block.y1 == lines[0].y1

    # Check all images are resized.
    for char in block.stream:
        assert char.image.w == image_size[0]
        assert char.image.h == image_size[1]


def test_multiple_blocks(test_complex_bw_image: Image):
    lines = test_complex_bw_image.detect_lines()
    blocks = Block.from_lines(lines)

    # Check that the block contains all the lines.
    assert len(blocks) == 4

    blocks.sort(key=lambda block: block.y1)
    # Check number of characters is approximately correct.
    assert 10 < len(blocks[0].stream) < 20
    assert 20 < len(blocks[1].stream) < 30
    assert 5 < len(blocks[2].stream) < 15
    assert 100 < len(blocks[3].stream)

    # Check that there is a single line for the first blocks.
    for i in range(3):
        assert np.all([char.spacing.v is None for char in blocks[i].stream])

    # Check that there are 3 distinct sections and 9 distinct lines in the last block.
    spaces = np.array([char.spacing.v for char in blocks[3].stream if char.spacing.v is not None])
    assert len(spaces) == 8
    assert np.count_nonzero(spaces > 0.5) == 2
    assert np.all(
        [(np.isclose(space, 0.0, atol=0.2) or np.isclose(space, 1.0, atol=0.2)) for space in spaces]
    )

    # Check all images are resized.
    for block in blocks:
        for char in block.stream:
            assert char.image.w == image_size[0]
            assert char.image.h == image_size[1]
