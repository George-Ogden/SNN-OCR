import numpy as np

from .image import CharacterSegment, Image, LineSegment, Segment


def test_detect_lines(test_bw_image: Image):
    lines = test_bw_image.detect_lines()

    # Check correct number of lines are detected.
    assert len(lines) == 8

    # Check that each line appears in the image.
    for line in lines:
        assert isinstance(line, LineSegment)
        assert line in test_bw_image


def test_characters(test_bw_image: Image):
    lines = test_bw_image.detect_lines()
    characters = lines[0].detect_characters()

    # Check correct number of characters are detected.
    assert len(characters) == 32

    # Check that each character appears in the line and image.
    for character in characters:
        assert isinstance(character, CharacterSegment)
        assert character in lines[0]
        assert character in test_bw_image


def test_trim(test_bw_image: Image):
    segment = Segment(test_bw_image.image, test_bw_image)
    trimmed = segment.trim()

    # Check that the trimmed segment is smaller than the original.
    assert 0 < trimmed.h < segment.h
    assert 0 < trimmed.w < segment.w
    assert trimmed.y1 > segment.y1
    assert trimmed.y2 < segment.y2
    assert trimmed.x1 > segment.x1
    assert trimmed.x2 < segment.x2

    # Check that the trimmed segment is still a segment of the original image.
    assert trimmed in test_bw_image
    assert trimmed in segment

    # Check that there is no whitespace in the trimmed segment.
    assert np.any(trimmed.image[0])
    assert np.any(trimmed.image[-1])
    assert np.any(trimmed.image[:, 0])
    assert np.any(trimmed.image[:, -1])
