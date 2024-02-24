from .image import Image


def test_detect_lines(test_bw_image: Image):
    lines = test_bw_image.detect_lines()

    # Check correct number of lines are detected.
    assert len(lines) == 8

    # Check that each line appears in the image.
    for line in lines:
        assert line in test_bw_image


def test_characters(test_bw_image: Image):
    lines = test_bw_image.detect_lines()
    characters = lines[0].detect_characters()

    # Check correct number of characters are detected.
    assert len(characters) == 32

    # Check that each character appears in the line and image.
    for character in characters:
        assert character in lines[0]
        assert character in test_bw_image
