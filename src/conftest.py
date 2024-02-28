import pytest

from .image import Image


@pytest.fixture
def test_image() -> Image:
    return Image.load("images/test.jpg")


@pytest.fixture
def test_bw_image(test_image: Image) -> Image:
    return Image(test_image.image < 128)
