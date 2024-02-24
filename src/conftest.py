import cv2
import pytest

from .image import Image


@pytest.fixture
def test_image() -> Image:
    return Image(cv2.imread("images/test.jpg"))


@pytest.fixture
def test_bw_image(test_image: Image) -> Image:
    return Image(cv2.cvtColor(test_image.image, cv2.COLOR_BGR2GRAY) < 128)
