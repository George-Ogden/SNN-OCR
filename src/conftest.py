import pytest

from .config import classes, image_size
from .image import Image
from .model import SNN


@pytest.fixture
def test_image() -> Image:
    return Image.load("images/test.jpg")


@pytest.fixture
def test_bw_image(test_image: Image) -> Image:
    return Image(test_image.image < 128)


@pytest.fixture
def test_complex_image() -> Image:
    return Image.load("images/test.png")


@pytest.fixture
def test_complex_bw_image(test_complex_image: Image) -> Image:
    return Image(test_complex_image.image < 128)


@pytest.fixture
def image_model() -> SNN:
    return SNN(input_size=image_size, num_outputs=len(classes))
