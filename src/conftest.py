import cv2
import numpy as np
import pytest


@pytest.fixture
def test_image() -> np.ndarray:
    return cv2.imread("images/test.jpg")


@pytest.fixture
def test_bw_image(test_image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) < 128
