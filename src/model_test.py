import pytest
import torch as th

from .config import image_size, num_characters
from .image import Image
from .model import LSTM, SNN


@pytest.fixture
def snn_model() -> SNN:
    return SNN((32, 28), 2)


@pytest.fixture
def lstm_model() -> LSTM:
    return LSTM(48)


def test_snn_forward(snn_model: SNN):
    x = th.rand(6, 1, 32, 28)
    output_spk, _ = snn_model(x)
    assert output_spk.shape == (SNN.num_steps, 6, 2)


def test_lstm_seq_forward(lstm_model: LSTM):
    x = th.randint(0, 48, size=(6, 32))
    output, hidden = lstm_model(x)
    assert output.shape == (6, 32, 48)


def test_lstm_step_forward(lstm_model: LSTM):
    x = th.randint(0, 48, size=(1, 32))
    hidden = lstm_model.c0, lstm_model.h0
    for i in range(6):
        output, hidden = lstm_model(x, hidden)
        x = th.argmax(output, dim=-1)
    assert x.shape == (1, 32)


def test_predict(image_model: SNN, test_bw_image: Image):
    characters = [
        character.resize_pad(image_size)
        for character in test_bw_image.detect_lines()[0].detect_characters()
    ]
    predictions = image_model.predict(characters)
    assert predictions.shape == (len(characters), num_characters)
    for prediction in predictions:
        assert th.isfinite(prediction).any()
