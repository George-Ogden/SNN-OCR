import pytest
import torch as th

from .model import LSTM, SNN


@pytest.fixture
def snn_model() -> SNN:
    return SNN((32, 48), 2)


@pytest.fixture
def model() -> LSTM:
    return LSTM(48)


def test_snn_forward(snn_model: SNN):
    x = th.rand(6, 1, 32, 48)
    output_spk, _ = snn_model(x)
    assert output_spk.shape == (SNN.num_steps, 6, 2)


def test_lstm_seq_forward(model: LSTM):
    x = th.randint(0, 48, size=(6, 32))
    output, hidden = model(x)
    assert output.shape == (6, 32, 48)


def test_lstm_step_forward(model: LSTM):
    x = th.randint(0, 48, size=(1, 32))
    hidden = model.c0, model.h0
    for i in range(6):
        output, hidden = model(x, hidden)
        x = th.argmax(output, dim=-1)
    assert x.shape == (1, 32)
