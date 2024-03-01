import pytest
import torch

from .model import SpikingNetwork


@pytest.fixture
def model() -> SpikingNetwork:
    return SpikingNetwork((32, 48), 2)


def test_forward(model: SpikingNetwork):
    x = torch.rand(6, 1, 32, 48)
    output_spk, _ = model(x)
    assert output_spk.shape == (SpikingNetwork.num_steps, 6, 2)
