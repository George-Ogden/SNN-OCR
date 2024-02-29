import pytest
import torch

from .model import SpikingNetwork


@pytest.fixture
def model() -> SpikingNetwork:
    return SpikingNetwork(10, 2)


def test_forward(model: SpikingNetwork):
    x = torch.rand(5, 10)
    output_spk, _ = model(x)
    assert output_spk.shape == (SpikingNetwork.num_steps, 5, 2)
