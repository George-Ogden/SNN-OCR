from typing import Tuple

import snntorch as snn
import torch as th
import torch.nn as nn


# Define Network
class SpikingNetwork(nn.Module):
    num_steps = 50
    beta = 0.95

    def __init__(self, input_size: Tuple[int, int], num_outputs: int):
        super().__init__()

        # Initialize layers
        self.input_size = input_size
        self.num_outputs = num_outputs
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        with th.no_grad():
            hidden_size = self.conv(th.zeros(1, 1, *input_size)).shape[-1]

        self.lif1 = snn.Leaky(beta=self.beta)
        self.fc = nn.Linear(hidden_size, num_outputs)
        self.lif2 = snn.Leaky(beta=self.beta)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.conv(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return th.stack(spk2_rec, dim=0), th.stack(mem2_rec, dim=0)
