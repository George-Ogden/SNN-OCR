from typing import Optional, Tuple

import snntorch as snn
import torch as th
import torch.nn as nn


# Define Network
class SNN(nn.Module):
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


# LSTM
class LSTM(nn.Module):
    hidden_size = 1024
    num_layers = 3

    def __init__(self, vocab_size: int):
        super(LSTM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.decoder = nn.Linear(self.hidden_size, vocab_size)
        self.h0 = nn.Parameter(th.zeros(self.num_layers, 1, self.hidden_size))
        self.c0 = nn.Parameter(th.zeros(self.num_layers, 1, self.hidden_size))

    def forward(
        self, x: th.Tensor, hidden_state: Optional[Tuple[th.Tensor, th.Tensor]] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        if hidden_state is None:
            hidden_state = th.tile(self.h0, (1, x.shape[0], 1)), th.tile(
                self.c0, (1, x.shape[0], 1)
            )
        x = self.encoder(x)
        x, hidden_state = self.lstm(x, hidden_state)
        x = self.decoder(x)
        return x, hidden_state
