import torch as th
import torch.nn as nn

from .beam import Beam


@th.no_grad()
def test_beam():
    embedding = nn.Embedding(64, 32)
    lstm = nn.LSTM(32, 64, 2, batch_first=True)

    beam = Beam(256, (th.zeros(2, 64), th.zeros(2, 64)))

    # Perform 8 rounds of beam search.
    for _ in range(8):
        input, hidden = beam.batch()
        output, hidden = lstm(embedding(input), hidden)
        log_probs = th.log_softmax(output, dim=-1)
        beam.update(log_probs, hidden)

    # Check sequences.
    sequence, log_prob = beam.most_probable()
    assert len(sequence) == 8
    # Better than chance.
    assert log_prob > -th.log(th.tensor(64, dtype=th.float)) * 8

    # Check other sequences.
    sequences, log_probs = beam.most_probable(4)
    assert sequences.shape == (4, 8)
    assert log_probs.shape == (4,)
    assert th.all(log_probs <= log_prob)
    assert th.all(sequences < 64)
