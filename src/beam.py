from typing import Optional, Tuple

import torch as th


class Beam:
    def __init__(
        self, beam_size: int, hidden_state: Tuple[th.Tensor, th.Tensor], pad_token: int = 0
    ):
        self.beam_size = beam_size
        self._hidden_state = tuple(state.unsqueeze(-2) for state in hidden_state)
        self._sequences = th.ones((1, 1), dtype=th.long) * pad_token
        self._log_probs = th.zeros(1, dtype=th.float)

    def update(self, delta_log_probs: th.Tensor, hidden: Tuple[th.Tensor, th.Tensor]):
        log_probs = th.unsqueeze(self._log_probs, -1) + th.squeeze(delta_log_probs, -2)
        top_log_probs, indices = th.topk(
            log_probs.flatten(), k=min(self.beam_size, delta_log_probs.numel())
        )
        indices = th.unravel_index(indices, log_probs.shape)
        self.update_hidden(hidden, indices)
        self.update_sequences(indices)
        self.update_log_probs(top_log_probs)

    def update_hidden(
        self, hidden: Tuple[th.Tensor, th.Tensor], indices: Tuple[th.Tensor, th.Tensor]
    ):
        self._hidden_state = (hidden[0][:, indices[0], :], hidden[1][:, indices[0], :])

    def update_sequences(self, indices: Tuple[th.Tensor, th.Tensor]):
        self._sequences = th.cat([self._sequences[indices[0], :], indices[1].unsqueeze(-1)], dim=-1)

    def update_log_probs(self, log_probs: th.Tensor):
        self._log_probs = log_probs

    def batch(self) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        return self._sequences[:, -1:], self._hidden_state

    def most_probable(self, k: Optional[int] = None) -> Tuple[th.Tensor, th.Tensor]:
        if k is None:
            return self._sequences[0, 1:], self._log_probs[0]
        return self._sequences[:k, 1:], self._log_probs[:k]
