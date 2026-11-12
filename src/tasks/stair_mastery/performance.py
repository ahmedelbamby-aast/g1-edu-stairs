"""Opt-in equivalent history writes for the pinned mjlab backend."""

from importlib.metadata import version

import torch
from mjlab.utils.buffers.circular_buffer import CircularBuffer

ORIGINAL_APPEND = CircularBuffer.append


def branchless_append(self, data):
  if data.shape[0] != self._batch_size:
    raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")
  data = data.to(self._device)
  if self._buffer is None:
    self._pointer = -1
    self._buffer = torch.empty((self._max_len, *data.shape), dtype=data.dtype, device=self._device)
  self._pointer = (self._pointer + 1) % self._max_len
  self._buffer[self._pointer] = data
  if self._max_len > 1:
    first = (self._num_pushes == 0).view(1, self._batch_size, *((1,) * (data.ndim - 1)))
    # Same first-frame backfill as mjlab, without a device-to-host boolean or
    # a dynamically sized advanced-index operation on every control substep.
    torch.where(first, data.unsqueeze(0), self._buffer, out=self._buffer)
  self._num_pushes += 1


def set_branchless_history(enabled):
  if enabled and version("mjlab") != "1.2.0":
    raise RuntimeError("History optimization was verified against pinned mjlab 1.2.0 only")
  CircularBuffer.append = branchless_append if enabled else ORIGINAL_APPEND
