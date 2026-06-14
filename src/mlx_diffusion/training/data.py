"""Minimal in-memory batching helpers for training.

For small/medium datasets that fit in unified memory. Larger pipelines can pass
any iterable of batches to ``DiffusionTrainer.fit``.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx


def batch_iterator(
    data: mx.array | tuple[mx.array, ...],
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> Iterator:
    """Yield mini-batches from one or more aligned arrays.

    A single array yields array batches; a tuple of arrays yields tuples of
    correspondingly-indexed batches (e.g. ``(images, labels)``).
    """
    arrays = data if isinstance(data, tuple) else (data,)
    n = arrays[0].shape[0]
    order = mx.array(list(range(n)))
    if shuffle:
        order = mx.random.permutation(n, key=mx.random.key(seed))

    stop = (n // batch_size) * batch_size if drop_last else n
    for start in range(0, stop, batch_size):
        idx = order[start : start + batch_size]
        batch = tuple(a[idx] for a in arrays)
        yield batch if isinstance(data, tuple) else batch[0]
