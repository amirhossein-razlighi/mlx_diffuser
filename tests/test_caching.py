"""Tests for the First-Block Cache decision logic (no model needed)."""

from __future__ import annotations

import mlx.core as mx

from mlx_diffuser.caching import FirstBlockCache


def test_disabled_never_reuses():
    cache = FirstBlockCache(0.0)
    assert not cache.enabled
    h = mx.ones((1, 4, 8))
    cache.residual = mx.zeros((1, 4, 8))  # even with a residual present
    for _ in range(5):
        assert cache.should_reuse(h) is False
    assert cache.skipped == 0


def test_first_steps_recompute_until_residual_exists():
    cache = FirstBlockCache(0.1)
    h = mx.ones((1, 4, 8))
    # No residual yet -> must recompute.
    assert cache.should_reuse(h) is False


def test_reuses_when_change_small_then_recomputes_when_accumulated():
    cache = FirstBlockCache(0.1)
    base = mx.ones((1, 4, 8))
    assert cache.should_reuse(base) is False  # primes prev_first
    cache.residual = mx.zeros((1, 4, 8))  # simulate a stored full-step residual

    # Tiny per-step change keeps the accumulator under threshold -> reuse.
    nudged = base + 0.001
    assert cache.should_reuse(nudged) is True
    assert cache.skipped == 1

    # A large jump pushes the accumulator over threshold -> recompute + reset.
    big = nudged + 5.0
    assert cache.should_reuse(big) is False


def test_reset_clears_state():
    cache = FirstBlockCache(0.1)
    cache.should_reuse(mx.ones((1, 2, 2)))
    cache.residual = mx.zeros((1, 2, 2))
    cache.reset()
    assert cache.prev_first is None and cache.residual is None
    assert cache.steps == 0 and cache.skipped == 0
