"""Tests for schedulers: process math, sampling, and exact-recovery properties."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_diffuser.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerScheduler,
    load_scheduler,
    make_betas,
)
from mlx_diffuser.schedulers.ddpm import DDPMConfig
from mlx_diffuser.schedulers.flow_match_euler import FlowMatchConfig

SHAPE = (2, 4, 4, 3)


@pytest.mark.parametrize("schedule", ["linear", "scaled_linear", "cosine"])
def test_beta_schedules(schedule):
    betas = make_betas(schedule, 100)
    assert betas.shape == (100,)
    acp = mx.cumprod(1.0 - betas)
    # alpha_cumprod is non-increasing and stays in (0, 1].
    assert mx.all(acp[1:] <= acp[:-1] + 1e-6).item()
    assert acp[0].item() <= 1.0 and acp[-1].item() > 0.0


def test_ddpm_add_noise_endpoints():
    sch = DDPMScheduler()
    x0 = mx.random.normal(SHAPE)
    noise = mx.random.normal(SHAPE)
    # t=0 is almost pure signal.
    xt0 = sch.add_noise(x0, noise, mx.array([0, 0]))
    assert mx.max(mx.abs(xt0 - x0)).item() < 0.05
    # Large t is much noisier than small t.
    far = sch.add_noise(x0, noise, mx.array([999, 999]))
    assert mx.mean(mx.abs(far - x0)).item() > mx.mean(mx.abs(xt0 - x0)).item()


def test_get_target_epsilon_and_v():
    x0 = mx.random.normal(SHAPE)
    noise = mx.random.normal(SHAPE)
    t = mx.array([10, 500])
    eps_sched = DDPMScheduler(DDPMConfig(prediction_type="epsilon"))
    assert mx.allclose(eps_sched.get_target(x0, noise, t), noise)
    v_sched = DDPMScheduler(DDPMConfig(prediction_type="v_prediction"))
    v = v_sched.get_target(x0, noise, t)
    assert v.shape == x0.shape


def test_ddim_exact_recovery():
    """Feeding the true epsilon through deterministic DDIM recovers x0."""
    sch = DDIMScheduler()
    sch.set_timesteps(50)
    x0 = mx.random.normal(SHAPE)
    noise = mx.random.normal(SHAPE)
    t0 = sch.timesteps[0]
    sample = sch.add_noise(x0, noise, mx.array([int(t0.item())] * SHAPE[0]))
    for t in sch.timesteps:
        sample = sch.step(noise, t, sample)  # model "predicts" true epsilon
    assert mx.max(mx.abs(sample - x0)).item() < 1e-3


def test_flow_match_endpoints_and_recovery():
    sch = FlowMatchEulerScheduler()
    x0 = mx.random.normal(SHAPE)
    noise = mx.random.normal(SHAPE)
    # Path endpoints.
    assert mx.allclose(sch.add_noise(x0, noise, mx.zeros((SHAPE[0],))), x0)
    assert mx.allclose(sch.add_noise(x0, noise, mx.ones((SHAPE[0],))), noise)
    assert mx.allclose(sch.get_target(x0, noise, mx.array([0.3, 0.7])), noise - x0)
    # Integrating the true velocity from sigma=1 back to 0 recovers x0 exactly.
    sch.set_timesteps(20)
    velocity = noise - x0
    sample = noise
    for t in sch.timesteps:
        sample = sch.step(velocity, t, sample)
    assert mx.max(mx.abs(sample - x0)).item() < 1e-4


def test_ddpm_sampling_runs_and_finite():
    sch = DDPMScheduler()
    sch.set_timesteps(10)
    sample = mx.random.normal(SHAPE)
    for t in sch.timesteps:
        sample = sch.step(mx.zeros(SHAPE), t, sample, key=mx.random.key(0))
    assert bool(mx.all(mx.isfinite(sample)).item())


def test_euler_sampling_runs():
    sch = EulerDiscreteScheduler()
    sch.set_timesteps(10)
    assert sch.sigmas.shape == (11,)
    sample = mx.random.normal(SHAPE) * sch.sigmas[0]
    for t in sch.timesteps:
        scaled = sch.scale_model_input(sample, t)
        assert scaled.shape == sample.shape
        sample = sch.step(mx.zeros(SHAPE), t, sample)
    assert bool(mx.all(mx.isfinite(sample)).item())


def test_scheduler_can_start_partway_through_schedule():
    sch = EulerDiscreteScheduler()
    sch.set_timesteps(10)
    sch.set_begin_index(6)
    assert sch._step_index == 6
    with pytest.raises(ValueError, match="begin_index"):
        sch.set_begin_index(10)


def test_sample_timesteps_shapes():
    key = mx.random.key(0)
    assert DDPMScheduler().sample_timesteps(8, key).shape == (8,)
    fm = FlowMatchEulerScheduler(FlowMatchConfig(shift=3.0)).sample_timesteps(8, key)
    assert fm.shape == (8,)
    assert bool(mx.all((fm >= 0) & (fm <= 1)).item())


def test_scheduler_save_load_roundtrip(tmp_path):
    sch = DDIMScheduler()
    sch.save_pretrained(tmp_path)
    loaded = load_scheduler(tmp_path)
    assert isinstance(loaded, DDIMScheduler)
    assert loaded.config.num_train_timesteps == sch.config.num_train_timesteps
