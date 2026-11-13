"""Teste de fumaça do engine vetorizado GPU/CPU."""
import torch

from arenas.seek_coin.arena.gpu_engine import GpuSeekCoin


def test_gpu_engine_cpu_smoke():
    engine = GpuSeekCoin(
        n_envs=4, n_chars=2, level=1, device="cpu", seed=0)
    assert engine.platforms.shape == (4, 8, 4)
    actions = torch.zeros((4, 2), dtype=torch.int64)
    out = engine.step(actions)
    assert out["coins"].shape == (4, 2)
    assert out["alive"].shape == (4, 2)
    for _ in range(20):
        out = engine.step(actions)
    assert torch.isfinite(engine.x).all()
    assert torch.isfinite(engine.y).all()
    obs = engine.observation()
    assert obs.shape == (4, 5, 20, 32)
    assert obs.dtype == torch.float32
