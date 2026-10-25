import torch

from dqn.agent import DQNAgent
from dqn.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_lightweight(tmp_path):
    agent = DQNAgent((5, 32, 20), 12)
    path = tmp_path / "c.pt"
    save_checkpoint(path, agent, extra={"a": 1}, episodes=5)
    payload = torch.load(path, weights_only=False)
    assert "buffer" not in payload and "env_state" not in payload and "rng_state" not in payload
    assert payload["episodes"] == 5
    agent2 = DQNAgent((5, 32, 20), 12)
    load_checkpoint(path, agent2, resume=True)  # sem env
    assert agent2.steps == agent.steps
