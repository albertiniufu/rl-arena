"""Testes do simulador econômico e do ambiente Gymnasium."""
import numpy as np
import pytest

from games.economy.config import EconomyConfig
from games.economy.env import EconomyEnv
from games.economy.model import Economy


def _small_config(**overrides):
    values = dict(n_customers=40, n_firms=3, products_per_firm=2,
                  max_steps=5, seed=7)
    values.update(overrides)
    return EconomyConfig(**values)


def test_reset_returns_metrics_and_reproducible():
    a = Economy(_small_config())
    b = Economy(_small_config())
    for _ in range(10):
        ma = a.step()
        mb = b.step()
    assert ma == pytest.approx(mb)


def test_invariants_after_many_steps():
    economy = Economy(_small_config(n_customers=60, n_firms=4))
    for _ in range(30):
        m = economy.step()
        assert 0.0 <= m["unemployment"] <= 1.0
        assert m["price_index"] > 0.0
        assert np.isfinite(m["gdp"])
        assert np.isfinite(m["firm_cash"])
        for firm in economy.firms:
            for product in firm.products:
                assert product.price > 0.0
                assert product.inventory >= -1e-9
                assert product.quality <= 1.0


def test_government_budget_accounting():
    economy = Economy(_small_config())
    for _ in range(5):
        economy.step()
    before = economy.government_funds
    m = economy.step()
    expected = (m["tax_revenue"] - m["government_spending"]
                - m["transfers"] - m["interest_paid"])
    assert m["government_funds"] - before == pytest.approx(expected)
    assert m["government_debt"] == pytest.approx(
        max(0.0, -m["government_funds"]))


def test_policy_values_are_clamped():
    economy = Economy(_small_config())
    economy.set_policy(income_tax=2.0, spending_ratio=-1.0, interest_rate=9.0)
    assert economy.income_tax == pytest.approx(0.6)
    assert economy.spending_ratio == pytest.approx(0.05)
    assert economy.interest_rate == pytest.approx(0.05)


def test_income_tax_affects_net_income():
    low = Economy(_small_config(seed=3))
    high = Economy(_small_config(seed=3))
    low.set_policy(income_tax=0.1)
    high.set_policy(income_tax=0.5)
    low.step()
    high.step()
    low_net = np.mean([c.net_income for c in low.customers if c.employed])
    high_net = np.mean([c.net_income for c in high.customers if c.employed])
    assert low_net > high_net


def test_env_reset_and_step_contract():
    env = EconomyEnv(_small_config(max_steps=3))
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert "metrics" in info

    truncated = False
    steps = 0
    while not truncated:
        action = np.array([0.2, 0.2, 0.01], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert terminated is False
        assert "metrics" in info
    assert steps == 3


def test_env_is_reproducible_with_same_seed():
    env = EconomyEnv(_small_config(max_steps=5))
    action = np.array([0.25, 0.2, 0.01], dtype=np.float32)

    env.reset(seed=42)
    first = [env.step(action)[0] for _ in range(5)]

    env.reset(seed=42)
    second = [env.step(action)[0] for _ in range(5)]

    assert np.allclose(np.stack(first), np.stack(second))
