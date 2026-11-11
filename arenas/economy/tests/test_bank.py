"""Testes do banco-agente e do ambiente de crédito."""
import numpy as np
import pytest

from arenas.economy.bank import Bank, BankConfig
from arenas.economy.bank_env import BankEnv
from arenas.economy.config import EconomyConfig

MACRO = {"growth": 0.005, "inflation": 0.02, "unemployment": 0.55}


def _bank_config(**overrides):
    values = dict(
        group_names=("A", "B"),
        group_pd=(0.01, 0.10),
        group_lgd=(0.40, 0.60),
        group_exposure=(1000.0, 500.0),
        group_demand=(100.0, 50.0),
        group_reference_spread=(0.05, 0.12),
        initial_capital=10_000.0,
        seed=1,
    )
    values.update(overrides)
    return BankConfig(**values)


def _action(approval=1.0, limit=1.0, spreads=(0.05, 0.12)):
    n = len(spreads)
    return np.column_stack([
        np.full(n, approval), np.full(n, limit), np.array(spreads)
    ])


def test_config_requires_matching_group_fields():
    with pytest.raises(ValueError):
        BankConfig(group_pd=(0.1,))


def test_raroc_and_capital_accounting():
    bank = Bank(_bank_config())
    metrics = bank.step(_action(), MACRO, inflation_target=0.02)
    assert metrics["net_income"] == pytest.approx(
        metrics["nii"] - metrics["expected_loss"] - metrics["opex"])
    assert metrics["raroc"] == pytest.approx(
        metrics["net_income"] / metrics["economic_capital"])
    assert metrics["capital"] == pytest.approx(
        10_000.0 + metrics["nii"] - metrics["opex"]
        - metrics["realized_loss"])
    assert metrics["ead_total"] == pytest.approx(sum(metrics["ead"]))


def test_approval_and_limit_scale_exposure():
    approve_none = Bank(_bank_config()).step(
        _action(approval=0.0), MACRO)
    approve_all = Bank(_bank_config()).step(
        _action(approval=1.0), MACRO)
    assert approve_none["ead_total"] == 0.0
    assert approve_all["ead_total"] > 0.0

    full_limit = Bank(_bank_config()).step(
        _action(limit=1.0), MACRO)
    half_limit = Bank(_bank_config()).step(
        _action(limit=0.5), MACRO)
    assert half_limit["ead_total"] == pytest.approx(
        0.5 * full_limit["ead_total"], rel=0.05)


def test_spread_reduces_demand():
    cheap = Bank(_bank_config()).step(
        _action(spreads=(0.0, 0.0)), MACRO)
    expensive = Bank(_bank_config()).step(
        _action(spreads=(0.25, 0.25)), MACRO)
    assert expensive["approved_total"] < cheap["approved_total"]
    assert all(e < c for e, c in zip(
        expensive["demand_factor"], cheap["demand_factor"]))


def test_high_risk_group_has_higher_pd_and_expected_loss():
    bank = Bank(_bank_config())
    metrics = bank.step(_action(), MACRO)
    pd = np.array(metrics["pd"])
    lgd = np.array(metrics["lgd"])
    assert pd[1] > pd[0]
    assert (pd[1] * lgd[1]) > (pd[0] * lgd[0])


def test_macro_stress_raises_pd():
    bank = Bank(_bank_config())
    group = bank.groups[0]
    base = bank.probability_of_default(
        group, {"growth": 0.01, "inflation": 0.02, "unemployment": 0.5},
        0.02)
    stressed = bank.probability_of_default(
        group, {"growth": -0.05, "inflation": 0.06, "unemployment": 0.8},
        0.02)
    assert stressed > base


def test_env_contract_and_reward():
    cfg = EconomyConfig(n_customers=20, n_firms=2, max_steps=3, seed=5)
    env = BankEnv(cfg, _bank_config(max_steps=3))
    obs, info = env.reset(seed=5)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert len(info["groups"]) == 2
    assert len(env.action_labels()) == 6

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(_action())
        metrics = info["bank"]
        expected = float(np.clip(metrics["raroc"], -2.0, 2.0))
        expected -= 0.5 * metrics["default_rate"]
        if (metrics["capital_ratio"]
                < env.bank_cfg.regulatory_capital_ratio):
            expected -= 1.0
        if metrics["failed"]:
            expected -= env.bank_cfg.failure_penalty
        assert reward == pytest.approx(expected)
        assert env.observation_space.contains(obs)
        steps += 1
    assert steps == 3


def test_env_is_deterministic_with_same_seed():
    def rollout():
        env = BankEnv(EconomyConfig(n_customers=20, n_firms=2, seed=9),
                      _bank_config(seed=9, max_steps=5))
        env.reset(seed=9)
        out = []
        for _ in range(5):
            obs, reward, terminated, truncated, _ = env.step(_action())
            out.append((obs.copy(), reward))
            if terminated or truncated:
                break
        return out

    first, second = rollout(), rollout()
    assert len(first) == len(second)
    for (obs_a, reward_a), (obs_b, reward_b) in zip(first, second):
        assert np.allclose(obs_a, obs_b)
        assert reward_a == pytest.approx(reward_b)


def test_bank_failure_terminates_episode():
    cfg = EconomyConfig(n_customers=20, n_firms=2, seed=2)
    env = BankEnv(cfg, _bank_config(seed=2, initial_capital=-1e9))
    env.reset(seed=2)
    _, reward, terminated, _, info = env.step(_action())
    assert info["bank"]["failed"] is True
    assert terminated is True
    assert reward < -env.bank_cfg.failure_penalty
