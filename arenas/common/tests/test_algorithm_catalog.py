"""Testes do catálogo de algoritmos por arena."""
from arenas.common.algorithm_catalog import algorithm_catalog
from arenas.registry import algorithms_for_arena


def test_algorithms_for_arena_matches_cli():
    assert algorithms_for_arena("risk") == (
        "dqn", "doubledqn_per_dueling", "ppo", "a2c")
    assert algorithms_for_arena("economy") == (
        "ppo_continuous", "ddpg", "td3", "sac")
    seek = set(algorithms_for_arena("seek_coin"))
    assert {"dqn", "ppo", "eppo", "ppo_aes", "gtr", "a2c", "mka",
            "doubledqn_per_dueling", "sac_discrete",
            "ippo", "mappo", "happo"} <= seek
    assert not {"ppo_continuous", "ddpg", "td3", "sac"} & seek


def test_catalog_lists_each_arena_with_docs():
    catalog = algorithm_catalog()
    assert [arena["name"] for arena in catalog["arenas"]] == [
        "seek_coin", "risk", "economy"]
    by_name = {arena["name"]: arena for arena in catalog["arenas"]}
    assert by_name["seek_coin"]["default_algorithm"] == "dqn"
    assert by_name["economy"]["action_space"] == "box"
    for arena in catalog["arenas"]:
        assert arena["algorithms"], arena["name"]
        assert arena["readme"].startswith("/docs/")
        for row in arena["algorithms"]:
            assert row["docs"], f"{arena['name']}/{row['name']} sem docs"
            assert all(doc["url"].startswith("/docs/")
                       for doc in row["docs"])


def test_catalog_metadata_and_lessons():
    catalog = algorithm_catalog()
    seek = next(a for a in catalog["arenas"] if a["name"] == "seek_coin")
    rows = {row["name"]: row for row in seek["algorithms"]}
    assert rows["ppo"]["on_policy"] is True
    assert rows["dqn"]["on_policy"] is False
    assert rows["ippo"]["multi_agent"] is True
    assert any(doc["label"] == "Aula (HTML)" for doc in rows["ppo"]["docs"])
    assert any(doc["label"] == "Aula (HTML)"
               for doc in rows["doubledqn_per_dueling"]["docs"])
    assert rows["dqn"]["default"] is True

    economy = next(a for a in catalog["arenas"] if a["name"] == "economy")
    sac = next(row for row in economy["algorithms"] if row["name"] == "sac")
    assert sac["continuous"] is True
    assert not any(doc["label"] == "Aula (HTML)" for doc in sac["docs"])
    assert any(doc["url"].startswith("/docs/arenas/economy/")
               for doc in sac["docs"])
