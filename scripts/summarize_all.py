#!/usr/bin/env python3
"""Resume a comparação controlada dos quatro algoritmos.

Lê os CSVs produzidos por scripts/compare_onpolicy.sh e pelo runner
DQN/Double DQN e escreve, em logs/compare:

- summary-all.txt
- training_summary_all.csv
- evaluation_summary_all.csv
- comparacao-all.md

Não inventa números: arquivo ou campo ausente aparece como n/d.
"""
import csv
import statistics
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/compare")
ALGS = [
    ("dqn", "DQN", "off-policy"),
    ("ddqn", "Double DQN + PER + dueling", "off-policy"),
    ("ppo", "PPO", "on-policy"),
    ("a2c", "A2C", "on-policy"),
]
CHECKPOINTS = [("ep050", "50 eps"), ("ep100", "100 eps"), ("best", "melhor")]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values):
    values = [v for v in values if v == v]
    return statistics.fmean(values) if values else float("nan")


def field_mean(rows, key):
    return mean([number(row.get(key)) for row in rows])


def fmt(value, digits=2, signed=False):
    if value != value:
        return "n/d"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def reward_stats(rows):
    rewards = [number(row.get("reward")) for row in rows]
    rewards = [r for r in rewards if r == r]
    if not rewards:
        return None
    return {
        "n": len(rewards),
        "mean": statistics.fmean(rewards),
        "std": statistics.pstdev(rewards),
        "median": statistics.median(rewards),
        "min": min(rewards),
        "max": max(rewards),
    }


def training_summary():
    result = {}
    for tag, label, kind in ALGS:
        path = OUT / f"training-{tag}.csv"
        if not path.exists():
            result[tag] = None
            continue
        rows = read_csv(path)
        rewards = [number(row.get("reward")) for row in rows]
        rewards = [r for r in rewards if r == r]
        if not rewards:
            result[tag] = None
            continue
        best_index = max(range(len(rewards)), key=rewards.__getitem__)
        window = min(20, len(rows))
        last = rows[-window:]
        result[tag] = {
            "label": label,
            "kind": kind,
            "episodes": len(rows),
            "first20": statistics.fmean(rewards[:window]),
            "last20": statistics.fmean(rewards[-window:]),
            "best": rewards[best_index],
            "best_episode": best_index + 1,
            "coins_last20": field_mean(last, "coins"),
            "deaths_last20": field_mean(last, "deaths"),
            "epsilon_final": number(rows[-1].get("epsilon")),
            "loss_last20": field_mean(last, "mean_loss"),
        }
    return result


def evaluation_summary():
    result = {}
    for tag, label, kind in ALGS:
        for ckpt, ckpt_label in CHECKPOINTS:
            path = OUT / f"evaluation-{tag}-{ckpt}.csv"
            if not path.exists():
                continue
            rows = read_csv(path)
            stats = reward_stats(rows)
            if stats is None:
                continue
            stats.update({
                "alg": tag,
                "label": label,
                "kind": kind,
                "checkpoint": ckpt_label,
                "coins": field_mean(rows, "coins"),
                "deaths": field_mean(rows, "deaths"),
                "duration": field_mean(rows, "duration_seconds"),
            })
            result[(tag, ckpt)] = stats
    return result


def paired_deltas():
    lines = []
    for tag, label, _ in ALGS:
        if tag == "dqn":
            continue
        for ckpt, ckpt_label in CHECKPOINTS:
            left_path = OUT / f"evaluation-dqn-{ckpt}.csv"
            right_path = OUT / f"evaluation-{tag}-{ckpt}.csv"
            if not (left_path.exists() and right_path.exists()):
                continue
            base = read_csv(left_path)
            other = read_csv(right_path)
            count = min(len(base), len(other))
            if count == 0:
                continue
            diffs = [number(other[i].get("reward")) - number(base[i].get("reward"))
                     for i in range(count)]
            diffs = [d for d in diffs if d == d]
            if not diffs:
                continue
            wins = sum(d > 0 for d in diffs)
            ties = sum(d == 0 for d in diffs)
            used = len(diffs)
            lines.append({
                "tag": tag,
                "label": label,
                "checkpoint": ckpt_label,
                "delta_mean": statistics.fmean(diffs),
                "delta_median": statistics.median(diffs),
                "wins": wins,
                "ties": ties,
                "losses": used - wins - ties,
                "n": used,
            })
    return lines


def ranking(evals, train):
    """Melhor média de avaliação de cada algoritmo, entre os checkpoints."""
    best = {}
    for (tag, _), item in evals.items():
        if tag not in best or item["mean"] > best[tag]["mean"]:
            best[tag] = dict(item)
    order = sorted(best.values(), key=lambda item: item["mean"], reverse=True)
    rows = []
    for position, item in enumerate(order, 1):
        tag = item["alg"]
        rows.append({
            "position": position,
            "tag": tag,
            "label": item["label"],
            "checkpoint": item["checkpoint"],
            "mean": item["mean"],
            "std": item["std"],
            "train_last20": train[tag]["last20"] if train.get(tag) else float("nan"),
        })
    return rows


def write_training_csv(train):
    fields = [
        "algoritmo", "tipo", "episodios", "reward_media_primeiros20",
        "reward_media_ultimos20", "reward_melhor", "episodio_melhor",
        "coins_media_ultimos20", "deaths_media_ultimos20",
        "epsilon_final", "loss_media_ultimos20",
    ]
    with (OUT / "training_summary_all.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for tag, _, _ in ALGS:
            item = train.get(tag)
            if item is None:
                continue
            writer.writerow({
                "algoritmo": item["label"],
                "tipo": item["kind"],
                "episodios": item["episodes"],
                "reward_media_primeiros20": fmt(item["first20"], 4),
                "reward_media_ultimos20": fmt(item["last20"], 4),
                "reward_melhor": fmt(item["best"], 4),
                "episodio_melhor": item["best_episode"],
                "coins_media_ultimos20": fmt(item["coins_last20"], 4),
                "deaths_media_ultimos20": fmt(item["deaths_last20"], 4),
                "epsilon_final": fmt(item["epsilon_final"], 4),
                "loss_media_ultimos20": fmt(item["loss_last20"], 6),
            })


def write_evaluation_csv(evals):
    fields = [
        "algoritmo", "tipo", "checkpoint", "episodios", "reward_media",
        "reward_desvio", "reward_mediana", "reward_min", "reward_max",
        "coins_media", "deaths_media",
        "duracao_media_segundos",
    ]
    with (OUT / "evaluation_summary_all.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for tag, _, _ in ALGS:
            for ckpt, _ in CHECKPOINTS:
                item = evals.get((tag, ckpt))
                if item is None:
                    continue
                writer.writerow({
                    "algoritmo": item["label"],
                    "tipo": item["kind"],
                    "checkpoint": item["checkpoint"],
                    "episodios": item["n"],
                    "reward_media": fmt(item["mean"], 4),
                    "reward_desvio": fmt(item["std"], 4),
                    "reward_mediana": fmt(item["median"], 4),
                    "reward_min": fmt(item["min"], 4),
                    "reward_max": fmt(item["max"], 4),
                    "coins_media": fmt(item["coins"], 4),
                    "deaths_media": fmt(item["deaths"], 4),
                    "duracao_media_segundos": fmt(item["duration"], 4),
                })


def protocol_text():
    return (
        "seed=7  adversário=rule  frame_stack=4  frame_skip=3  batch=64  "
        "treino=100 episódios; avaliação gulosa com 20 episódios, seed=1000, "
        "adversário=rule. DQN/Double DQN: lr=2.5e-4, alvo fixo C=10. "
        "PPO: lr=2.5e-4, rollout=1024, 4 épocas. "
        "A2C: lr=7e-4, n_steps=64. PPO e A2C usam 1 ambiente."
    )


def write_text(train, evals, deltas, ranks):
    lines = ["COMPARAÇÃO CONTROLADA — DQN, Double DQN, PPO e A2C", protocol_text(), ""]
    lines.append("TREINO")
    for tag, label, _ in ALGS:
        item = train.get(tag)
        if item is None:
            lines.append(f"  {label}: n/d")
            continue
        lines.append(
            f"  {label}: primeiros20={item['first20']:.2f}  "
            f"últimos20={item['last20']:.2f}  melhor={item['best']:.2f} "
            f"(ep {item['best_episode']})  epsilon_final="
            f"{fmt(item['epsilon_final'], 3)}")
    lines.append("")
    lines.append("AVALIAÇÃO")
    for tag, _, _ in ALGS:
        for ckpt, _ in CHECKPOINTS:
            item = evals.get((tag, ckpt))
            if item is None:
                continue
            lines.append(
                f"  {item['label']} @ {item['checkpoint']}: "
                f"reward={item['mean']:.2f} ± {item['std']:.2f} "
                f"(mediana {item['median']:.1f}, faixa "
                f"{item['min']:.0f}..{item['max']:.0f})  "
                f"moedas={item['coins']:.2f}  mortes={item['deaths']:.2f}")
    if not evals:
        lines.append("  n/d")
    lines.append("")
    lines.append("DELTA PAREADO CONTRA DQN (outro − DQN)")
    for item in deltas:
        lines.append(
            f"  {item['label']} @ {item['checkpoint']}: "
            f"média {item['delta_mean']:+.2f}, "
            f"mediana {item['delta_median']:+.1f}, "
            f"vitórias/empates/derrotas={item['wins']}/{item['ties']}/"
            f"{item['losses']} em {item['n']} episódios")
    if not deltas:
        lines.append("  n/d")
    lines.append("")
    lines.append("RANKING (melhor checkpoint de cada algoritmo na avaliação)")
    for item in ranks:
        lines.append(
            f"  {item['position']}. {item['label']} @ {item['checkpoint']}: "
            f"{item['mean']:.2f} ± {item['std']:.2f}  "
            f"(treino últimos20={fmt(item['train_last20'])})")
    if not ranks:
        lines.append("  n/d")
    lines.append("")
    lines.append("Uma seed e 20 episódios de avaliação: diferenças pequenas "
                 "não devem ser tratadas como significativas.")
    text = "\n".join(lines) + "\n"
    (OUT / "summary-all.txt").write_text(text)
    print(text, end="")


def write_markdown(train, evals, deltas, ranks):
    md = ["# Comparação controlada — DQN, Double DQN, PPO e A2C", ""]
    md.append(f"Protocolo: {protocol_text()}")
    md.append("")
    md.append("## Treino")
    md.append("")
    md.append("| algoritmo | tipo | eps | primeiros 20 | últimos 20 | melhor | ep. do melhor | moedas (últ. 20) | mortes (últ. 20) | epsilon final |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, _, _ in ALGS:
        item = train.get(tag)
        if item is None:
            md.append("| n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d |")
            continue
        md.append(
            f"| {item['label']} | {item['kind']} | {item['episodes']} | "
            f"{item['first20']:.2f} | {item['last20']:.2f} | "
            f"{item['best']:.2f} | {item['best_episode']} | "
            f"{fmt(item['coins_last20'])} | {fmt(item['deaths_last20'])} | "
            f"{fmt(item['epsilon_final'], 3)} |")
    md.append("")
    md.append("## Avaliação gulosa")
    md.append("")
    md.append("| algoritmo | checkpoint | eps | reward médio | desvio | mediana | mín | máx | moedas | mortes | duração (s) |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, _, _ in ALGS:
        for ckpt, _ in CHECKPOINTS:
            item = evals.get((tag, ckpt))
            if item is None:
                continue
            md.append(
                f"| {item['label']} | {item['checkpoint']} | {item['n']} | "
                f"{item['mean']:.2f} | {item['std']:.2f} | "
                f"{item['median']:.1f} | {item['min']:.0f} | "
                f"{item['max']:.0f} | {item['coins']:.2f} | "
                f"{item['deaths']:.2f} | "
                f"{fmt(item['duration'])} |")
    if not evals:
        md.append("| n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d |")
    md.append("")
    md.append("## Delta pareado contra DQN")
    md.append("")
    md.append("Cada linha compara o mesmo checkpoint com o DQN no mesmo "
              "episódio de avaliação (mesma seed de reset; as trajetórias "
              "divergem quando as políticas diferem).")
    md.append("")
    md.append("| comparação | checkpoint | delta médio | delta mediano | V/E/D | n |")
    md.append("|---|---|---:|---:|---:|---:|")
    for item in deltas:
        md.append(
            f"| {item['label']} − DQN | {item['checkpoint']} | "
            f"{item['delta_mean']:+.2f} | {item['delta_median']:+.1f} | "
            f"{item['wins']}/{item['ties']}/{item['losses']} | {item['n']} |")
    if not deltas:
        md.append("| n/d | n/d | n/d | n/d | n/d | n/d |")
    md.append("")
    md.append("## Ranking")
    md.append("")
    md.append("Melhor checkpoint de cada algoritmo pela média da avaliação.")
    md.append("")
    md.append("| posição | algoritmo | checkpoint | reward médio | desvio | últimos 20 no treino |")
    md.append("|---:|---|---|---:|---:|---:|")
    for item in ranks:
        md.append(
            f"| {item['position']} | {item['label']} | {item['checkpoint']} | "
            f"{item['mean']:.2f} | {item['std']:.2f} | "
            f"{fmt(item['train_last20'])} |")
    if not ranks:
        md.append("| n/d | n/d | n/d | n/d | n/d | n/d |")
    md.append("")
    md.append("## Limitações")
    md.append("")
    md.append("- Uma única seed; a avaliação tem 20 episódios, então "
              "diferenças pequenas estão dentro do ruído.")
    md.append("- PPO e A2C são on-policy e rodaram com um ambiente; "
              "DQN/Double DQN poderiam usar `--envs`, mas rodaram com um "
              "para manter o protocolo comparável.")
    md.append("- O A2C usa `lr=7e-4` (valor usual para actor-critic com um "
              "worker), enquanto os demais usam `2.5e-4`; isso é uma escolha "
              "de configuração razoável por algoritmo, não um controle "
              "estrito de learning rate.")
    md.append("- `best.pt` é escolhido durante o treino; os snapshots "
              "`ep050` e `ep100` permitem comparar orçamentos iguais.")
    md.append("")
    md.append("Curvas: `tensorboard --logdir logs/runs` (runs "
              "`dqn_configs_rule_fs4_fk3_lr0.00025_b64_seed7`, "
              "`doubledqn_per_dueling_configs_rule_fs4_fk3_lr0.00025_b64_seed7`, "
              "`ppo_configs_rule_fs4_fk3_lr0.00025_b64_seed7` e "
              "`a2c_configs_rule_fs4_fk3_lr0.0007_b64_seed7`).")
    md.append("")
    (OUT / "comparacao-all.md").write_text("\n".join(md))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = training_summary()
    evals = evaluation_summary()
    deltas = paired_deltas()
    ranks = ranking(evals, train)
    write_training_csv(train)
    write_evaluation_csv(evals)
    write_text(train, evals, deltas, ranks)
    write_markdown(train, evals, deltas, ranks)


if __name__ == "__main__":
    main()
