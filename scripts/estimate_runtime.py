"""Project the configured launch schedule from measured smoke phases, without launching models."""

import argparse
import json
import math
from pathlib import Path

from main import build_runs, load_config, model_run_groups
from scripts.smoke_training import queue_seconds


def run_seconds(config, training_row, serial_row, checkpoints):
    rates = training_row["timing_projection"]["rates"]
    reference = serial_row["timing_projection"]
    expert, online = config.training.expert, config.training.online
    expert_updates = int(expert.updates) if expert.enabled else 0
    updates = int(online.updates) if int(online.steps) else 0
    calls = math.ceil(int(online.steps) / (int(config.env.env_num) * int(config.env.action_repeat)))
    read, compute = (rates[name]["seconds_per_call"] for name in ("sample", "expert"))
    offline = [expert_updates * max(read, compute), expert_updates * (read + compute)]
    collection = calls * rates["collect"]["seconds_per_call"]
    episode_steps = math.ceil(int(config.env.time_limit) / int(config.env.action_repeat))
    eval_rates = reference.get("evaluation_seconds_per_vector_step", {})

    def evaluation(batch):
        if str(batch) in eval_rates:
            return eval_rates[str(batch)] * episode_steps
        # This fallback is explicitly labelled: planner throughput need not scale linearly with batch size.
        return reference["rates"]["collect"]["seconds_per_call"] * batch / int(config.env.env_num) * episode_steps

    periodic = 0.0
    if int(online.steps) and int(online.eval_every) and int(config.env.eval_episode_num):
        periodic = (1 + math.ceil(int(online.steps) / int(online.eval_every))) * evaluation(
            int(config.env.eval_episode_num)
        )
    if expert_updates and int(expert.eval_every) and int(config.env.eval_episode_num):
        periodic += math.ceil(expert_updates / int(expert.eval_every)) * evaluation(int(config.env.eval_episode_num))
    final = len(checkpoints) * evaluation(int(config.evaluation.final.episodes))
    return {
        "offline": offline,
        "updates": updates * rates["online"]["seconds_per_call"],
        "collect": collection,
        "periodic": periodic,
        "final": final,
        "startup": training_row["timing_projection"].get("startup_seconds", 0)
        + training_row["timing_projection"].get("cold_overhead_seconds", 0),
        "evaluation_basis": "short evaluation batch timings"
        if all(
            str(batch) in eval_rates
            for batch in (int(config.env.eval_episode_num), int(config.evaluation.final.episodes))
        )
        else "collection-rate extrapolation",
    }


def estimate(matrix, reports):
    measured = {}
    for report in reports:
        for row in report["runs"]:
            key = row.get("case", "serial"), row["name"]
            if row["status"] == "PASS" and row.get("timing_projection"):
                measured[key] = row
            else:
                measured[key] = None
    scenarios = {name: build_runs(matrix, name, matrix.scenario_configs[name]) for name in matrix.scenarios}
    rows = []
    for label, runs, workers in model_run_groups(matrix, scenarios):
        groups = [(label, runs)] if workers > 1 else [(run.name, [run]) for run in runs]
        for name, group in groups:
            parts, missing = [], []
            for run in group:
                key = f"{run.scenario}/{run.family}/{run.variant}"
                config = load_config(run.config, run.overrides)
                training = measured.get((f"scenarios_{workers}" if workers > 1 else "serial", key))
                serial = measured.get(("serial", key))
                if workers == 1 and run.family == "temporal_straightening":
                    planner_case = f"gradient_{config.jepa_model.planner.gradient_batch_size}", key
                    if planner_case in measured:
                        training = serial = measured[planner_case]
                if (
                    training is None
                    or serial is None
                    or "collect" not in training["timing_projection"]["rates"]
                    or "collect" not in serial["timing_projection"]["rates"]
                ):
                    missing.append(key)
                    continue
                parts.append(run_seconds(config, training, serial, matrix.evaluation.checkpoints))
            row = {"group": name, "workers": workers, "missing": missing}
            if not missing:
                row.update(
                    offline=[queue_seconds([part["offline"][i] for part in parts], workers) / 3600 for i in (0, 1)],
                    **{
                        key: queue_seconds([part[key] for part in parts], workers) / 3600
                        for key in ("updates", "collect", "periodic")
                    },
                    final=sum(part["final"] for part in parts) / 3600,
                    total=[
                        (
                            queue_seconds(
                                [
                                    part["offline"][i]
                                    + sum(part[key] for key in ("updates", "collect", "periodic", "startup"))
                                    for part in parts
                                ],
                                workers,
                            )
                            + sum(part["final"] for part in parts)
                        )
                        / 3600
                        for i in (0, 1)
                    ],
                    evaluation_basis=sorted({part["evaluation_basis"] for part in parts}),
                )
            rows.append(row)
    return rows


def render(rows):
    lines = [
        "# Runtime Estimate",
        "",
        "Hours of wall time, in launch order. Concurrent scenario runs count as one group.",
        "Offline = expert pretraining; online = gradient updates plus collection, with periodic evaluations interspersed.",
        "Final = final.pt and best.pt policy evaluation, serialized after each group. State-prediction evaluation is unmeasured.",
        "",
        "| Group | Workers | Offline | Online Updates | Collection | Periodic Eval | Final Eval | Total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["missing"]:
            lines.append(f"| {row['group']} | {row['workers']} | Unmeasured | | | | | |")
            continue
        low, high = row["offline"]
        first, last = row["total"]
        lines.append(
            f"| {row['group']} | {row['workers']} | {low:.2f}-{high:.2f} | {row['updates']:.2f} | {row['collect']:.2f} | {row['periodic']:.2f} | {row['final']:.2f} | {first:.2f}-{last:.2f} |"
        )
    totals = [sum(row["total"][i] for row in rows if not row["missing"]) for i in (0, 1)]
    lines += [
        "",
        f"Measured-group subtotal: {totals[0]:.1f}-{totals[1]:.1f} hours ({totals[0] / 24:.1f}-{totals[1] / 24:.1f} days).",
        f"Coverage: {sum(not row['missing'] for row in rows)}/{len(rows)} groups.",
        "",
        "## Assumptions",
        "- Historical projection: no assumed speedup for optimizations not present in the supplied measurements.",
        "- Prefetch ranges bound read/compute overlap, not statistical uncertainty.",
        "- Phase columns estimate each phase separately. Group totals schedule whole training jobs, then sum serialized final evaluations; columns need not sum exactly for concurrent groups.",
        "- Evaluation uses measured short episodes at matching batch sizes when available; otherwise it scales collection time by episode count. Neither is a measured full-episode guarantee.",
        "- Concurrent periodic evaluation contention, state-prediction evaluation, checkpoint I/O, dataset staging, and expert-data collection are not measured.",
        "- Uses the first measured seed for each configuration and assumes fresh, complete training. It does not subtract completed/resumed work.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        nargs="+",
        required=True,
        help="Later reports replace matching earlier cases; missing groups are not guessed.",
    )
    parser.add_argument("--config-name", default="dmc_benchmark")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("runs/runtime_estimate"))
    args = parser.parse_args()
    matrix = load_config(args.config_name, args.override)
    rows = estimate(matrix, [json.loads(path.read_text()) for path in args.report])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "schedule.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    text = render(rows)
    (args.output / "schedule.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
