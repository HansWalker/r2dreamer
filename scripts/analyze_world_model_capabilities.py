"""Read-only presentation analysis of the original 39 archived evaluations.

No model loading, training, simulator execution, or averaging of unlike units.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = [*(f"dreamer/{v}" for v in ("gru", "sliding_window", "mamba3", "s5", "hyena")),
          *(f"storm/{v}" for v in ("transformer", "sliding_window", "mamba3", "s5", "hyena")),
          "tdmpc2/default", "leworldmodel/default", "temporal_straightening/default"]
TASKS = ["cartpole_balance_sparse", "reacher", "ball_in_cup"]
LABELS = ["D-GRU", "D-Window", "D-Mamba3", "D-S5", "D-Hyena", "S-Transformer",
          "S-Window", "S-Mamba3", "S-S5", "S-Hyena", "TD-MPC2", "LeWM", "TS"]
METRICS = {"observed_rmse": "With real observations", "rmse": "Without future observations",
           "persistence_rmse": "Hold last estimated state"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(root, tasks=TASKS):
    result = []
    for task in tasks:
        reference = None
        for model in MODELS:
            path = Path(root) / task / model / "seed_0/evaluation.json"
            r = json.loads(path.read_text())
            q, identity = r["physical_state_prediction"], r["run_identity"]
            if (identity["scenario"], identity["model_family"] + "/" + identity["model_variant"],
                    r["training_seed"], r["online_updates"], r["environment_steps"]) != (task, model, 0, 10000, 80000):
                raise ValueError(f"Wrong original experiment identity: {path}")
            if not q or q["evaluation_fitting"] or q["dataset_split"] != "heldout":
                raise ValueError(f"Missing or fitted/non-held-out predictions: {path}")
            comparison = (q["windows"], q["state_coordinates"], q["horizons"], q["context_length"], r["dataset_identity"])
            if reference is not None and comparison != reference:
                raise ValueError(f"Unmatched physical evaluation windows: {path}")
            reference = comparison
            for cohort in [q, *q["cohorts"].values()]:
                for metric in METRICS:
                    values = np.asarray([list(v.values()) for v in cohort[metric].values()])
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError(f"Invalid errors: {path}")
            result.append((task, model, r, path))
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_correlation(x, y):
    from scipy.stats import rankdata
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def coordinate_label(task, coordinate):
    if coordinate.startswith(("sin(", "cos(")) or task == "cartpole_balance_sparse" and coordinate in ("position[1]", "position[2]"):
        return coordinate + " [dimensionless]"
    if coordinate.startswith("velocity"):
        unit = "rad/s" if task == "reacher" or task == "cartpole_balance_sparse" and coordinate == "velocity[1]" else "m/s"
    else:
        unit = "m"
    return coordinate + f" [{unit}]"


def plot_records(data, output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors = plt.get_cmap("tab20").colors
    with PdfPages(output / "state_estimation_and_forecasting.pdf") as pdf:
        for task in TASKS:
            task_rows = [(m, r) for t, m, r, _ in data if t == task]
            coords = task_rows[0][1]["physical_state_prediction"]["state_coordinates"]
            for cohort in ("all", "uniform", "motion"):
                fig, axes = plt.subplots(2, len(coords), figsize=(3.0 * len(coords), 7), squeeze=False)
                for row, metric in enumerate(("observed_rmse", "rmse")):
                    for col, coord in enumerate(coords):
                        ax = axes[row, col]
                        for model, r in task_rows:
                            i = MODELS.index(model)
                            q = r["physical_state_prediction"]
                            q = q if cohort == "all" else q["cohorts"][cohort]
                            h = sorted(map(int, q[metric]))
                            ax.plot(h, [max(q[metric][str(k)][coord], 1e-12) for k in h],
                                    color=colors[i], label=LABELS[i], marker=".", linewidth=1.3)
                        ax.set(xscale="log", yscale="log", xlabel="Future step", title=coordinate_label(task, coord))
                        ax.grid(alpha=.2)
                        if col == 0:
                            ax.set_ylabel(METRICS[metric] + "\nRMSE")
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=9)
                fig.suptitle(f"{task} — {cohort} windows — original final checkpoints\nShared windows; fixed physical readouts; lower error is better", fontsize=13)
                fig.tight_layout(rect=(0, .09, 1, .91))
                pdf.savefig(fig)
                fig.savefig(output / f"errors_{task}_{cohort}.png", dpi=150)
                plt.close(fig)
    with PdfPages(output / "prediction_and_control.pdf") as pdf:
        for task in TASKS:
            task_rows = [(m, r) for t, m, r, _ in data if t == task]
            coords = task_rows[0][1]["physical_state_prediction"]["state_coordinates"]
            fig, axes = plt.subplots(2, len(coords), figsize=(3.0 * len(coords), 7), squeeze=False)
            for row, metric in enumerate(("observed_rmse", "rmse")):
                for col, coord in enumerate(coords):
                    ax = axes[row, col]
                    x, y = [], []
                    for model, r in task_rows:
                        i = MODELS.index(model)
                        error = r["physical_state_prediction"][metric]["25"][coord]
                        x.append(error)
                        y.append(r["mean_return"])
                        ax.scatter(max(error, 1e-12), y[-1], color=colors[i], label=LABELS[i], s=36)
                        ax.annotate(LABELS[i], (max(error, 1e-12), y[-1]), fontsize=6, xytext=(3, 3), textcoords="offset points")
                    rho = rank_correlation(x, y)
                    ax.set(xscale="log", ylim=(-.04 * max(y), 1.12 * max(y)), xlabel=coordinate_label(task, coord),
                           title=f"H25; rank correlation = {rho:.2f}" if rho is not None else "H25; correlation undefined")
                    ax.grid(alpha=.2)
                    if col == 0:
                        ax.set_ylabel(METRICS[metric] + "\nFinal episode return")
            fig.suptitle(f"{task}: estimation / prediction versus control\nOne fit per variant; exploratory association, not a causal or significance test", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, .9))
            pdf.savefig(fig)
            fig.savefig(output / f"control_{task}.png", dpi=150)
            plt.close(fig)
    # One compact, dimensionless overview; each baseline belongs to its own model.
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    for ax, task in zip(axes, TASKS):
        coords = next(r for t, _, r, _ in data if t == task)["physical_state_prediction"]["state_coordinates"]
        values = np.array([[next(r["forecast_over_hold"] for r in rows if
                               (r["task"], r["model"], r["cohort"], r["horizon"], r["coordinate"]) == (task, m, "all", 25, c))
                            for c in coords] for m in MODELS], dtype=float)
        im = ax.imshow(np.log2(np.clip(values, .125, 8)), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        for i in range(len(MODELS)):
            for j in range(len(coords)):
                ax.text(j, i, f"{values[i,j]:.1f}" if np.isfinite(values[i,j]) else "n/a", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(coords)), coords, rotation=70, ha="right", fontsize=8)
        ax.set_yticks(range(len(MODELS)), LABELS, fontsize=9)
        ax.set_title(task)
    fig.suptitle("25-step forecast RMSE / hold-last-estimate RMSE\nBelow 1 beats holding the model's own last estimate; this is not a shared true-state baseline")
    fig.tight_layout(rect=(0, 0, .94, .90))
    bar = fig.colorbar(im, cax=fig.add_axes((.95, .25, .012, .5)), ticks=[-3, -2, -1, 0, 1, 2, 3])
    bar.ax.set_yticklabels(["≤0.125", "0.25", "0.5", "1", "2", "4", "≥8"])
    fig.savefig(output / "forecast_vs_hold.png", dpi=180)
    fig.savefig(output / "forecast_vs_hold.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("local/run_results"))
    parser.add_argument("--output", type=Path, default=Path("local/reports/world_model_capabilities"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    data = records(args.archive_root)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, correlations = [], []
    for task, model, r, _ in data:
        q = r["physical_state_prediction"]
        for cohort, values in [("all", q), *q["cohorts"].items()]:
            for h in q["horizons"]:
                for c in q["state_coordinates"]:
                    observed, forecast, hold = [values[k][str(h)][c] for k in METRICS]
                    rows.append(dict(task=task, model=model, cohort=cohort, horizon=h, coordinate=c,
                                     observed_rmse=observed, forecast_rmse=forecast, hold_rmse=hold,
                                     forecast_over_hold=forecast / hold if hold else None,
                                     forecast_over_observed=forecast / observed if observed else None,
                                     return_mean=r["mean_return"], return_se=r["return_standard_error"]))
    for task in TASKS:
        coords = next(r for t, _, r, _ in data if t == task)["physical_state_prediction"]["state_coordinates"]
        for cohort in ("all", "uniform", "motion"):
            for h in (1, 5, 10, 25, 50, 100):
                for c in coords:
                    for group in ("all", "dreamer", "storm"):
                        subset = [r for r in rows if (r["task"], r["cohort"], r["horizon"], r["coordinate"]) == (task, cohort, h, c)
                                  and (group == "all" or r["model"].startswith(group + "/"))]
                        for metric in ("observed_rmse", "forecast_rmse"):
                            correlations.append(dict(task=task, cohort=cohort, horizon=h, coordinate=c, family=group,
                                                     metric=metric, models=len(subset), spearman=rank_correlation(
                                                         [r[metric] for r in subset], [r["return_mean"] for r in subset])))
    write_csv(args.output / "physical_errors.csv", rows)
    write_csv(args.output / "control_associations.csv", correlations)
    audit = {"evaluations": len(data), "same_windows_within_task": True, "new_training": False,
             "state_windows_per_evaluation": sorted({r["physical_state_prediction"]["state_windows"] for _, _, r, _ in data}),
             "sources": [{"path": str(p), "sha256": sha(p), "checkpoint_id": r["checkpoint_id"]} for _, _, r, p in data]}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not args.no_plots:
        plot_records(data, args.output, rows)
    text = """# State estimation, forecasting and control in the original experiment

Completed analyses 1 and 2: all 13 variants on all three tasks, using only the
39 original final evaluations. No retraining or new controller runs. Every model
within a task was measured on identical 128 held-out windows (64 uniform and
64 higher-motion windows), with the same 64-frame observed prefix.

## Presentation files

- `forecast_vs_hold.pdf` / `.png`: compact 25-step forecast-to-hold error ratios.
- `state_estimation_and_forecasting.pdf`: nine pages, one per task/cohort;
  current-observation decoding and open-loop forecasts, by coordinate and horizon.
- `prediction_and_control.pdf`: three pages linking physical errors to original
  50-episode final returns. PNGs are also supplied for each task.
- `physical_errors.csv`: exact per-coordinate values and within-model ratios.
- `control_associations.csv`: descriptive rank correlations for all variants,
  and separately within Dreamer and STORM, for every coordinate/cohort/horizon.
- `audit.json`: source hashes, checkpoint identities and matching checks.

## Interpretation rules

Physical errors measure the frozen representation plus its existing supervised
readout, not all information recoverable from the representation. Observed errors
use actual future images at the scored time; open-loop forecasts do not. Their
gap does not uniquely identify dynamics error because readouts and histories
also differ. Do not subtract these RMSEs as an error decomposition.

The archived persistence baseline holds the model's last decoded state, not the
true simulator state. Its quality differs across models. Ratios below one show
improvement over that model's own baseline; use absolute coordinate errors for
between-model comparisons. No common aggregate is formed across meters, angular
velocities and dimensionless sine/cosine coordinates. Wrapped-angle RMSE cannot
be reconstructed from the archived aggregate sine/cosine errors.

The two halves of the window bank are reported separately. Expert trajectories
can be nearly stationary and do not cover all failed on-policy states. Prediction
and policy returns are measured on different trajectories. Associations therefore
do not demonstrate that prediction error caused a control result. Lower error and
higher return produce a negative correlation. No significance claims are made:
there is one training seed, 13 cross-family points, or five points within a family.
Coordinates, horizons and cohorts are not independent training replications.

The parent state-estimation study and this visual-control study use different
inputs, model capacities and training protocols. Present them as complementary
questions, not a combined architecture leaderboard. The history-length test is
a separate frozen-checkpoint evaluation, not a repair or reinterpretation of
these original results.
"""
    (args.output / "README.md").write_text(text)
    print(f"Analyzed {len(data)} evaluations; wrote {len(rows)} coordinate/horizon rows to {args.output}")


if __name__ == "__main__":
    main()
