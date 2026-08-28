#!/usr/bin/env python3
"""Report trainable parameter budgets for every DMC comparison model."""

import argparse
import contextlib
import io
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch import nn

REPO_DIR = Path(__file__).resolve().parents[1]

with initialize_config_dir(config_dir=str(REPO_DIR / "configs"), version_base=None):
    BENCHMARK = compose(config_name="dmc_benchmark")
SCENARIOS = tuple(BENCHMARK.scenarios)


class _WeightOnlyNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))


class _Mamba3Parameters(nn.Module):
    """Parameter shapes from mamba-ssm 2.3.2.post1, without its CUDA kernels."""

    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        is_outproj_norm=False,
        is_mimo=False,
        mimo_rank=4,
        **_,
    ):
        super().__init__()
        inner = int(expand * d_model)
        heads = inner // int(headdim)
        rank = int(mimo_rank) if is_mimo else 1
        rope_angles = int(int(d_state) * float(rope_fraction)) // 2
        projection = 2 * inner + 2 * int(d_state) * int(ngroups) * rank + 3 * heads + rope_angles
        self.in_proj = nn.Linear(int(d_model), projection, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(heads))
        self.B_bias = nn.Parameter(torch.empty(heads, rank, int(d_state)))
        self.C_bias = nn.Parameter(torch.empty(heads, rank, int(d_state)))
        self.B_norm = _WeightOnlyNorm(int(d_state))
        self.C_norm = _WeightOnlyNorm(int(d_state))
        self.D = nn.Parameter(torch.empty(heads))
        if is_mimo:
            shape = (heads, rank, int(headdim))
            self.mimo_x = nn.Parameter(torch.empty(shape))
            self.mimo_z = nn.Parameter(torch.empty(shape))
            self.mimo_o = nn.Parameter(torch.empty(shape))
        if is_outproj_norm:
            self.norm = _WeightOnlyNorm(inner)
        self.out_proj = nn.Linear(inner, int(d_model), bias=False)


def install_mamba_proxy():
    import models.shared.mamba3 as mamba3_module

    if mamba3_module.Mamba3 is not None:
        return False
    mamba3_module.Mamba3 = _Mamba3Parameters
    return True


USING_MAMBA_PROXY = install_mamba_proxy()


MODELS = {}
FAMILIES = {}
for family, variants in BENCHMARK.models.items():
    for variant, config in variants.items():
        name = family if variant == "default" else f"{family}_{variant}"
        config = config.config if not isinstance(config, str) else config
        MODELS[name] = config
        FAMILIES[name] = family

GROUPS = ("encoder", "dynamics", "decoder", "heads", "controller")
PAIRS = tuple(
    (baseline, name)
    for family, baseline in (("dreamer", "dreamer_gru"), ("storm", "storm_transformer"))
    for name, model_family in FAMILIES.items()
    if model_family == family and name != baseline
)
# Trainable image-model counts from each original implementation.
REFERENCE_COUNTS = {
    "dreamer": {
        "encoder": 690_624,
        "dynamics": 5_778_944,
        "decoder": 6_985_667,
        "heads": 2_232_576,
        "controller": 2_238_219,
    },
    "storm": {
        "encoder": 690_624,
        "dynamics": 9_748_992,
        "decoder": 4_884_931,
        "heads": 1_184_000,
        "controller": 2_236_680,
    },
    "tdmpc2": {"encoder": 48_864, "dynamics": 793_088, "heads": 3_493_470, "controller": 533_516},
    "leworldmodel": {"encoder": 6_294_144, "dynamics": 11_740_334},
    "temporal_straightening": {"dynamics": 20_122_700, "decoder": 10_140_163},
}


def parameter_count(*modules, trainable=True):
    parameters = {
        id(parameter): parameter for module in modules if module is not None for parameter in module.parameters()
    }
    return sum(parameter.numel() for parameter in parameters.values() if not trainable or parameter.requires_grad)


def model_groups(family, model):
    if family == "dreamer":
        return {
            "encoder": (model.encoder,),
            "dynamics": (model.rssm,),
            "decoder": (model.decoder,),
            "heads": (model.reward, model.cont),
            "controller": (model.actor, model.value),
        }
    if family == "storm":
        world_model = model.world_model
        actor_critic = model.actor_critic
        return {
            "encoder": (world_model.encoder,),
            "dynamics": (world_model.posterior, world_model.sequence_core, world_model.prior),
            "decoder": (world_model.decoder,),
            "heads": (world_model.reward, world_model.termination),
            "controller": (actor_critic.actor, actor_critic.critic),
        }
    if family == "tdmpc2":
        return {
            "encoder": (model.encoder,),
            "dynamics": (model.dynamics,),
            "decoder": (),
            "heads": (model.reward, model.qs),
            "controller": (model.policy,),
        }
    if family in {"leworldmodel", "temporal_straightening"}:
        return {
            "encoder": (model.encoder, model.projector),
            "dynamics": (model.action_encoder, model.predictor, model.pred_projector),
            "decoder": (model.decoder,),
            "heads": (),
            "controller": (model.goal_readout,),
        }
    raise ValueError(f"No parameter grouping for model family {family!r}")


def build(config_name, scenario, overrides=()):
    from models.dreamer.model import DreamerModel
    from models.leworldmodel import LeWorldModel
    from models.storm.model import StormModel
    from models.tdmpc2 import TDMPC2
    from models.temporal_straightening import TemporalStraightening

    config = compose(
        config_name=config_name,
        overrides=[f"scenario={scenario}", "device=cpu", *overrides],
    )
    family_name = str(config.model_family)
    constructors = {
        "dreamer": lambda: DreamerModel(config.model, config.model_io),
        "storm": lambda: StormModel(config, config.model_io),
        "tdmpc2": lambda: TDMPC2(config, config.model_io),
        "leworldmodel": lambda: LeWorldModel(config, config.model_io),
        "temporal_straightening": lambda: TemporalStraightening(config, config.model_io),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        model = constructors[family_name]().to(config.device)
    groups = model_groups(family_name, model)
    counts = {name: parameter_count(*groups[name]) for name in GROUPS}
    trainable = parameter_count(model)
    grouped = sum(counts.values())
    if grouped != trainable:
        raise RuntimeError(f"{config_name} grouped {grouped:,} parameters, but the model has {trainable:,}.")
    frozen = parameter_count(model, trainable=False) - trainable
    return counts | {"trainable": trainable, "frozen": frozen}


def print_report(rows):
    headers = ("model", "scenario", *GROUPS, "trainable", "frozen")
    widths = {
        header: max(
            len(header),
            *(len(f"{row[header]:,}" if isinstance(row[header], int) else str(row[header])) for row in rows),
        )
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            text = f"{value:,}" if isinstance(value, int) else value
            values.append(text.rjust(widths[header]) if isinstance(value, int) else text.ljust(widths[header]))
        print("  ".join(values))

    print()
    for scenario in sorted({row["scenario"] for row in rows}):
        totals = [row["trainable"] for row in rows if row["scenario"] == scenario]
        print(f"{scenario:12} min={min(totals):,} max={max(totals):,} spread={max(totals) - min(totals):,}")

    print()
    lookup = {(row["model"], row["scenario"]): row["trainable"] for row in rows}
    pair_gaps = []
    for scenario in sorted({row["scenario"] for row in rows}):
        for left, right in PAIRS:
            left_key = (left, scenario)
            right_key = (right, scenario)
            if left_key not in lookup or right_key not in lookup:
                continue
            gap = abs(lookup[left_key] - lookup[right_key])
            pair_gaps.append(gap)
            print(f"{scenario:12} {left} vs {right}: gap={gap:,}")
    return pair_gaps


def proportion_errors(rows):
    errors = []
    for row in rows:
        reference = REFERENCE_COUNTS[FAMILIES[row["model"]]]
        reference_total = sum(reference.values())
        current_total = sum(row[group] for group in reference)
        for group, count in reference.items():
            fit = (row[group] / current_total) / (count / reference_total)
            errors.append((abs(fit - 1), row["model"], row["scenario"], group, fit))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--max-spread", type=int, default=65_000)
    parser.add_argument("--max-pair-gap", type=int, default=2_000)
    parser.add_argument("--max-proportion-error", type=float, default=0.1)
    parser.add_argument("--target", type=int, default=5_250_000)
    parser.add_argument("--target-tolerance", type=int, default=60_000)
    args = parser.parse_args()

    rows = []
    with initialize_config_dir(config_dir=str(REPO_DIR / "configs"), version_base=None):
        for scenario in args.scenarios:
            for name in args.models:
                rows.append({
                    "model": name,
                    "scenario": scenario,
                    **build(MODELS[name], scenario, args.override),
                })
    if USING_MAMBA_PROXY and any("mamba" in row["model"] for row in rows):
        print("Mamba3 CUDA package unavailable; using its exact parameter shapes for this report.\n")
    pair_gaps = print_report(rows)
    totals = [row["trainable"] for row in rows]
    spreads = []
    for scenario in {row["scenario"] for row in rows}:
        values = [row["trainable"] for row in rows if row["scenario"] == scenario]
        if len(values) > 1:
            spreads.append(max(values) - min(values))
    if spreads and max(spreads) > args.max_spread:
        raise SystemExit(f"Parameter spread {max(spreads):,} exceeds {args.max_spread:,}.")
    if pair_gaps and max(pair_gaps) > args.max_pair_gap:
        raise SystemExit(f"Paired recurrent gap {max(pair_gaps):,} exceeds {args.max_pair_gap:,}.")
    fits = proportion_errors(rows)
    if fits:
        print("\nWorst original-component proportional deviation by model:")
        for name in args.models:
            model_fit = max((item for item in fits if item[1] == name), default=None)
            if model_fit:
                fit_error, _, fit_scenario, fit_group, relative_fit = model_fit
                detail = f"{fit_group:10}  {fit_scenario} (relative fit {relative_fit:.3f})"
                print(f"{name:28} {fit_error:6.1%}  {detail}")
    worst_fit = max(fits, default=None)
    if worst_fit:
        error, model, scenario, group, fit = worst_fit
        print(
            f"\nWorst original-component proportional deviation: {model} {scenario} {group} "
            f"{error:.1%} (relative fit {fit:.3f})"
        )
    if worst_fit and error > args.max_proportion_error:
        raise SystemExit(
            f"{model} {scenario} {group} proportional fit {fit:.3f} deviates by {error:.1%}, "
            f"exceeding the {args.max_proportion_error:.1%} tolerance."
        )
    if totals and max(abs(total - args.target) for total in totals) > args.target_tolerance:
        raise SystemExit(f"A model is more than {args.target_tolerance:,} parameters from the {args.target:,} target.")


if __name__ == "__main__":
    main()
