"""Collect expert DeepMind Control Suite rollouts with TD-MPC2 checkpoints."""

import argparse

from dmc_expert.collection import collect, load_config
from training.protocol import freeze_implementation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default="dmc_expert_collection",
        help="Hydra collection config name.",
    )
    parser.add_argument("--task", action="append", help="Collect only this DMC domain/task; repeat as needed.")
    parser.add_argument("--override", action="append", default=[], help="Hydra override; repeat as needed.")
    args = parser.parse_args()
    config = load_config(args.config_name, args.override)
    freeze_implementation()
    if args.task:
        config.tasks = args.task
    collect(config)


if __name__ == "__main__":
    main()
