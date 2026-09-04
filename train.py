from pathlib import Path

import hydra

import tools
from training.trainer import train


@hydra.main(version_base=None, config_path="configs", config_name=None)
def main(config):
    tools.configure_randomness(config.seed, config.deterministic_run)
    logdir = Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Run | output={logdir}")

    checkpoint = Path(config.resume_from).expanduser() if config.resume_from else None
    with tools.Logger(logdir) as logger:
        train(config, logger, logdir, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()
