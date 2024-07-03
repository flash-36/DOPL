from dopo.registry import TRAINING_FUNCTIONS
import logging
import wandb
from dopo.utils import set_seed, save_results


log = logging.getLogger(__name__)


def train(cfg, env):
    """Train and evaluate the model based on the given configuration."""

    for seeds in range(cfg.num_seeds):
        set_seed(seeds)
        print("*" * 40, f"Training Seed {seeds + 1}", "*" * 40)
        wandb.init(
            project="dopo",
            name=f"{cfg.exp.name}_seed_{seeds + 1}",
            config=dict(cfg),
            reinit=True,
        )

        results_dict = train_algos(env, cfg)
        wandb.finish()
        # Save results dict in hydra output
        save_results(results_dict, seeds)


def train_algos(env, cfg):
    """Run each algo and aggregate results"""
    results_dict = {}
    for algo_name in cfg.algos:
        log.info(f"Training {algo_name}")
        results_dict[algo_name] = TRAINING_FUNCTIONS[algo_name](env, cfg)
    return results_dict
