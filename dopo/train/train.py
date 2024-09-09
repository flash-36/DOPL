from dopo.registry import TRAINING_FUNCTIONS
import logging
import wandb
from dopo.utils import set_seed, save_results
import numpy as np

log = logging.getLogger(__name__)


def train(cfg, env):
    """Train and evaluate the model based on the given configuration."""
    for seed in range(cfg.num_seeds):
        print_string = ("*" * 10, f"Seed {seed + 1}", "*" * 10)
        log.info(" ".join(print_string))
        results_dict = train_algos(env, cfg, seed)
        save_results(results_dict, seed)  # Save results dict in hydra output


def train_algos(env, cfg, seed):
    """Run each algo and aggregate results"""
    results_dict = {}
    for algo_name in cfg.algos:
        set_seed(seed)
        config_dict = dict(cfg)
        config_dict["algo"] = algo_name
        wandb.init(
            project="dopo",
            name=f"{cfg.exp.name}",
            job_type=f"{algo_name}_seed_{seed + 1}",
            config=config_dict,
            reinit=False,
        )
        log.info(f"Training {algo_name}")
        results = TRAINING_FUNCTIONS[algo_name](env, cfg)
        results_dict[algo_name] = results
        log.info(f"Run time for {algo_name}: {results['run_time']:.2f} seconds")
        wandb.finish()
    return results_dict
