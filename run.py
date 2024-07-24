import hydra
from omegaconf import DictConfig

from dopo.utils import (
    load_environment_configuration,
    initialize_environment,
)
from dopo.train import get_opt_performance, train
from dopo.plot import (
    plot_training_performance,
    plot_dopl_estimation_errors,
)
import warnings
import logging

logging.getLogger("pyomo").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


@hydra.main(config_path="conf", config_name="cpap_dpo", version_base=None)
def main(cfg: DictConfig):
    # instantiate env
    P_list, R_list, arm_constraint = load_environment_configuration(cfg)
    env = initialize_environment(cfg, P_list, R_list, arm_constraint)
    get_opt_performance(env)  # Compute asymptotic optimal cost for env

    # Perform training
    train(cfg, env)

    # Plot results
    plot_training_performance(cfg)
    plot_dopl_estimation_errors(cfg)


if __name__ == "__main__":
    main()
