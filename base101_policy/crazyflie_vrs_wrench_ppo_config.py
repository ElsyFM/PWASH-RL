"""Shared PPO configuration for the Crazyflie direct-wrench NASA-VRS task."""


def build_ppo_cfg(
    experiment_name: str = "crazyflie_vrs_vtrigger_175_wrench",
    max_iterations: int = 2000,
    num_steps_per_env: int = 24,
    save_interval: int = 100,
) -> dict:
    """Return the PPO setup shared by velocity-triggered training and playback."""
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "value_loss_coef": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            # RSL-RL ignored the old *_hidden_dimensions spelling and trained this run
            # with its default [256, 256, 256] architecture. Keep this shape so that
            # evaluation can load existing checkpoints from this run.
            "actor_hidden_dims": [256, 256, 256],
            "critic_hidden_dims": [256, 256, 256],
            "init_noise_std": 0.8,
        },
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": experiment_name,
            "max_iterations": max_iterations,
        },
        "num_steps_per_env": num_steps_per_env,
        "save_interval": save_interval,
    }
