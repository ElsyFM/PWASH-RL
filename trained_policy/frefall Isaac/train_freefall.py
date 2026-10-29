import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Baseline Crazyflie Trajectory Policy")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import os
from rsl_rl.runners import OnPolicyRunner
from crazyflie_ypath_freefall import CrazyflieYPathEnv, CrazyflieYPathEnvCfg


def main():
    # 1. Initialize environment
    env_cfg = CrazyflieYPathEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = CrazyflieYPathEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # 2. PPO Configuration
    ppo_cfg = {
        "algorithm": {
	    "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "value_loss_coef": 1.0,
        },
        "policy": {
	    "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dimensions": [256, 128, 64],
            "critic_hidden_dimensions": [256, 128, 64],
            "init_noise_std": 0.3,
        },

	"obs_groups": {
    	    "policy": ["policy"],
	},

        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": "crazyflie_freefall_oct8",
            "max_iterations": 2000,
            
        },
	"num_steps_per_env": 24,
        "save_interval": 100,
    }

    # 3. Train policy
    log_dir = os.path.join("logs", ppo_cfg["runner"]["experiment_name"])
    runner = OnPolicyRunner(env, ppo_cfg, log_dir=log_dir, device=env.device)

    print("[INFO] Starting Recovery RL Training from Baseline Checkpoint...")
    runner.learn(num_learning_iterations=2000)

    # 4. Save model
    policy_path = os.path.join(log_dir, "freefall_policy.pt")
    runner.save(policy_path)
    print(f"[SUCCESS] Low-throttle model saved to: {policy_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()