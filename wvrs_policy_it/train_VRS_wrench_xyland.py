r"""Train a fresh velocity-triggered direct-wrench Crazyflie policy with NASA-inspired VRS.

Five phases: ascent -> motor-off freefall until vz <= -1.75 m/s -> powered VRS braking/capture
near 0.65 m -> powered climb to 1.70 m -> landing. Run from crazyflie_ypath with Isaac Sim Python:
  & "$env:ISAACSIM_PATH\python.bat" -u train_VRS_wrench.py --headless --num_envs 256 --max_iterations 2000

Fresh training is the default. A checkpoint is compatible only if trained with this 18-observation,
4-action five-phase task; earlier 17-observation checkpoints must not be resumed.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train direct-wrench Crazyflie recovery policy with NASA-style VRS model")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments; 128-256 is a conservative start for an 8 GB GPU")
parser.add_argument("--max_iterations", type=int, default=2000, help="PPO training iterations")
parser.add_argument("--num_steps_per_env", type=int, default=24, help="Rollout steps per environment per iteration")
parser.add_argument("--save_interval", type=int, default=100, help="Save an automatic checkpoint every N iterations")
parser.add_argument("--experiment_name", type=str, default="VRS_v2", help="Folder name under ./logs")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--resume", type=str, default="", help="Optional compatible 18-observation / 4-action checkpoint from this five-phase task only")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be >= 1")
if args_cli.max_iterations < 1:
    parser.error("--max_iterations must be >= 1")
if args_cli.num_steps_per_env < 1:
    parser.error("--num_steps_per_env must be >= 1")
if args_cli.save_interval < 1:
    parser.error("--save_interval must be >= 1")
if args_cli.resume and not os.path.isfile(args_cli.resume):
    parser.error(f"Resume checkpoint not found: {args_cli.resume}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac-dependent modules only after Kit has launched.
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from crazyflie_ypath_VRS_nasa_wrench_xyland import CrazyflieYPathEnv, CrazyflieYPathEnvCfg
from crazyflie_vrs_wrench_ppo_config import build_ppo_cfg


def main() -> None:
    env = None
    try:
        torch.manual_seed(args_cli.seed)

        env_cfg = CrazyflieYPathEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if getattr(args_cli, "device", None):
            env_cfg.sim.device = args_cli.device

        # VRS is enabled during powered descending braking Phase 2. Phase 1 is motor-off
        # until vertical speed reaches -1.75 m/s, then Phase 2 tries to capture near 0.65 m.
        print(f"[VRS WRENCH TRAIN] environments={args_cli.num_envs}")
        print(f"[VRS WRENCH TRAIN] iterations={args_cli.max_iterations}; seed={args_cli.seed}")
        print(f"[VRS WRENCH TRAIN] NASA-style VRS enabled={env_cfg.vrs_enabled}; kappa={env_cfg.vrs_kappa}; f={env_cfg.vrs_strength}")
        print("[VRS WRENCH TRAIN] Sequence: ascent -> motor-off freefall to vz<=-1.75 -> powered VRS braking/capture near z=0.65 -> climb to z=1.70 -> landing.")

        env = CrazyflieYPathEnv(cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)

        ppo_cfg = build_ppo_cfg(
            experiment_name=args_cli.experiment_name,
            max_iterations=args_cli.max_iterations,
            num_steps_per_env=args_cli.num_steps_per_env,
            save_interval=args_cli.save_interval,
        )
        log_dir = os.path.abspath(os.path.join("logs", args_cli.experiment_name))
        os.makedirs(log_dir, exist_ok=True)
        runner = OnPolicyRunner(env, ppo_cfg, log_dir=log_dir, device=env.device)

        if args_cli.resume:
            resume_path = os.path.abspath(args_cli.resume)
            print(f"[VRS WRENCH TRAIN] Resuming VRS-specific training checkpoint: {resume_path}")
            runner.load(resume_path)
        else:
            print("[VRS WRENCH TRAIN] Starting a new velocity-triggered direct-wrench VRS policy from scratch.")
            print("[VRS WRENCH TRAIN] This task uses 18 observations and 4 actions; older 17-observation checkpoints are incompatible.")

        runner.learn(num_learning_iterations=args_cli.max_iterations)

        final_path = os.path.join(log_dir, "vrs_wrench_policy.pt")
        runner.save(final_path)
        print(f"[VRS WRENCH TRAIN] Final direct-wrench checkpoint saved: {final_path}")
        print(f"[VRS WRENCH TRAIN] Periodic checkpoints are in: {log_dir}")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
