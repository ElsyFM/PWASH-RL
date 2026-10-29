#!/usr/bin/env python3
"""Diagnostic-only Crazyflie policy test.

Connects over Crazyradio, reads state-estimator telemetry, builds the 17-element
observation used by crazyflie_ypath_freefall.py, and evaluates freefall_policy.pt.

SAFETY: This script NEVER sends setpoints, motor commands, or flight commands.
It is for checking radio, telemetry, tensor dimensions, and policy inference only.
It does not fly the trajectory.
"""

import argparse
import math
import os
import sys
import time

import torch
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"
DEFAULT_POLICY = "freefall_policy.pt"


class Actor(torch.nn.Module):
    """Actor architecture inferred from the supplied PPO checkpoint/config."""
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(17, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 4),
        )

    def forward(self, obs):
        return self.actor(obs)


def quaternion_wxyz_from_euler(roll, pitch, yaw):
    """Euler angles in radians -> quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def load_actor(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain 'model_state_dict'.")
    state = checkpoint["model_state_dict"]
    actor = Actor()
    actor_state = {k: v for k, v in state.items() if k.startswith("actor.")}
    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    # 'std' belongs to the stochastic policy distribution, not the actor MLP.
    if missing or unexpected:
        raise RuntimeError(f"Actor checkpoint mismatch. Missing={missing}, unexpected={unexpected}")
    actor.eval()
    return actor, checkpoint.get("iter", "unknown")


def make_observation(data, phase):
    # Training order: position(3), linear velocity(3), quaternion wxyz(4),
    # angular velocity(3), phase one-hot(4). Isaac angular velocity is rad/s;
    # cflib gyro logs are commonly degrees/s, so convert to radians/s.
    pos = [float(data[k]) for k in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z")]
    vel = [float(data[k]) for k in ("stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz")]
    roll, pitch, yaw = [math.radians(float(data[k])) for k in
                        ("stateEstimate.roll", "stateEstimate.pitch", "stateEstimate.yaw")]
    quat = list(quaternion_wxyz_from_euler(roll, pitch, yaw))
    ang = [math.radians(float(data[k])) for k in ("gyro.x", "gyro.y", "gyro.z")]
    one_hot = [1.0 if i == phase else 0.0 for i in range(4)]
    values = pos + vel + quat + ang + one_hot
    if len(values) != 17:
        raise RuntimeError(f"Expected 17 observation values, got {len(values)}")
    return values


def main():
    parser = argparse.ArgumentParser(description="Diagnostic-only Crazyflie RL policy test (no flight commands).")
    parser.add_argument("--uri", default=DEFAULT_URI, help=f"Crazyradio URI (default: {DEFAULT_URI})")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="Path to freefall_policy.pt")
    parser.add_argument("--phase", type=int, choices=range(4), default=0,
                        help="One-hot phase supplied to the policy: 0=ascent, 1=freefall, 2=recovery, 3=landing")
    parser.add_argument("--duration", type=float, default=10.0, help="Telemetry duration in seconds")
    parser.add_argument("--period-ms", type=int, default=50, help="Telemetry logging period in milliseconds")
    args = parser.parse_args()

    if not os.path.isfile(args.policy):
        sys.exit(f"ERROR: checkpoint not found: {args.policy}\nPut it beside this script or pass --policy /full/path/freefall_policy.pt")

    actor, iteration = load_actor(args.policy)
    print(f"Loaded checkpoint iteration: {iteration}")
    print("Actor architecture from checkpoint weights: 17 -> 256 -> 256 -> 256 -> 4 (ELU hidden layers)")
    print("MODE: DIAGNOSTIC ONLY — no setpoints or motor commands will be sent.")
    print(f"Phase input: {args.phase}; duration: {args.duration:.1f}s")

    cflib.crtp.init_drivers(enable_debug_driver=False)
    variables = [
        ("stateEstimate.x", "float"), ("stateEstimate.y", "float"), ("stateEstimate.z", "float"),
        ("stateEstimate.vx", "float"), ("stateEstimate.vy", "float"), ("stateEstimate.vz", "float"),
        ("stateEstimate.roll", "float"), ("stateEstimate.pitch", "float"), ("stateEstimate.yaw", "float"),
        ("gyro.x", "float"), ("gyro.y", "float"), ("gyro.z", "float"),
    ]

    sample_count = 0
    last_print = 0.0

    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        log_conf = LogConfig(name="PolicyDiag", period_in_ms=max(20, args.period_ms))
        for name, kind in variables:
            log_conf.add_variable(name, kind)

        def on_log(timestamp, data, logconf):
            nonlocal sample_count, last_print
            try:
                obs_values = make_observation(data, args.phase)
                obs = torch.tensor(obs_values, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    actions = actor(obs).squeeze(0).tolist()
                sample_count += 1
                now = time.monotonic()
                if now - last_print >= 0.25:
                    last_print = now
                    print(
                        f"z={obs_values[2]:+.3f}m  vz={obs_values[5]:+.3f}m/s  "
                        f"obs={len(obs_values)}  policy_action="
                        f"[{actions[0]:+.3f}, {actions[1]:+.3f}, {actions[2]:+.3f}, {actions[3]:+.3f}]"
                    )
            except Exception as exc:
                print(f"Telemetry/policy error: {exc}", file=sys.stderr)

        def on_log_error(logconf, msg):
            print(f"Logging error: {msg}", file=sys.stderr)

        log_conf.data_received_cb.add_callback(on_log)
        log_conf.error_cb.add_callback(on_log_error)
        log_conf.start()
        print("Connected. Reading state estimates and evaluating policy; this script will not command flight.")
        try:
            time.sleep(max(0.0, args.duration))
        finally:
            log_conf.stop()

    print(f"Finished. Valid telemetry/policy samples: {sample_count}")
    print("Reminder: this test verifies inference dimensions only; it does not validate real-flight control or action scaling.")


if __name__ == "__main__":
    main()
