#!/usr/bin/env python3
"""
Crazyflie 2.1+ flight with the VRS direct-wrench policy (vrs_wrench_policy.pt).

Same path as the Isaac task crazyflie_ypath_VRS_nasa_wrench_xyland.py:
    phase 0  ascent to --top (default 1.70 m)            onboard position control
             + hover --hover-seconds (prop wash builds, hover thrust measured)
    phase 1  free fall (motors ~off) until vz <= -restart speed
    phase 2  POLICY brakes and captures at 0.60-0.70 m, |vz| <= 0.20 m/s
    phase 3  POLICY climbs back to 1.70 m
    phase 4  POLICY lands (or onboard landing with --landing onboard)

Who flies what:
    * Ascent and hover: the onboard position controller. The sim had no hover, but
      the hover is needed to measure the real hover thrust for the thrust mapping
      and it lets the wake from the drone settle below it before the drop.
    * Free fall: thrust --hold-thrust (2000, ~3% lift) so the onboard attitude
      controller stays on. Thrust 0 switches it off and the drone tumbles.
    * Phases 2-4: the policy's thrust, action[0]:
          thrust_N = (clip(a0, -1, 1) + 1) * 0.29          (env _apply_action)
      The policy's body torques (action[1:3] * 0.001 N m) cannot be sent through
      cflib, so the onboard attitude controller keeps the drone level instead
      (optionally with --xy-hold, which tilts it up to 3 deg toward the take-off x/y).

Observation (18 values, env _get_observations):
    [x, y, z (local), vx, vy, vz (world), qw, qx, qy, qz, wx, wy, wz (world, rad/s),
     phase one-hot(5)]
    z fed to the policy = Loco z - (top - 1.70) in phases 2-3, so the policy's
    1.70 m target sits at --top. In phase 4 it is Loco z - pad height, so the
    policy lands on the foam (its "floor" is z = 0).

Restart speed: the policy was trained to restart at -2.9 m/s. The sim motors give
T/W 1.9 instantly; your drone gave ~1.4 with ~0.05 s spin-up. Run with the policy
in the loop and the real motors, -2.9 m/s from 1.70 m hits the floor. The default
here is 1.75 m/s, which you have flown before (ratio vz/vh ~0.41 at the switch, with
the peak during braking ~0.55). The script simulates the policy with your measured
thrust before every drop and refuses if the lowest point is below the floor margin.

Safety: without --arm, no setpoints are sent (dry run). Ctrl+C once = controlled
landing, twice = motors off. Geofence, telemetry timeout, phase timeouts and a
floor threshold are all active. Power-cycle the drone before every run.

Usage:
    python3 vrs_wrench_flight.py --pad-height 0.33                    # dry run
    python3 vrs_wrench_flight.py --arm --ceiling 2.4 --pad-height 0.33 --xy-hold
"""

import argparse
import csv
import math
import os
import sys
import threading
import time

import numpy as np

# --------------------------------------------------------------------------
# Constants from crazyflie_ypath_VRS_nasa_wrench_xyland.py
# --------------------------------------------------------------------------
MASS_KG = 0.031156
G = 9.81
HOVER_FORCE_N = MASS_KG * G
SIM_THRUST_SCALE_N = 0.29          # thrust_N = (a0 + 1) * 0.29 -> max 0.58 N, T/W 1.9
SIM_THRUST_TO_WEIGHT = 2 * SIM_THRUST_SCALE_N / HOVER_FORCE_N
N_OBS = 18

START_HEIGHT = 0.20
TOP_HEIGHT = 1.70                  # freefall_release_height_m / climb_complete_height_m
TRAINED_RESTART_SPEED = 2.9        # freefall_restart_velocity_mps = -2.9
CAPTURE_LOW, CAPTURE_HIGH = 0.60, 0.70
CAPTURE_VZ = 0.20
CAPTURE_FAIL = 0.55
BRAKE_TIMEOUT = 2.0                # brake_timeout_steps 200 x 0.01 s
CLIMB_TIMEOUT = 5.0                # climb_timeout_steps 500
LANDING_TIMEOUT = 8.0              # landing_timeout_steps 800

# Hover induced velocity v_h = sqrt(mg / (2 rho A_total)) = 4.28 m/s
# (momentum theory; Bauersfeld et al. RA-L 2024 Eq. 1, your notebook value).
V_HOVER_INDUCED = 4.28

# Crazyflie 2.x per-motor thrust curve: thrust_N = A*pwm^2 + B*pwm + C
THRUST_A = 2.130295e-11
THRUST_B = 1.032633e-6
THRUST_C = 5.484560e-4
PWM_MIN, PWM_MAX = 10001, 65535

CONTROL_HZ = 100.0
DT = 1.0 / CONTROL_HZ

DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"
DEFAULT_POLICY = "vrs_wrench_policy.pt"

PHASE_NAMES = {0: "ASCENT", 1: "FREEFALL", 2: "BRAKE/CAPTURE", 3: "CLIMB", 4: "LANDING"}


class SafetyAbort(Exception):
    pass


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
def load_actor_weights(path):
    """Load the actor MLP (18 -> 256 -> 256 -> 256 -> 4) as numpy arrays."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    shapes = {"actor.0.weight": (256, N_OBS), "actor.2.weight": (256, 256),
              "actor.4.weight": (256, 256), "actor.6.weight": (4, 256)}
    for k, s in shapes.items():
        if k not in sd or tuple(sd[k].shape) != s:
            got = tuple(sd[k].shape) if k in sd else None
            raise RuntimeError(f"Checkpoint mismatch at {k}: expected {s}, got {got}. "
                               "Is this the 18-observation VRS wrench policy?")
    return {k: v.detach().cpu().numpy().astype(np.float64)
            for k, v in sd.items() if k.startswith("actor.")}


class Policy:
    def __init__(self, weights):
        self.layers = [(weights[f"actor.{i}.weight"], weights[f"actor.{i}.bias"])
                       for i in (0, 2, 4, 6)]

    def __call__(self, obs):
        h = np.asarray(obs, dtype=np.float64)
        for i, (w, b) in enumerate(self.layers):
            h = w @ h + b
            if i < 3:
                h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))  # ELU
        return h


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def build_observation(v, phase, origin_xy, z_shift):
    q = np.array([v["stateEstimate.qw"], v["stateEstimate.qx"],
                  v["stateEstimate.qy"], v["stateEstimate.qz"]])
    n = np.linalg.norm(q)
    if n < 1e-6:
        raise SafetyAbort("Invalid estimator quaternion")
    q = q / n
    gyro_world = quat_to_rotmat(q) @ np.radians(
        [v["gyro.x"], v["gyro.y"], v["gyro.z"]])
    obs = np.zeros(N_OBS)
    obs[0] = v["stateEstimate.x"] - origin_xy[0]
    obs[1] = v["stateEstimate.y"] - origin_xy[1]
    obs[2] = v["stateEstimate.z"] - z_shift
    obs[3:6] = [v["stateEstimate.vx"], v["stateEstimate.vy"], v["stateEstimate.vz"]]
    obs[6:10] = q
    obs[10:13] = gyro_world
    obs[13 + phase] = 1.0
    if not np.isfinite(obs).all():
        raise SafetyAbort("Non-finite observation")
    return obs


# --------------------------------------------------------------------------
# Thrust conversion
# --------------------------------------------------------------------------
def motor_force(pwm):
    return THRUST_A * pwm ** 2 + THRUST_B * pwm + THRUST_C


def pwm_for_force_ratio(ratio, hover_pwm):
    """PWM that gives `ratio` x the thrust measured at hover_pwm."""
    target = max(0.0, ratio) * motor_force(hover_pwm)
    disc = THRUST_B ** 2 - 4 * THRUST_A * (THRUST_C - target)
    pwm = (-THRUST_B + math.sqrt(max(disc, 0.0))) / (2 * THRUST_A)
    return int(np.clip(pwm, PWM_MIN, PWM_MAX))


def policy_thrust(raw_action, hover_pwm, floor_thrust):
    """Policy action[0] -> Crazyflie thrust command. Never below floor_thrust,
    so the onboard attitude controller stays active when the policy asks for 0 N."""
    a0 = float(np.clip(raw_action[0], -1.0, 1.0))
    thrust_n = (a0 + 1.0) * SIM_THRUST_SCALE_N
    pwm = pwm_for_force_ratio(thrust_n / HOVER_FORCE_N, hover_pwm)
    if thrust_n < 0.02 * HOVER_FORCE_N:
        pwm = floor_thrust
    return max(pwm, floor_thrust), a0, thrust_n


def measured_thrust_to_weight(hover_pwm):
    return motor_force(PWM_MAX) / motor_force(hover_pwm)


# --------------------------------------------------------------------------
# Pre-flight check: simulate the POLICY in the loop with real motors
# --------------------------------------------------------------------------
def simulate_maneuver(policy, restart_speed, tw_max, motor_lag, latency, spin_down=0.0, floor=0.0):
    """1-D vertical simulation of phases 1-3 from TOP_HEIGHT with the policy
    choosing the thrust. Motors: first-order lag `motor_lag`, commands delayed by
    `latency`, thrust capped at tw_max x weight. No drag, no VRS loss (VRS would
    only make it worse; the policy compensates for the sim's VRS model itself).
    Heights are in the policy frame; `floor` is the pad top in that frame."""
    dt = 0.002
    z, vz, force, t = TOP_HEIGHT, 0.0, 0.0, 0.0
    phase, t_phase = 1, 0.0
    queue = []
    out = {"restart_z": None, "peak_v": 0.0, "low_z": math.inf, "capture_t": None,
           "capture_z": None, "climb_t": None, "result": "timeout"}
    obs = np.zeros(N_OBS)
    obs[6] = 1.0
    while t < 10.0:
        if phase == 1:
            cmd = 0.0
            if vz <= -restart_speed:
                phase, t_phase = 2, t
                out["restart_z"] = z
        else:
            obs[2], obs[5] = z, vz
            obs[13:] = 0.0
            obs[13 + phase] = 1.0
            a0 = float(np.clip(policy(obs)[0], -1, 1))
            cmd = min((a0 + 1) * SIM_THRUST_SCALE_N / HOVER_FORCE_N, tw_max)
        queue.append(cmd)
        delayed = queue.pop(0) if len(queue) > int(round(latency / dt)) else 0.0
        if phase == 1 and t < spin_down:
            delayed = 0.64          # props spinning down right after the cut (~0.36 g fall)
        force += (delayed - force) * min(1.0, dt / max(motor_lag, dt))
        vz += G * (force - 1.0) * dt
        z += vz * dt
        t += dt
        out["peak_v"] = max(out["peak_v"], -vz)
        if phase >= 2:
            out["low_z"] = min(out["low_z"], z)
        if z <= floor:
            out["result"] = f"hits the floor at {-vz:.2f} m/s"
            out["low_z"] = z
            return out
        if phase == 2 and CAPTURE_LOW <= z <= CAPTURE_HIGH and abs(vz) <= CAPTURE_VZ:
            phase = 3
            out["capture_t"], out["capture_z"], t_phase = t - t_phase, z, t
        elif phase == 2 and t - t_phase > BRAKE_TIMEOUT:
            out["result"] = "no capture within 2 s"
            return out
        elif phase == 3 and z >= TOP_HEIGHT:
            out["climb_t"] = t - t_phase
            out["result"] = "ok"
            return out
    return out


def print_sim(label, s):
    rz = f"{s['restart_z']:.2f}" if s["restart_z"] is not None else "n/a"
    cap = (f"capture at {s['capture_z']:.2f} m after {s['capture_t']:.2f} s"
           if s["capture_t"] is not None else "no capture")
    print(f"  {label}: restart at {rz} m, peak {s['peak_v']:.2f} m/s, lowest {s['low_z']:+.2f} m, "
          f"{cap} -> {s['result']}")


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------
class Telemetry:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = {}
        self.last = 0.0
        self.count = 0

    def update(self, data):
        with self.lock:
            self.values.update({k: float(x) for k, x in data.items()})
            self.last = time.monotonic()
            self.count += 1

    def get(self):
        with self.lock:
            return dict(self.values), time.monotonic() - self.last, self.count


LOG_BLOCKS = {
    "Pos": ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
            "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"],
    "Att": ["stateEstimate.qw", "stateEstimate.qx", "stateEstimate.qy",
            "stateEstimate.qz", "stabilizer.thrust"],
    "Gyro": ["gyro.x", "gyro.y", "gyro.z", "pm.vbat"],
}


def start_logging(cf, telemetry):
    from cflib.crazyflie.log import LogConfig
    configs = []
    for name, variables in LOG_BLOCKS.items():
        lc = LogConfig(name=name, period_in_ms=10)
        for var in variables:
            lc.add_variable(var, "float")
        lc.data_received_cb.add_callback(lambda ts, d, c: telemetry.update(d))
        lc.error_cb.add_callback(lambda c, msg: print(f"Log error {c.name}: {msg}",
                                                      file=sys.stderr))
        cf.log.add_config(lc)
        if not lc.valid:
            raise RuntimeError(f"Crazyflie rejected log block {name}")
        lc.start()
        configs.append(lc)
    return configs


def tilt_deg(v):
    qx, qy = v["stateEstimate.qx"], v["stateEstimate.qy"]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))


def yaw_deg_from(v):
    w, x, y, z = (v["stateEstimate.qw"], v["stateEstimate.qx"],
                  v["stateEstimate.qy"], v["stateEstimate.qz"])
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def stop_motors(cf):
    for _ in range(10):
        cf.commander.send_stop_setpoint()
        time.sleep(0.005)


# --------------------------------------------------------------------------
# Flight
# --------------------------------------------------------------------------
class Flight:
    def __init__(self, cf, telemetry, policy, args):
        self.cf = cf
        self.tel = telemetry
        self.policy = policy
        self.args = args
        self.top = args.top
        self.off = args.top - TOP_HEIGHT          # policy z = Loco z - off (phases 2-3)
        self.rows = []
        self.t0 = time.monotonic()
        self.next_tick = self.t0
        self.state = "INIT"
        self.hover_samples = []
        self.hover_pwm = None
        self.origin = None
        self.yaw = 0.0
        self.last = None
        self.stats = {}

    # ---- helpers ---------------------------------------------------------
    def tick(self):
        self.next_tick += DT
        time.sleep(max(0.0, self.next_tick - time.monotonic()))

    def read(self, check=True):
        v, age, _ = self.tel.get()
        if age > self.args.telemetry_timeout:
            raise SafetyAbort(f"Telemetry stale ({age * 1000:.0f} ms)")
        self.last = v
        if check:
            self.geofence(v)
        return v

    def geofence(self, v):
        z = v["stateEstimate.z"]
        ceiling = min(self.top + self.args.ceiling_margin, self.args.ceiling - 0.2)
        if z > ceiling:
            raise SafetyAbort(f"Above ceiling: z={z:.2f} m > {ceiling:.2f} m")
        drift = math.hypot(v["stateEstimate.x"] - self.origin[0],
                           v["stateEstimate.y"] - self.origin[1])
        if drift > self.args.max_drift:
            raise SafetyAbort(f"Horizontal drift {drift:.2f} m > {self.args.max_drift} m")
        if time.monotonic() - self.t0 > self.args.max_seconds:
            raise SafetyAbort("Maximum flight time reached")

    def log(self, v, **extra):
        self.rows.append({"t": time.monotonic() - self.t0, "state": self.state,
                          **v, **extra})

    def pos_setpoint(self, z):
        self.cf.commander.send_position_setpoint(self.origin[0], self.origin[1],
                                                 z, self.yaw)

    def ramp_to(self, z_from, z_to, speed):
        steps = max(1, int(abs(z_to - z_from) / speed * CONTROL_HZ))
        for i in range(1, steps + 1):
            v = self.read()
            z_cmd = z_from + (z_to - z_from) * i / steps
            self.pos_setpoint(z_cmd)
            self.log(v, z_cmd=z_cmd)
            self.tick()

    def hold(self, z, seconds, record_hover=False):
        for _ in range(int(seconds * CONTROL_HZ)):
            v = self.read()
            self.pos_setpoint(z)
            if record_hover:
                self.hover_samples.append(v.get("stabilizer.thrust", 0.0))
            self.log(v, z_cmd=z)
            self.tick()

    def unlock(self):
        for _ in range(20):
            self.cf.commander.send_setpoint(0, 0, 0, 0)
            self.tick()

    def policy_step(self, v, phase, z_shift):
        """One control step with the policy's thrust. Returns (raw, pwm)."""
        raw = self.policy(build_observation(v, phase, self.origin, z_shift))
        pwm, a0, thrust_n = policy_thrust(raw, self.hover_pwm, self.args.hold_thrust)
        roll, pitch = self.xy_cmd(v)
        self.cf.commander.send_setpoint(roll, pitch, 0.0, pwm)
        vz = v["stateEstimate.vz"]
        self.log(v, phase=phase, thrust_cmd=pwm, a0_clipped=a0, thrust_N=thrust_n,
                 tilt_deg=tilt_deg(v), roll_cmd=roll, pitch_cmd=pitch,
                 vz_over_vh=vz / V_HOVER_INDUCED, z_policy=v["stateEstimate.z"] - z_shift,
                 raw0=raw[0], raw1=raw[1], raw2=raw[2], raw3=raw[3])
        return raw, pwm

    # ---- the maneuver ----------------------------------------------------
    def run(self):
        a = self.args
        v = self.read(check=False)
        self.origin = (v["stateEstimate.x"], v["stateEstimate.y"])
        self.yaw = yaw_deg_from(v)
        z_ground = v["stateEstimate.z"]
        self.unlock()

        # ---- PHASE 0: ascent (onboard) + hover ----
        self.state = "P0_TAKEOFF"
        print(f"PHASE 0  take-off to {a.takeoff_z:.2f} m")
        self.ramp_to(z_ground, a.takeoff_z, 0.4)
        self.hold(a.takeoff_z, 1.0)
        self.state = "P0_CLIMB"
        print(f"PHASE 0  climb to {self.top:.2f} m")
        self.ramp_to(a.takeoff_z, self.top, a.climb_speed)
        self.state = "P0_HOVER"
        print(f"PHASE 0  hover {a.hover_seconds:.1f} s (wake builds, hover thrust measured)")
        self.hold(self.top, a.hover_seconds, record_hover=True)
        samples = self.hover_samples[len(self.hover_samples) // 3:]
        self.hover_pwm = float(np.median(samples)) if samples else 0.0
        if not 20000 <= self.hover_pwm <= 58000:
            raise SafetyAbort(f"Implausible hover thrust {self.hover_pwm:.0f}; aborting before the drop "
                              "(power-cycle the drone and retry)")
        tw_meas = measured_thrust_to_weight(self.hover_pwm)
        tw = min(tw_meas, a.thrust_to_weight)
        sim = simulate_maneuver(self.policy, a.restart_speed, tw, a.motor_lag, a.latency, a.spin_down,
                                a.pad_height - self.off)
        print(f"  hover thrust {self.hover_pwm:.0f} -> max T/W ~{tw_meas:.2f} (planning with {tw:.2f})")
        print_sim("prediction with measured thrust", sim)
        low_loco = sim["low_z"] + self.off
        if low_loco < a.floor_limit and not a.accept_floor_impact:
            raise SafetyAbort(f"Predicted lowest point {low_loco:+.2f} m < {a.floor_limit:.2f} m. "
                              "Landing instead of dropping.")

        # ---- PHASE 1: free fall ----
        self.state = "P1_FREEFALL"
        backup_z = self.top - a.restart_speed ** 2 / (2 * G) - 0.10
        print(f"PHASE 1  free fall (thrust {a.hold_thrust}) until vz <= -{a.restart_speed:.2f} m/s "
              f"(backup: z <= {backup_z:.2f} m)")
        t_drop = time.monotonic()
        z_cut = self.read(check=False)["stateEstimate.z"]
        vz_peak = 0.0
        while True:
            v = self.read()
            z, vz = v["stateEstimate.z"], v["stateEstimate.vz"]
            vz_peak = min(vz_peak, vz)
            roll, pitch = self.xy_cmd(v)
            self.cf.commander.send_setpoint(roll, pitch, 0, a.hold_thrust)
            self.log(v, phase=1, thrust_cmd=a.hold_thrust, tilt_deg=tilt_deg(v),
                     roll_cmd=roll, pitch_cmd=pitch, vz_over_vh=vz / V_HOVER_INDUCED)
            if vz <= -a.restart_speed or z <= backup_z:
                break
            if z <= a.impact_cut_z:
                stop_motors(self.cf)
                raise SafetyAbort(f"Floor threshold during the free fall (z={z:.2f} m)")
            if time.monotonic() - t_drop > 1.5:
                raise SafetyAbort("Free fall lasted > 1.5 s without reaching the restart speed")
            self.tick()
        t_ff = time.monotonic() - t_drop
        trig = "speed" if vz <= -a.restart_speed else "BACKUP HEIGHT"
        print(f"  restart ({trig}): z={z:.3f} m vz={vz:+.2f} m/s after {t_ff:.3f} s, "
              f"dropped {z_cut - z:.2f} m (Loco)")
        self.stats.update(restart_z=z - self.off, restart_v=-vz, peak_v=-vz_peak,
                          tilt_max=tilt_deg(v), vrs_ratio_max=-vz / V_HOVER_INDUCED)
        self.tick()

        # ---- PHASE 2: policy brakes and captures ----
        self.state = "P2_BRAKE"
        print(f"PHASE 2  POLICY braking, capture band {CAPTURE_LOW}-{CAPTURE_HIGH} m (policy frame), "
              f"|vz| <= {CAPTURE_VZ}")
        t_ph = time.monotonic()
        z_min = z
        brake = []
        captured = False
        while True:
            v = self.read()
            z, vz = v["stateEstimate.z"], v["stateEstimate.vz"]
            zp = z - self.off
            dt_ph = time.monotonic() - t_ph
            z_min = min(z_min, z)
            self.stats["peak_v"] = max(self.stats["peak_v"], -vz)
            self.stats["tilt_max"] = max(self.stats["tilt_max"], tilt_deg(v))
            # Powered descent is where VRS can occur: track vz / v_h while thrust is on.
            if vz < 0:
                self.stats["vrs_ratio_max"] = max(self.stats["vrs_ratio_max"], -vz / V_HOVER_INDUCED)
            if 0.06 <= dt_ph <= 0.25 and vz < 0:
                brake.append((dt_ph, vz))
            if len(brake) >= 3:
                ts, vs = np.array(brake).T
                acc = float(np.polyfit(ts, vs, 1)[0])
                self.stats["brake_acc"] = acc
                self.stats["implied_tw"] = 1.0 + acc / G
            self.policy_step(v, 2, self.off)
            if z <= a.impact_cut_z:
                self.state = "FLOOR_IMPACT"
                stop_motors(self.cf)
                self.stats["low_z"] = z_min - self.off
                print(f"  FLOOR THRESHOLD z={z:.3f} m (vz={vz:+.2f}) -> motors stopped")
                return
            if CAPTURE_LOW <= zp <= CAPTURE_HIGH and abs(vz) <= CAPTURE_VZ:
                captured = True
                break
            if dt_ph > a.brake_timeout:
                print(f"  no capture in {a.brake_timeout:.1f} s (z={zp:.2f} m policy frame, vz={vz:+.2f}); "
                      "continuing to the climb (the sim would end the episode here)")
                break
            self.tick()
        self.stats.update(low_z=z_min - self.off, captured=captured,
                          capture_t=dt_ph if captured else None, capture_z=zp)
        if z_min - self.off < CAPTURE_FAIL:
            print(f"  note: lowest {z_min - self.off:.2f} m was below the sim's capture-fail line "
                  f"{CAPTURE_FAIL} m (the sim would have ended the episode)")
        print(f"  {'captured' if captured else 'leaving phase 2'} at z={zp:.3f} m vz={vz:+.2f} after "
              f"{dt_ph:.2f} s; lowest {z_min - self.off:.3f} m (policy frame)")
        self.tick()

        # ---- PHASE 3: policy climb ----
        self.state = "P3_CLIMB"
        print(f"PHASE 3  POLICY climb to {self.top:.2f} m")
        t_ph = time.monotonic()
        handed_back = False
        while True:
            v = self.read()
            z, vz = v["stateEstimate.z"], v["stateEstimate.vz"]
            self.policy_step(v, 3, self.off)
            if z <= a.impact_cut_z:
                stop_motors(self.cf)
                raise SafetyAbort(f"Floor threshold during the climb (z={z:.2f} m)")
            if z - self.off >= TOP_HEIGHT:
                break
            apex = z + max(vz, 0.0) ** 2 / (2 * G)
            if apex >= self.top + a.overshoot:
                print(f"  predicted apex {apex:.2f} m > top + {a.overshoot} -> onboard hold")
                handed_back = True
                break
            if time.monotonic() - t_ph > CLIMB_TIMEOUT:
                raise SafetyAbort("Policy did not climb to the top in 5 s")
            self.tick()
        self.stats["climb_t"] = time.monotonic() - t_ph
        print(f"  reached z={z:.3f} m after {self.stats['climb_t']:.2f} s")

        # ---- PHASE 4: landing ----
        if a.landing == "onboard" or handed_back:
            self.state = "P4_ONBOARD_HOLD"
            self.next_tick = time.monotonic()
            self.hold(self.top, 1.0)
            self.state = "P4_ONBOARD_LAND"
            print("PHASE 4  onboard landing")
            self.land_from(self.top)
            return
        self.policy_landing()

    def policy_landing(self):
        """Phase 4 with the policy. Its floor is z = 0, so the observation is
        shifted by the pad height: it lands on the foam, not 0.33 m below it."""
        a = self.args
        self.state = "P4_LANDING"
        shift = a.pad_height
        print(f"PHASE 4  POLICY landing on the pad ({shift:.2f} m)")
        t_ph = time.monotonic()
        while True:
            v = self.read()
            z, vz = v["stateEstimate.z"], v["stateEstimate.vz"]
            zp = z - shift
            xy = math.hypot(v["stateEstimate.x"] - self.origin[0], v["stateEstimate.y"] - self.origin[1])
            self.policy_step(v, 4, shift)
            if zp <= a.touchdown_z:
                stop_motors(self.cf)
                self.stats.update(touchdown_v=-vz, touchdown_xy=xy)
                print(f"  TOUCHDOWN z={zp:.3f} m above pad, vz={vz:+.2f} m/s, {xy:.2f} m from take-off "
                      f"({time.monotonic() - t_ph:.2f} s)")
                self.cf.commander.send_notify_setpoint_stop()
                return
            if vz < -a.max_landing_speed:
                print(f"  landing too fast ({vz:+.2f} m/s) -> onboard controller takes over")
                self.next_tick = time.monotonic()
                self.land_from(max(z, a.ground_z + 0.2))
                return
            if time.monotonic() - t_ph > LANDING_TIMEOUT:
                print("  policy landing timed out -> onboard landing")
                self.land_from(max(z, a.ground_z + 0.1))
                return
            self.tick()

    # ---- horizontal hold ----------------------------------------------------
    def xy_cmd(self, v, target=None):
        """Small roll/pitch (deg) toward the take-off x/y. Thrust untouched."""
        a = self.args
        if not a.xy_hold and target is None:
            return 0.0, 0.0
        tx, ty = target if target is not None else self.origin
        ex, ey = tx - v["stateEstimate.x"], ty - v["stateEstimate.y"]
        ax = a.xy_kp * ex - a.xy_kd * v["stateEstimate.vx"]
        ay = a.xy_kp * ey - a.xy_kd * v["stateEstimate.vy"]
        a_max = G * math.tan(math.radians(a.xy_max_tilt))
        n = math.hypot(ax, ay)
        if n > a_max:
            ax, ay = ax * a_max / n, ay * a_max / n
        yaw = math.radians(yaw_deg_from(v))
        axb = math.cos(yaw) * ax + math.sin(yaw) * ay
        ayb = -math.sin(yaw) * ax + math.cos(yaw) * ay
        pitch = a.xy_sign_pitch * math.degrees(math.atan2(axb, G))
        roll = a.xy_sign_roll * math.degrees(math.atan2(ayb, G))
        return roll, pitch

    def land_from(self, z_from, speed=0.4):
        ground = self.args.ground_z
        steps = max(1, int(max(z_from - ground, 0.05) / speed * CONTROL_HZ))
        for i in range(1, steps + 1):
            z_cmd = z_from + (ground - z_from) * i / steps
            try:
                v = self.read(check=False)
                self.log(v, z_cmd=z_cmd)
            except SafetyAbort:
                pass
            self.pos_setpoint(z_cmd)
            self.tick()
        for _ in range(int(0.5 * CONTROL_HZ)):
            self.pos_setpoint(ground)
            self.tick()
        self.cf.commander.send_stop_setpoint()
        self.cf.commander.send_notify_setpoint_stop()

    def emergency_land(self, reason):
        print(f"\n*** ABORT: {reason} -> controlled landing (Ctrl+C again = motors off) ***")
        self.state = "EMERGENCY_LAND"
        if self.origin is None:
            self.cf.commander.send_stop_setpoint()
            return
        z = (self.last or {}).get("stateEstimate.z", self.top)
        self.next_tick = time.monotonic()
        for _ in range(int(0.5 * CONTROL_HZ)):
            self.pos_setpoint(max(z, self.args.ground_z + 0.3))
            self.tick()
        self.land_from(max(z, self.args.ground_z + 0.3))


def print_comparison(st, sim_ref, hover_pwm):
    if not st:
        return
    def fmt(x):
        return f"{x:8.2f}" if isinstance(x, (int, float)) and x is not None and math.isfinite(x) else "     n/a"
    sim_acc = None
    rows = [
        ("restart height (m)", sim_ref.get("restart_z"), st.get("restart_z")),
        ("peak descent speed (m/s)", sim_ref.get("peak_v"), st.get("peak_v")),
        ("peak vz/vh (VRS ratio)", sim_ref.get("peak_v", 0) / V_HOVER_INDUCED, st.get("vrs_ratio_max")),
        ("lowest point (m)", sim_ref.get("low_z"), st.get("low_z")),
        ("capture time (s)", sim_ref.get("capture_t"), st.get("capture_t")),
        ("climb time (s)", sim_ref.get("climb_t"), st.get("climb_t")),
        ("braking accel (m/s^2)", sim_acc, st.get("brake_acc")),
        ("implied thrust/weight", SIM_THRUST_TO_WEIGHT, st.get("implied_tw")),
        ("max tilt (deg)", 0.0, st.get("tilt_max")),
    ]
    print("\nSIM (policy, T/W 1.9, instant motors, same restart speed) vs REAL  [policy frame]")
    print(f"  {'':28}{'sim':>8}{'real':>8}")
    for name, s_val, r_val in rows:
        print(f"  {name:28}{fmt(s_val)}{fmt(r_val)}")
    if hover_pwm:
        print(f"  max T/W from the measured hover thrust: {measured_thrust_to_weight(hover_pwm):.2f}")
    if "touchdown_v" in st:
        print(f"  touchdown {st['touchdown_v']:.2f} m/s, {st['touchdown_xy']:.2f} m from take-off "
              f"(sim 'landed' needs <= 0.5 m/s and <= 0.10 m)")
    print("  VRS bands for vz/vh: Johnson 2005 ~0.5-1.5; Drees 0.62-1.53; Longobardo 2025 (quad) deep VRS 0.8-1.3.")


def save_rows(rows, path):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "t", k != "state", k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved log: {path} ({len(rows)} rows)")


def arm(cf):
    for obj in ("supervisor", "platform"):
        fn = getattr(getattr(cf, obj, None), "send_arming_request", None)
        if fn:
            try:
                fn(True)
                return True
            except Exception:
                pass
    return False


def main():
    p = argparse.ArgumentParser(description="Crazyflie 2.1+ flight with the VRS direct-wrench policy")
    p.add_argument("--uri", default=DEFAULT_URI)
    p.add_argument("--policy", default=DEFAULT_POLICY)
    p.add_argument("--arm", action="store_true", help="Actually fly. Without it: dry run, no setpoints.")
    p.add_argument("--ceiling", type=float, default=None, help="Room height (m). Required with --arm.")
    p.add_argument("--top", type=float, default=TOP_HEIGHT, help="Release / climb height (m), Loco frame")
    p.add_argument("--pad-height", type=float, default=0.0, help="Foam pad height (m), e.g. 0.33")
    p.add_argument("--restart-speed", type=float, default=1.75,
                   help=f"Free fall ends at this descent speed (m/s). Trained: {TRAINED_RESTART_SPEED}")
    p.add_argument("--landing", choices=["policy", "onboard"], default="policy",
                   help="Phase 4: the policy lands on the pad, or the onboard controller does")
    p.add_argument("--hover-seconds", type=float, default=2.5)
    p.add_argument("--climb-speed", type=float, default=0.5)
    p.add_argument("--hold-thrust", type=int, default=2000,
                   help="Thrust during free fall / when the policy asks for 0 N (keeps attitude control on)")
    p.add_argument("--xy-hold", action="store_true", help="Small roll/pitch to hold the take-off x/y")
    p.add_argument("--xy-kp", type=float, default=1.0)
    p.add_argument("--xy-kd", type=float, default=1.2)
    p.add_argument("--xy-max-tilt", type=float, default=3.0)
    p.add_argument("--xy-sign-pitch", type=int, choices=[-1, 1], default=1)
    p.add_argument("--xy-sign-roll", type=int, choices=[-1, 1], default=-1)
    p.add_argument("--thrust-to-weight", type=float, default=1.4, help="Assumed real max T/W (your logs: ~1.4)")
    p.add_argument("--motor-lag", type=float, default=0.05)
    p.add_argument("--latency", type=float, default=0.03)
    p.add_argument("--spin-down", type=float, default=0.15,
                   help="Seconds the props keep some lift after the cut (your logs: 0.1-0.2)")
    p.add_argument("--floor-margin", type=float, default=0.15,
                   help="Minimum predicted clearance above the pad (m)")
    p.add_argument("--accept-floor-impact", action="store_true")
    p.add_argument("--impact-cut-z", type=float, default=0.12, help="Motors stop this close to the pad (m)")
    p.add_argument("--brake-timeout", type=float, default=BRAKE_TIMEOUT)
    p.add_argument("--touchdown-z", type=float, default=0.05, help="Policy landing: motors stop at this height above the pad")
    p.add_argument("--max-landing-speed", type=float, default=1.2,
                   help="Policy landing: hand to onboard control if descending faster (m/s)")
    p.add_argument("--overshoot", type=float, default=0.15)
    p.add_argument("--ground-z", type=float, default=0.03)
    p.add_argument("--max-drift", type=float, default=0.6)
    p.add_argument("--ceiling-margin", type=float, default=0.3)
    p.add_argument("--telemetry-timeout", type=float, default=0.15)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--dry-seconds", type=float, default=10.0)
    p.add_argument("--log", default="vrs_wrench_flight_log.csv")
    args = p.parse_args()

    top, pad = args.top, args.pad_height
    off = top - TOP_HEIGHT
    args.takeoff_z = max(START_HEIGHT, pad + 0.2)
    args.ground_z += pad
    args.impact_cut_z += pad
    args.floor_limit = pad + args.floor_margin
    if args.arm and args.ceiling is None:
        sys.exit("--arm needs --ceiling <room height in m>.")
    if args.ceiling is not None and top > args.ceiling - 0.4:
        sys.exit(f"--top {top} m is within 0.4 m of the {args.ceiling} m ceiling.")
    if not os.path.isfile(args.policy):
        sys.exit(f"Policy not found: {args.policy}")
    policy = Policy(load_actor_weights(args.policy))
    print(f"Loaded policy {args.policy} (18 obs, 5 phases)")

    # ---- plan: the policy in the loop, sim motors vs real motors ----
    sim_ref = simulate_maneuver(policy, args.restart_speed, SIM_THRUST_TO_WEIGHT, 0.001, 0.0)
    real = simulate_maneuver(policy, args.restart_speed, args.thrust_to_weight, args.motor_lag,
                             args.latency, args.spin_down, pad - off)
    trained = simulate_maneuver(policy, TRAINED_RESTART_SPEED, args.thrust_to_weight, args.motor_lag,
                                args.latency, args.spin_down, pad - off)
    print(f"Flight plan (heights in the policy frame = Loco z {0.0 - off + 0.0:+.2f} m; pad {pad:.2f} m):")
    print(f"  pad -> {top:.2f} m hover {args.hover_seconds}s -> free fall to -{args.restart_speed} m/s "
          f"-> policy capture 0.60-0.70 m -> policy climb to 1.70 -> {args.landing} landing")
    print_sim("sim motors (T/W 1.9)", sim_ref)
    print_sim(f"real motors (T/W {args.thrust_to_weight})", real)
    if args.restart_speed != TRAINED_RESTART_SPEED:
        print_sim(f"trained -{TRAINED_RESTART_SPEED} m/s, real motors", trained)
    low_loco = real["low_z"] + off
    if low_loco < args.floor_limit:
        print(f"  !! predicted lowest point {low_loco:+.2f} m (Loco) is below pad {pad:.2f} + "
              f"margin {args.floor_margin} = {args.floor_limit:.2f} m. Lower --restart-speed.")
        if args.arm and not args.accept_floor_impact:
            sys.exit("Refusing to arm.")

    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

    cflib.crtp.init_drivers()
    telemetry = Telemetry()
    flight = None
    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(2.0)
        configs = start_logging(cf, telemetry)
        deadline = time.monotonic() + 5
        while telemetry.get()[2] < 20 and time.monotonic() < deadline:
            time.sleep(0.05)
        v, age, n = telemetry.get()
        if n < 20 or age > 0.2:
            sys.exit("No fresh telemetry. Power-cycle the drone and check the Loco deck.")
        print(f"Telemetry OK. z={v['stateEstimate.z']:.3f} m, battery {v.get('pm.vbat', 0):.2f} V")
        if abs(v["stateEstimate.z"] - pad) > 0.15:
            msg = (f"Drone on the pad reads z={v['stateEstimate.z']:+.2f} m but --pad-height is {pad:.2f} m.")
            if args.arm:
                sys.exit(msg)
            print("  !! " + msg)
        if v.get("pm.vbat", 4.2) < 3.8:
            print("  !! Battery below 3.8 V. Charge it: braking needs every bit of thrust.")

        try:
            if not args.arm:
                print("DRY RUN: no setpoints are sent. Live policy thrust for each phase "
                      "(at hover PWM 42000):")
                origin = (v["stateEstimate.x"], v["stateEstimate.y"])
                end = time.monotonic() + args.dry_seconds
                while time.monotonic() < end:
                    v, _, _ = telemetry.get()
                    parts = []
                    for ph in (2, 3, 4):
                        shift = pad if ph == 4 else off
                        raw = policy(build_observation(v, ph, origin, shift))
                        pwm, a0, thr = policy_thrust(raw, 42000, args.hold_thrust)
                        parts.append(f"P{ph} a0={a0:+.2f} {thr:.2f}N pwm={pwm}")
                    print(f"  z={v['stateEstimate.z']:+.3f} vz={v['stateEstimate.vz']:+.2f} | " + " | ".join(parts))
                    time.sleep(0.25)
                return

            print("\n*** ARMED FLIGHT. Clear area, foam below, hand near the kill switch. ***")
            input("Press ENTER to take off (Ctrl+C to cancel)... ")
            arm(cf)
            time.sleep(0.5)
            flight = Flight(cf, telemetry, policy, args)
            try:
                flight.run()
                print("DONE.")
                print_comparison(flight.stats, sim_ref, flight.hover_pwm)
            except (SafetyAbort, KeyboardInterrupt, Exception) as exc:
                reason = "Ctrl+C" if isinstance(exc, KeyboardInterrupt) else str(exc)
                try:
                    flight.emergency_land(reason)
                except KeyboardInterrupt:
                    print("\n*** EMERGENCY STOP: motors off ***")
                    try:
                        cf.loc.send_emergency_stop()
                    except Exception:
                        pass
                    cf.commander.send_stop_setpoint()
                print_comparison(flight.stats, sim_ref, flight.hover_pwm)
                if not isinstance(exc, (SafetyAbort, KeyboardInterrupt)):
                    raise
        finally:
            if flight:
                save_rows(flight.rows, args.log)
            for c in configs:
                try:
                    c.stop()
                except Exception:
                    pass
            try:
                cf.commander.send_stop_setpoint()
            except Exception:
                pass


if __name__ == "__main__":
    main()
