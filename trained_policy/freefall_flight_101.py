#!/usr/bin/env python3
"""
Crazyflie 2.1+ freefall maneuver flown with freefall_policy.pt.

Path (heights above the floor):
    ground -> 0.20 m (settle) -> --top (default 1.70 m; hover, measure hover thrust)
    -> motors off, free fall to the recovery height
    -> POLICY recovery back up to the top -> hold -> land at 0 m

Recovery height: the sim used 0.695 m, but a real 2.1+ cannot stop a 4 m/s
fall in 0.695 m. With --recover-height auto (default) the script picks the
lowest trigger whose predicted dip still clears the floor by --floor-margin.

Who flies what:
    * Take-off, climb, hover, final hold and landing: the Crazyflie's own
      onboard position controller (position setpoints). Policy not used.
    * Free fall: zero thrust, exactly like phase 1 in the Isaac env.
    * Recovery (phase 2): the policy's thrust output (action[0]) drives the
      motors. Roll/pitch are held level and yaw rate is 0 by the onboard
      attitude controller, because Isaac's body torques have no faithful
      equivalent in the Crazyflie commander interface.

Isaac conventions used (taken from your earlier hardware scripts):
    obs  = [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz (world, rad/s),
            phase one-hot(4)]                                   -> 17 values
    act  = clip(raw, -1, 1);  thrust_N = (a0 + 1) * 0.29  (T/W 1.9, m=31.156 g)

Safety:
    * Without --arm the script only connects, reads telemetry and runs the
      policy. It sends NO setpoints.
    * Before arming it predicts the lowest point of the recovery dip and
      refuses to fly if that is below the floor margin. It also refuses
      if --top is closer than 0.4 m to --ceiling.
    * Ctrl+C once: controlled landing at the current x/y.
      Ctrl+C twice: emergency motor stop (drone drops).
    * Geofence, telemetry-timeout and flight-time limits all trigger a
      controlled landing.
    Fly over a net or foam, with nobody in the area.

Usage:
    python freefall_flight.py                       # dry run, no motors
    python freefall_flight.py --arm --ceiling 2.5   # real flight (room height in m)

Loco Positioning: the anchor coordinates must have z = 0 at the floor. The
script checks that the drone reads z ~ 0 sitting on the ground before arming.
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
# Constants from the Isaac environment / training
# --------------------------------------------------------------------------
MASS_KG = 0.031156
G = 9.81
HOVER_FORCE_N = MASS_KG * G
SIM_THRUST_SCALE_N = 0.29          # thrust_N = (a0 + 1) * 0.29
SIM_THRUST_TO_WEIGHT = 1.9

START_HEIGHT = 0.20
TOP_HEIGHT = 1.70
FREEFALL_RECOVER_HEIGHT = 0.695
RECOVERY_TRIGGER_VZ = -4.5
PHASE_FREEFALL = 1
PHASE_RECOVERY = 2

# What the policy did in Isaac (your sim log), for the sim-vs-real comparison.
SIM_REF = {"switch_z": 0.616, "switch_v": 4.01, "peak_v": 4.18, "low_z": 0.476}

# Crazyflie 2.x per-motor thrust curve: thrust_N = A*pwm^2 + B*pwm + C
THRUST_A = 2.130295e-11
THRUST_B = 1.032633e-6
THRUST_C = 5.484560e-4
PWM_MIN, PWM_MAX = 10001, 65535

CONTROL_HZ = 100.0
DT = 1.0 / CONTROL_HZ

DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"
DEFAULT_POLICY = "freefall_policy.pt"


class SafetyAbort(Exception):
    pass


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
def load_actor_weights(path):
    """Load the actor MLP (17 -> 256 -> 256 -> 256 -> 4) as numpy arrays."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    shapes = {"actor.0.weight": (256, 17), "actor.2.weight": (256, 256),
              "actor.4.weight": (256, 256), "actor.6.weight": (4, 256)}
    for k, s in shapes.items():
        if k not in sd or tuple(sd[k].shape) != s:
            raise RuntimeError(f"Checkpoint mismatch at {k}: expected {s}")
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
    """Observation in the simulator's frame: x/y relative to the take-off
    point, z shifted by (top - 1.70) so the policy's target matches --top."""
    q = np.array([v["stateEstimate.qw"], v["stateEstimate.qx"],
                  v["stateEstimate.qy"], v["stateEstimate.qz"]])
    n = np.linalg.norm(q)
    if n < 1e-6:
        raise SafetyAbort("Invalid estimator quaternion")
    q = q / n
    gyro_world = quat_to_rotmat(q) @ np.radians(
        [v["gyro.x"], v["gyro.y"], v["gyro.z"]])
    obs = np.zeros(17)
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


def policy_thrust_pwm(raw_action, hover_pwm):
    a0 = float(np.clip(raw_action[0], -1.0, 1.0))
    thrust_n = (a0 + 1.0) * SIM_THRUST_SCALE_N
    return pwm_for_force_ratio(thrust_n / HOVER_FORCE_N, hover_pwm), a0


# --------------------------------------------------------------------------
# Pre-flight physics check
# --------------------------------------------------------------------------
def predict_dip(top, recover, thrust_to_weight, motor_lag, latency, trigger_lead=0.0):
    """Step-simulate drop + full-thrust recovery. Returns (speed at trigger,
    lowest height). Motors respond as a first-order lag; commands arrive
    `latency` seconds late (radio + telemetry). Drag is ignored (pessimistic)."""
    dt = 0.001
    z, vz, force, t = top, 0.0, 0.0, 0.0
    t_trig = None
    v_trig = 0.0
    while True:
        if t_trig is None and (z + vz * trigger_lead <= recover or vz <= RECOVERY_TRIGGER_VZ):
            t_trig, v_trig = t, -vz
        cmd = thrust_to_weight if (t_trig is not None and t >= t_trig + latency) else 0.0
        force += (cmd - force) * dt / max(motor_lag, dt)
        vz += G * (force - 1.0) * dt
        z += vz * dt
        t += dt
        if t_trig is not None and vz >= 0:
            return v_trig, z
        if t > 5:
            return v_trig, -math.inf


def auto_recover_height(top, floor_limit, tw, args):
    """Lowest recovery trigger whose predicted dip stays above floor_limit.
    Returns None if even a tiny drop isn't safe."""
    best = None
    r = top - 0.05
    while r > FREEFALL_RECOVER_HEIGHT - 1e-9:
        if predict_dip(top, r, tw, args.motor_lag, args.latency, args.trigger_lead)[1] < floor_limit:
            break
        best = round(r, 3)
        r -= 0.005
    return best


def measured_thrust_to_weight(hover_pwm):
    return motor_force(PWM_MAX) / motor_force(hover_pwm)


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
    """Angle between the drone's up-axis and vertical (0 = level)."""
    qx, qy = v["stateEstimate.qx"], v["stateEstimate.qy"]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))


def yaw_deg_from(v):
    w, x, y, z = (v["stateEstimate.qw"], v["stateEstimate.qx"],
                  v["stateEstimate.qy"], v["stateEstimate.qz"])
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


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
        self.recover = args.recover_z
        self.off = args.top - TOP_HEIGHT        # z shift fed to the policy
        self.rows = []
        self.t0 = time.monotonic()
        self.next_tick = self.t0
        self.state = "INIT"
        self.hover_samples = []
        self.hover_pwm = None
        self.origin = None
        self.yaw = 0.0
        self.last = None

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

    # ---- the maneuver ----------------------------------------------------
    def run(self):
        a = self.args
        v = self.read(check=False)
        self.origin = (v["stateEstimate.x"], v["stateEstimate.y"])
        self.yaw = yaw_deg_from(v)
        z_ground = v["stateEstimate.z"]

        # Unlock the commander's thrust lock with zero setpoints.
        for _ in range(20):
            self.cf.commander.send_setpoint(0, 0, 0, 0)
            self.tick()

        self.state = "TAKEOFF"
        print(f"TAKEOFF to {a.takeoff_z:.2f} m")
        self.ramp_to(z_ground, a.takeoff_z, 0.4)
        self.hold(a.takeoff_z, 1.0)

        self.state = "CLIMB"
        top = self.top
        print(f"CLIMB to {top:.2f} m")
        self.ramp_to(a.takeoff_z, top, a.climb_speed)

        self.state = "HOVER"
        print(f"HOVER {a.hover_seconds:.1f} s at {top:.2f} m (prop wash builds, hover thrust measured)")
        self.hold(top, a.hover_seconds, record_hover=True)
        samples = self.hover_samples[len(self.hover_samples) // 3:]
        self.hover_pwm = float(np.median(samples)) if samples else 0.0
        if not 20000 <= self.hover_pwm <= 58000:
            raise SafetyAbort(f"Implausible hover thrust {self.hover_pwm:.0f}; aborting before drop")
        tw_meas = measured_thrust_to_weight(self.hover_pwm)
        tw = min(tw_meas, a.thrust_to_weight)
        print(f"  measured hover thrust = {self.hover_pwm:.0f} -> max thrust/weight ~{tw_meas:.2f}"
              f" (planning with {tw:.2f})")
        if a.recover_height == "auto" and a.switch_at_speed is None:
            new = auto_recover_height(top, a.floor_limit, tw, a)
            if new is None:
                raise SafetyAbort(f"Only {tw_meas:.2f}x thrust: no drop is safe. Landing instead.")
            if abs(new - self.recover) > 0.004:
                print(f"  recovery height re-planned with measured thrust: "
                      f"{self.recover:.3f} -> {new:.3f} m")
            self.recover = new
        plan_z = a.plan_z if a.switch_at_speed is not None else self.recover
        _, z_low = predict_dip(top, plan_z, tw, a.motor_lag, a.latency, a.trigger_lead)
        print(f"  predicted lowest point with measured thrust: {z_low:+.2f} m")
        if z_low < a.floor_limit and not a.accept_floor_impact:
            raise SafetyAbort(f"Measured thrust predicts the dip reaches {z_low:+.2f} m "
                              f"(< {a.floor_limit:.2f}). Landing instead of dropping.")

        # ---- FREEFALL (phase 1): motors off ----
        self.state = "FREEFALL"
        recover_z = self.recover
        mode_txt = {"off": "motors off", "idle": "idle thrust",
                    "hold": f"near-zero thrust {a.hold_thrust} (attitude control stays on)",
                    "hybrid": f"FREE FALL (thrust {a.hold_thrust}) until the switch, then the POLICY "
                              f"flies the descent to {a.phase2_height} m with its phase-1 input",
                    "policy": "POLICY controls thrust (phase-1 obs, like the sim)"}[a.freefall_mode]
        print(f"FREEFALL: {mode_txt}; recovery at z <= {recover_z:.3f} m or vz <= {RECOVERY_TRIGGER_VZ}")
        t_drop = time.monotonic()
        z_cut = self.read(check=False)["stateEstimate.z"]
        t_switch = (math.sqrt(2 * a.switch_after_drop / G) + a.spin_down
                    if a.switch_after_drop else None)
        if t_switch:
            print(f"  time-based switch: {a.switch_after_drop:.3f} m of free fall -> "
                  f"{t_switch:.3f} s after the cut (not using Loco height)")
        vz_peak = 0.0
        while True:
            v = self.read()
            z, vz = v["stateEstimate.z"], v["stateEstimate.vz"]
            vz_peak = min(vz_peak, vz)
            extra = {"tilt_deg": tilt_deg(v)}
            if a.freefall_mode == "off":
                cmd = 0
            elif a.freefall_mode == "idle":
                cmd = PWM_MIN
            elif a.freefall_mode in ("hold", "hybrid"):
                cmd = a.hold_thrust      # tiny thrust: motors almost off, attitude control stays on
            else:
                raw = self.policy(build_observation(v, PHASE_FREEFALL, self.origin, self.off))
                cmd, a0 = policy_thrust_pwm(raw, self.hover_pwm)
                if raw[0] <= -1.0:
                    cmd = 0          # policy asks for zero thrust -> motors fully off
                extra.update(a0_clipped=a0, raw0=raw[0])
            self.cf.commander.send_setpoint(0, 0, 0, cmd)
            self.log(v, thrust_cmd=cmd, **extra)
            # Lead the trigger by the telemetry delay so we don't start late.
            if a.switch_at_speed is not None:
                if vz <= -a.switch_at_speed or z <= recover_z:
                    break
            elif t_switch is not None:
                if time.monotonic() - t_drop >= t_switch or z <= recover_z:
                    break
            elif z + vz * a.trigger_lead <= recover_z or vz <= RECOVERY_TRIGGER_VZ:
                break
            if a.freefall_mode == "policy" and z <= a.impact_cut_z:
                for _ in range(10):
                    self.cf.commander.send_stop_setpoint()
                    time.sleep(0.005)
                raise SafetyAbort(f"Floor threshold during the drop (z={z:.2f} m)")
            limit = 1.5 if a.freefall_mode != "policy" else 4.0
            if time.monotonic() - t_drop > limit:
                raise SafetyAbort(f"Drop lasted > {limit:.0f} s without reaching the switch point")
            self.tick()
        t_ff = time.monotonic() - t_drop
        phys = 0.5 * G * max(t_ff - a.spin_down, 0) ** 2
        print(f"  recovery start: z={z:.3f} m vz={vz:+.2f} m/s after {t_ff:.3f} s")
        print(f"  drop by Loco: {z_cut - z:.2f} m | by free-fall physics (1/2 g t^2): {phys:.2f} m"
              f"  -> if physics is much bigger, the Loco height is lagging")
        self.stats = {"switch_z": z, "switch_v": -vz, "peak_v": -vz_peak,
                      "tilt_switch": tilt_deg(v), "tilt_max": tilt_deg(v)}
        self.tick()

        # ---- RECOVERY (phase 2): policy thrust ----
        if a.freefall_mode == "hybrid":
            self.policy_descent()
            z = self.last["stateEstimate.z"]

        self.state = "RECOVERY"
        t_rec = time.monotonic()
        z_min = z
        brake = []                      # (t, vz) samples to measure real deceleration
        while True:
            v = self.read()
            z, vz = v["stateEstimate.z"], v["stateEstimate.vz"]
            z_min = min(z_min, z)
            self.stats["peak_v"] = max(self.stats["peak_v"], -vz)
            self.stats["tilt_max"] = max(self.stats["tilt_max"], tilt_deg(v))
            dt_rec = time.monotonic() - t_rec
            if 0.06 <= dt_rec <= 0.25 and vz < 0:
                brake.append((dt_rec, vz))
            if len(brake) >= 3:
                ts, vs = np.array(brake).T
                acc = float(np.polyfit(ts, vs, 1)[0])      # m/s^2, + = slowing the fall
                self.stats["brake_acc"] = acc
                self.stats["implied_tw"] = 1.0 + acc / G
            obs = build_observation(v, PHASE_RECOVERY, self.origin, self.off)
            raw = self.policy(obs)
            pwm, a0 = policy_thrust_pwm(raw, self.hover_pwm)
            self.cf.commander.send_setpoint(0.0, 0.0, 0.0, pwm)
            self.log(v, thrust_cmd=pwm, a0_clipped=a0, tilt_deg=tilt_deg(v),
                     raw0=raw[0], raw1=raw[1], raw2=raw[2], raw3=raw[3])
            # Floor threshold: it's touching down anyway, so stop the props
            # instead of letting full throttle launch it off the pad.
            if z <= a.impact_cut_z:
                self.state = "FLOOR_IMPACT"
                for _ in range(10):
                    self.cf.commander.send_stop_setpoint()
                    time.sleep(0.005)
                print(f"  FLOOR THRESHOLD z={z:.3f} m (vz={vz:+.2f} m/s) -> motors stopped "
                      f"({time.monotonic() - t_rec:.2f} s into recovery)")
                self.stats.update(low_z=z, impact_v=-vz)
                return {"vz_peak": vz_peak, "z_min": z, "hover_pwm": self.hover_pwm,
                        "impact_vz": vz}
            # Ceiling threshold: if it would coast past the top, hand back to
            # the onboard position controller now so it brakes.
            # Only hand back once the fall has actually been stopped: a switch high up
            # (timed switch, or Loco reading high) must not end recovery instantly.
            apex = z + max(vz, 0.0) ** 2 / (2 * G)
            stopped = dt_rec > 0.15 and vz > -0.1
            if (vz > 0 and apex >= self.top + a.overshoot) or (stopped and z >= self.top - 0.15):
                break
            if time.monotonic() - t_rec > a.recovery_timeout:
                raise SafetyAbort("Policy did not climb back in time")
            self.tick()
        print(f"  recovered in {time.monotonic() - t_rec:.2f} s, lowest z={z_min:.3f} m")
        self.stats["low_z"] = z_min

        # ---- back to onboard control: hold then land ----
        self.state = "HOLD"
        self.hold(top, 1.0)
        self.state = "LAND"
        print("LAND")
        self.land_from(top)
        return {"vz_peak": vz_peak, "z_min": z_min, "hover_pwm": self.hover_pwm}

    def policy_descent(self):
        """Hybrid mode: after the free fall, the policy flies the rest of the
        descent with its phase-1 input (as in training / your old runs) until
        phase2_height, then the normal phase-2 recovery takes over."""
        a = self.args
        self.state = "POLICY_DESCENT"
        print(f"POLICY DESCENT: policy (phase 1) brings it down to {a.phase2_height} m")
        t0 = time.monotonic()
        while True:
            v = self.read()
            z, vz = v["stateEstimate.z"], v["stateEstimate.vz"]
            self.stats["peak_v"] = max(self.stats["peak_v"], -vz)
            raw = self.policy(build_observation(v, PHASE_FREEFALL, self.origin, self.off))
            cmd, a0 = policy_thrust_pwm(raw, self.hover_pwm)
            if raw[0] <= -1.0:
                cmd = a.hold_thrust          # keep attitude control alive at "zero" thrust
            self.cf.commander.send_setpoint(0, 0, 0, cmd)
            self.log(v, thrust_cmd=cmd, a0_clipped=a0, raw0=raw[0], tilt_deg=tilt_deg(v))
            if z <= a.impact_cut_z:
                for _ in range(10):
                    self.cf.commander.send_stop_setpoint()
                    time.sleep(0.005)
                raise SafetyAbort(f"Floor threshold during the policy descent (z={z:.2f} m)")
            if z <= a.phase2_height:
                print(f"  reached {a.phase2_height} m: z={z:.3f} vz={vz:+.2f} m/s "
                      f"after {time.monotonic() - t0:.2f} s")
                self.stats["phase2_z"], self.stats["phase2_v"] = z, -vz
                return
            if time.monotonic() - t0 > 4.0:
                print("  policy descent timed out (it hovered); switching to phase 2 here")
                return
            self.tick()

    def land_from(self, z_from, speed=0.4):
        ground = self.args.ground_z
        steps = max(1, int((z_from - ground) / speed * CONTROL_HZ))
        for i in range(1, steps + 1):
            z_cmd = z_from + (ground - z_from) * i / steps
            try:
                v = self.read(check=False)
                self.log(v, z_cmd=z_cmd)
            except SafetyAbort:
                pass  # telemetry gone: the onboard estimator still flies it down
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
        z = (self.last or {}).get("stateEstimate.z", self.top)
        if self.origin is None:
            self.cf.commander.send_stop_setpoint()
            return
        self.next_tick = time.monotonic()
        # Hold briefly so the onboard controller catches the drone first.
        for _ in range(int(0.5 * CONTROL_HZ)):
            self.pos_setpoint(max(z, self.args.ground_z + 0.3))
            self.tick()
        self.land_from(max(z, self.args.ground_z + 0.3))


def print_comparison(st, hover_pwm):
    """Sim vs real, so the gap points at its cause."""
    if not st:
        return
    real_tw_max = measured_thrust_to_weight(hover_pwm) if hover_pwm else float("nan")
    sim_acc = SIM_REF["switch_v"] ** 2 / (2 * (SIM_REF["switch_z"] - SIM_REF["low_z"]))
    rows = [
        ("switch height (m)", SIM_REF["switch_z"], st.get("switch_z")),
        ("speed at switch (m/s)", SIM_REF["switch_v"], st.get("switch_v")),
        ("peak descent speed (m/s)", SIM_REF["peak_v"], st.get("peak_v")),
        ("lowest point (m)", SIM_REF["low_z"], st.get("low_z")),
        ("braking accel (m/s^2)", sim_acc, st.get("brake_acc")),
        ("implied thrust/weight", 1 + sim_acc / G, st.get("implied_tw")),
        ("tilt at switch (deg)", 0.0, st.get("tilt_switch")),
        ("max tilt in recovery (deg)", 0.0, st.get("tilt_max")),
    ]
    print("\nSIM vs REAL")
    print(f"  {'':26}{'sim':>8}{'real':>8}")
    for name, s_val, r_val in rows:
        r_txt = f"{r_val:8.2f}" if r_val is not None else "     n/a"
        print(f"  {name:26}{s_val:8.2f}{r_txt}")
    print(f"  max thrust/weight the real motors can give (from hover): {real_tw_max:.2f}")
    print("  If the sim's implied T/W is far above both real numbers, the sim's thrust is the gap.")
    if st.get("tilt_max", 0) > 30:
        print("  !! Large tilt: thrust was pointing sideways. Try --freefall-mode hold.")


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
    p = argparse.ArgumentParser(description="Crazyflie 2.1+ freefall maneuver with RL recovery")
    p.add_argument("--uri", default=DEFAULT_URI)
    p.add_argument("--policy", default=DEFAULT_POLICY)
    p.add_argument("--arm", action="store_true", help="Actually fly. Without it: dry run, no setpoints.")
    p.add_argument("--ceiling", type=float, default=None,
                   help="Room height in metres (floor to ceiling). Required with --arm.")
    p.add_argument("--top", type=float, default=TOP_HEIGHT, help="Hover/top height (m)")
    p.add_argument("--recover-height", default="auto",
                   help="Height where the policy takes over: 'auto' (deepest safe drop) or metres")
    p.add_argument("--hover-seconds", type=float, default=2.5,
                   help="Hover at the top before the drop (prop wash builds up)")
    p.add_argument("--climb-speed", type=float, default=0.5, help="m/s for the first climb")
    p.add_argument("--switch-after-drop", type=float, default=None,
                   help="Switch to the policy after this many metres of free fall, timed with "
                        "t = sqrt(2d/g) instead of waiting for the Loco height (e.g. 0.423)")
    p.add_argument("--spin-down", type=float, default=0.0,
                   help="Extra seconds added to the timed switch for the motors spinning down")
    p.add_argument("--hold-thrust", type=int, default=2000,
                   help="Thrust value for --freefall-mode hold (0-65535; ~2000 is almost no lift)")
    p.add_argument("--switch-at-speed", type=float, default=None,
                   help="End the free fall when descent speed reaches this (m/s). Uses the IMU-based "
                        "velocity, which reacts faster than the Loco height. E.g. 2.0")
    p.add_argument("--phase2-height", type=float, default=FREEFALL_RECOVER_HEIGHT,
                   help="Hybrid mode: height where the policy switches from phase 1 to phase 2 (default 0.695)")
    p.add_argument("--freefall-mode", choices=["off", "idle", "hold", "hybrid", "policy"], default="off",
                   help="hybrid = free fall, then the policy flies the descent to --phase2-height and back up "
                        "(the 0 -> 1.7 -> 0.695 -> 1.7 -> 0 path); "
                        "off = motors off (attitude control off too!); hold = near-zero thrust that keeps "
                        "attitude control on; idle = minimum PWM; policy = the policy controls thrust "
                        "during the drop with the phase-1 input (what your sim most likely did)")
    p.add_argument("--sim-replay", action="store_true",
                   help="Fly the sim's maneuver: 1.70 m top, switch at 0.695 m, sim low point ~0.476 m. "
                        "Prints a sim-vs-real comparison afterwards. Needs --accept-floor-impact.")
    p.add_argument("--thrust-to-weight", type=float, default=1.4,
                   help="Assumed real max T/W for the floor check (sim used 1.9)")
    p.add_argument("--motor-lag", type=float, default=0.05,
                   help="Motor spin-up time constant for the floor check (s); fits your t030 log")
    p.add_argument("--latency", type=float, default=0.03, help="Radio + telemetry delay for the floor check (s)")
    p.add_argument("--trigger-lead", type=float, default=0.03, help="Telemetry delay compensation (s)")
    p.add_argument("--floor-margin", type=float, default=0.30,
                   help="Minimum predicted clearance (m); Loco z noise is ~0.1 m")
    p.add_argument("--accept-floor-impact", action="store_true",
                   help="Fly even if the floor check predicts the dip goes below the margin")
    p.add_argument("--impact-cut-z", type=float, default=0.12,
                   help="Floor threshold above the pad: motors stop if recovery gets this low (m)")
    p.add_argument("--overshoot", type=float, default=0.10,
                   help="Hand recovery back to onboard control if the predicted apex exceeds top + this (m)")
    p.add_argument("--pad-height", type=float, default=0.0,
                   help="Height of the foam/pad top the drone sits on (m). Treated as the floor.")
    p.add_argument("--ground-z", type=float, default=0.03, help="Landing setpoint height")
    p.add_argument("--max-drift", type=float, default=0.6)
    p.add_argument("--ceiling-margin", type=float, default=0.3)
    p.add_argument("--telemetry-timeout", type=float, default=0.15)
    p.add_argument("--recovery-timeout", type=float, default=3.0)
    p.add_argument("--max-seconds", type=float, default=25.0)
    p.add_argument("--dry-seconds", type=float, default=10.0)
    p.add_argument("--log", default="freefall_flight_log.csv")
    args = p.parse_args()

    if args.sim_replay:
        args.top = TOP_HEIGHT
        args.recover_height = str(FREEFALL_RECOVER_HEIGHT)
        args.trigger_lead = 0.0                    # sim switches exactly at 0.695
        print(f"SIM REPLAY: top {TOP_HEIGHT} m, switch at {FREEFALL_RECOVER_HEIGHT} m. Sim reference: "
              f"switch {SIM_REF['switch_z']} m at {SIM_REF['switch_v']} m/s, peak {SIM_REF['peak_v']} m/s, "
              f"low point {SIM_REF['low_z']} m.")
    top = args.top
    pad = args.pad_height
    args.takeoff_z = max(START_HEIGHT, pad + 0.2)
    args.ground_z += pad
    args.impact_cut_z += pad
    floor_limit = pad + args.floor_margin
    if args.arm and args.ceiling is None:
        sys.exit("--arm needs --ceiling <room height in m>.")
    if args.ceiling is not None and top > args.ceiling - 0.4:
        sys.exit(f"--top {top} m is within 0.4 m of the {args.ceiling} m ceiling "
                 f"(ceiling suction + Loco noise). Use --top {args.ceiling - 0.4:.2f} or lower.")
    dip = lambda r: predict_dip(top, r, args.thrust_to_weight, args.motor_lag,
                                args.latency, args.trigger_lead)
    args.floor_limit = floor_limit
    if args.recover_height == "auto":
        recover = auto_recover_height(top, floor_limit, args.thrust_to_weight, args)
        if recover is None:
            sys.exit("No drop is safe with these settings; lower the pad or raise --top.")
    else:
        recover = float(args.recover_height)
    if args.switch_at_speed is not None:
        # The free fall ends when this speed is reached: plan the floor check from there.
        # Plan the floor check from where that speed is reached in free fall; the
        # backup height trigger (only if the speed trigger fails) sits 10 cm lower.
        recover = top - args.switch_at_speed ** 2 / (2 * G)
    args.plan_z = recover
    args.recover_z = recover - (0.10 if args.switch_at_speed is not None else 0.0)
    v_bottom, z_low = dip(recover)
    t_fall = math.sqrt(2 * max(top - recover, 0) / G)
    print("Flight plan:")
    print(f"  pad {pad:.2f} -> {args.takeoff_z:.2f} -> {top:.2f} m (hover {args.hover_seconds}s)"
          f" -> drop to {recover:.3f} m -> policy back to ~{top:.2f} -> land")
    print(f"  free fall ~{t_fall:.2f} s, ~{v_bottom:.2f} m/s at recovery start (no drag)")
    print(f"  predicted lowest point (T/W {args.thrust_to_weight}, motor lag {args.motor_lag}s, "
          f"latency {args.latency}s): {z_low:+.2f} m")
    if z_low < floor_limit:
        print(f"  !! Recovery dip reaches {z_low:+.2f} m, below {floor_limit:.2f} m "
              f"(pad {pad:.2f} + margin {args.floor_margin}). "
              f"Raise --recover-height (or use auto).")
        if args.arm and not args.accept_floor_impact:
            sys.exit("Refusing to arm. Pass --accept-floor-impact only if you mean to hit the floor.")

    if not os.path.isfile(args.policy):
        sys.exit(f"Policy not found: {args.policy}")
    policy = Policy(load_actor_weights(args.policy))
    print(f"Loaded policy {args.policy}")

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
            sys.exit("No fresh telemetry. Check the positioning deck / log variables.")
        print(f"Telemetry OK. z={v['stateEstimate.z']:.3f} m, battery {v.get('pm.vbat', 0):.2f} V")
        if abs(v["stateEstimate.z"] - pad) > 0.15:
            msg = (f"Drone on the pad reads z={v['stateEstimate.z']:+.2f} m but --pad-height is {pad:.2f} m. "
                   "Set --pad-height to the measured foam height. If that doesn't match, the Loco anchor z values "
                   "must be measured from the floor (and the anchors must be placed well above and "
                   "below the flight heights) or every height in this script is wrong.")
            if args.arm:
                sys.exit(msg)
            print("  !! " + msg)
        if v.get("pm.vbat", 4.2) < 3.8:
            print("  !! Battery below 3.8 V. Charge it: thrust margin matters for the recovery.")

        try:
            if not args.arm:
                print("DRY RUN: no setpoints are sent. Showing the policy's phase-2 output.")
                origin = (v["stateEstimate.x"], v["stateEstimate.y"])
                end = time.monotonic() + args.dry_seconds
                while time.monotonic() < end:
                    v, _, _ = telemetry.get()
                    raw = policy(build_observation(v, PHASE_RECOVERY, origin, top - TOP_HEIGHT))
                    pwm, a0 = policy_thrust_pwm(raw, 42000)
                    print(f"  z={v['stateEstimate.z']:+.3f} vz={v['stateEstimate.vz']:+.2f} "
                          f"raw={np.round(raw, 2)} a0={a0:+.2f} pwm@hover42000={pwm}")
                    time.sleep(0.25)
                return

            print("\n*** ARMED FLIGHT. Clear area, net/foam below, hand near the kill switch. ***")
            input("Press ENTER to take off (Ctrl+C to cancel)... ")
            arm(cf)
            time.sleep(0.5)
            flight = Flight(cf, telemetry, policy, args)
            try:
                result = flight.run()
                print(f"DONE. peak descent {result['vz_peak']:+.2f} m/s, "
                      f"lowest z {result['z_min']:.3f} m, hover thrust {result['hover_pwm']:.0f}")
                print_comparison(getattr(flight, "stats", {}), result["hover_pwm"])
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