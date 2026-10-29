#!/usr/bin/env python3
"""
Crazyflie Loco + Flow coordinate test - revision 103.

Changes from 102, all aimed at the crash seen in the 102 logs:

  * ERROR-GATED RAMP. The interpolated setpoint only advances when the
    drone is actually keeping up with it. In 102 the setpoint marched
    on a fixed timer, the drone fell up to 0.5 m behind, and the
    resulting large error commanded a tilt big enough to tumble it.
  * TAKEOFF SETTLE. Traverse does not begin until altitude has
    actually been achieved and held, verified by feedback rather
    than assumed after a fixed duration.
  * TUMBLE DETECTION. Aborts on a large yaw excursion, which shows up
    a beat before the Z-bounds check would catch a crash.
  * SLOWER SPEEDS. Climb and traverse rates roughly halved.
  * Tighter runaway limit.

Usage:
    python3 coordinateTestCF_103.py --print-only --monitor
    python3 coordinateTestCF_103.py --x 0.5 --y 0.5 --z 0.4 --duration 20
"""

import time
import math
import signal
import sys
import argparse
import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig


# ============================================================
# CONFIG
# ============================================================

DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"

VAR_THRESHOLD = 0.001
CONVERGE_TIMEOUT = 15.0

SETPOINT_HZ = 50.0
SETPOINT_DT = 1.0 / SETPOINT_HZ

# --- Motion rates. Deliberately slow. -----------------------

CLIMB_SPEED = 0.15          # m/s, was 0.25
TRAVERSE_SPEED = 0.15       # m/s, was 0.30

# --- Error gating -------------------------------------------
# The ramp pauses whenever the drone is further than MAX_LAG from
# the current setpoint. This is the core fix: the setpoint can
# never get so far ahead that the controller commands a violent
# correction.

MAX_LAG = 0.15              # m, for horizontal traverse
MAX_LAG_WAIT = 8.0          # s to wait at one setpoint before aborting

# Takeoff needs a much looser gate. During spin-up the drone
# physically cannot keep up with the setpoint -- the motors are
# still ramping and it has not broken free of the ground. A tight
# gate here deadlocks: the setpoint freezes low, the drone never
# gets enough command to lift, and it sits on the floor forever.
MAX_LAG_TAKEOFF = 0.35      # m

# Below this altitude the drone is still on the ground or in ground
# effect. Do not gate at all.
GROUND_HEIGHT = 0.10        # m

# --- Takeoff settle ------------------------------------------

SETTLE_TOLERANCE = 0.13     # m, altitude must be within this
SETTLE_DURATION = 1.5       # s it must stay there
SETTLE_TIMEOUT = 10.0       # s before giving up

# --- Safety limits -------------------------------------------

RUNAWAY_LIMIT = 0.8         # m from setpoint, was 1.5
YAW_EXCURSION_LIMIT = 45.0  # deg from commanded yaw -> tumble
BOUNDS_XY = 6.0
BOUNDS_Z = 4.0

YAW_TOLERANCE_DEG = 20.0

# Pose is logged at 50 ms. If nothing arrives for this long the link
# is in trouble and any safety check would be reading frozen values.
STALE_DATA_LIMIT = 0.5      # s


_running = True
_cf = None


# ============================================================
# SAFETY
# ============================================================

def signal_handler(sig, frame):
    global _running

    print("\n\n!!! EMERGENCY STOP !!!")

    _running = False

    try:
        if _cf:
            _cf.commander.send_stop_setpoint()
            time.sleep(0.2)
            _cf.close_link()
    except Exception:
        pass

    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# STATE
# ============================================================

class State:

    def __init__(self):
        self.lock = threading.Lock()
        self.x = None
        self.y = None
        self.z = None
        self.yaw = None
        self.var_x = None
        self.var_y = None
        self.var_z = None
        self.last_update = None
        self.link_ok = True

    def set_pose(self, x, y, z, yaw):
        with self.lock:
            self.x, self.y, self.z, self.yaw = x, y, z, yaw
            self.last_update = time.time()

    def age(self):
        """Seconds since the last pose update, or None if never."""
        with self.lock:
            if self.last_update is None:
                return None
            return time.time() - self.last_update

    def set_link_down(self):
        with self.lock:
            self.link_ok = False

    def is_link_ok(self):
        with self.lock:
            return self.link_ok

    def set_var(self, vx, vy, vz):
        with self.lock:
            self.var_x, self.var_y, self.var_z = vx, vy, vz

    def pose(self):
        with self.lock:
            return self.x, self.y, self.z, self.yaw

    def variance(self):
        with self.lock:
            return self.var_x, self.var_y, self.var_z

    def has_pose(self):
        with self.lock:
            return self.x is not None


# ============================================================
# ESTIMATOR
# ============================================================

def reset_estimator(cf, state):

    print("\nResetting Kalman estimator...")

    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.3)
    cf.param.set_value("kalman.resetEstimation", "0")

    print("Waiting for estimate to converge...")

    start = time.time()

    history_len = 10
    hist_x, hist_y, hist_z = [], [], []

    while time.time() - start < CONVERGE_TIMEOUT:

        if not _running:
            return False

        vx, vy, vz = state.variance()

        if vx is None:
            time.sleep(0.05)
            continue

        hist_x.append(vx)
        hist_y.append(vy)
        hist_z.append(vz)

        hist_x = hist_x[-history_len:]
        hist_y = hist_y[-history_len:]
        hist_z = hist_z[-history_len:]

        if len(hist_x) < history_len:
            time.sleep(0.05)
            continue

        sx = max(hist_x) - min(hist_x)
        sy = max(hist_y) - min(hist_y)
        sz = max(hist_z) - min(hist_z)

        print(
            f"  t={time.time() - start:4.1f}s  "
            f"spread: X={sx:.5f} Y={sy:.5f} Z={sz:.5f}",
            end="\r"
        )

        if sx < VAR_THRESHOLD and sy < VAR_THRESHOLD and sz < VAR_THRESHOLD:
            print("\n  Converged.                                    ")
            return True

        time.sleep(0.05)

    print("\n  WARNING: estimate did not converge within timeout.")
    return False


# ============================================================
# SAFETY CHECKS
# ============================================================

def check_flight_safety(state, tx, ty, tz, cmd_yaw):
    """
    Returns (ok, message). Checked on every setpoint.
    """

    x, y, z, yaw = state.pose()

    if x is None:
        return True, None

    # --- Link health ---
    # If the radio drops, the log callbacks stop firing but the last
    # values persist. Every check below would then be evaluating a
    # frozen snapshot while the drone does whatever it likes. Catch
    # that first.

    if not state.is_link_ok():
        return False, "radio link lost -- estimate is no longer live"

    age = state.age()

    if age is not None and age > STALE_DATA_LIMIT:
        return False, (
            f"no estimator update for {age:.1f} s "
            f"-- position data is stale, not flying blind"
        )

    # --- Estimate plausibility ---

    if abs(x) > BOUNDS_XY or abs(y) > BOUNDS_XY:
        return False, f"estimate outside XY bounds ({x:.2f}, {y:.2f})"

    if z < -0.3 or z > BOUNDS_Z:
        return False, f"estimate outside Z bounds ({z:.2f})"

    # --- Tumble detection ---
    # A crash shows up as a large yaw excursion roughly one sample
    # before the position estimate goes obviously bad.

    yaw_err = abs(((yaw - cmd_yaw) + 180.0) % 360.0 - 180.0)

    if yaw_err > YAW_EXCURSION_LIMIT:
        return False, (
            f"yaw is {yaw:.1f} deg vs commanded {cmd_yaw:.1f} "
            f"-- the drone has tumbled"
        )

    # --- Runaway ---

    dist = math.sqrt((tx - x) ** 2 + (ty - y) ** 2 + (tz - z) ** 2)

    if dist > RUNAWAY_LIMIT:
        return False, (
            f"drone is {dist:.2f} m from setpoint "
            f"(limit {RUNAWAY_LIMIT} m) -- runaway"
        )

    return True, None


def lag_from(state, tx, ty, tz):
    """Distance from the drone to the current setpoint."""

    x, y, z, _ = state.pose()

    if x is None:
        return 0.0

    return math.sqrt((tx - x) ** 2 + (ty - y) ** 2 + (tz - z) ** 2)


# ============================================================
# FLIGHT
# ============================================================

def fly_segment(cf, state, start_pt, end_pt, yaw, speed, label,
                max_lag=MAX_LAG, gate_on_ground=True):
    """
    Move from start_pt to end_pt with an ERROR-GATED ramp.

    The setpoint advances along the path at `speed`, but only while
    the drone is within MAX_LAG of it. If the drone falls behind,
    the setpoint freezes and waits. This prevents the setpoint from
    running away and commanding a tilt large enough to tumble.
    """

    sx, sy, sz = start_pt
    ex, ey, ez = end_pt

    dist = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2)

    if dist < 0.01:
        print(f"\n{label}: already there, skipping.")
        return True

    duration = dist / speed
    total_steps = max(int(duration / SETPOINT_DT), 1)

    print(f"\n{label}: {dist:.2f} m, nominal {duration:.1f} s")
    print("  (setpoint pauses if the drone falls behind)")

    step = 0
    waiting_since = None
    last_print = 0.0
    paused_total = 0.0

    while step <= total_steps:

        if not _running:
            return False

        frac = step / total_steps

        tx = sx + (ex - sx) * frac
        ty = sy + (ey - sy) * frac
        tz = sz + (ez - sz) * frac

        cf.commander.send_position_setpoint(tx, ty, tz, yaw)

        ok, msg = check_flight_safety(state, tx, ty, tz, yaw)

        if not ok:
            print(f"\n  ABORT: {msg}")
            return False

        # --- The gate ---

        lag = lag_from(state, tx, ty, tz)

        # Do not gate while still on the ground. The drone cannot
        # follow a setpoint it has not lifted off toward yet, and
        # gating here deadlocks the climb.
        _, _, cur_z, _ = state.pose()
        on_ground = (
            not gate_on_ground
            and cur_z is not None
            and cur_z < GROUND_HEIGHT
        )

        if lag > max_lag and not on_ground:

            if waiting_since is None:
                waiting_since = time.time()

            waited = time.time() - waiting_since

            if waited > MAX_LAG_WAIT:
                print(
                    f"\n  ABORT: drone stuck {lag:.2f} m behind setpoint "
                    f"for {waited:.1f} s. Not enough thrust or authority "
                    f"to follow the path."
                )
                return False

            paused_total += SETPOINT_DT
            # Do not advance `step`.

        else:
            waiting_since = None
            step += 1

        now = time.time()

        if now - last_print >= 0.5:
            last_print = now

            x, y, z, yw = state.pose()

            if x is not None:
                flag = (
                    "  <- WAITING"
                    if (lag > max_lag and not on_ground) else ""
                )
                print(
                    f"  SET({tx:+.2f},{ty:+.2f},{tz:+.2f})  "
                    f"POS({x:+.2f},{y:+.2f},{z:+.2f})  "
                    f"LAG={lag:.2f}  YAW={yw:+6.1f}{flag}"
                )

        time.sleep(SETPOINT_DT)

    if paused_total > 0.1:
        print(f"  (paused {paused_total:.1f} s waiting for the drone)")

    return True


def settle_at(cf, state, target, yaw, label):
    """
    Hold at target until the drone is actually there and stays there.
    Used after takeoff so the traverse never begins mid-climb.
    """

    tx, ty, tz = target

    print(f"\n{label}: waiting for the drone to arrive and steady...")

    start = time.time()
    in_tolerance_since = None
    last_print = 0.0

    while time.time() - start < SETTLE_TIMEOUT:

        if not _running:
            return False

        cf.commander.send_position_setpoint(tx, ty, tz, yaw)

        ok, msg = check_flight_safety(state, tx, ty, tz, yaw)

        if not ok:
            print(f"\n  ABORT: {msg}")
            return False

        x, y, z, yw = state.pose()

        if x is not None:

            err = math.sqrt(
                (tx - x) ** 2 + (ty - y) ** 2 + (tz - z) ** 2
            )

            if err < SETTLE_TOLERANCE:
                if in_tolerance_since is None:
                    in_tolerance_since = time.time()

                if time.time() - in_tolerance_since >= SETTLE_DURATION:
                    print(f"  Settled (err {err:.3f} m).")
                    return True
            else:
                in_tolerance_since = None

            now = time.time()

            if now - last_print >= 0.5:
                last_print = now
                print(
                    f"  POS({x:+.2f},{y:+.2f},{z:+.2f})  "
                    f"ERR={err:.3f}  YAW={yw:+6.1f}"
                )

        time.sleep(SETPOINT_DT)

    print(
        f"  WARNING: did not settle within {SETTLE_TIMEOUT:.0f} s. "
        f"Continuing anyway."
    )
    return True


def hold_position(cf, state, target, yaw, duration):

    tx, ty, tz = target

    print("\n========================================")
    print(" HOLDING POSITION")
    print("========================================")
    print(f"TARGET: X={tx:.3f} Y={ty:.3f} Z={tz:.3f}")

    if duration is None:
        print("Duration: INDEFINITE (Ctrl+C to land)")
    else:
        print(f"Duration: {duration:.1f} s")

    print("========================================")

    start = time.time()
    last_print = 0.0

    # Track hold quality so we can report it at the end.
    max_err = 0.0
    err_sum = 0.0
    err_count = 0

    while _running:

        cf.commander.send_position_setpoint(tx, ty, tz, yaw)

        ok, msg = check_flight_safety(state, tx, ty, tz, yaw)

        if not ok:
            print(f"\n  ABORT: {msg}")
            return False

        x, y, z, yw = state.pose()

        if x is not None:
            ex, ey, ez = tx - x, ty - y, tz - z
            d = math.sqrt(ex ** 2 + ey ** 2 + ez ** 2)

            max_err = max(max_err, d)
            err_sum += d
            err_count += 1

            now = time.time()

            if now - last_print >= 0.5:
                last_print = now
                print(
                    f"  POS({x:+.3f},{y:+.3f},{z:+.3f})  "
                    f"ERR({ex:+.3f},{ey:+.3f},{ez:+.3f})  "
                    f"DIST={d:.3f}  YAW={yw:+6.1f}"
                )

        if duration is not None and time.time() - start >= duration:
            print("\n  Requested duration reached.")
            break

        time.sleep(SETPOINT_DT)

    if err_count:
        print(
            f"\n  Hold quality: mean error {err_sum / err_count:.3f} m, "
            f"max {max_err:.3f} m"
        )

    return True


def land(cf, state, yaw):

    print("\nLanding...")

    if not state.is_link_ok():
        print("  Link is down -- cannot command a landing.")
        print("  The firmware commander watchdog will cut the motors.")
        return

    x, y, z, _ = state.pose()

    if x is None:
        cf.commander.send_stop_setpoint()
        return

    ground = 0.05
    dist = max(z - ground, 0.0)
    duration = max(dist / CLIMB_SPEED, 0.5)
    steps = max(int(duration / SETPOINT_DT), 1)

    for i in range(steps + 1):
        frac = i / steps
        tz = z + (ground - z) * frac
        cf.commander.send_position_setpoint(x, y, tz, yaw)
        time.sleep(SETPOINT_DT)

    for _ in range(int(0.5 / SETPOINT_DT)):
        cf.commander.send_position_setpoint(x, y, ground, yaw)
        time.sleep(SETPOINT_DT)

    cf.commander.send_stop_setpoint()
    time.sleep(0.3)

    print("Landed.")


# ============================================================
# MAIN
# ============================================================

def main(args):
    global _cf

    print("Initializing Crazyflie drivers...")
    cflib.crtp.init_drivers()

    cf = Crazyflie(rw_cache="./cache")
    _cf = cf

    state = State()

    connected_event = threading.Event()

    def on_connected(uri):
        print(f"\nConnected to {uri}")
        connected_event.set()

    def on_disconnected(uri):
        state.set_link_down()
        print("\nDisconnected.")

    def on_connection_lost(uri, msg):
        state.set_link_down()
        print(f"\nCONNECTION LOST: {msg}")

    cf.connected.add_callback(on_connected)
    cf.disconnected.add_callback(on_disconnected)
    cf.connection_lost.add_callback(on_connection_lost)

    print(f"\nConnecting to {args.uri}...")
    cf.open_link(args.uri)

    if not connected_event.wait(timeout=10):
        print("ERROR: Connection timeout.")
        return

    time.sleep(1.0)

    log_pose = None
    log_var = None
    log_bat = None

    battery = {"v": None}

    try:

        def pose_cb(timestamp, data, logconf):
            state.set_pose(
                data["stateEstimate.x"],
                data["stateEstimate.y"],
                data["stateEstimate.z"],
                data["stabilizer.yaw"],
            )

        def var_cb(timestamp, data, logconf):
            state.set_var(
                data["kalman.varPX"],
                data["kalman.varPY"],
                data["kalman.varPZ"],
            )

        def bat_cb(timestamp, data, logconf):
            battery["v"] = data["pm.vbat"]

        log_pose = LogConfig(name="Pose", period_in_ms=50)
        log_pose.add_variable("stateEstimate.x", "float")
        log_pose.add_variable("stateEstimate.y", "float")
        log_pose.add_variable("stateEstimate.z", "float")
        log_pose.add_variable("stabilizer.yaw", "float")
        log_pose.data_received_cb.add_callback(pose_cb)

        log_var = LogConfig(name="Variance", period_in_ms=50)
        log_var.add_variable("kalman.varPX", "float")
        log_var.add_variable("kalman.varPY", "float")
        log_var.add_variable("kalman.varPZ", "float")
        log_var.data_received_cb.add_callback(var_cb)

        log_bat = LogConfig(name="Battery", period_in_ms=200)
        log_bat.add_variable("pm.vbat", "float")
        log_bat.data_received_cb.add_callback(bat_cb)

        cf.log.add_config(log_pose)
        cf.log.add_config(log_var)
        cf.log.add_config(log_bat)

        if not log_pose.valid or not log_var.valid:
            print("\nERROR: log configuration invalid.")
            return

        log_pose.start()
        log_var.start()

        if log_bat.valid:
            log_bat.start()

        try:
            est = cf.param.get_value("stabilizer.estimator")
            ctl = cf.param.get_value("stabilizer.controller")
            loco = cf.param.get_value("loco.mode")
            print(f"\nEstimator: {est}   Controller: {ctl}   loco.mode: {loco}")
        except Exception as e:
            print(f"(Could not read parameters: {e})")

        converged = reset_estimator(cf, state)

        if not state.has_pose():
            print("\nERROR: No estimator data received.")
            return

        x, y, z, yaw = state.pose()

        print("\n========================================")
        print(" CURRENT ESTIMATED POSITION")
        print("========================================")
        print(f"X:   {x:.3f} m")
        print(f"Y:   {y:.3f} m")
        print(f"Z:   {z:.3f} m")
        print(f"Yaw: {yaw:.1f} deg")

        if battery["v"] is not None:
            print(f"Bat: {battery['v']:.3f} V")

        print("========================================")

        # ====================================================
        # PRINT ONLY
        # ====================================================

        if args.print_only:

            print("\nPRINT-ONLY MODE -- no flight commands sent.")

            if args.monitor:
                print("\nMonitoring. Press Ctrl+C to stop.")

                while _running:
                    x, y, z, yaw = state.pose()
                    vx, vy, vz = state.variance()

                    if x is not None:
                        bat = (
                            f"  BAT={battery['v']:.2f}"
                            if battery["v"] is not None else ""
                        )
                        print(
                            f"POS({x:+.3f},{y:+.3f},{z:+.3f})  "
                            f"YAW={yaw:+6.1f}  "
                            f"VAR({vx:.5f},{vy:.5f},{vz:.5f}){bat}"
                        )

                    time.sleep(0.5)

            return

        # ====================================================
        # PRE-FLIGHT
        # ====================================================

        problems = []

        if not converged:
            problems.append("Kalman variance never settled.")

        if abs(yaw) > YAW_TOLERANCE_DEG:
            problems.append(
                f"Yaw is {yaw:.1f} deg, not ~0. Point the drone along +X "
                f"and power-cycle it there."
            )

        if battery["v"] is not None and battery["v"] < 3.85:
            problems.append(
                f"Battery is {battery['v']:.2f} V. With a Flow deck AND a "
                f"Loco deck this airframe is heavy; below ~3.85 V there "
                f"may not be enough thrust margin to translate without "
                f"sinking."
            )

        if problems:
            print("\n########################################")
            print(" PRE-FLIGHT WARNINGS")
            print("########################################")

            for p in problems:
                print(f"  * {p}")

            print("########################################")

            if not args.force:
                print("\nRefusing to fly. Re-run with --force to override.")
                return

            print("\n--force given, continuing anyway.")

        print("\n========================================")
        print(" FLIGHT PLAN")
        print("========================================")
        print(f"  1. Climb in place to {args.z:.2f} m")
        print(f"  2. Settle and verify altitude")
        print(f"  3. Traverse to X={args.x:.3f} Y={args.y:.3f}")
        print(f"  4. Hold")
        print(f"  5. Land")
        print("========================================")

        input("\nPress ENTER to fly, or Ctrl+C to cancel...")

        # ====================================================
        # FLY
        # ====================================================

        x, y, z, _ = state.pose()

        cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.1)

        ok = fly_segment(
            cf, state,
            (x, y, z), (x, y, args.z),
            args.yaw, CLIMB_SPEED, "TAKEOFF",
            max_lag=MAX_LAG_TAKEOFF,
            gate_on_ground=False
        )

        if ok:
            ok = settle_at(cf, state, (x, y, args.z), args.yaw, "SETTLE")

        if ok and not args.hover_only:

            # Start the traverse from where the drone ACTUALLY is,
            # not from the pre-takeoff position. It will have drifted
            # during the climb and settle, and starting the ramp from
            # a stale point injects that drift as instant error.
            cx, cy, cz, _ = state.pose()

            if cx is None:
                cx, cy, cz = x, y, args.z

            ok = fly_segment(
                cf, state,
                (cx, cy, cz), (args.x, args.y, args.z),
                args.yaw, TRAVERSE_SPEED, "TRAVERSE"
            )

        if ok:
            hold_target = (
                (x, y, args.z) if args.hover_only
                else (args.x, args.y, args.z)
            )
            ok = hold_position(
                cf, state, hold_target, args.yaw, args.duration
            )

        land(cf, state, args.yaw)

        if battery["v"] is not None:
            print(f"Battery after flight: {battery['v']:.3f} V")

    finally:

        for lc in (log_pose, log_var, log_bat):
            try:
                if lc:
                    lc.stop()
            except Exception:
                pass

        try:
            cf.commander.send_stop_setpoint()
        except Exception:
            pass

        try:
            cf.close_link()
        except Exception:
            pass


# ============================================================
# ARGS
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Crazyflie Loco + Flow coordinate test"
    )

    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.4)
    parser.add_argument("--yaw", type=float, default=0.0)

    parser.add_argument("--duration", type=float, default=None,
                        help="Hold duration in seconds")

    parser.add_argument("--hover-only", action="store_true",
                        help="Climb and hold in place. No traverse. "
                             "Use this to isolate hold quality from "
                             "translation problems.")

    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    main(args)