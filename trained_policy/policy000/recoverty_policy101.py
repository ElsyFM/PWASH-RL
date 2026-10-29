"""
Crazyflie RL trajectory test using recovery_policy.pt.

Explicit physical trajectory:

    1.6 m  ────────────────┐
                            │
                            ↓
    0.3 m  ────────────────┐
                            │
                            ↑
    1.6 m  ────────────────┘

The policy itself was trained around HOME_POS[2] = 0.80 m.

Rather than changing the policy's native reference, the script shifts
the Z observation so that:

    desired altitude == actual altitude
        -> policy sees Z = 0.80 m

    actual altitude > desired altitude
        -> policy sees Z > 0.80 m
        -> policy reduces thrust

    actual altitude < desired altitude
        -> policy sees Z < 0.80 m
        -> policy increases thrust

This makes the learned hover controller track the explicit altitude
trajectory.

DRY_RUN = True is strongly recommended for the first test.
"""

import csv
import math
import threading
import time

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper


# ============================================================================
# Configuration
# ============================================================================

URI = uri_helper.uri_from_env(
    default='radio://0/80/2M/E7E7E7E7E7'
)

CHECKPOINT = 'recovery_policy.pt'
FLIGHT_LOG = 'policy_trajectory.csv'

# IMPORTANT:
# Start with True. Change to False only after checking the dry-run behavior.
DRY_RUN = False

USING_MOCAP = False


# ---------------------------------------------------------------------------
# Explicit trajectory
# ---------------------------------------------------------------------------

START_HEIGHT = 1.80
BOTTOM_HEIGHT = 0.60
END_HEIGHT = 2.20

TOP_HOLD_SECONDS = 1.0
DESCEND_SECONDS = 4.0
BOTTOM_HOLD_SECONDS = 1.0
ASCEND_SECONDS = 4.0
FINAL_DESCEND_SECONDS = 3.0

POLICY_SECONDS = 25.0

CONTROL_HZ = 100


# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------

# Horizontal drift from the position where the policy takes control.
MAX_DRIFT_M = 0.6

# Absolute altitude safety envelope.
MIN_SAFE_ALTITUDE = 0.30
MAX_SAFE_ALTITUDE = 2.2

MAX_TILT_DEG = 45.0

TELEMETRY_TIMEOUT_S = 0.15


# ============================================================================
# Policy conventions
# ============================================================================

# IMPORTANT:
# This remains 0.80 because that is where the checkpoint's actor expects
# its nominal hover observation.
#
# DO NOT change this to 1.60 just because the physical trajectory starts
# at 1.60 m.
HOME_POS = np.array([
    -0.107,
    0.167,
    0.80
])

HOME_YAW = math.radians(5.7)

THRUST_AT_HOVER = 2.02

THRUST_GAIN = 0.4
RATE_GAIN = 5.0

MAX_RATE = math.radians(400)

MIN_THRUST_RATIO = 0.3
MAX_THRUST_RATIO = 1.8


# ============================================================================
# Policy network
# ============================================================================

class Policy:
    """Actor network: 13 -> 256 -> 256 -> 256 -> 4, ELU activations."""

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, obs):
        h = np.asarray(obs, dtype=np.float64)

        for i, (weight, bias) in enumerate(self.layers):
            h = weight @ h + bias

            if i < len(self.layers) - 1:
                h = np.where(
                    h > 0,
                    h,
                    np.expm1(np.minimum(h, 0))
                )

        return h

    def self_check(self):
        """
        At the policy's native hover point, the network should request
        approximately hover thrust and no rotation.
        """

        q = [
            math.cos(HOME_YAW / 2),
            0,
            0,
            math.sin(HOME_YAW / 2)
        ]

        obs = np.concatenate([
            HOME_POS,
            [0, 0, 0],
            q,
            [0, 0, 0]
        ])

        action = self(obs)

        print(
            'Policy self-check:',
            np.round(action, 4)
        )

        if (
            abs(action[0] - THRUST_AT_HOVER) > 0.05
            or np.abs(action[1:]).max() > 0.05
        ):
            raise RuntimeError(
                'Checkpoint does not match the expected policy '
                f'conventions. Output at hover: '
                f'{np.round(action, 3)}'
            )


def load_policy(path):

    import torch

    checkpoint = torch.load(
        path,
        map_location='cpu',
        weights_only=True
    )

    state = checkpoint.get(
        'model_state_dict',
        checkpoint
    )

    layers = []

    for i, shape in (
        (0, (256, 13)),
        (2, (256, 256)),
        (4, (256, 256)),
        (6, (4, 256)),
    ):

        weight = state[f'actor.{i}.weight']
        bias = state[f'actor.{i}.bias']

        if tuple(weight.shape) != shape:
            raise ValueError(
                f'actor.{i}.weight is '
                f'{tuple(weight.shape)}, '
                f'expected {shape}'
            )

        layers.append((
            weight.double().numpy(),
            bias.double().numpy()
        ))

    policy = Policy(layers)
    policy.self_check()

    return policy


# ============================================================================
# Quaternion / attitude helpers
# ============================================================================

def quat_multiply(a, b):

    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,

        w1*x2 + x1*w2 + y1*z2 - z1*y2,

        w1*y2 - x1*z2 + y1*w2 + z1*x2,

        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def rotation_matrix(q):
    """Body -> world rotation for quaternion w,x,y,z."""

    w, x, y, z = q

    return np.array([
        [
            1 - 2*(y*y + z*z),
            2*(x*y - w*z),
            2*(x*z + w*y)
        ],
        [
            2*(x*y + w*z),
            1 - 2*(x*x + z*z),
            2*(y*z - w*x)
        ],
        [
            2*(x*z - w*y),
            2*(y*z + w*x),
            1 - 2*(x*x + y*y)
        ]
    ])


def yaw_of(q):

    w, x, y, z = q

    return math.atan2(
        2*(w*z + x*y),
        1 - 2*(y*y + z*z)
    )


def tilt_deg(q):

    return math.degrees(
        math.acos(
            max(
                -1.0,
                min(
                    1.0,
                    1 - 2*(q[1]**2 + q[2]**2)
                )
            )
        )
    )


def roll_pitch_deg(q):

    w, x, y, z = q

    roll = math.atan2(
        2*(w*x + y*z),
        1 - 2*(x*x + y*y)
    )

    pitch = math.asin(
        max(
            -1.0,
            min(
                1.0,
                2*(w*y - z*x)
            )
        )
    )

    return (
        math.degrees(roll),
        math.degrees(pitch)
    )


# ============================================================================
# Policy frame
# ============================================================================

class PolicyFrame:

    def __init__(self, ref_pos, ref_yaw):

        self.ref_pos = np.array(
            ref_pos,
            dtype=float
        )

        turn = HOME_YAW - ref_yaw

        c = math.cos(turn)
        s = math.sin(turn)

        self.rot = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

        self.turn = np.array([
            math.cos(turn / 2),
            0,
            0,
            math.sin(turn / 2)
        ])

    def observation(
        self,
        state,
        desired_height
    ):
        """
        Convert real-world state into the policy's expected observation.

        X/Y are referenced to the original handover position.

        Z is different:

            policy_z = HOME_Z + (actual_z - desired_z)

        Therefore the policy always sees HOME_Z when the real drone
        is exactly at the desired altitude.
        """

        q = state['quat']

        # X/Y stay relative to the handover point. This deployment wrapper is
    # intentionally retained for the trajectory experiment; the training
    # state used raw position, but the checkpoint self-check establishes
    # HOME_POS as its native hover reference.
        pos_xy = (
            self.rot @
            (state['pos'] - self.ref_pos)
        )

        # Replace only the Z component with the tracking error.
        policy_z = (
            HOME_POS[2]
            + (state['pos'][2] - desired_height)
        )

        pos = np.array([
            pos_xy[0],
            pos_xy[1],
            policy_z
        ])

        # TRAINING-MATCHED STATE COMPONENTS
        # The uploaded Isaac environment feeds world-frame velocity,
        # quaternion, and angular velocity directly to the actor.
        #
        # Crazyflie's gyro is measured in the body frame, so convert it
        # back to world frame before giving it to the policy.
        vel_world = state['vel']
        ang_vel_world = (
            rotation_matrix(q) @ np.radians(state['gyro'])
        )

        # Keep the position wrapper above because this deployment uses the
        # checkpoint's native HOME_POS as the hover reference while asking
        # the physical drone to follow the explicit altitude trajectory.
        # Do NOT replace this with raw world position in this experiment.
        return np.concatenate([
            pos,
            vel_world,
            q,
            ang_vel_world
        ])


# ============================================================================
# Explicit altitude trajectory
# ============================================================================

# ------------------------------------------------------------
# EXPLICIT POLICY TRAJECTORY
# 1.80 m -> 0.60 m -> 2.20 m
# ------------------------------------------------------------



def desired_height(t):
    """
    Explicit trajectory:

        0-1 s:    hold 1.80 m
        1-5 s:    descend 1.80 -> 0.60 m
        5-6 s:    hold 0.60 m
        6-10 s:   ascend 0.60 -> 2.20 m
    """

    # Hold at 1.80 m
    if t < TOP_HOLD_SECONDS:
        return START_HEIGHT

    # Descend: 1.80 -> 0.60 m
    t_descend = t - TOP_HOLD_SECONDS

    if t_descend < DESCEND_SECONDS:
        progress = t_descend / DESCEND_SECONDS
        return START_HEIGHT + progress * (BOTTOM_HEIGHT - START_HEIGHT)

    # Hold at 0.60 m
    t_bottom = t_descend - DESCEND_SECONDS

    if t_bottom < BOTTOM_HOLD_SECONDS:
        return BOTTOM_HEIGHT

    # Ascend: 0.60 -> 2.20 m
    t_ascend = t_bottom - BOTTOM_HOLD_SECONDS

    if t_ascend < ASCEND_SECONDS:
        progress = t_ascend / ASCEND_SECONDS
        return BOTTOM_HEIGHT + progress * (END_HEIGHT - BOTTOM_HEIGHT)
        # Descend: 2.20 -> 0.00 m
    
    t_final_descent = t_ascend - ASCEND_SECONDS

    if t_final_descent < FINAL_DESCEND_SECONDS:
        progress = t_final_descent / FINAL_DESCEND_SECONDS
        return END_HEIGHT * (1.0 - progress)

    # Stay at 0.00 m
    return 0.00

   

def command_from_action(action):

    ratio = float(
        np.clip(
            1
            + THRUST_GAIN * (
                action[0] - THRUST_AT_HOVER
            ),
            MIN_THRUST_RATIO,
            MAX_THRUST_RATIO
        )
    )

    rates = np.clip(
        RATE_GAIN * np.asarray(action[1:4]),
        -MAX_RATE,
        MAX_RATE
    )

    return ratio, rates


# Crazyflie 2.x per-motor thrust model.
_A = 2.130295e-11
_B = 1.032633e-6
_C = 5.484560e-4


def thrust_pwm(ratio, hover_pwm):

    target = ratio * (
        _A * hover_pwm**2
        + _B * hover_pwm
        + _C
    )

    discriminant = (
        _B**2
        - 4 * _A * (_C - target)
    )

    if discriminant < 0:
        raise RuntimeError(
            'Invalid thrust conversion.'
        )

    pwm = (
        -_B
        + math.sqrt(discriminant)
    ) / (2 * _A)

    return int(
        min(
            max(pwm, 0),
            60000
        )
    )


def send_command(
    cf,
    ratio,
    rates,
    hover_pwm
):

    roll, pitch, yaw = np.degrees(rates)

    cf.commander.send_setpoint(
        roll,
        pitch,
        -yaw,
        thrust_pwm(
            ratio,
            hover_pwm
        )
    )


# ============================================================================
# Telemetry
# ============================================================================

LOG_BLOCKS = [

    (
        'Position',
        [
            'stateEstimate.x',
            'stateEstimate.y',
            'stateEstimate.z',
            'stateEstimate.vx',
            'stateEstimate.vy',
            'stateEstimate.vz'
        ]
    ),

    (
        'Attitude',
        [
            'stateEstimate.qw',
            'stateEstimate.qx',
            'stateEstimate.qy',
            'stateEstimate.qz'
        ]
    ),

    (
        'Rates',
        [
            'gyro.x',
            'gyro.y',
            'gyro.z',
            'stabilizer.thrust'
        ]
    )
]


class Telemetry:

    def __init__(self):

        self._lock = threading.Lock()

        self._values = {}
        self._stamps = {}
        self._configs = []

    def start(self, cf):

        for name, variables in LOG_BLOCKS:

            conf = LogConfig(
                name=name,
                period_in_ms=10
            )

            for var in variables:
                conf.add_variable(
                    var,
                    'float'
                )

            cf.log.add_config(conf)

            conf.data_received_cb.add_callback(
                self._on_data
            )

            conf.start()

            self._configs.append(conf)

    def _on_data(
        self,
        timestamp,
        data,
        logconf
    ):

        with self._lock:

            self._values.update(data)

            self._stamps[
                logconf.name
            ] = time.monotonic()

    def snapshot(self):

        with self._lock:

            values = dict(
                self._values
            )

            stamps = list(
                self._stamps.values()
            )

        if len(stamps) < len(LOG_BLOCKS):
            return {
                'age': math.inf
            }

        return {

            'pos': np.array([
                values['stateEstimate.x'],
                values['stateEstimate.y'],
                values['stateEstimate.z']
            ]),

            'vel': np.array([
                values['stateEstimate.vx'],
                values['stateEstimate.vy'],
                values['stateEstimate.vz']
            ]),

            'quat': np.array([
                values['stateEstimate.qw'],
                values['stateEstimate.qx'],
                values['stateEstimate.qy'],
                values['stateEstimate.qz']
            ]),

            'gyro': np.array([
                values['gyro.x'],
                values['gyro.y'],
                values['gyro.z']
            ]),

            'thrust': values[
                'stabilizer.thrust'
            ],

            'age': (
                time.monotonic()
                - min(stamps)
            )
        }

    def wait_until_ready(
        self,
        timeout=5.0
    ):

        deadline = (
            time.monotonic()
            + timeout
        )

        while (
            self.snapshot()['age']
            > TELEMETRY_TIMEOUT_S
        ):

            if time.monotonic() > deadline:
                raise RuntimeError(
                    'No telemetry from the Crazyflie. '
                    'Check the radio link.'
                )

            time.sleep(0.05)

    def stop(self):

        for conf in self._configs:
            conf.stop()


# ============================================================================
# Positioning / estimator
# ============================================================================

def find_positioning_deck(cf):

    for param in (
        'deck.bcFlow2',
        'deck.bcLighthouse4',
        'deck.bcLoco'
    ):

        try:

            if int(
                cf.param.get_value(param)
            ):
                return param.split(
                    '.bc'
                )[1]

        except (
            KeyError,
            ValueError
        ):
            pass

    return None


def reset_estimator(scf):

    cf = scf.cf

    cf.param.set_value(
        'kalman.resetEstimation',
        1
    )

    time.sleep(0.1)

    cf.param.set_value(
        'kalman.resetEstimation',
        0
    )

    conf = LogConfig(
        name='KalmanVariance',
        period_in_ms=500
    )

    for axis in 'XYZ':
        conf.add_variable(
            f'kalman.varP{axis}',
            'float'
        )

    history = []

    deadline = (
        time.monotonic()
        + 15
    )

    with SyncLogger(
        scf,
        conf
    ) as logger:

        for _, data, _ in logger:

            history.append([
                data[
                    f'kalman.varP{axis}'
                ]
                for axis in 'XYZ'
            ])

            recent = np.array(
                history[-10:]
            )

            if (
                len(recent) == 10
                and (
                    recent.max(0)
                    - recent.min(0)
                ).max() < 0.001
            ):
                return

            if time.monotonic() > deadline:
                raise RuntimeError(
                    'Position estimate did not settle. '
                    'Check the positioning system.'
                )


def measure_hover_thrust(
    telemetry,
    seconds=2.0
):

    samples = []

    end = (
        time.monotonic()
        + seconds
    )

    while time.monotonic() < end:

        samples.append(
            telemetry.snapshot()[
                'thrust'
            ]
        )

        time.sleep(0.02)

    hover = float(
        np.median(samples)
    )

    if not 15000 < hover < 62000:

        raise RuntimeError(
            f'Hover thrust {hover:.0f} looks wrong; '
            'not handing over to the policy.'
        )

    return hover


# ============================================================================
# Safety
# ============================================================================

def safety_violation(
    state,
    start
):

    if (
        state['age']
        > TELEMETRY_TIMEOUT_S
    ):
        return (
            f"telemetry "
            f"{state['age'] * 1000:.0f} ms old"
        )

    drift = float(
        np.linalg.norm(
            state['pos'][:2]
            - start['pos'][:2]
        )
    )

    if drift > MAX_DRIFT_M:

        return (
            f'drifted {drift:.2f} m'
        )

    altitude = state['pos'][2]

    if altitude < MIN_SAFE_ALTITUDE:

        return (
            f'too low ({altitude:.2f} m)'
        )

    if altitude > MAX_SAFE_ALTITUDE:

        return (
            f'too high ({altitude:.2f} m)'
        )

    tilt = tilt_deg(
        state['quat']
    )

    if tilt > MAX_TILT_DEG:

        return (
            f'tilted {tilt:.0f} deg'
        )

    return None


# ============================================================================
# Policy flight
# ============================================================================

def fly_trajectory(
    cf,
    policy,
    telemetry,
    hover_pwm,
    rows
):

    start = telemetry.snapshot()

    frame = PolicyFrame(
        start['pos'],
        yaw_of(start['quat'])
    )

    period = 1.0 / CONTROL_HZ

    t0 = time.monotonic()
    next_tick = t0

    last_print = -1.0

    while True:

        now = time.monotonic()

        t = now - t0

        if t >= POLICY_SECONDS:

            print(
                'Trajectory complete.'
            )

            return (
                '1.8 -> 0.6 -> 2.2 -> 0.0 completed'
            )

        state = telemetry.snapshot()

        problem = safety_violation(
            state,
            start
        )

        if problem:

            return (
                f'SAFETY STOP: {problem}'
            )

        desired_z = desired_height(t)

        obs = frame.observation(
            state,
            desired_z
        )

        action = policy(obs)

        if not np.all(
            np.isfinite(action)
        ):

            return (
                'policy output is not finite'
            )

        ratio, rates = command_from_action(
            action
        )

        send_command(
            cf,
            ratio,
            rates,
            hover_pwm
        )

        actual_z = state['pos'][2]

        z_error = (
            actual_z
            - desired_z
        )

        rows.append([
            t,
            actual_z,
            desired_z,
            z_error,
            *obs,
            *action,
            ratio,
            *np.degrees(rates)
        ])

        # Print approximately 5 times per second.
        if (
            t - last_print
            >= 0.2
        ):

            print(
                f't={t:5.2f}s '
                f'actual={actual_z:5.2f} m '
                f'target={desired_z:5.2f} m '
                f'error={z_error:+5.2f} m '
                f'thrust={ratio*100:5.1f}%'
            )

            last_print = t

        next_tick += period

        delay = (
            next_tick
            - time.monotonic()
        )

        if delay > 0:

            time.sleep(delay)

        else:

            next_tick = time.monotonic()


# ============================================================================
# Landing
# ============================================================================

def land(cf):

    hl = cf.high_level_commander

    try:

        time.sleep(1.0)

        hl.land(
            0.0,
            2.5,
            yaw=None
        )

        time.sleep(3.0)

        hl.stop()

    except KeyboardInterrupt:

        print(
            'Cutting motors.'
        )

        cf.commander.send_stop_setpoint()

        hl.stop()


# ============================================================================
# Flight setup
# ============================================================================

def test_flight(
    cf,
    policy,
    telemetry
):

    hl = cf.high_level_commander

    rows = []

    airborne = False

    try:

        cf.param.set_value(
            'flightmode.stabModeRoll',
            0
        )

        cf.param.set_value(
            'flightmode.stabModePitch',
            0
        )

        try:

            cf.param.set_value(
                'commander.enHighLevel',
                1
            )

        except KeyError:

            pass

        cf.supervisor.send_arming_request(
            True
        )

        time.sleep(0.5)

        cf.commander.send_setpoint(
            0,
            0,
            0,
            0
        )

        cf.commander.send_notify_setpoint_stop()

        print()
        print(
            f'Taking off to '
            f'{START_HEIGHT:.2f} m...'
        )

        airborne = True

        hl.takeoff(START_HEIGHT, 2.5, yaw=None)

        # Give the onboard controller enough time
        # to settle at 1.6 m.
        time.sleep(4.0)

        state = telemetry.snapshot()

        print(
            f'Physical altitude after takeoff: '
            f'{state["pos"][2]:.3f} m'
        )

        hover_pwm = measure_hover_thrust(
            telemetry
        )

        print(
            f'Hover thrust: '
            f'{hover_pwm:.0f}'
        )

        print()
        print(
            'Starting explicit trajectory:'
        )
        print(
            '1.80 m -> 0.60 m -> 2.20 m -> 0.00 m'
        )
        print(
            f'    total policy time: '
            f'{POLICY_SECONDS:.1f} s'
        )
        print()

        try:

            reason = fly_trajectory(
                cf,
                policy,
                telemetry,
                hover_pwm,
                rows
            )

        except KeyboardInterrupt:

            reason = 'Ctrl+C'

        finally:

            # Give the onboard controller control again.
            cf.commander.send_notify_setpoint_stop()

        print(
            f'Policy stopped: {reason}'
        )

    finally:

        if airborne:

            print(
                'Returning control to onboard '
                'controller and landing.'
            )

            land(cf)

        cf.param.set_value(
            'flightmode.stabModeRoll',
            1
        )

        cf.param.set_value(
            'flightmode.stabModePitch',
            1
        )

        cf.supervisor.send_arming_request(
            False
        )

        if rows:

            save_log(rows)


# ============================================================================
# Logging
# ============================================================================

def save_log(rows):

    header = [
        't',
        'actual_z',
        'desired_z',
        'z_error',

        'policy_x',
        'policy_y',
        'policy_z',

        'vx_body',
        'vy_body',
        'vz_body',

        'qw',
        'qx',
        'qy',
        'qz',

        'wx',
        'wy',
        'wz',

        'out_thrust',
        'out_roll',
        'out_pitch',
        'out_yaw',

        'thrust_ratio',

        'roll_rate_dps',
        'pitch_rate_dps',
        'yaw_rate_dps'
    ]

    with open(
        FLIGHT_LOG,
        'w',
        newline=''
    ) as f:

        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(
            np.round(
                np.array(rows, dtype=float),
                5
            ).tolist()
        )

    print(
        f'Saved {len(rows)} policy steps '
        f'to {FLIGHT_LOG}'
    )


# ============================================================================
# Dry run
# ============================================================================

def dry_run(
    policy,
    telemetry
):

    start = telemetry.snapshot()

    frame = PolicyFrame(
        start['pos'],
        yaw_of(start['quat'])
    )

    print()
    print(
        'DRY RUN: motors remain OFF.'
    )
    print(
        'The script will evaluate the policy '
        'against the explicit trajectory.'
    )
    print()
    print(
        'Trajectory:'
    )
    print(
        '    1.60 m -> 0.30 m -> 1.60 m'
    )
    print()
    print(
        'Press Ctrl+C to stop.'
    )
    print()

    try:

        t0 = time.monotonic()

        while True:

            t = time.monotonic() - t0

            desired_z = desired_height(t)

            state = telemetry.snapshot()

            if state['age'] > TELEMETRY_TIMEOUT_S:

                print(
                    'Telemetry stale.'
                )

                time.sleep(0.2)

                continue

            obs = frame.observation(
                state,
                desired_z
            )

            action = policy(obs)

            ratio, rates = command_from_action(
                action
            )

            actual_z = state['pos'][2]

            error = (
                actual_z
                - desired_z
            )

            r, p, y = np.degrees(
                rates
            )

            print(
                f't={t:5.2f}s '
                f'actual={actual_z:5.2f} '
                f'target={desired_z:5.2f} '
                f'error={error:+5.2f} '
                f'roll={r:+6.1f} '
                f'pitch={p:+6.1f} '
                f'yaw={y:+6.1f} '
                f'thrust={ratio*100:5.1f}%'
            )

            if t >= POLICY_SECONDS:

                print(
                    'Dry-run trajectory finished.'
                )

                break

            time.sleep(0.2)

    except KeyboardInterrupt:

        print(
            'Dry run stopped.'
        )


# ============================================================================
# Main
# ============================================================================

def main():

    policy = load_policy(
        CHECKPOINT
    )

    cflib.crtp.init_drivers()

    with SyncCrazyflie(
        URI,
        cf=Crazyflie(
            rw_cache='./cache'
        )
    ) as scf:

        cf = scf.cf

        deck = find_positioning_deck(
            cf
        )

        if (
            deck is None
            and not USING_MOCAP
        ):

            raise SystemExit(
                'No Flow, Lighthouse or Loco deck '
                'found, and the policy needs position. '
                'If you use mocap, set USING_MOCAP = True.'
            )

        print(
            f'Position from: '
            f'{deck or "mocap"}'
        )

        print(
            'Waiting for the position estimate '
            'to settle...'
        )

        reset_estimator(
            scf
        )

        telemetry = Telemetry()

        telemetry.start(
            cf
        )

        try:

            telemetry.wait_until_ready()

            if DRY_RUN:

                dry_run(
                    policy,
                    telemetry
                )

            else:

                test_flight(
                    cf,
                    policy,
                    telemetry
                )

        finally:

            telemetry.stop()

            time.sleep(0.1)


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:

        print(
            'Stopped.'
        )