import math
import numpy as np
import torch


CHECKPOINT = "recovery_policy.pt"

HOME_POS = np.array([
    -0.107,
    0.167,
    0.80
])

HOME_YAW = math.radians(5.7)

THRUST_AT_HOVER = 2.02


class Policy:

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


def load_policy(path):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    layers = []

    for i, shape in (
        (0, (256, 13)),
        (2, (256, 256)),
        (4, (256, 256)),
        (6, (4, 256)),
    ):

        weight = state[f"actor.{i}.weight"]
        bias = state[f"actor.{i}.bias"]

        if tuple(weight.shape) != shape:
            raise ValueError(
                f"Unexpected shape for actor.{i}: "
                f"{tuple(weight.shape)}"
            )

        layers.append((
            weight.double().numpy(),
            bias.double().numpy()
        ))

    return Policy(layers)


def make_observation(
    actual_height,
    desired_height,
    vertical_velocity
):

    # Policy-frame position.
    #
    # X/Y are zero because we only want to test
    # vertical behavior.

    policy_z = (
        HOME_POS[2]
        + (actual_height - desired_height)
    )

    pos = np.array([
        0.0,
        0.0,
        policy_z
    ])

    # Body-frame velocity.
    vel = np.array([
        0.0,
        0.0,
        vertical_velocity
    ])

    # Hover attitude.
    q = np.array([
        math.cos(HOME_YAW / 2),
        0.0,
        0.0,
        math.sin(HOME_YAW / 2)
    ])

    # Zero gyro.
    gyro = np.array([
        0.0,
        0.0,
        0.0
    ])

    return np.concatenate([
        pos,
        vel,
        q,
        gyro
    ])


policy = load_policy(CHECKPOINT)


print()
print("Vertical recovery test")
print("======================")
print()
print("Columns:")
print("actual_z  target_z  vz       policy_thrust")
print()


target = 0.60

for vz in [
    0.0,
    -0.25,
    -0.50,
    -0.75,
    -1.00,
    -1.50,
]:

    actual = 0.60

    obs = make_observation(
        actual,
        target,
        vz
    )

    action = policy(obs)

    print(
        f"{actual:8.2f} "
        f"{target:9.2f} "
        f"{vz:7.2f} "
        f"{action[0]:14.4f}"
    )


print()
print("Below target + descending")
print("==========================")
print()

for actual in [
    0.60,
    0.55,
    0.50,
    0.45,
    0.40,
    0.35,
]:

    vz = -0.50

    obs = make_observation(
        actual,
        target,
        vz
    )

    action = policy(obs)

    print(
        f"actual={actual:.2f} "
        f"error={actual-target:+.2f} "
        f"vz={vz:+.2f} "
        f"thrust={action[0]:.4f}"
    )