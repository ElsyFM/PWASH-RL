"""Velocity-triggered direct-wrench Crazyflie environment with an experimental NASA-inspired VRS model.

Policy actions remain four high-level values: collective thrust and direct body torques X/Y/Z.
Five phases: powered ascent; motor-off freefall until vz <= freefall_restart_velocity_mps (-2.9 m/s); powered braking/capture
near 0.65 m; powered climb to 1.70 m; landing. The Johnson AHS-SF2004 Tables 3/4 mean-inflow
correction with first-order lag is applied during powered descending braking and is coupled to
aggregate thrust through an experimental fixed-power approximation. This is a reduced-order
prototype, not a calibrated Crazyflie model. Per-motor RPM/spin-up and the residual wake/air-column
model are not included.

References:
https://github.com/learnsyslab/crazyflow/blob/main/crazyflow/drones/params.toml
https://github.com/learnsyslab/crazyflow/blob/main/crazyflow/dynamics/first_principles/dynamics.py
"""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply_inverse


@configclass
class CrazyflieYPathEnvCfg(DirectRLEnvCfg):
    """Configuration for direct collective-thrust/body-torque control with VRS."""

    decimation = 2
    episode_length_s = 15.0  # full sequence needs roughly 8-10 s; 10 s left almost no slack
    num_actions = 4       # [collective thrust, direct body torque x/y/z]
    num_observations = 19 # [pos(3), vel(3), quat(4), ang_vel(3), phase(6)]
    num_states = 0
    observation_space = 19
    action_space = 4

    # Idealized motor-off freefall until downward speed reaches this threshold.
    instant_motor_cutoff_in_freefall = True
    # Motors cut when the drone reaches this height (was hard-coded 1.60 m, which contradicted the 1.70 m climb target).
    freefall_release_height_m = 1.70
    freefall_restart_velocity_mps = -2.9
    capture_target_height_m = 0.65
    capture_upper_height_m = 0.70   # capture band is 0.65 m +/- 0.05 m
    capture_lower_height_m = 0.60
    capture_vertical_speed_limit_mps = 0.20
    capture_fail_height_m = 0.55    # fail only well below the band
    # Per-phase step limits (control step = 0.01 s). Exceeding one ends the episode as a failure so the
    # policy cannot farm shaping reward by lingering in a phase.
    brake_timeout_steps = 200
    climb_timeout_steps = 500
    landing_timeout_steps = 800
    climb_target_height_m = 1.70
    climb_complete_height_m = 1.70
    # Phase 4 (HOVER): hold hover_height_m for hover_duration_steps control steps (0.01 s each), then land.
    hover_height_m = 1.70
    hover_duration_steps = 200   # 2.0 s
    # Diagnostic-only override; keep disabled for training and normal playback.
    vrs_test_hold_recovery_phase = False

    # Johnson (AHS-SF2004) VRS model switches/parameters. The velocity-boundary
    # constants are inside the implementation and are nondimensional per the paper.
    vrs_enabled = True
    vrs_air_density_kg_m3 = 1.204
    vrs_prop_radius_m = 0.0235
    vrs_kappa = 1.12                 # hover induced-loss factor; calibrate for CF2x
    vrs_strength = 1.0               # f multiplier from the paper's VRS correction
    vrs_min_rotor_thrust_n = 0.005   # avoid normalizing near zero thrust
    vrs_min_thrust_scale = 0.35      # numerical/safety floor, NOT a NASA parameter
    vrs_tau_min_s = 0.01
    vrs_tau_max_s = 0.12
    # Precompute Johnson's normalized curves once, then bilinearly interpolate each step.
    # This follows the paper's real-time recommendation to precompute and interpolate the model.
    vrs_lut_z_min = -2.5
    vrs_lut_z_max = 0.25
    vrs_lut_z_points = 281
    vrs_lut_x_min = 0.0
    vrs_lut_x_max = 1.2
    vrs_lut_x_points = 121

    sim: SimulationCfg = SimulationCfg(
        device="cuda",
        dt=1.0 / 200.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=1.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    # Preserve the existing parallelization for now. Start with fewer envs while debugging.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)


class CrazyflieYPathEnv(DirectRLEnv):
    """Ascent -> velocity-triggered freefall -> powered capture -> climb -> landing.

    Phase 0: powered ascent to the release height.
    Phase 1: motor-off freefall until vz <= configured restart speed.
    Phase 2: powered braking/capture at z ~= 0.65 m; VRS is enabled during descent.
    Phase 3: powered climb to 1.70 m.
    Phase 4: landing.

    The policy commands a net body wrench (collective thrust plus x/y/z torques); no
    individual motor RPM or spin-up dynamics are simulated in this direct-wrench variant.
    """

    cfg: CrazyflieYPathEnvCfg

    def __init__(self, cfg: CrazyflieYPathEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.num_actions = self.cfg.action_space

        # Direct-wrench actuation: preserve the older environment's normalized-action mapping.
        # action[0] -> total vertical thrust in N; actions[1:4] -> direct body torque (N*m).
        # The VRS computation uses four equal virtual rotor disks only to estimate mean inflow;
        # there are no simulated motor RPM states, motor lag, mixer limits, or per-rotor outputs.
        self.takeoff_mass_kg = 0.031156
        self.arm_length_m = 0.03253
        self.total_thrust_max_n = 0.58  # matches prior direct-wrench mapping: (a0 + 1) * 0.29 N
        self.mixing_matrix = torch.tensor(
            [[-1.0, -1.0, 1.0, 1.0],
             [-1.0, 1.0, 1.0, -1.0],
             [-1.0, 1.0, -1.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )

        self.drag_coeff_body = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )

        # Match the user's measured Crazyflie takeoff mass (31.156 g), preserving USD inertia ratios.
        native_masses = self.robot.root_physx_view.get_masses()
        native_inertias = self.robot.root_physx_view.get_inertias()
        total_native_mass = native_masses[0].sum()
        mass_scale = self.takeoff_mass_kg / total_native_mass
        scaled_masses = native_masses * mass_scale
        scaled_inertias = native_inertias * mass_scale
        env_indices = torch.arange(self.robot.root_physx_view.count, dtype=torch.int32, device="cpu")
        self.robot.root_physx_view.set_masses(scaled_masses, env_indices)
        self.robot.root_physx_view.set_inertias(scaled_inertias, env_indices)
        self.capture_complete_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.recovery_complete_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.release_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.hover_complete_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.phase_start_time = torch.zeros(self.num_envs, device=self.device)
        self._last_phase = self.phase.clone()
        self.phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase_timeout_limits = torch.tensor(
            [10**9, 10**9, self.cfg.brake_timeout_steps, self.cfg.climb_timeout_steps, 10**9, self.cfg.landing_timeout_steps],
            dtype=torch.long, device=self.device,
        )

        # Four equal virtual disks provide a coarse local-flow estimate for VRS,
        # while the physical actuation remains one net thrust and one body-torque vector.
        self.vrs_delta_inflow = torch.zeros((self.num_envs, 4), device=self.device)
        self.vrs_active = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self.vrs_thrust_scale = torch.ones((self.num_envs, 4), device=self.device)
        self.vrs_vertical_ratio = torch.zeros((self.num_envs, 4), device=self.device)
        self.vrs_horizontal_ratio = torch.zeros((self.num_envs, 4), device=self.device)
        self.vrs_target_delta = torch.zeros((self.num_envs, 4), device=self.device)

        self.vrs_rotor_disk_area_m2 = torch.pi * self.cfg.vrs_prop_radius_m ** 2
        self._build_vrs_lookup_tables()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=False)

    @staticmethod
    def _zero_intercept_cubic_coefficients(
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: torch.Tensor,
        slope_a: torch.Tensor,
        y_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve p(x)=b*x+c*x^2+d*x^3 from p(A), p'(A), and p(B)."""
        x_a_safe = torch.where(x_a.abs() < 1.0e-5, torch.where(x_a >= 0, 1.0e-5, -1.0e-5), x_a)
        x_b_safe = torch.where(x_b.abs() < 1.0e-5, torch.where(x_b >= 0, 1.0e-5, -1.0e-5), x_b)
        dx = x_a - x_b
        dx_safe = torch.where(dx.abs() < 1.0e-5, torch.where(dx >= 0, 1.0e-5, -1.0e-5), dx)
        f = (y_a / x_a_safe - y_b / x_b_safe) / dx_safe
        d_resid = slope_a - y_a / x_a_safe
        denom = x_a_safe * dx_safe
        denom = torch.where(denom.abs() < 1.0e-6, torch.where(denom >= 0, 1.0e-6, -1.0e-6), denom)
        d = (d_resid - x_a_safe * f) / denom
        c = f - d * (x_a_safe + x_b_safe)
        b = y_a / x_a_safe - c * x_a_safe - d * x_a_safe.square()
        return b, c, d

    @staticmethod
    def _momentum_inflow(z: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nondimensional momentum-theory induced inflow and slope.

        The forward-flight equation v=1/sqrt(x^2+(z+v)^2) is solved with damped
        fixed-point iterations for x>0. The axial branches are used at x ~= 0.
        These are the paper's baseline momentum-theory branches, not the VRS correction.
        """
        half_z = 0.5 * z
        root_up = torch.sqrt(half_z.square() + 1.0)
        v_up = -half_z + root_up
        root_dn = torch.sqrt(torch.clamp(half_z.square() - 1.0, min=1.0e-8))
        v_dn = -half_z - root_dn
        axial_v = torch.where(z < -2.0, v_dn, v_up)

        v_forward = torch.clamp(axial_v, min=1.0e-3)
        for _ in range(24):
            target = torch.rsqrt(torch.clamp(x.square() + (z + v_forward).square(), min=1.0e-8))
            v_forward = 0.5 * v_forward + 0.5 * target
        v = torch.where(x < 1.0e-4, axial_v, v_forward)

        slope_up = -0.5 + z / (4.0 * torch.sqrt(half_z.square() + 1.0))
        slope_dn = -0.5 - z / (4.0 * root_dn)
        axial_slope = torch.where(z < -2.0, slope_dn, slope_up)
        denom_power = torch.clamp(x.square() + (z + v).square(), min=1.0e-8).pow(1.5)
        implicit_denom = denom_power + z + v
        implicit_denom = torch.where(
            implicit_denom.abs() < 1.0e-5,
            torch.where(implicit_denom >= 0, 1.0e-5, -1.0e-5),
            implicit_denom,
        )
        forward_slope = -(z + v) / implicit_denom
        slope = torch.where(x < 1.0e-4, axial_slope, forward_slope)
        return v, slope

    def _johnson_baseline_inflow(
        self, z: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Johnson A/B/C baseline induced inflow, using nondimensional velocities."""
        vx_c = 0.75
        ratio = torch.clamp(x / vx_c, min=0.0, max=1.0)
        z_a = -1.5 + 0.2 * ratio.square()
        z_b = -2.1 + 0.2 * ratio.square()
        z_b = z_b + torch.where(
            ratio > 0.5,
            0.7 * (z_a - z_b) * (2.0 * ratio - 1.0).clamp(min=0.0).pow(3),
            torch.zeros_like(z_b),
        )

        v_a, slope_a = self._momentum_inflow(z_a, x)
        v_b, _ = self._momentum_inflow(z_b, x)
        b, c, d = self._zero_intercept_cubic_coefficients(z_a, z_b, v_a, slope_a, v_b)
        v_poly = b * z + c * z.square() + d * z.pow(3)
        slope_poly = b + 2.0 * c * z + 3.0 * d * z.square()

        v_momentum, slope_momentum = self._momentum_inflow(z, x)
        use_polynomial = (x < vx_c) & (z > z_b) & (z < z_a)
        return (
            torch.where(use_polynomial, v_poly, v_momentum),
            torch.where(use_polynomial, slope_poly, slope_momentum),
        )

    def _johnson_vrs_inflow(
        self, z: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Table 3/Table 4 mean VRS inflow correction.

        Returns baseline v, target delta-v (all nondimensional by vh), and the
        instantaneous Table-4 VRS-region mask. The piecewise cubic curves connect
        the paper's D, N, X and E points. See Johnson, AHS-SF2004, Tables 3-4.
        """
        v_base, _ = self._johnson_baseline_inflow(z, x)

        z_d = torch.full_like(z, -0.2)
        z_n0 = torch.full_like(z, -0.45)
        z_x0 = torch.full_like(z, -1.5)
        z_e0 = torch.full_like(z, -2.0)
        vx_m = 0.95
        x_ratio = torch.clamp(x / vx_m, min=0.0, max=1.0)
        remaining = torch.clamp(1.0 - x_ratio.square(), min=0.0)
        z_n = 0.5 * (z_n0 + z_x0) + 0.5 * (z_n0 - z_x0) * remaining.pow(0.2)
        z_x = 0.5 * (z_n0 + z_x0) - 0.5 * (z_n0 - z_x0) * remaining.pow(1.5)
        z_e = z_e0 + (z_x - z_x0)

        v_n, dv_n = self._johnson_baseline_inflow(z_n, x)
        v_x, dv_x = self._johnson_baseline_inflow(z_x, x)
        delta_n = 0.85 - (z_n + v_n)  # Table 4: (Vz+v)_N
        delta_x = 1.25 - (z_x + v_x)  # Table 4: (Vz+v)_X
        slope_n = -(1.0 + dv_n)
        slope_x = -(1.0 + dv_x)

        # Segment D-N: interpolate from N's correction/slope to zero value/slope at D.
        h_dn = torch.clamp(z_d - z_n, min=1.0e-5)
        t_dn = ((z - z_n) / h_dn).clamp(0.0, 1.0)
        h00 = 2.0 * t_dn.pow(3) - 3.0 * t_dn.square() + 1.0
        h10 = t_dn.pow(3) - 2.0 * t_dn.square() + t_dn
        delta_dn = h00 * delta_n + h10 * h_dn * slope_n

        # Segment N-X: connect both Table-4 endpoints and their specified slopes.
        h_nx = torch.clamp(z_n - z_x, min=1.0e-5)
        t_nx = ((z - z_x) / h_nx).clamp(0.0, 1.0)
        h00 = 2.0 * t_nx.pow(3) - 3.0 * t_nx.square() + 1.0
        h10 = t_nx.pow(3) - 2.0 * t_nx.square() + t_nx
        h01 = -2.0 * t_nx.pow(3) + 3.0 * t_nx.square()
        h11 = t_nx.pow(3) - t_nx.square()
        delta_nx = (
            h00 * delta_x + h10 * h_nx * slope_x
            + h01 * delta_n + h11 * h_nx * slope_n
        )

        # Segment X-E: Table 3's zero-intercept polynomial; endpoint slope at E is not matched.
        b_e, c_e, d_e = self._zero_intercept_cubic_coefficients(
            z_x, z_e, delta_x, slope_x, torch.zeros_like(delta_x)
        )
        delta_xe = b_e * z + c_e * z.square() + d_e * z.pow(3)

        on_dn = (z >= z_n) & (z <= z_d)
        on_nx = (z >= z_x) & (z < z_n)
        on_xe = (z > z_e) & (z < z_x)
        in_vrs_region = (z < z_d) & (z > z_e) & (x < vx_m)
        delta = torch.where(
            on_dn, delta_dn,
            torch.where(on_nx, delta_nx, torch.where(on_xe, delta_xe, torch.zeros_like(z))),
        )
        delta = torch.where(in_vrs_region, delta, torch.zeros_like(delta))
        return v_base, delta, in_vrs_region

    def _build_vrs_lookup_tables(self) -> None:
        """Precompute Johnson's nondimensional inflow curves on CPU once per env."""
        z_grid = torch.linspace(
            self.cfg.vrs_lut_z_min, self.cfg.vrs_lut_z_max,
            self.cfg.vrs_lut_z_points, dtype=torch.float64, device="cpu",
        )
        x_grid = torch.linspace(
            self.cfg.vrs_lut_x_min, self.cfg.vrs_lut_x_max,
            self.cfg.vrs_lut_x_points, dtype=torch.float64, device="cpu",
        )
        zz, xx = torch.meshgrid(z_grid, x_grid, indexing="ij")
        with torch.no_grad():
            v_base, delta_vrs, _ = self._johnson_vrs_inflow(zz, xx)
        self._vrs_z_lut = z_grid.to(device=self.device, dtype=torch.float32)
        self._vrs_x_lut = x_grid.to(device=self.device, dtype=torch.float32)
        self._vrs_base_lut = v_base.to(device=self.device, dtype=torch.float32)
        self._vrs_delta_lut = delta_vrs.to(device=self.device, dtype=torch.float32)
        self._vrs_z_lut_scale = (self.cfg.vrs_lut_z_points - 1) / (
            self.cfg.vrs_lut_z_max - self.cfg.vrs_lut_z_min
        )
        self._vrs_x_lut_scale = (self.cfg.vrs_lut_x_points - 1) / (
            self.cfg.vrs_lut_x_max - self.cfg.vrs_lut_x_min
        )

    def _lookup_johnson_inflow(
        self, z: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilinear interpolation of the precomputed nondimensional (v_base, delta-v) table."""
        zf = ((z - self.cfg.vrs_lut_z_min) * self._vrs_z_lut_scale).clamp(
            0.0, self.cfg.vrs_lut_z_points - 1.0001
        )
        xf = ((x - self.cfg.vrs_lut_x_min) * self._vrs_x_lut_scale).clamp(
            0.0, self.cfg.vrs_lut_x_points - 1.0001
        )
        z0 = torch.floor(zf).to(torch.long)
        x0 = torch.floor(xf).to(torch.long)
        z1 = z0 + 1
        x1 = x0 + 1
        fz = zf - z0.to(zf.dtype)
        fx = xf - x0.to(xf.dtype)

        b00 = self._vrs_base_lut[z0, x0]
        b01 = self._vrs_base_lut[z0, x1]
        b10 = self._vrs_base_lut[z1, x0]
        b11 = self._vrs_base_lut[z1, x1]
        d00 = self._vrs_delta_lut[z0, x0]
        d01 = self._vrs_delta_lut[z0, x1]
        d10 = self._vrs_delta_lut[z1, x0]
        d11 = self._vrs_delta_lut[z1, x1]

        base_low = b00 + fx * (b01 - b00)
        base_high = b10 + fx * (b11 - b10)
        delta_low = d00 + fx * (d01 - d00)
        delta_high = d10 + fx * (d11 - d10)
        v_base = base_low + fz * (base_high - base_low)
        delta_vrs = delta_low + fz * (delta_high - delta_low)
        return v_base, delta_vrs

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.capture_complete_event.zero_()
        self.recovery_complete_event.zero_()
        self.release_event.zero_()
        self.hover_complete_event.zero_()

        pos_z = self.robot.data.root_pos_w[:, 2]
        vel_z = self.robot.data.root_lin_vel_w[:, 2]

        # Phase 0 -> 1: cut motor force at the release height.
        start_freefall = (self.phase == 0) & (pos_z >= self.cfg.freefall_release_height_m)
        self.release_event.copy_(start_freefall)
        self.phase = torch.where(start_freefall, torch.ones_like(self.phase), self.phase)

        # Phase 1 -> 2: restart the motors as soon as downward speed reaches freefall_restart_velocity_mps.
        # This is velocity-triggered, not a height-triggered motor restart.
        start_braking = (self.phase == 1) & (vel_z <= self.cfg.freefall_restart_velocity_mps)
        self.phase = torch.where(start_braking, torch.full_like(self.phase, 2), self.phase)

        # Phase 2 -> 3: only leave the braking/capture phase after reaching a narrow
        # altitude band around 0.65 m with low vertical speed. This discourages a
        # fast pass through the target altitude from being counted as a capture.
        captured = (
            (self.phase == 2)
            & (pos_z <= self.cfg.capture_upper_height_m)
            & (pos_z >= self.cfg.capture_lower_height_m)
            & (torch.abs(vel_z) <= self.cfg.capture_vertical_speed_limit_mps)
        )
        self.capture_complete_event.copy_(captured)
        self.phase = torch.where(captured, torch.full_like(self.phase, 3), self.phase)

        # Phase 3 -> 4: after capturing near 0.65 m, climb back to the requested 1.70 m.
        if self.cfg.vrs_test_hold_recovery_phase:
            start_landing = torch.zeros_like(self.phase, dtype=torch.bool)
        else:
            start_landing = (self.phase == 3) & (pos_z >= self.cfg.climb_complete_height_m)
        self.recovery_complete_event.copy_(start_landing)
        # (start_landing here means: climb finished -> start the 2 s HOVER phase, phase 4.)
        self.phase = torch.where(start_landing, torch.full_like(self.phase, 4), self.phase)

        # Phase 4 -> 5: after hovering for hover_duration_steps control steps, begin the landing.
        # phase_steps counts steps already spent in the phase when this check runs, hence the -1.
        hover_done = (self.phase == 4) & (self.phase_steps >= self.cfg.hover_duration_steps - 1)
        self.hover_complete_event.copy_(hover_done)
        self.phase = torch.where(hover_done, torch.full_like(self.phase, 5), self.phase)

        # Per-phase step counter and timeout flag (consumed by _get_rewards and _get_dones).
        changed = self.phase != self._last_phase
        self.phase_steps = torch.where(changed, torch.zeros_like(self.phase_steps), self.phase_steps + 1)
        self._last_phase = self.phase.clone()
        self.phase_timeout = self.phase_steps > self.phase_timeout_limits[self.phase]

        self.target_pos[:, 0:2] = 0.0
        self.target_pos[:, 2] = torch.where(
            self.phase == 0,
            self.cfg.climb_target_height_m,
            torch.where(
                (self.phase == 1) | (self.phase == 2),
                self.cfg.capture_target_height_m,
                torch.where((self.phase == 3) | (self.phase == 4), self.cfg.climb_target_height_m, 0.0),
            ),
        )

    def _apply_action(self) -> None:
        """Apply direct wrench; VRS attenuates net thrust during Phase 2 powered braking."""
        dt = float(self.cfg.sim.dt)
        freefall_mask = self.phase == 1

        # Keep the original direct-wrench action scale and four action semantics. The new 18-D
        # observation includes a fifth phase indicator, so old 17-D checkpoints are incompatible.
        thrust_clean_total = torch.clamp((self.actions[:, 0] + 1.0) * 0.29, min=0.0, max=0.58)
        proxy_rotor_thrust = (thrust_clean_total / 4.0).unsqueeze(-1).expand(-1, 4)

        body_velocity = quat_apply_inverse(
            self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w
        )
        body_omega = quat_apply_inverse(
            self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w
        )

        thrust_scale = torch.ones_like(proxy_rotor_thrust)
        active_now = torch.zeros_like(proxy_rotor_thrust, dtype=torch.bool)
        z_ratio = torch.zeros_like(proxy_rotor_thrust)
        x_ratio = torch.zeros_like(proxy_rotor_thrust)
        delta_target = torch.zeros_like(proxy_rotor_thrust)

        if self.cfg.vrs_enabled:
            rotor_xyz = torch.stack(
                [
                    -self.mixing_matrix[1] * self.arm_length_m,
                    self.mixing_matrix[0] * self.arm_length_m,
                    torch.zeros(4, device=self.device),
                ], dim=-1,
            )
            arm_xyz = rotor_xyz.unsqueeze(0).expand(self.num_envs, -1, -1)
            omega_at_arm = torch.cross(
                body_omega.unsqueeze(1).expand(-1, 4, -1), arm_xyz, dim=-1
            )
            rotor_air_velocity = body_velocity.unsqueeze(1) + omega_at_arm
            rotor_vz = rotor_air_velocity[:, :, 2]
            rotor_vx = torch.linalg.vector_norm(rotor_air_velocity[:, :, :2], dim=-1)

            rotor_vh = torch.sqrt(
                proxy_rotor_thrust.clamp(min=0.0)
                / (2.0 * self.cfg.vrs_air_density_kg_m3 * self.vrs_rotor_disk_area_m2)
            )
            rotor_has_thrust = proxy_rotor_thrust >= self.cfg.vrs_min_rotor_thrust_n
            vh_safe = rotor_vh.clamp(min=1.0e-3)
            z_ratio = (rotor_vz / vh_safe).clamp(min=-6.0, max=3.0)
            x_ratio = (rotor_vx / vh_safe).clamp(min=0.0, max=1.5)

            v_base, delta_vrs_raw = self._lookup_johnson_inflow(z_ratio, x_ratio)
            x_over_m = (x_ratio / 0.95).clamp(0.0, 1.0)
            remain = (1.0 - x_over_m.square()).clamp(min=0.0)
            z_x = 0.5 * (-0.45 - 1.5) - 0.5 * (-0.45 + 1.5) * remain.pow(1.5)
            z_e = -2.0 + (z_x + 1.5)
            in_vrs_region = (z_ratio < -0.2) & (z_ratio > z_e) & (x_ratio < 0.95)
            # VRS is relevant here during powered descent/braking; the later upward
            # climb phase does not apply this attenuation.
            active_now = (
                (self.phase.unsqueeze(-1) == 2)
                & rotor_has_thrust
                & in_vrs_region
            )
            delta_target = torch.where(active_now, delta_vrs_raw, torch.zeros_like(delta_vrs_raw))

            tau_vrs = (
                0.7 * 2.0 * torch.pi * self.cfg.vrs_prop_radius_m / vh_safe
            ).clamp(min=self.cfg.vrs_tau_min_s, max=self.cfg.vrs_tau_max_s)
            self.vrs_delta_inflow = self.vrs_delta_inflow + (
                dt * (delta_target - self.vrs_delta_inflow) / tau_vrs
            )
            self.vrs_delta_inflow = torch.where(
                freefall_mask.unsqueeze(-1), torch.zeros_like(self.vrs_delta_inflow), self.vrs_delta_inflow
            )

            # Approximate fixed-power thrust attenuation using Johnson's induced-inflow correction.
            # The same scale is aggregated across four equal virtual disks; this intentionally does
            # not model individual motor RPM, motor lag, per-rotor saturation or rotor-asymmetric torque.
            w_base = z_ratio + self.cfg.vrs_kappa * v_base
            w_corrected = z_ratio + self.cfg.vrs_kappa * (
                v_base + self.cfg.vrs_strength * self.vrs_delta_inflow
            )
            effect_mask = (
                (self.phase.unsqueeze(-1) == 2)
                & rotor_has_thrust
                & (active_now | (self.vrs_delta_inflow.abs() > 1.0e-5))
            )
            scale_from_power = (
                w_base.clamp(min=0.05) / w_corrected.clamp(min=0.05)
            ).clamp(min=self.cfg.vrs_min_thrust_scale, max=1.0)
            thrust_scale = torch.where(effect_mask, scale_from_power, torch.ones_like(scale_from_power))

        self.vrs_active.copy_(active_now)
        self.vrs_thrust_scale.copy_(thrust_scale)
        self.vrs_vertical_ratio.copy_(z_ratio)
        self.vrs_horizontal_ratio.copy_(x_ratio)
        self.vrs_target_delta.copy_(delta_target)

        # Apply the mean scale to the single collective-thrust command. Freefall is truly motor-off.
        thrust_total = (proxy_rotor_thrust * thrust_scale).sum(dim=-1)
        torque_body = self.actions[:, 1:] * 0.001
        thrust_total = torch.where(freefall_mask, torch.zeros_like(thrust_total), thrust_total)
        torque_body = torch.where(freefall_mask.unsqueeze(-1), torch.zeros_like(torque_body), torque_body)

        # Legacy direct-wrench environment had no modeled aerodynamic drag; keep that actuator
        # behavior here for rapid transfer and reuse of its checkpoints.
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces[:, 0, 2] = thrust_total
        self.robot.set_external_force_and_torque(forces, torque_body.unsqueeze(1), body_ids=[0])

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Per-env local position: world position includes the env-grid offset, which would feed the
        # policy tens of metres of meaningless x/y that differs for every environment.
        pos = self.robot.data.root_pos_w - self.scene.env_origins
        vel = self.robot.data.root_lin_vel_w
        quat = self.robot.data.root_quat_w
        ang_vel = self.robot.data.root_ang_vel_w
        phase_one_hot = torch.nn.functional.one_hot(self.phase, num_classes=6).float()
        obs = torch.cat([pos, vel, quat, ang_vel, phase_one_hot], dim=-1)
        return {"policy": obs}

    def get_observations(self):
        return self._get_observations()

    def _get_rewards(self) -> torch.Tensor:
        pos = self.robot.data.root_pos_w
        vel = self.robot.data.root_lin_vel_w
        pos_local = pos - self.scene.env_origins
        xy_error = torch.norm(pos_local[:, :2], dim=-1)
        horizontal_speed = torch.linalg.norm(vel[:, :2], dim=-1)
        pos_error = torch.norm(pos_local - self.target_pos, dim=-1)
        freefall_mask = (self.phase == 1).float()
        # No tracking reward in motor-off freefall: the policy cannot influence it, and a low value
        # there would otherwise make lingering below the release height look better than releasing.
        ascent_mask = (self.phase == 0).float()
        r_track = torch.exp(-pos_error) * ascent_mask  # ascent only; later phases use their own shaping
        # One-time bonus for completing the ascent so reaching the release height is never penalized.
        r_release = 20.0 * self.release_event.float()
        brake_mask = (self.phase == 2).float()
        climb_mask = (self.phase == 3).float()
        hover_mask = (self.phase == 4).float()
        landing_mask = (self.phase == 5).float()

        # Phase 1 intentionally has no target descent-velocity reward: the transition
        # is triggered by vz <= freefall_restart_velocity_mps (-2.9 m/s), then powered braking begins.
        # Phase 2 rewards capture near 0.65 m with low vertical speed, and explicitly
        # penalizes passing below the requested capture height.
        brake_altitude_error = torch.abs(pos[:, 2] - self.cfg.capture_target_height_m)

        # Stronger, smoother reward for approaching the capture altitude.
        r_brake_altitude = torch.exp(-4.0 * brake_altitude_error) * brake_mask

        # Reward low vertical speed primarily when near the actual capture altitude.
        r_brake_velocity = (
            torch.exp(-torch.abs(vel[:, 2]) / 0.25)
            * torch.exp(-4.0 * brake_altitude_error)
            * brake_mask
        )
        capture_overshoot = torch.clamp(
            self.cfg.capture_lower_height_m - pos[:, 2], min=0.0
        ) * brake_mask
        r_capture_complete = 30.0 * self.capture_complete_event.float()

        # Phase 3 is the powered climb from the capture zone to 1.70 m.
        climb_velocity_error = torch.abs(vel[:, 2] - 0.4)
        r_climb_velocity = torch.exp(-climb_velocity_error) * climb_mask
        climb_height_progress = torch.clamp(
            (pos[:, 2] - self.cfg.capture_target_height_m)
            / (self.cfg.climb_target_height_m - self.cfg.capture_target_height_m),
            min=0.0, max=1.0,
        )
        r_climb_height = climb_height_progress * climb_mask
        r_climb_complete = 20.0 * self.recovery_complete_event.float()

        # Phase 4 is a timed hover at hover_height_m: reward holding height with low vertical speed.
        hover_height_error = torch.abs(pos_local[:, 2] - self.cfg.hover_height_m)
        r_hover = (
            0.5 * torch.exp(-5.0 * hover_height_error)
            + 0.5 * torch.exp(-2.0 * torch.abs(vel[:, 2]))
        ) * hover_mask
        r_hover_complete = 10.0 * self.hover_complete_event.float()

        quat = self.robot.data.root_quat_w
        upright_penalty = 1.0 - quat[:, 0].square()
        # Actions in true freefall are ignored by the dynamics and not penalized.
        action_penalty = self.actions.square().sum(dim=-1) * (1.0 - freefall_mask)

        # Encourage faster descent above 0.50 m, then slow for touchdown.
        landing_target_vz = -0.30 - 0.40 * torch.clamp(
            pos_local[:, 2] / 0.50, min=0.0, max=1.0
        )
        landing_velocity_error = torch.abs(vel[:, 2] - landing_target_vz)
        r_landing_velocity = torch.exp(-2.0 * landing_velocity_error) * landing_mask
        landing_height = torch.clamp(pos_local[:, 2], min=0.0)
        r_landing_height = torch.exp(-2.0 * landing_height) * landing_mask
        landed = (
            (self.phase == 5)
            & (pos_local[:, 2] <= 0.03)
            & (torch.abs(vel[:, 2]) <= 0.5)
            & (xy_error <= 0.10)
            & (horizontal_speed <= 0.25)
        ).float()
        r_landed_bonus = 100.0 * landed
        capture_failed = (self.phase == 2) & (pos[:, 2] < self.cfg.capture_fail_height_m)
        boundary_excess = torch.clamp(xy_error - 0.50, min=0.0)
        r_boundary = -5.0 * boundary_excess
        r_out_of_bounds = -10.0 * (xy_error > 0.75).float()
        crashed_now = (pos[:, 2] < 0.02) & (landed < 0.5)
        r_crash = -10.0 * crashed_now.float()
        # Lateral shaping: hold position everywhere (motors-off freefall excluded) and prioritise (0, 0) in landing.
        r_lateral = (
            -0.3 * horizontal_speed * (1.0 - freefall_mask)
            - 1.0 * xy_error * landing_mask
            - 0.5 * horizontal_speed * landing_mask
            - 0.5 * horizontal_speed * hover_mask
        )

        total_reward = (
            0.5 * r_track
            + r_release
            + 0.5 * r_brake_altitude
            + 0.25 * r_brake_velocity
            + r_capture_complete
            + r_climb_complete
            + 1.0 * r_hover
            + r_hover_complete
            + 0.5 * r_climb_velocity
            + 0.5 * r_climb_height
            + r_landed_bonus
            - 30.0 * capture_overshoot
            - 10.0 * capture_failed.float()
            - 10.0 * self.phase_timeout.float()
            - 0.2 * upright_penalty
            - 0.01 * action_penalty
            + 1.0 * r_landing_velocity
            + 0.2 * r_landing_height
            - 0.3 * xy_error
            + r_boundary
            + r_out_of_bounds
            + r_crash
            + r_lateral
        )

        # RSL-RL collects extras["log"] during rollout and reports the averaged
        # values alongside the normal iteration/reward statistics. These phase
        # percentages are the average share of parallel environments in each phase
        # over that iteration's rollout window, not just a single instantaneous snapshot.
        phase = self.phase
        self.extras["log"] = {
            "Phase/ascent_pct": (phase == 0).float().mean() * 100.0,
            "Phase/freefall_pct": (phase == 1).float().mean() * 100.0,
            "Phase/brake_capture_pct": (phase == 2).float().mean() * 100.0,
            "Phase/climb_pct": (phase == 3).float().mean() * 100.0,
            "Phase/hover_pct": (phase == 4).float().mean() * 100.0,
            "Phase/landing_pct": (phase == 5).float().mean() * 100.0,
            "Phase/mean_altitude_m": pos[:, 2].mean(),
            "Phase/mean_vertical_speed_mps": vel[:, 2].mean(),
            "VRS/active_env_pct": self.vrs_active.any(dim=-1).float().mean() * 100.0,
            "Events/capture_success_pct_per_step": self.capture_complete_event.float().mean() * 100.0,
            "Events/climb_complete_pct_per_step": self.recovery_complete_event.float().mean() * 100.0,
            "Events/hover_complete_pct_per_step": self.hover_complete_event.float().mean() * 100.0,
            "Events/capture_failure_pct_per_step": capture_failed.float().mean() * 100.0,
            "Events/phase_timeout_pct_per_step": self.phase_timeout.float().mean() * 100.0,
            "Events/landed_pct_per_step": landed.mean() * 100.0,
        }
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.robot.data.root_pos_w
        vel = self.robot.data.root_lin_vel_w
        pos_local = pos - self.scene.env_origins
        xy_error = torch.norm(pos_local[:, :2], dim=-1)
        horizontal_speed = torch.linalg.norm(vel[:, :2], dim=-1)

        landed = (
            (self.phase == 5)
            & (pos_local[:, 2] <= 0.03)
            & (torch.abs(vel[:, 2]) <= 0.5)
            & (xy_error <= 0.10)
            & (horizontal_speed <= 0.25)
        )
        crashed = (pos[:, 2] < 0.02) & ~landed
        # Overshooting well below the 0.65 m capture target ends that episode as a failure.
        capture_failed = (self.phase == 2) & (pos[:, 2] < self.cfg.capture_fail_height_m)
        out_of_bounds = torch.norm(pos_local[:, :2], dim=-1) > 0.75
        time_out = self.episode_length_buf >= self.max_episode_length
        done = crashed | capture_failed | out_of_bounds | landed | self.phase_timeout
        return done, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        self.phase[env_ids] = 0
        self.phase_start_time[env_ids] = 0.0
        self._last_phase[env_ids] = 0
        self.capture_complete_event[env_ids] = False
        self.recovery_complete_event[env_ids] = False
        self.hover_complete_event[env_ids] = False
        self.release_event[env_ids] = False
        self.phase_steps[env_ids] = 0
        self.phase_timeout[env_ids] = False
        self.vrs_delta_inflow[env_ids] = 0.0
        self.vrs_active[env_ids] = False
        self.vrs_thrust_scale[env_ids] = 1.0
        self.vrs_vertical_ratio[env_ids] = 0.0
        self.vrs_horizontal_ratio[env_ids] = 0.0
        self.vrs_target_delta[env_ids] = 0.0

        default_pos = (
            self.robot.data.default_root_state[env_ids, :3]
            + self.scene.env_origins[env_ids]
        )
        default_rot = self.robot.data.default_root_state[env_ids, 3:7]
        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_rot], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)
