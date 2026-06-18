"""
diffdrive.kinematics
====================
Exact kinematic model for a differential-drive (unicycle) robot.

State vector:  q = [x, y, θ]   (metres, metres, radians)
Control input: u = [v, ω]       (m/s, rad/s)
                or equivalently  u_wheel = [v_L, v_R]  (m/s each wheel)

Kinematic equations:
    ẋ  = v · cos(θ)
    ẏ  = v · sin(θ)
    θ̇  = ω

where:
    v  = (v_R + v_L) / 2          linear  velocity
    ω  = (v_R - v_L) / L          angular velocity
    L  = wheel-base (track width)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Robot geometry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RobotParams:
    """Physical parameters of the differential-drive robot."""
    wheel_base:   float = 0.30   # L  – distance between wheels  [m]
    wheel_radius: float = 0.05   # r  – wheel radius              [m]
    max_speed:    float = 1.00   # |v| upper bound                [m/s]
    max_omega:    float = 3.14   # |ω| upper bound                [rad/s]

    def clip_controls(self, v: float, omega: float):
        """Hard-limit controls to physical bounds."""
        v     = float(np.clip(v,     -self.max_speed, self.max_speed))
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))
        return v, omega

    def wheel_speeds(self, v: float, omega: float):
        """Convert (v, ω) → (v_L, v_R)."""
        v_L = v - omega * self.wheel_base / 2
        v_R = v + omega * self.wheel_base / 2
        return v_L, v_R

    def body_controls(self, v_L: float, v_R: float):
        """Convert (v_L, v_R) → (v, ω)."""
        v     = (v_R + v_L) / 2
        omega = (v_R - v_L) / self.wheel_base
        return v, omega


# ─────────────────────────────────────────────────────────────────────────────
# ODE right-hand side
# ─────────────────────────────────────────────────────────────────────────────

def ddrive_ode(q: np.ndarray, v: float, omega: float) -> np.ndarray:
    """
    dq/dt for given state q = [x, y, θ] and controls (v, ω).
    Pure kinematics — no mass, no friction.
    """
    x, y, theta = q
    return np.array([
        v * np.cos(theta),
        v * np.sin(theta),
        omega,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Integrators
# ─────────────────────────────────────────────────────────────────────────────

def integrate_euler(q: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    """Forward Euler – O(dt) accuracy."""
    return q + dt * ddrive_ode(q, v, omega)


def integrate_rk4(q: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    """Classic 4th-order Runge-Kutta – O(dt⁴) accuracy."""
    k1 = ddrive_ode(q,            v, omega)
    k2 = ddrive_ode(q + dt/2*k1,  v, omega)
    k3 = ddrive_ode(q + dt/2*k2,  v, omega)
    k4 = ddrive_ode(q + dt*k3,    v, omega)
    return q + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# ─────────────────────────────────────────────────────────────────────────────
# Exact arc solution (zero-order hold on controls)
# ─────────────────────────────────────────────────────────────────────────────

def integrate_exact(q: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    """
    Closed-form integration assuming constant (v, ω) over dt.
    Exact for straight lines (ω≈0) and circular arcs (ω≠0).
    """
    x, y, theta = q
    if abs(omega) < 1e-9:                  # straight line
        x_new = x + v * dt * np.cos(theta)
        y_new = y + v * dt * np.sin(theta)
        theta_new = theta
    else:                                  # circular arc
        R = v / omega                      # turning radius
        theta_new = theta + omega * dt
        x_new = x + R * (np.sin(theta_new) - np.sin(theta))
        y_new = y - R * (np.cos(theta_new) - np.cos(theta))
    return np.array([x_new, y_new, theta_new])


INTEGRATORS = {
    'euler': integrate_euler,
    'rk4':   integrate_rk4,
    'exact': integrate_exact,
}


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """Container for a simulated trajectory."""
    t:      np.ndarray          # (N,)   time stamps [s]
    x:      np.ndarray          # (N,)   x position  [m]
    y:      np.ndarray          # (N,)   y position  [m]
    theta:  np.ndarray          # (N,)   heading     [rad]
    v:      np.ndarray          # (N,)   linear vel  [m/s]
    omega:  np.ndarray          # (N,)   angular vel [rad/s]
    label:  str = ''

    @property
    def arc_length(self) -> float:
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        return float(np.sum(np.hypot(dx, dy)))

    @property
    def final_pose(self):
        return self.x[-1], self.y[-1], self.theta[-1]

    def curvature(self) -> np.ndarray:
        """Signed curvature κ = ω / v (returns 0 where v ≈ 0)."""
        with np.errstate(invalid='ignore', divide='ignore'):
            kappa = np.where(np.abs(self.v) > 1e-6,
                             self.omega / self.v, 0.0)
        return kappa


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────

class DiffDriveSimulator:
    """
    Step-by-step simulator for a differential-drive robot.

    Usage
    -----
    sim = DiffDriveSimulator(params, integrator='rk4')
    traj = sim.run(control_fn, q0=[0,0,0], T=10.0, dt=0.02)

    control_fn(t, q) -> (v, omega)   — called at every time step
    """

    def __init__(self,
                 params: Optional[RobotParams] = None,
                 integrator: str = 'rk4',
                 add_noise: bool = False,
                 noise_std: float = 0.002):
        self.params     = params or RobotParams()
        self.integrator = INTEGRATORS[integrator]
        self.add_noise  = add_noise
        self.noise_std  = noise_std

    def run(self,
            control_fn: Callable,
            q0: list | np.ndarray = None,
            T: float = 10.0,
            dt: float = 0.02,
            label: str = '') -> Trajectory:
        """
        Simulate for T seconds with fixed time step dt.

        Parameters
        ----------
        control_fn : callable(t, q) → (v, omega)
        q0         : initial state [x0, y0, theta0]  (default origin, facing East)
        T          : total duration [s]
        dt         : time step     [s]
        label      : human-readable name for plots
        """
        q0 = np.array(q0 if q0 is not None else [0.0, 0.0, 0.0], dtype=float)

        steps = int(np.round(T / dt)) + 1
        ts    = np.linspace(0, T, steps)

        xs     = np.empty(steps)
        ys     = np.empty(steps)
        thetas = np.empty(steps)
        vs     = np.empty(steps)
        omegas = np.empty(steps)

        q = q0.copy()
        for i, t in enumerate(ts):
            v, omega = control_fn(t, q)
            v, omega = self.params.clip_controls(v, omega)

            xs[i], ys[i], thetas[i] = q
            vs[i], omegas[i] = v, omega

            if i < steps - 1:
                q = self.integrator(q, v, omega, dt)
                if self.add_noise:
                    q += np.random.randn(3) * self.noise_std
                q[2] = (q[2] + np.pi) % (2 * np.pi) - np.pi  # wrap θ ∈ (-π, π]

        return Trajectory(t=ts, x=xs, y=ys, theta=thetas,
                          v=vs, omega=omegas, label=label)
