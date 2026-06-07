"""
diffdrive.controllers
=====================
A library of control laws for the differential-drive robot.

Each controller is a callable:  control_fn(t, q) → (v, omega)

Controllers provided
--------------------
ConstantControl      – fixed (v, ω) forever
TimedSequence        – schedule of (duration, v, ω) segments
SineWaveControl      – oscillating ω for S-curves
PointTracker         – proportional controller to drive to a goal point
PathFollower         – pure-pursuit controller for waypoint lists
FigureEight          – parametric figure-8 reference tracker
Spiral               – logarithmic / Archimedean spiral
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Open-loop controllers
# ─────────────────────────────────────────────────────────────────────────────

class ConstantControl:
    """Drive at fixed linear and angular velocity."""
    def __init__(self, v: float = 0.5, omega: float = 0.0):
        self.v, self.omega = v, omega

    def __call__(self, t, q):
        return self.v, self.omega


class TimedSequence:
    """
    Execute a list of (duration, v, omega) segments in order.

    Example
    -------
    TimedSequence([(2, 0.5, 0), (1, 0, 1.0), (2, 0.5, 0)])
    → go straight 2 s, pivot 1 s, go straight 2 s
    """
    def __init__(self, segments: List[Tuple[float, float, float]]):
        self.segments   = segments
        self.boundaries = np.cumsum([s[0] for s in segments])

    def __call__(self, t, q):
        for i, boundary in enumerate(self.boundaries):
            if t <= boundary + 1e-9:
                return self.segments[i][1], self.segments[i][2]
        return 0.0, 0.0   # stopped


class SineWaveControl:
    """
    Constant forward speed with sinusoidally varying angular velocity.
    Produces S-curves and slaloms.
    """
    def __init__(self, v: float = 0.4, omega_amp: float = 0.8, freq: float = 0.3):
        self.v         = v
        self.omega_amp = omega_amp
        self.freq      = freq          # Hz

    def __call__(self, t, q):
        omega = self.omega_amp * np.sin(2 * np.pi * self.freq * t)
        return self.v, omega


class SpiralControl:
    """
    Archimedean spiral: omega decreases as radius grows.
    v is constant; omega = v / (a + b*t).
    """
    def __init__(self, v: float = 0.4, a: float = 0.3, b: float = 0.05):
        self.v, self.a, self.b = v, a, b

    def __call__(self, t, q):
        omega = self.v / max(self.a + self.b * t, 0.05)
        return self.v, omega


class FigureEightControl:
    """Open-loop figure-eight via alternating signed arcs."""
    def __init__(self, v: float = 0.4, omega_abs: float = 0.6, half_period: float = 5.0):
        self.v          = v
        self.omega_abs  = omega_abs
        self.half_period = half_period

    def __call__(self, t, q):
        sign = 1 if int(t / self.half_period) % 2 == 0 else -1
        return self.v, sign * self.omega_abs


# ─────────────────────────────────────────────────────────────────────────────
# Closed-loop controllers
# ─────────────────────────────────────────────────────────────────────────────

class PointTracker:
    """
    Proportional heading + distance controller.
    Drives the robot to a goal (gx, gy) and stops.

    Algorithm
    ---------
    1. Compute heading error Δα = atan2(gy-y, gx-x) - θ   (wrapped)
    2. ω  = k_alpha · Δα    (turn toward goal)
    3. v  = k_rho  · ρ · cos(Δα)   (go forward; back off if facing wrong way)
    """
    def __init__(self, goal: Tuple[float, float],
                 k_rho: float = 0.6, k_alpha: float = 2.0,
                 v_max: float = 0.5, stop_radius: float = 0.05):
        self.gx, self.gy = goal
        self.k_rho, self.k_alpha = k_rho, k_alpha
        self.v_max       = v_max
        self.stop_radius = stop_radius

    def __call__(self, t, q):
        x, y, theta = q
        dx = self.gx - x
        dy = self.gy - y
        rho   = np.hypot(dx, dy)
        if rho < self.stop_radius:
            return 0.0, 0.0

        alpha = np.arctan2(dy, dx) - theta
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi   # wrap to (-π, π]

        omega = self.k_alpha * alpha
        v     = min(self.k_rho * rho, self.v_max) * np.cos(alpha)
        return float(v), float(omega)


class WaypointFollower:
    """
    Pure-pursuit style waypoint follower.
    Advances to next waypoint when within `lookahead` metres.
    """
    def __init__(self, waypoints: List[Tuple[float, float]],
                 v: float = 0.4, k_alpha: float = 3.0,
                 lookahead: float = 0.15):
        self.waypoints = list(waypoints)
        self.v         = v
        self.k_alpha   = k_alpha
        self.lookahead = lookahead
        self._idx      = 0

    def __call__(self, t, q):
        if self._idx >= len(self.waypoints):
            return 0.0, 0.0

        x, y, theta = q
        gx, gy = self.waypoints[self._idx]

        dist  = np.hypot(gx - x, gy - y)
        if dist < self.lookahead:
            self._idx += 1
            if self._idx >= len(self.waypoints):
                return 0.0, 0.0
            gx, gy = self.waypoints[self._idx]

        alpha = np.arctan2(gy - y, gx - x) - theta
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        omega = self.k_alpha * alpha
        v     = self.v * max(np.cos(alpha), 0.0)   # slow when off-heading
        return float(v), float(omega)

    def reset(self):
        self._idx = 0


class ReferenceTracker:
    """
    Tracks a smooth reference trajectory (x_r(t), y_r(t), θ_r(t)) using
    the Kanayama feedback linearisation controller.

        v  = v_r·cos(e_θ) + k_x · e_x
        ω  = ω_r           + v_r·(k_y · e_y + k_θ · sin(e_θ))

    where (e_x, e_y, e_θ) are pose errors in the *body* frame.
    """
    def __init__(self,
                 ref_fn,        # callable(t) → (x_r, y_r, theta_r, v_r, omega_r)
                 k_x: float = 1.5,
                 k_y: float = 3.0,
                 k_theta: float = 2.5):
        self.ref_fn  = ref_fn
        self.k_x, self.k_y, self.k_theta = k_x, k_y, k_theta

    def __call__(self, t, q):
        x, y, theta = q
        xr, yr, thr, vr, wr = self.ref_fn(t)

        # Pose error in world frame
        ex_w = xr - x
        ey_w = yr - y
        eth  = (thr - theta + np.pi) % (2 * np.pi) - np.pi

        # Rotate error into body frame
        ex =  np.cos(theta) * ex_w + np.sin(theta) * ey_w
        ey = -np.sin(theta) * ex_w + np.cos(theta) * ey_w

        v     = vr * np.cos(eth) + self.k_x * ex
        omega = wr + vr * (self.k_y * ey + self.k_theta * np.sin(eth))
        return float(v), float(omega)


# ─────────────────────────────────────────────────────────────────────────────
# Reference trajectory generators  (used with ReferenceTracker)
# ─────────────────────────────────────────────────────────────────────────────

def circle_reference(radius: float = 1.5, speed: float = 0.4):
    """Constant-speed circle in the XY plane."""
    omega_r = speed / radius
    def ref(t):
        xr    = radius * np.cos(omega_r * t)
        yr    = radius * np.sin(omega_r * t)
        thr   = omega_r * t + np.pi / 2
        return xr, yr, thr, speed, omega_r
    return ref


def lemniscate_reference(a: float = 1.5, speed: float = 0.35):
    """
    Lemniscate of Bernoulli  (figure-eight in the XY plane).
    Parametric: x = a·cos(s) / (1+sin²(s)),  y = a·sin(s)·cos(s) / (1+sin²(s))
    s increases at a rate chosen so arc-speed ≈ `speed`.
    """
    # Numerical speed parameter
    ds_dt = speed / a   # approximate; good enough for tracking

    def ref(t):
        s   = ds_dt * t
        denom = 1 + np.sin(s) ** 2
        xr  = a * np.cos(s) / denom
        yr  = a * np.sin(s) * np.cos(s) / denom

        # Derivative for heading
        denom2   = denom ** 2
        dxds = (-a * np.sin(s) * denom - a * np.cos(s) * 2 * np.sin(s) * np.cos(s)) / denom2
        dyds = ( a * (np.cos(2*s)) * denom - a * np.sin(s)*np.cos(s) * 2*np.sin(s)*np.cos(s)) / denom2
        thr  = np.arctan2(dyds, dxds)

        ds   = ds_dt
        v_r  = ds * np.hypot(dxds, dyds)
        # curvature-based omega_r (finite difference is fine here)
        eps  = 1e-5
        s2   = s + eps
        d2 = 1 + np.sin(s2)**2
        dx2 = (-a*np.sin(s2)*d2 - a*np.cos(s2)*2*np.sin(s2)*np.cos(s2)) / d2**2
        dy2 = ( a*(np.cos(2*s2))*d2 - a*np.sin(s2)*np.cos(s2)*2*np.sin(s2)*np.cos(s2)) / d2**2
        thr2 = np.arctan2(dy2, dx2)
        omega_r = (thr2 - thr) / (eps / ds_dt)

        return xr, yr, thr, float(v_r), float(omega_r)
    return ref
