# Differential Drive Robot Simulator

A clean, well-documented Python simulation of a **differential-drive (unicycle) robot** — the most common mobile robot configuration. Implements exact kinematics, multiple control laws, and publication-quality trajectory plots.

---

## What Is a Differential Drive Robot?

A differential-drive robot has **two independently driven wheels** on a common axle. Steering is achieved by varying the relative speed of each wheel — no separate steering mechanism needed.

```
         ┌───────┐
 ○ vL ── │  🤖   │ ── vR ○
         └───────┘
              ↑ heading θ
```

### Kinematic equations

```
ẋ  = v · cos(θ)          v = (vR + vL) / 2      linear velocity
ẏ  = v · sin(θ)          ω = (vR - vL) / L      angular velocity
θ̇  = ω                   L = wheel base [m]
```

State vector `q = [x, y, θ]` — position and heading.  
Control input `u = [v, ω]` — forward speed and turn rate.

---

## Project Structure

```
diffdrive/
├── src/
│   ├── kinematics.py    # ODE model, integrators, DiffDriveSimulator
│   ├── controllers.py   # Open- and closed-loop control laws
│   ├── visualise.py     # Plotting utilities (dark theme)
│   └── simulate.py      # Main runner — generates all figures
├── tests/
│   └── test_kinematics.py
├── outputs/             # Auto-generated figures
└── README.md
```

---

## Quick Start

```bash
# Install dependencies (all standard)
pip install numpy matplotlib scipy

# Run all simulations  →  figures appear in outputs/
cd src
python simulate.py

# Run unit tests
cd ..
python tests/test_kinematics.py
```

---

## Scenarios

| # | Figure | What it shows |
|---|--------|---------------|
| 1 | `01_scenario_grid.png` | 8 open-loop scenarios: straight, circle, S-curve, spiral, figure-eight, timed sequences |
| 2 | `02_integrators.png` | Euler vs RK4 vs Exact integration — accuracy vs step-size |
| 3 | `03_point_tracker.png` | Proportional closed-loop controller driving to goal points |
| 4 | `04_waypoint_follower.png` | Pure-pursuit style multi-waypoint navigation |
| 5 | `05_reference_tracking.png` | Kanayama tracker following a circle and a lemniscate |
| 6 | `06_state_timeseries.png` | Full state + control time-series for a single run |
| 7 | `07_noise_sensitivity.png` | How odometry noise accumulates along a trajectory |
| 8 | `08_tracking_error.png` | Position and heading error convergence from a perturbed start |

---

## API Reference

### `kinematics.py`

#### `RobotParams`
```python
RobotParams(
    wheel_base   = 0.30,   # L  [m]
    wheel_radius = 0.05,   # r  [m]
    max_speed    = 1.00,   # |v| [m/s]
    max_omega    = 3.14,   # |ω| [rad/s]
)
```

#### `DiffDriveSimulator`
```python
sim = DiffDriveSimulator(
    params     = RobotParams(),
    integrator = 'rk4',      # 'euler' | 'rk4' | 'exact'
    add_noise  = False,
    noise_std  = 0.002,       # [m] per step
)

traj = sim.run(
    control_fn,              # callable(t, q) → (v, ω)
    q0  = [0, 0, 0],         # initial pose [x, y, θ]
    T   = 10.0,              # duration [s]
    dt  = 0.02,              # time step [s]
    label = 'my run',
)
```

#### `Trajectory`
```python
traj.t          # (N,) time stamps
traj.x, .y      # (N,) positions
traj.theta       # (N,) headings
traj.v, .omega  # (N,) controls
traj.arc_length  # float — total path length [m]
traj.final_pose  # (x, y, θ) at end
traj.curvature() # (N,) κ = ω/v
```

### `controllers.py`

| Class | Type | Description |
|-------|------|-------------|
| `ConstantControl(v, omega)` | Open | Fixed velocity and turn rate |
| `TimedSequence(segments)` | Open | Schedule of `(duration, v, ω)` segments |
| `SineWaveControl(v, amp, freq)` | Open | Sinusoidal `ω` for S-curves |
| `SpiralControl(v, a, b)` | Open | Archimedean spiral |
| `FigureEightControl(v, omega_abs, period)` | Open | Alternating signed arcs |
| `PointTracker(goal)` | Closed | Proportional point-to-point |
| `WaypointFollower(waypoints)` | Closed | Pure-pursuit waypoint list |
| `ReferenceTracker(ref_fn)` | Closed | Kanayama feedback linearisation |

**Writing A controller:**
```python
def my_ctrl(t, q):
    x, y, theta = q
    v     = 0.4
    omega = 0.3 * np.sin(t)
    return v, omega

traj = sim.run(my_ctrl, T=10.0)
```

---

## Integrator Comparison

| Method | Order | Notes |
|--------|-------|-------|
| `euler` | O(dt) | Simple; drifts fast at large dt |
| `rk4` | O(dt⁴) | Default; good balance of speed/accuracy |
| `exact` | ∞ | Closed-form arc — perfect for constant (v,ω) |

For constant controls over each time step, `exact` is theoretically perfect. RK4 at `dt=0.02 s` gives sub-millimetre error per second for typical motions.

---

## Theory Notes

### Homogeneous transform (frame changes)

The robot's pose defines a frame:

```
T_WB = [cos(θ)  -sin(θ)  x]
       [sin(θ)   cos(θ)  y]
       [  0        0     1]
```

A point `p_B` in the body frame is expressed in world as `p_W = T_WB · p_B`.  
Inverse: `T_BW = T_WB⁻¹` uses `R.T` — no expensive matrix inversion needed.

### Turning radius

For `ω ≠ 0`, the robot follows a circular arc of radius `R = v / ω`.  
- `ω > 0` → turns left (CCW)
- `ω < 0` → turns right (CW)
- `ω = 0` → straight line (R → ∞)

### Kanayama controller

The `ReferenceTracker` implements the Kanayama (1990) feedback linearisation:

```
e = [cos(θ)  sin(θ)  0] [xr-x ]
    [-sin(θ) cos(θ)  0] [yr-y ]
    [  0       0     1] [θr-θ ]

v  = v_r · cos(e_θ) + k_x · e_x
ω  = ω_r + v_r · (k_y · e_y + k_θ · sin(e_θ))
```

This guarantees asymptotic stability for smooth references.

---
*Built with NumPy · Matplotlib · SciPy*
