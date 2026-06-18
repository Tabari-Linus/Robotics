"""
diffdrive.simulate
==================
Run all demonstration scenarios and save figures to outputs/.

Run:   python simulate.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from kinematics   import DiffDriveSimulator, RobotParams
from controllers  import (
    ConstantControl, TimedSequence, SineWaveControl,
    SpiralControl, FigureEightControl,
    PointTracker, WaypointFollower, ReferenceTracker,
    circle_reference, lemniscate_reference,
)
from visualise    import (
    plot_trajectories, plot_controls,
    plot_state_timeseries, plot_comparison_grid,
    plot_tracking_error,
)

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

params = RobotParams(wheel_base=0.30, wheel_radius=0.05,
                     max_speed=1.0,   max_omega=3.14)
sim    = DiffDriveSimulator(params, integrator='rk4')


# ═══════════════════════════════════════════════════════════════════════════
# 1  OPEN-LOOP SCENARIO COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print('── Scenario 1: open-loop comparisons …')

scenarios_ol = [
    {
        'ctrl':  ConstantControl(v=0.5, omega=0.0),
        'label': 'Straight line\nv=0.5, ω=0',
        'T': 5.0,
    },
    {
        'ctrl':  ConstantControl(v=0.4, omega=0.5),
        'label': 'Circle\nv=0.4, ω=0.5',
        'T': 13.0,
    },
    {
        'ctrl':  ConstantControl(v=0.4, omega=-0.5),
        'label': 'Circle (CW)\nv=0.4, ω=-0.5',
        'T': 13.0,
    },
    {
        'ctrl':  TimedSequence([
            (2.0, 0.5,  0.0),
            (1.5, 0.0,  np.pi/3),
            (2.0, 0.5,  0.0),
            (1.5, 0.0, -np.pi/3),
            (2.0, 0.5,  0.0),
        ]),
        'label': 'Square-ish\nstraight + pivot',
        'T': 9.0,
    },
    {
        'ctrl':  SineWaveControl(v=0.4, omega_amp=0.9, freq=0.25),
        'label': 'S-Curve slalom\nω=A·sin(2πft)',
        'T': 12.0,
    },
    {
        'ctrl':  SpiralControl(v=0.35, a=0.5, b=0.04),
        'label': 'Archimedean spiral\nω=v/(a+bt)',
        'T': 20.0,
    },
    {
        'ctrl':  FigureEightControl(v=0.4, omega_abs=0.55, half_period=5.7),
        'label': 'Figure-eight\nalternating arcs',
        'T': 22.0,
    },
    {
        'ctrl':  TimedSequence([
            (0.5, 0.0,  np.pi),    # spin 180°
            (3.0, 0.6,  0.0),      # straight
            (1.0, 0.0,  np.pi/2),  # turn 90°
            (2.0, 0.5,  0.3),      # arc
            (1.0, 0.0, -np.pi/2),  # turn back
            (2.0, 0.4,  0.0),
        ]),
        'label': 'Custom sequence\nspin+straight+arc',
        'T': 9.5,
    },
]

scen_data = []
for sc in scenarios_ol:
    traj = sim.run(sc['ctrl'], q0=[0,0,0], T=sc['T'], dt=0.02, label=sc['label'])
    scen_data.append({'traj': traj, 'title': sc['label']})

fig_grid = plot_comparison_grid(scen_data, cols=4,
                                 savepath=f'{OUT}/01_scenario_grid.png')
plt.close(fig_grid)
print('   saved 01_scenario_grid.png')


# ═══════════════════════════════════════════════════════════════════════════
# 2  INTEGRATOR ACCURACY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print('── Scenario 2: integrator accuracy …')

ctrl_circle = ConstantControl(v=0.5, omega=0.7)
q0          = [0.0, 0.0, 0.0]

trajs_int = []
for method, dt, label in [
    ('euler', 0.10, 'Euler  dt=0.10 s'),
    ('euler', 0.02, 'Euler  dt=0.02 s'),
    ('rk4',   0.10, 'RK4    dt=0.10 s'),
    ('exact',  0.10, 'Exact  dt=0.10 s'),
]:
    s = DiffDriveSimulator(params, integrator=method)
    t = s.run(ctrl_circle, q0=q0, T=9.0, dt=dt, label=label)
    trajs_int.append(t)

# Reference: exact with very fine dt
sim_ref = DiffDriveSimulator(params, integrator='exact')
traj_ref = sim_ref.run(ctrl_circle, q0=q0, T=9.0, dt=0.001, label='Exact dt=0.001 s (ref)')

fig_int, ax_int = plot_trajectories(
    trajs_int + [traj_ref],
    title='Integrator Accuracy: Euler vs RK4 vs Exact\n(circular arc, same control)',
    show_poses=0,
    show_speed_colour=False,
    savepath=f'{OUT}/02_integrators.png',
)
plt.close(fig_int)
print('   saved 02_integrators.png')


# ═══════════════════════════════════════════════════════════════════════════
# 3  POINT-TO-POINT CLOSED-LOOP CONTROL
# ═══════════════════════════════════════════════════════════════════════════
print('── Scenario 3: point tracker …')

goals  = [(3.0, 0.0), (3.0, 2.5), (0.0, 2.5), (0.0, 0.0)]
starts = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 2.5, np.pi), (0.0, 2.5, np.pi)]
colors = ['#4fc3f7','#f48fb1','#a5d6a7','#ffd54f']

plt.rcParams.update({'figure.facecolor':'#0d0d1a','axes.facecolor':'#12122a',
                     'axes.edgecolor':'#2a2a55','axes.labelcolor':'#d0d0ff',
                     'xtick.color':'#7070a0','ytick.color':'#7070a0',
                     'grid.color':'#1c1c3a','text.color':'#d0d0ff','font.family':'monospace'})

fig_pt, ax_pt = plt.subplots(figsize=(8, 7))
ax_pt.set_title('Point-to-Point Closed-Loop Control\n(proportional heading + distance)',
                fontsize=12, color='#d0d0ff')
ax_pt.set_aspect('equal'); ax_pt.grid(True)
ax_pt.set_xlabel('x [m]'); ax_pt.set_ylabel('y [m]')

for (gx, gy), q0, col in zip(goals, starts, colors):
    ctrl = PointTracker((gx, gy), k_rho=0.8, k_alpha=2.5, v_max=0.5)
    traj = sim.run(ctrl, q0=list(q0), T=8.0, dt=0.02,
                   label=f'→ ({gx},{gy})')
    ax_pt.plot(traj.x, traj.y, color=col, lw=2.2)
    ax_pt.plot(traj.x[0], traj.y[0], 's', color=col, ms=8, zorder=6)
    ax_pt.plot(gx, gy, '*', color=col, ms=14, zorder=7,
               markeredgecolor='white', markeredgewidth=0.5)

ax_pt.set_xlim(-0.5, 4); ax_pt.set_ylim(-0.5, 3.5)
fig_pt.tight_layout()
fig_pt.savefig(f'{OUT}/03_point_tracker.png', dpi=130, bbox_inches='tight')
plt.close(fig_pt)
print('   saved 03_point_tracker.png')


# ═══════════════════════════════════════════════════════════════════════════
# 4  WAYPOINT FOLLOWER
# ═══════════════════════════════════════════════════════════════════════════
print('── Scenario 4: waypoint follower …')

# Three different waypoint paths
paths = {
    'Square loop': [(2,0),(2,2),(0,2),(0,0)],
    'Star points':  [(2,0),(0.6,0.8),(0,2.4),(1.4,1.4),(2.6,2.4),(2,0.8)],
    'Slalom gates': [(0.5*i, 0.6*(-1)**i) for i in range(1,10)],
}

fig_wp, axes_wp = plt.subplots(1, 3, figsize=(15, 5))
fig_wp.suptitle('Waypoint Follower  (pure-pursuit style)',
                fontsize=12, color='#d0d0ff')

for ax, (name, wpts) in zip(axes_wp, paths.items()):
    ctrl = WaypointFollower(wpts, v=0.4, k_alpha=3.0, lookahead=0.18)
    traj = sim.run(ctrl, q0=[0,0,0], T=30.0, dt=0.02, label=name)

    ax.set_title(name, fontsize=10, color='#d0d0ff')
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

    from visualise import _speed_cmap_segments
    lc = _speed_cmap_segments(traj.x, traj.y, np.abs(traj.v))
    ax.add_collection(lc); ax.autoscale_view()

    wpts_np = np.array(wpts)
    ax.scatter(wpts_np[:,0], wpts_np[:,1],
               color='#ffd54f', s=60, zorder=6, marker='D', label='Waypoints')
    ax.plot(traj.x[0], traj.y[0], 's', color='#4fc3f7', ms=9, zorder=7, label='Start')
    ax.legend(fontsize=8)

plt.tight_layout()
fig_wp.savefig(f'{OUT}/04_waypoint_follower.png', dpi=130, bbox_inches='tight')
plt.close(fig_wp)
print('   saved 04_waypoint_follower.png')


# ═══════════════════════════════════════════════════════════════════════════
# 5  REFERENCE TRACKING (circle + lemniscate)
# ═══════════════════════════════════════════════════════════════════════════
print('── Scenario 5: reference tracking …')

fig_tr, axes_tr = plt.subplots(1, 2, figsize=(14, 6))
fig_tr.suptitle('Kanayama Reference Tracker  —  Closed-Loop Trajectory Following',
                fontsize=12, color='#d0d0ff')

for ax, (name, ref_fn, q0, T) in zip(axes_tr, [
    ('Circle  (r=1.5 m)',    circle_reference(radius=1.5, speed=0.4),
     [1.5, 0.0, np.pi/2], 24.0),
    ('Lemniscate  (a=1.5 m)', lemniscate_reference(a=1.5, speed=0.32),
     [1.5, 0.0, np.pi/2], 30.0),
]):
    ctrl  = ReferenceTracker(ref_fn, k_x=1.5, k_y=3.0, k_theta=2.5)
    traj  = sim.run(ctrl, q0=q0, T=T, dt=0.02, label=name)

    # Reference path
    ts_ref  = np.linspace(0, T, 1000)
    refs    = np.array([ref_fn(t) for t in ts_ref])
    ax.plot(refs[:,0], refs[:,1], '--', color='#555588', lw=1.2, label='Reference')

    from visualise import _speed_cmap_segments
    lc = _speed_cmap_segments(traj.x, traj.y, np.abs(traj.v))
    ax.add_collection(lc); ax.autoscale_view()

    ax.set_title(name, fontsize=10, color='#d0d0ff')
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.plot(traj.x[0], traj.y[0], 's', color='#4fc3f7', ms=9, zorder=6)
    ax.legend(fontsize=8)

plt.tight_layout()
fig_tr.savefig(f'{OUT}/05_reference_tracking.png', dpi=130, bbox_inches='tight')
plt.close(fig_tr)
print('   saved 05_reference_tracking.png')


# ═══════════════════════════════════════════════════════════════════════════
# 6  STATE TIME-SERIES  (detailed view of one trajectory)
# ═══════════════════════════════════════════════════════════════════════════
print('── Scenario 6: state time-series …')

ctrl_detail = SineWaveControl(v=0.4, omega_amp=0.85, freq=0.28)
traj_detail = sim.run(ctrl_detail, q0=[0,0,0], T=14.0, dt=0.02,
                      label='S-Curve (SineWave controller)')

fig_ts = plot_state_timeseries(traj_detail,
                                savepath=f'{OUT}/06_state_timeseries.png')
plt.close(fig_ts)
print('   saved 06_state_timeseries.png')


# ═══════════════════════════════════════════════════════════════════════════
# 7  NOISE SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════
print('── Scenario 7: noise sensitivity …')

ctrl_noise = ConstantControl(v=0.5, omega=0.4)
trajs_noise = []
np.random.seed(42)
for noise, label in [(0.0,    'No noise'),
                     (0.002,  'σ=0.002 m'),
                     (0.005,  'σ=0.005 m'),
                     (0.012,  'σ=0.012 m')]:
    s = DiffDriveSimulator(params, integrator='rk4',
                           add_noise=noise>0, noise_std=noise)
    t = s.run(ctrl_noise, q0=[0,0,0], T=15.0, dt=0.02, label=label)
    trajs_noise.append(t)

fig_n, ax_n = plot_trajectories(
    trajs_noise,
    title='Effect of Odometry Noise on Circular Arc\n(constant controls)',
    show_poses=0, show_speed_colour=False,
    savepath=f'{OUT}/07_noise_sensitivity.png',
)
plt.close(fig_n)
print('   saved 07_noise_sensitivity.png')


# ═══════════════════════════════════════════════════════════════════════════
# 8  TRACKING ERROR PLOT
# ═══════════════════════════════════════════════════════════════════════════
print('── Scenario 8: tracking error …')

ref_fn2 = circle_reference(radius=1.5, speed=0.4)
ctrl8   = ReferenceTracker(ref_fn2, k_x=1.5, k_y=3.0, k_theta=2.5)

# Perturbed start to show convergence
traj8   = sim.run(ctrl8, q0=[1.0, 0.5, 0.2], T=20.0, dt=0.02,
                  label='Circle tracking (perturbed start)')

fig_err = plot_tracking_error(traj8, ref_fn2,
                               savepath=f'{OUT}/08_tracking_error.png')
plt.close(fig_err)
print('   saved 08_tracking_error.png')


# ═══════════════════════════════════════════════════════════════════════════
# 9  SUMMARY MOSAIC
# ═══════════════════════════════════════════════════════════════════════════
print('── Building summary mosaic …')

import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

files = [
    ('01_scenario_grid.png',      'Open-loop scenario grid'),
    ('02_integrators.png',        'Integrator accuracy'),
    ('03_point_tracker.png',      'Point-to-point control'),
    ('04_waypoint_follower.png',  'Waypoint follower'),
    ('05_reference_tracking.png', 'Reference tracking'),
    ('06_state_timeseries.png',   'State time-series'),
    ('07_noise_sensitivity.png',  'Noise sensitivity'),
    ('08_tracking_error.png',     'Tracking error'),
]

fig_m = plt.figure(figsize=(20, 18), facecolor='#0d0d1a')
fig_m.suptitle('Differential Drive Robot — Full Simulation Suite',
               fontsize=16, color='#d0d0ff', y=1.005)

gs = GridSpec(4, 2, figure=fig_m, hspace=0.06, wspace=0.04)
ax_list = [fig_m.add_subplot(gs[r, c]) for r in range(4) for c in range(2)]

for ax, (fname, title) in zip(ax_list, files):
    path = f'{OUT}/{fname}'
    if os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img); ax.axis('off')
        ax.set_title(title, fontsize=9, color='#a0a0cc', pad=3)

fig_m.savefig(f'{OUT}/00_summary_mosaic.png', dpi=100, bbox_inches='tight')
plt.close(fig_m)
print('   saved 00_summary_mosaic.png')

print('\n✅  All figures written to outputs/')
