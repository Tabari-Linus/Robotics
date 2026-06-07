"""
diffdrive.visualise
===================
Publication-quality plots for differential-drive simulations.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from typing import List, Optional, Tuple

# ── Dark theme ──────────────────────────────────────────────────────────────
BG   = '#0d0d1a'
PANEL= '#12122a'
GRID = '#1c1c3a'
TEXT = '#d0d0ff'
AXES = '#2a2a55'

PLT_RC = {
    'figure.facecolor': BG,
    'axes.facecolor':   PANEL,
    'axes.edgecolor':   AXES,
    'axes.labelcolor':  TEXT,
    'xtick.color':      '#7070a0',
    'ytick.color':      '#7070a0',
    'grid.color':       GRID,
    'grid.linewidth':   0.7,
    'text.color':       TEXT,
    'font.family':      'monospace',
    'figure.dpi':       130,
    'legend.facecolor': '#0d0d22',
    'legend.edgecolor': AXES,
    'legend.labelcolor': TEXT,
}

PALETTE = [
    '#4fc3f7', '#f48fb1', '#a5d6a7', '#ffd54f',
    '#ce93d8', '#ff8a65', '#80cbc4', '#ef9a9a',
]


def _set_theme():
    plt.rcParams.update(PLT_RC)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _speed_cmap_segments(x, y, speed, cmap='plasma'):
    """Build a LineCollection coloured by speed magnitude."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap,
                        norm=plt.Normalize(0, max(speed.max(), 0.01)))
    lc.set_array(speed[:-1])
    lc.set_linewidth(2.2)
    return lc


def draw_robot_pose(ax, x, y, theta, length=0.18, color='white', alpha=1.0):
    """Draw a tiny arrow showing robot orientation."""
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                 arrowprops=dict(arrowstyle='->', color=color,
                                 lw=1.5, alpha=alpha))
    ax.plot(x, y, 'o', color=color, ms=4, alpha=alpha, zorder=5)


def draw_robot_body(ax, x, y, theta, L=0.12, W=0.08, color='white', alpha=0.85):
    """Draw a small rectangle representing the robot body."""
    corners = np.array([
        [ L/2,  W/2],
        [-L/2,  W/2],
        [-L/2, -W/2],
        [ L/2, -W/2],
    ])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    rotated = (R @ corners.T).T + np.array([x, y])
    poly = plt.Polygon(rotated, closed=True,
                       facecolor=color, edgecolor='white',
                       alpha=alpha * 0.4, lw=0.8, zorder=4)
    ax.add_patch(poly)


# ─────────────────────────────────────────────────────────────────────────────
# Main plotting functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectories(trajectories, title='Robot Trajectories',
                      show_poses: int = 12,
                      show_speed_colour: bool = True,
                      reference=None,
                      savepath: Optional[str] = None):
    """
    Plot XY trajectories with optional speed colouring and pose arrows.

    Parameters
    ----------
    trajectories    : list of Trajectory objects
    title           : figure title
    show_poses      : number of pose arrows drawn per trajectory
    show_speed_colour : colour the line by instantaneous speed
    reference       : optional (xs, ys) tuple for a reference path
    savepath        : if given, save figure to this path
    """
    _set_theme()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title(title, pad=14, fontsize=13, color=TEXT)
    ax.set_xlabel('x  [m]'); ax.set_ylabel('y  [m]')
    ax.set_aspect('equal'); ax.grid(True)

    if reference is not None:
        ax.plot(*reference, '--', color='#555588', lw=1.2,
                label='Reference', zorder=1)

    for i, traj in enumerate(trajectories):
        col = PALETTE[i % len(PALETTE)]

        if show_speed_colour:
            lc = _speed_cmap_segments(traj.x, traj.y,
                                      np.abs(traj.v), cmap='plasma')
            ax.add_collection(lc)
            # Proxy patch for legend
            patch = mpatches.Patch(color=col, label=traj.label or f'Traj {i}')
            ax.add_patch(patch)
        else:
            ax.plot(traj.x, traj.y, color=col, lw=2,
                    label=traj.label or f'Traj {i}', zorder=3)

        # Start marker
        ax.plot(traj.x[0], traj.y[0], 's', color=col, ms=9, zorder=6)
        # End marker
        ax.plot(traj.x[-1], traj.y[-1], '*', color=col, ms=12, zorder=6)

        # Pose arrows
        if show_poses > 0:
            idxs = np.linspace(0, len(traj.t) - 1, show_poses, dtype=int)
            for idx in idxs:
                draw_robot_pose(ax, traj.x[idx], traj.y[idx],
                                traj.theta[idx], color=col, alpha=0.55)

    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches='tight')
    return fig, ax


def plot_controls(trajectories, savepath: Optional[str] = None):
    """Time-series plot of v(t) and ω(t) for each trajectory."""
    _set_theme()
    n   = len(trajectories)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    fig.suptitle('Control Inputs Over Time', fontsize=13, color=TEXT)

    for i, traj in enumerate(trajectories):
        col   = PALETTE[i % len(PALETTE)]
        label = traj.label or f'Traj {i}'
        axes[0].plot(traj.t, traj.v,     color=col, lw=1.8, label=label)
        axes[1].plot(traj.t, traj.omega, color=col, lw=1.8, label=label)

    axes[0].set_ylabel('v  [m/s]');   axes[0].grid(True); axes[0].legend(fontsize=8)
    axes[1].set_ylabel('ω  [rad/s]'); axes[1].grid(True); axes[1].set_xlabel('t  [s]')
    axes[0].axhline(0, color=AXES, lw=0.8)
    axes[1].axhline(0, color=AXES, lw=0.8)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches='tight')
    return fig


def plot_state_timeseries(traj, savepath: Optional[str] = None):
    """Plot x(t), y(t), θ(t), v(t), ω(t), κ(t) for a single trajectory."""
    _set_theme()
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(f'State & Control Time-Series  —  {traj.label}',
                 fontsize=12, color=TEXT)

    kappa = traj.curvature()
    t = traj.t

    plots = [
        (axes[0,0], t, traj.x,       'x  [m]',        PALETTE[0]),
        (axes[1,0], t, traj.y,       'y  [m]',        PALETTE[1]),
        (axes[2,0], t, np.degrees(traj.theta), 'θ  [°]', PALETTE[2]),
        (axes[0,1], t, traj.v,       'v  [m/s]',      PALETTE[3]),
        (axes[1,1], t, traj.omega,   'ω  [rad/s]',    PALETTE[4]),
        (axes[2,1], t, kappa,        'κ  [1/m]',      PALETTE[5]),
    ]
    for ax, x, y, ylabel, col in plots:
        ax.plot(x, y, color=col, lw=1.6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True)
        ax.axhline(0, color=AXES, lw=0.7)

    axes[2,0].set_xlabel('t  [s]')
    axes[2,1].set_xlabel('t  [s]')
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches='tight')
    return fig


def plot_comparison_grid(scenarios: List[dict], cols=3, savepath=None):
    """
    Grid of XY trajectory subplots, one per scenario.
    Each scenario dict: {'traj': Trajectory, 'title': str}
    """
    _set_theme()
    n    = len(scenarios)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5.5 * cols, 5 * rows),
                             squeeze=False)
    fig.suptitle('Scenario Comparison', fontsize=14, color=TEXT, y=1.01)

    for k, sc in enumerate(scenarios):
        ax   = axes[k // cols][k % cols]
        traj = sc['traj']
        col  = PALETTE[k % len(PALETTE)]

        ax.set_title(sc.get('title', traj.label), fontsize=9, color=TEXT)
        ax.set_aspect('equal'); ax.grid(True)
        ax.set_xlabel('x [m]', fontsize=8); ax.set_ylabel('y [m]', fontsize=8)

        lc = _speed_cmap_segments(traj.x, traj.y, np.abs(traj.v))
        ax.add_collection(lc)
        ax.autoscale_view()

        # Pose arrows every ~20 steps
        step = max(len(traj.t) // 14, 1)
        for idx in range(0, len(traj.t), step):
            draw_robot_body(ax, traj.x[idx], traj.y[idx],
                            traj.theta[idx], color=col)

        ax.plot(traj.x[0],  traj.y[0],  's', color=col, ms=8, zorder=6)
        ax.plot(traj.x[-1], traj.y[-1], '*', color=col, ms=11, zorder=6)

        info = (f'arc={traj.arc_length:.2f}m  '
                f'T={traj.t[-1]:.1f}s\n'
                f'final ({traj.x[-1]:.2f},{traj.y[-1]:.2f})')
        ax.text(0.02, 0.97, info, transform=ax.transAxes,
                va='top', fontsize=6.5, color='#9999cc',
                bbox=dict(facecolor='#0d0d1a', edgecolor=AXES, pad=2))

    # Hide unused axes
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_visible(False)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches='tight')
    return fig


def plot_tracking_error(traj, ref_fn, savepath=None):
    """
    Plot position and heading tracking error over time for a reference tracker.
    ref_fn(t) → (xr, yr, thr, vr, wr)
    """
    _set_theme()
    refs = np.array([ref_fn(ti) for ti in traj.t])
    xr, yr, thr = refs[:,0], refs[:,1], refs[:,2]

    pos_err   = np.hypot(xr - traj.x, yr - traj.y)
    theta_err = np.abs((thr - traj.theta + np.pi) % (2*np.pi) - np.pi)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig.suptitle(f'Tracking Error  —  {traj.label}', fontsize=12, color=TEXT)

    a1.plot(traj.t, pos_err,              color=PALETTE[0], lw=1.8)
    a1.fill_between(traj.t, 0, pos_err,  color=PALETTE[0], alpha=0.15)
    a1.set_ylabel('Position error [m]');  a1.grid(True)

    a2.plot(traj.t, np.degrees(theta_err), color=PALETTE[1], lw=1.8)
    a2.fill_between(traj.t, 0, np.degrees(theta_err), color=PALETTE[1], alpha=0.15)
    a2.set_ylabel('Heading error [°]'); a2.grid(True)
    a2.set_xlabel('t  [s]')

    for ax in (a1, a2):
        ax.axhline(0, color=AXES, lw=0.7)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches='tight')
    return fig
