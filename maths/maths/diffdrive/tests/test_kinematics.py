"""
tests/test_kinematics.py
========================
Unit tests for the differential-drive kinematics module.
Run with:  python tests/test_kinematics.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from kinematics import (
    integrate_euler, integrate_rk4, integrate_exact,
    DiffDriveSimulator, RobotParams,
)
from controllers import ConstantControl

PASS = '✅ PASS'
FAIL = '❌ FAIL'

def check(name, cond, tol=1e-6):
    ok = bool(np.all(np.abs(np.asarray(cond)) < tol))
    print(f'  {PASS if ok else FAIL}  {name}')
    return ok


# ─────────────────────────────────────────────────────────────────────────────
print('══ RobotParams ═══════════════════════════════')
p = RobotParams(wheel_base=0.30)

vL, vR = p.wheel_speeds(0.5, 0.0)
check('Straight: v_L == v_R', vL - vR)
check('Straight: v == (vL+vR)/2', (vL + vR)/2 - 0.5)

vL2, vR2 = p.wheel_speeds(0.0, 1.0)
check('Spin: v_L == -v_R', vL2 + vR2)

v_back, w_back = p.body_controls(vL2, vR2)
check('body_controls inverse of wheel_speeds (v)', v_back)
check('body_controls inverse of wheel_speeds (w)', w_back - 1.0)

# ─────────────────────────────────────────────────────────────────────────────
print('\n══ Integrators ═══════════════════════════════')

# Straight line: distance should equal v*dt
q0 = np.array([0.0, 0.0, 0.0])
v, w, dt = 1.0, 0.0, 0.5
for name, fn in [('euler', integrate_euler),
                 ('rk4',   integrate_rk4),
                 ('exact',  integrate_exact)]:
    q1 = fn(q0, v, w, dt)
    check(f'{name}: straight x = v*dt', q1[0] - v*dt)
    check(f'{name}: straight y = 0',    q1[1])
    check(f'{name}: straight θ = 0',    q1[2])

# Circular arc: final radius should equal v/omega
q0 = np.array([0.0, 0.0, 0.0])
v, w, T = 0.5, 0.5, 2*np.pi/0.5   # one full circle
dt = 0.001
q = q0.copy()
for _ in range(int(T/dt)):
    q = integrate_rk4(q, v, w, dt)
check('RK4 circle: returns to start x',   q[0], tol=0.01)
check('RK4 circle: returns to start y',   q[1], tol=0.01)
th_wrap = (q[2] + np.pi) % (2*np.pi) - np.pi
check('RK4 circle: heading wraps to 0',   th_wrap, tol=0.02)

# ─────────────────────────────────────────────────────────────────────────────
print('\n══ Simulator ══════════════════════════════════')
sim  = DiffDriveSimulator(RobotParams(), integrator='exact')
ctrl = ConstantControl(v=0.5, omega=0.0)
traj = sim.run(ctrl, q0=[0,0,0], T=4.0, dt=0.02, label='test')

check('Trajectory length > 0', 1 - int(len(traj.t) > 0))
check('Straight x = v*T at end', traj.x[-1] - 0.5*4.0, tol=0.01)
check('Straight y = 0 throughout', np.max(np.abs(traj.y)), tol=1e-9)
check('Straight theta = 0',        np.max(np.abs(traj.theta)), tol=1e-9)
check('Arc length ≈ v*T', traj.arc_length - 0.5*4.0, tol=0.01)

# ─────────────────────────────────────────────────────────────────────────────
print('\n══ Control clipping ════════════════════════════')
p2 = RobotParams(max_speed=0.5, max_omega=1.0)
v_c, w_c = p2.clip_controls(10.0, 99.0)
check('v clipped to max_speed',  v_c - 0.5)
check('w clipped to max_omega',  w_c - 1.0)
v_c2, w_c2 = p2.clip_controls(-10.0, -99.0)
check('v clipped to -max_speed', v_c2 + 0.5)
check('w clipped to -max_omega', w_c2 + 1.0)

print('\nAll tests complete.')
