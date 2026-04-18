"""
Racing MPC Solver Generator — acados / SQP-RTI
===============================================
Run ONCE on the Jetson to generate the C solver:
    python3 generate_racing_solver.py

Model:  Kinematic bicycle, state=[x, y, theta, v], controls=[delta, a]
Params: [x_ref, y_ref, theta_ref, v_ref] per stage

Tuned for racing: higher speed, looser position tracking,
tighter yaw tracking for cornering precision.
"""

import numpy as np
import os
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import SX, vertcat, sin, cos, tan

# ─────────────────────────────────────────────────────────────────────────────
# RACING PARAMETERS — tune these
# ─────────────────────────────────────────────────────────────────────────────
N   = 20        # horizon steps
DT  = 0.05      # timestep [s] → 1.0s lookahead
L   = 0.33      # wheelbase [m]

# Cost weights [x, y, theta, v]
# High yaw weight → precise cornering
# Low position weight → don't fight the path, follow it smoothly
Q_diag  = [3.0,  3.0,  8.0,  2.0]   # stage state cost
Qf_diag = [10.0, 10.0, 20.0, 5.0]   # terminal state cost
R_diag  = [0.5,  0.05]              # [steer, accel] input cost

# Constraints
V_MAX     = 6.0    # max speed [m/s] — push this up for racing
V_MIN     = 0.0
A_MAX     = 8.0    # max accel [m/s^2]
DELTA_MAX = 0.4189 # max steer [rad]

# ─────────────────────────────────────────────────────────────────────────────
def create_model():
    model = AcadosModel()
    model.name = "f1tenth_racing"

    x     = SX.sym("x")
    y     = SX.sym("y")
    theta = SX.sym("theta")
    v     = SX.sym("v")
    states = vertcat(x, y, theta, v)

    delta = SX.sym("delta")
    a     = SX.sym("a")
    controls = vertcat(delta, a)

    x_ref     = SX.sym("x_ref")
    y_ref     = SX.sym("y_ref")
    theta_ref = SX.sym("theta_ref")
    v_ref     = SX.sym("v_ref")
    params = vertcat(x_ref, y_ref, theta_ref, v_ref)

    # Nonlinear bicycle dynamics
    f = vertcat(
        v * cos(theta),
        v * sin(theta),
        (v / L) * tan(delta),
        a
    )

    model.x = states
    model.u = controls
    model.p = params
    model.f_expl_expr = f

    return model


def create_ocp():
    ocp   = AcadosOcp()
    model = create_model()
    ocp.model = model

    nx = 4
    nu = 2
    np_ = 4

    ocp.dims.N = N

    # ── External cost (most flexible with params) ──────────────────────────
    x_sym = model.x
    u_sym = model.u
    p_sym = model.p

    x_e  = x_sym[0] - p_sym[0]
    y_e  = x_sym[1] - p_sym[1]
    th_e = x_sym[2] - p_sym[2]
    v_e  = x_sym[3] - p_sym[3]

    Q  = np.diag(Q_diag)
    Qf = np.diag(Qf_diag)
    R  = np.diag(R_diag)

    stage_cost = (
        Q[0,0] * x_e**2  +
        Q[1,1] * y_e**2  +
        Q[2,2] * th_e**2 +
        Q[3,3] * v_e**2  +
        R[0,0] * u_sym[0]**2 +
        R[1,1] * u_sym[1]**2
    )

    terminal_cost = (
        Qf[0,0] * x_e**2  +
        Qf[1,1] * y_e**2  +
        Qf[2,2] * th_e**2 +
        Qf[3,3] * v_e**2
    )

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost   = stage_cost
    ocp.model.cost_expr_ext_cost_e = terminal_cost

    # ── Constraints ───────────────────────────────────────────────────────
    ocp.constraints.lbu = np.array([-DELTA_MAX, -A_MAX])
    ocp.constraints.ubu = np.array([ DELTA_MAX,  A_MAX])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-1e6, -1e6, -1e6, V_MIN])
    ocp.constraints.ubx = np.array([ 1e6,  1e6,  1e6, V_MAX])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(np_)

    # ── Solver options — SQP-RTI for real-time ────────────────────────────
    ocp.solver_options.tf              = N * DT
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"   # real-time iteration
    ocp.solver_options.qp_solver       = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx  = "GAUSS_NEWTON"
    ocp.solver_options.num_stages      = 4
    ocp.solver_options.num_steps       = 1
    ocp.solver_options.print_level     = 0

    ocp.code_export_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "c_generated_code"
    )

    return ocp


if __name__ == "__main__":
    print("Generating racing MPC solver (acados SQP-RTI)...")
    print(f"  Horizon: N={N}, dt={DT}s → {N*DT:.1f}s lookahead")
    print(f"  V_max: {V_MAX} m/s")

    ocp = create_ocp()
    solver = AcadosOcpSolver(ocp, json_file="f1tenth_racing.json",
                              generate=True, build=True)

    print("\nDone. Generated:")
    print("  c_generated_code/acados_solver_f1tenth_racing.h")
    print("  c_generated_code/libacados_ocp_solver_f1tenth_racing.so")
    print("\nNext: build the ROS2 node with colcon")