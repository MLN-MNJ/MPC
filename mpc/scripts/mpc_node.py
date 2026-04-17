#!/usr/bin/env python3
"""
MPC Node for F1Tenth — Python / CVXPY + OSQP
=============================================
Subscribes:  /ego_racecar/odom
Publishes:   /drive

Approach: Linearized kinematic bicycle model, QP solved via CVXPY → OSQP
State:    z = [x, y, v, yaw]
Controls: u = [accel, steer]
"""

import math
import numpy as np
import cvxpy

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from dataclasses import dataclass, field
from utils import nearest_point, calc_interpolated_ref_trajectory


# ─────────────────────────────────────────────────────────────────────────────
# Config — tune weights here
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class mpc_config:
    NXK: int = 4
    NU:  int = 2
    TK:  int = 8          # prediction horizon

    # Input cost [accel, steer]
    Rk:  np.ndarray = field(default_factory=lambda: np.diag([0.01, 100.0]))
    # Input rate-of-change cost
    Rdk: np.ndarray = field(default_factory=lambda: np.diag([0.01, 100.0]))
    # State error cost [x, y, v, yaw]
    Qk:  np.ndarray = field(default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]))
    # Terminal state cost
    Qfk: np.ndarray = field(default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]))

    DTK:       float = 0.1        # timestep [s]
    WB:        float = 0.33       # wheelbase [m]
    MIN_STEER: float = -0.4189
    MAX_STEER: float =  0.4189
    MAX_DSTEER:float = math.pi    # max steering rate [rad/s]
    MAX_SPEED: float =  6.0
    MIN_SPEED: float =  0.0
    MAX_ACCEL: float =  3.0
    V_REF:     float =  2.0       # reference speed [m/s]


@dataclass
class State:
    x:   float = 0.0
    y:   float = 0.0
    v:   float = 0.0
    yaw: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize_angle(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def get_linearized_model(v: float, phi: float, delta: float,
                          cfg: mpc_config):
    """
    Linearize kinematic bicycle model around (v, phi, delta).
    Returns discrete-time A, B, C such that:
        z_{t+1} = A*z_t + B*u_t + C
    State order: [x=0, y=1, v=2, yaw=3]
    Input order: [accel=0, steer=1]
    """
    dt = cfg.DTK

    # Continuous Jacobians
    A = np.eye(4)
    A[0, 2] =  dt * math.cos(phi)
    A[0, 3] = -dt * v * math.sin(phi)
    A[1, 2] =  dt * math.sin(phi)
    A[1, 3] =  dt * v * math.cos(phi)
    A[3, 2] =  dt * math.tan(delta) / cfg.WB

    B = np.zeros((4, 2))
    B[2, 0] = dt                                               # dv/da
    B[3, 1] = dt * v / (cfg.WB * math.cos(delta) ** 2)       # dyaw/ddelta

    # Affine term (from linearization around operating point)
    C = np.zeros(4)
    C[0] =  dt * v * math.sin(phi) * phi
    C[1] = -dt * v * math.cos(phi) * phi
    C[3] = -dt * v * delta / (cfg.WB * math.cos(delta) ** 2)

    return A, B, C


def predict_motion(x0: np.ndarray, oa: np.ndarray, od: np.ndarray,
                   cfg: mpc_config) -> np.ndarray:
    """
    Simulate T steps with nonlinear model for linearization points.
    Returns path_predict: (4, T+1)
    """
    T = cfg.TK
    path = np.zeros((4, T + 1))
    path[:, 0] = x0

    state = State(x=x0[0], y=x0[1], v=x0[2], yaw=x0[3])
    for t in range(T):
        delta = np.clip(od[t], cfg.MIN_STEER, cfg.MAX_STEER)
        state.x   += state.v * math.cos(state.yaw) * cfg.DTK
        state.y   += state.v * math.sin(state.yaw) * cfg.DTK
        state.yaw  = normalize_angle(
            state.yaw + (state.v / cfg.WB) * math.tan(delta) * cfg.DTK)
        state.v    = np.clip(state.v + oa[t] * cfg.DTK,
                             cfg.MIN_SPEED, cfg.MAX_SPEED)
        path[0, t+1] = state.x
        path[1, t+1] = state.y
        path[2, t+1] = state.v
        path[3, t+1] = state.yaw

    return path


# ─────────────────────────────────────────────────────────────────────────────
# MPC Node
# ─────────────────────────────────────────────────────────────────────────────
class MPC(Node):
    def __init__(self):
        super().__init__('mpc_node')

        self.config = mpc_config()
        self.oa     = np.zeros(self.config.TK)
        self.odelta = np.zeros(self.config.TK)

        # ── Load waypoints ──────────────────────────────────────────────────
        wp_path = self.declare_parameter('waypoints_file', 'waypoints.csv') \
                      .get_parameter_value().string_value
        self._load_waypoints(wp_path)

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        self.ref_pub = self.create_publisher(
            MarkerArray, '/mpc_ref_traj', 10)

        # ── Init MPC ─────────────────────────────────────────────────────────
        self.mpc_prob_init()
        self.get_logger().info('MPC node ready (CVXPY/OSQP)')

    # ── Waypoint loading ──────────────────────────────────────────────────────
    def _load_waypoints(self, path: str):
        data = np.loadtxt(path, delimiter=',')
        self.wx   = data[:, 0]
        self.wy   = data[:, 1]
        self.wyaw = data[:, 2]
        # Constant reference speed — replace with curvature-based profile if desired
        self.wv   = np.full(len(self.wx), self.config.V_REF)
        self.get_logger().info(f'Loaded {len(self.wx)} waypoints from {path}')

    # ── MPC problem setup ─────────────────────────────────────────────────────
    def mpc_prob_init(self):
        """
        Build CVXPY decision variables and problem structure.
        The problem is rebuilt each iteration in mpc_prob_solve
        (A, B, C matrices change with linearization point).
        Warm starting is handled by OSQP internally.
        """
        # Stored here so mpc_prob_solve can access variable shapes
        self._T   = self.config.TK
        self._NXK = self.config.NXK
        self._NU  = self.config.NU
        self.get_logger().info('MPC problem structure initialized')

    # ── Per-iteration solve ───────────────────────────────────────────────────
    def mpc_prob_solve(self, ref_traj: np.ndarray,
                       path_predict: np.ndarray,
                       x0: np.ndarray):
        """
        Build and solve the MPC QP for current timestep.

        ref_traj:     (4, T+1) reference trajectory
        path_predict: (4, T+1) predicted motion for linearization
        x0:           (4,)     current state

        Returns: (success, oa, odelta, ox, oy, oyaw)
        """
        T   = self.config.TK
        cfg = self.config

        # ── 1. Compute linearized dynamics along predicted path ─────────────
        A_list, B_list, C_list = [], [], []
        for t in range(T):
            A, B, C = get_linearized_model(
                path_predict[2, t],   # v at t
                path_predict[3, t],   # yaw at t
                0.0,                  # delta operating point (use 0)
                cfg
            )
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)

        # ── 2. Decision variables ────────────────────────────────────────────
        zk = cvxpy.Variable((cfg.NXK, T + 1))   # states
        uk = cvxpy.Variable((cfg.NU,  T))        # controls

        # ── 3. Objective ─────────────────────────────────────────────────────
        # J = Σ_t [(z_t - z_ref_t)' Q (z_t - z_ref_t)
        #         + u_t' R u_t
        #         + (u_t - u_{t-1})' Rd (u_t - u_{t-1})]
        #   + (z_T - z_ref_T)' Qf (z_T - z_ref_T)
        cost = 0
        for t in range(T):
            cost += cvxpy.quad_form(zk[:, t] - ref_traj[:, t], cfg.Qk)
            cost += cvxpy.quad_form(uk[:, t], cfg.Rk)
            if t > 0:
                cost += cvxpy.quad_form(uk[:, t] - uk[:, t - 1], cfg.Rdk)
        cost += cvxpy.quad_form(zk[:, T] - ref_traj[:, T], cfg.Qfk)

        # ── 4. Constraints ───────────────────────────────────────────────────
        constraints = []

        # (a) Initial state
        constraints.append(zk[:, 0] == x0)

        # (b) Linearized dynamics: z_{t+1} = A_t*z_t + B_t*u_t + C_t
        for t in range(T):
            constraints.append(
                zk[:, t + 1] == A_list[t] @ zk[:, t]
                               + B_list[t] @ uk[:, t]
                               + C_list[t]
            )

        # (c) Speed bounds
        constraints.append(zk[2, :] >= cfg.MIN_SPEED)
        constraints.append(zk[2, :] <= cfg.MAX_SPEED)

        # (d) Control bounds
        constraints.append(uk[0, :] >= -cfg.MAX_ACCEL)
        constraints.append(uk[0, :] <=  cfg.MAX_ACCEL)
        constraints.append(uk[1, :] >= cfg.MIN_STEER)
        constraints.append(uk[1, :] <= cfg.MAX_STEER)

        # (e) Steering rate limit
        max_ddelta = cfg.MAX_DSTEER * cfg.DTK
        for t in range(T - 1):
            constraints.append(
                cvxpy.abs(uk[1, t + 1] - uk[1, t]) <= max_ddelta)

        # ── 5. Solve ─────────────────────────────────────────────────────────
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(
            solver=cvxpy.OSQP,
            warm_start=True,
            verbose=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=5000,
        )

        if prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
            self.get_logger().warn(f'CVXPY solver status: {prob.status}')
            return False, None, None, None, None, None

        oa     = uk.value[0, :]
        odelta = uk.value[1, :]
        ox     = zk.value[0, :]
        oy     = zk.value[1, :]
        oyaw   = zk.value[3, :]

        return True, oa, odelta, ox, oy, oyaw

    # ── Outer MPC loop (SQP-style iteration) ─────────────────────────────────
    def linear_mpc_control(self, ref_path, x0, oa, odelta):
        """
        Predict motion → linearize → solve QP → return controls.
        Single SQP iteration (sufficient for low-speed tracking).
        """
        path_predict = predict_motion(x0, oa, odelta, self.config)
        success, mpc_a, mpc_delta, ox, oy, oyaw = \
            self.mpc_prob_solve(ref_path, path_predict, x0)

        if not success:
            return oa, odelta, None, None, None, path_predict

        return mpc_a, mpc_delta, ox, oy, oyaw, path_predict

    # ── Odom callback ─────────────────────────────────────────────────────────
    def odom_callback(self, odom_msg: Odometry):
        if self.wx is None:
            return

        # ── Extract state ────────────────────────────────────────────────────
        q = odom_msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        vehicle_state = State(
            x   = odom_msg.pose.pose.position.x,
            y   = odom_msg.pose.pose.position.y,
            v   = math.hypot(odom_msg.twist.twist.linear.x,
                             odom_msg.twist.twist.linear.y),
            yaw = yaw,
        )

        # ── Reference trajectory ─────────────────────────────────────────────
        ref_path = calc_interpolated_ref_trajectory(
            vehicle_state.x, vehicle_state.y,
            self.wx, self.wy, self.wv, self.wyaw,
            self.config.DTK, self.config.TK,
        )  # shape (4, TK+1): rows = [x, y, v, yaw]

        x0 = np.array([vehicle_state.x, vehicle_state.y,
                        vehicle_state.v, vehicle_state.yaw])

        # ── Solve MPC ────────────────────────────────────────────────────────
        self.oa, self.odelta, ox, oy, oyaw, _ = self.linear_mpc_control(
            ref_path, x0, self.oa, self.odelta)

        # ── Publish ──────────────────────────────────────────────────────────
        steer_cmd = float(self.odelta[0])
        speed_cmd = float(np.clip(
            vehicle_state.v + self.oa[0] * self.config.DTK,
            self.config.MIN_SPEED, self.config.MAX_SPEED))

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = steer_cmd
        drive_msg.drive.speed          = speed_cmd
        self.drive_pub.publish(drive_msg)

        # ── Visualize ref trajectory ─────────────────────────────────────────
        if ox is not None:
            self._publish_markers(ox, oy)

    def _publish_markers(self, ox, oy):
        ma = MarkerArray()
        for i, (x, y) in enumerate(zip(ox, oy)):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns = 'mpc_pred'; m.id = i
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.05
            m.scale.x = m.scale.y = m.scale.z = 0.1
            m.color.r = 1.0; m.color.g = 0.5; m.color.b = 0.0; m.color.a = 0.8
            ma.markers.append(m)
        self.ref_pub.publish(ma)


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = MPC()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()