/**
 * MPC Node for F1Tenth — C++ / OSQP-Eigen
 * =========================================
 * Subscribes:  /ego_racecar/odom
 * Publishes:   /drive
 *
 * QP form: min (1/2) z'Pz + q'z   s.t. l <= Az <= u
 * Decision variables: z = [x_0,...,x_T, u_0,...,u_{T-1}]
 *   x_t = [x, y, v, yaw]    (NX=4)
 *   u_t = [accel, steer]    (NU=2)
 *
 * P, A sparsity pattern built ONCE in mpc_prob_init().
 * Each iteration updates:
 *   - A dynamics rows  (new A_block, B_block from linearization)
 *   - l, u bounds      (C term + x0)
 *   - q gradient       (reference trajectory)
 */

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "rclcpp/rclcpp.hpp"

#include "mpc/mpc_utils.hpp"

using namespace mpc;

class MPC : public rclcpp::Node {
public:
    MPC() : Node("mpc_node") {
        auto wp = this->declare_parameter<std::string>("waypoints_file", "waypoints.csv");
        load_waypoints(wp);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10,
            std::bind(&MPC::odom_callback, this, std::placeholders::_1));
        drive_pub_ = this->create_publisher<
            ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        viz_pub_ = this->create_publisher<
            visualization_msgs::msg::MarkerArray>("/mpc_pred_traj", 10);

        config_ = Config();
        oa_.assign(config_.TK, 0.0);
        odelta_.assign(config_.TK, 0.0);

        A_block_.resize(config_.TK);
        B_block_.resize(config_.TK);
        C_block_.resize(config_.TK);
        for (int t = 0; t < config_.TK; ++t) {
            A_block_[t] = Eigen::Matrix4d::Identity();
            B_block_[t] = Eigen::Matrix<double, NX, NU>::Zero();
            C_block_[t] = Eigen::Vector4d::Zero();
        }

        mpc_prob_init();
        RCLCPP_INFO(get_logger(), "MPC node ready (C++/OSQP-Eigen)");
    }

private:
    Config config_;
    Eigen::MatrixXd waypoints_;          // (4 x N): [x; y; v; yaw]
    std::vector<double> oa_, odelta_;    // warm-start buffers

    std::vector<Eigen::Matrix4d>               A_block_;
    std::vector<Eigen::Matrix<double, NX, NU>> B_block_;
    std::vector<Eigen::Vector4d>               C_block_;

    Eigen::SparseMatrix<double> P_, A_;
    Eigen::VectorXd l_, u_, q_;
    OsqpEigen::Solver solver_;

    // Row offsets in constraint matrix A (set in init)
    int row_x0_, row_dyn_, row_ubnd_, row_vbnd_, row_dsteer_;
    int n_constraints_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub_;

    // ─────────────────────────────────────────────────────────────────────────
    void load_waypoints(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            RCLCPP_FATAL(get_logger(), "Cannot open waypoints: %s", path.c_str());
            rclcpp::shutdown(); return;
        }
        std::vector<double> xs, ys, yaws;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line); std::string tok;
            std::getline(ss, tok, ','); xs.push_back(std::stod(tok));
            std::getline(ss, tok, ','); ys.push_back(std::stod(tok));
            std::getline(ss, tok, ','); yaws.push_back(std::stod(tok));
        }
        int n = static_cast<int>(xs.size());
        waypoints_.resize(4, n);
        for (int i = 0; i < n; ++i) {
            waypoints_(0, i) = xs[i];
            waypoints_(1, i) = ys[i];
            waypoints_(2, i) = 2.0;       // constant reference speed
            waypoints_(3, i) = yaws[i];
        }
        RCLCPP_INFO(get_logger(), "Loaded %d waypoints", n);
    }

    // ─────────────────────────────────────────────────────────────────────────
    Eigen::MatrixXd calc_ref_trajectory(const State& s) {
        return calcInterpolatedRefTrajectory(
            s.x, s.y,
            waypoints_.row(0).transpose(),
            waypoints_.row(1).transpose(),
            waypoints_.row(2).transpose(),
            waypoints_.row(3).transpose(),
            config_.DTK, config_.TK);
    }

    Eigen::MatrixXd predict_motion(const Eigen::Vector4d& x0,
                                   const std::vector<double>& oa,
                                   const std::vector<double>& od) {
        int T = config_.TK;
        Eigen::MatrixXd p(NX, T + 1);
        p.col(0) = x0;
        State s; s.x=x0(X); s.y=x0(Y); s.v=x0(V); s.yaw=x0(YAW);
        for (int t = 0; t < T; ++t) {
            s = updateState(s, oa[t], od[t], config_);
            p(X,t+1)=s.x; p(Y,t+1)=s.y; p(V,t+1)=s.v; p(YAW,t+1)=s.yaw;
        }
        return p;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Build sparse A from current A_block_, B_block_
    // ─────────────────────────────────────────────────────────────────────────
    void build_A_matrix() {
        const int T = config_.TK;
        const int n = numVars(T);
        std::vector<Eigen::Triplet<double>> tri;
        tri.reserve(n_constraints_ * 5);

        // ── Part 0: Initial state  x_0 = x0  (equality)
        for (int i = 0; i < NX; ++i)
            tri.push_back({row_x0_ + i, stateIdx(0, i, T), 1.0});

        // ── Part 1: Dynamics  x_{t+1} - A_t x_t - B_t u_t = C_t  (equality)
        for (int t = 0; t < T; ++t) {
            int rb = row_dyn_ + t * NX;
            for (int i = 0; i < NX; ++i) {
                tri.push_back({rb + i, stateIdx(t+1, i, T), 1.0});         // +I
                for (int j = 0; j < NX; ++j) {
                    double v = -A_block_[t](i, j);
                    if (std::abs(v) > 1e-12)
                        tri.push_back({rb + i, stateIdx(t, j, T), v});     // -A_t
                }
                for (int j = 0; j < NU; ++j) {
                    double v = -B_block_[t](i, j);
                    if (std::abs(v) > 1e-12)
                        tri.push_back({rb + i, inputIdx(t, j, T), v});     // -B_t
                }
            }
        }

        // ── Part 2: Input bounds (one row per input component per timestep)
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < NU; ++j)
                tri.push_back({row_ubnd_ + t*NU + j, inputIdx(t, j, T), 1.0});

        // ── Part 3: Speed bound on v component of each state
        for (int t = 0; t <= T; ++t)
            tri.push_back({row_vbnd_ + t, stateIdx(t, V, T), 1.0});

        // ── Part 4: Steering rate  |steer_{t+1} - steer_t| <= max_ddelta
        for (int t = 0; t < T - 1; ++t) {
            tri.push_back({row_dsteer_ + t, inputIdx(t+1, STEER, T),  1.0});
            tri.push_back({row_dsteer_ + t, inputIdx(t,   STEER, T), -1.0});
        }

        A_.resize(n_constraints_, n);
        A_.setFromTriplets(tri.begin(), tri.end());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Build QP once: P (Hessian), A structure, static bounds
    // ─────────────────────────────────────────────────────────────────────────
    void mpc_prob_init() {
        const int T = config_.TK;
        const int n = numVars(T);  // (T+1)*NX + T*NU

        // ── Row layout ───────────────────────────────────────────────────────
        row_x0_     = 0;
        row_dyn_    = row_x0_     + NX;
        row_ubnd_   = row_dyn_    + T * NX;
        row_vbnd_   = row_ubnd_   + T * NU;
        row_dsteer_ = row_vbnd_   + (T + 1);
        n_constraints_ = row_dsteer_ + (T - 1);

        // ════════════════════════════════════════════════════════════════════
        // HESSIAN P  (upper triangular)
        // OSQP minimizes (1/2) z'Pz  → multiply quadratic coefficients by 2
        // ════════════════════════════════════════════════════════════════════
        std::vector<Eigen::Triplet<double>> Pt;
        Pt.reserve(n * 2);

        // ── Objective part 1: input cost  Σ u_t' Rk u_t ─────────────────
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < NU; ++j) {
                int idx = inputIdx(t, j, T);
                Pt.push_back({idx, idx, 2.0 * config_.Rk(j, j)});
            }

        // ── Objective part 2: state tracking  Σ (z_t - z_ref_t)' Q (z_t - z_ref_t)
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < NX; ++i) {
                int idx = stateIdx(t, i, T);
                Pt.push_back({idx, idx, 2.0 * config_.Qk(i, i)});
            }
        // Terminal term with Qfk
        for (int i = 0; i < NX; ++i) {
            int idx = stateIdx(T, i, T);
            Pt.push_back({idx, idx, 2.0 * config_.Qfk(i, i)});
        }

        // ── Objective part 3: input rate cost  Σ (u_{t+1}-u_t)' Rdk (u_{t+1}-u_t)
        // Diagonal contributions:
        //   t=0 and t=T-1 appear in ONE difference term → factor 2
        //   t=1..T-2 appear in TWO difference terms     → factor 4
        // Off-diagonal (upper triangular): -2*Rdk between u_t and u_{t+1}
        for (int t = 0; t < T; ++t) {
            double df = (t == 0 || t == T-1) ? 2.0 : 4.0;
            for (int j = 0; j < NU; ++j) {
                int idx = inputIdx(t, j, T);
                Pt.push_back({idx, idx, df * config_.Rdk(j, j)});
            }
            if (t < T - 1) {
                for (int j = 0; j < NU; ++j) {
                    int r = inputIdx(t,   j, T);
                    int c = inputIdx(t+1, j, T);
                    Pt.push_back({r, c, -2.0 * config_.Rdk(j, j)});  // upper triangle
                }
            }
        }

        P_.resize(n, n);
        P_.setFromTriplets(Pt.begin(), Pt.end());

        // ════════════════════════════════════════════════════════════════════
        // CONSTRAINT MATRIX A  (initial build; dynamics updated each solve)
        // ════════════════════════════════════════════════════════════════════
        build_A_matrix();

        // ════════════════════════════════════════════════════════════════════
        // BOUNDS l, u
        // Dynamic rows (x0, dynamics C) are set in mpc_prob_solve.
        // Static rows (input/speed/rate bounds) set here.
        // ════════════════════════════════════════════════════════════════════
        l_ = Eigen::VectorXd::Constant(n_constraints_, -1e9);
        u_ = Eigen::VectorXd::Constant(n_constraints_,  1e9);

        // Input bounds
        for (int t = 0; t < T; ++t) {
            int r = row_ubnd_ + t * NU;
            l_(r+ACCEL)=-config_.MAX_ACCEL; u_(r+ACCEL)= config_.MAX_ACCEL;
            l_(r+STEER)= config_.MIN_STEER; u_(r+STEER)= config_.MAX_STEER;
        }
        // Speed bounds
        for (int t = 0; t <= T; ++t) {
            l_(row_vbnd_+t) = config_.MIN_SPEED;
            u_(row_vbnd_+t) = config_.MAX_SPEED;
        }
        // Steering rate bounds
        double max_dd = config_.MAX_DSTEER * config_.DTK;
        for (int t = 0; t < T-1; ++t) {
            l_(row_dsteer_+t) = -max_dd;
            u_(row_dsteer_+t) =  max_dd;
        }

        // ════════════════════════════════════════════════════════════════════
        // INITIALIZE OSQP
        // ════════════════════════════════════════════════════════════════════
        q_ = Eigen::VectorXd::Zero(n);

        solver_.settings()->setVerbosity(false);
        solver_.settings()->setWarmStart(true);
        solver_.settings()->setPolish(true);
        solver_.settings()->setAbsoluteTolerance(1e-4);
        solver_.settings()->setRelativeTolerance(1e-4);
        solver_.settings()->setMaxIteration(5000);

        solver_.data()->setNumberOfVariables(n);
        solver_.data()->setNumberOfConstraints(n_constraints_);
        solver_.data()->setHessianMatrix(P_);
        solver_.data()->setGradient(q_);
        solver_.data()->setLinearConstraintsMatrix(A_);
        solver_.data()->setLowerBound(l_);
        solver_.data()->setUpperBound(u_);

        solver_.initSolver();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Per-iteration solve: update matrices and call OSQP
    // ─────────────────────────────────────────────────────────────────────────
    std::tuple<bool, Eigen::VectorXd, Eigen::VectorXd,
               Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
    mpc_prob_solve(const Eigen::MatrixXd& ref_traj,
                   const Eigen::MatrixXd& path_predict,
                   const Eigen::Vector4d& x0) {
        const int T = config_.TK;

        // ── 1. Recompute linearized dynamics along predicted path ─────────
        for (int t = 0; t < T; ++t) {
            getLinearizedModel(
                path_predict(V,   t),
                path_predict(YAW, t),
                0.0,
                config_,
                A_block_[t], B_block_[t], C_block_[t]);
        }

        // ── 2. Rebuild A with new dynamics and push to solver ─────────────
        build_A_matrix();
        solver_.updateLinearConstraintsMatrix(A_);

        // ── 3. Update bounds (initial state + dynamics C term) ─────────────
        // Initial state: hard equality  x_0 = x0
        for (int i = 0; i < NX; ++i) {
            l_(row_x0_+i) = x0(i);
            u_(row_x0_+i) = x0(i);
        }
        // Dynamics equality RHS = C_t
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < NX; ++i) {
                int row = row_dyn_ + t*NX + i;
                l_(row) = C_block_[t](i);
                u_(row) = C_block_[t](i);
            }
        solver_.updateBounds(l_, u_);

        // ── 4. Update gradient q = -2 Q x_ref (with yaw normalization) ──────
        q_.setZero();
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < NX; ++i) {
                double ref_val = ref_traj(i, t);
                if (i == YAW) {
                    double diff = ref_val - x0(YAW);
                    while (diff >  M_PI) diff -= 2.0 * M_PI;
                    while (diff < -M_PI) diff += 2.0 * M_PI;
                    ref_val = x0(YAW) + diff;
                }
                q_(stateIdx(t, i, T)) = -2.0 * config_.Qk(i,i) * ref_val;
            }
        }
        for (int i = 0; i < NX; ++i) {
            double ref_val = ref_traj(i, T);
            if (i == YAW) {
                double diff = ref_val - x0(YAW);
                while (diff >  M_PI) diff -= 2.0 * M_PI;
                while (diff < -M_PI) diff += 2.0 * M_PI;
                ref_val = x0(YAW) + diff;
            }
            q_(stateIdx(T, i, T)) = -2.0 * config_.Qfk(i,i) * ref_val;
        }
        solver_.updateGradient(q_);
        
        // ── 5. Solve ─────────────────────────────────────────────────────
        if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
            RCLCPP_WARN(get_logger(), "OSQP solver failed");
            return {false,{},{},{},{},{}};
        }

        Eigen::VectorXd sol = solver_.getSolution();

        Eigen::VectorXd oa(T), odelta(T), ox(T+1), oy(T+1), oyaw(T+1);
        for (int t = 0; t <= T; ++t) {
            ox(t)   = sol(stateIdx(t, X,   T));
            oy(t)   = sol(stateIdx(t, Y,   T));
            oyaw(t) = sol(stateIdx(t, YAW, T));
        }
        for (int t = 0; t < T; ++t) {
            oa(t)     = sol(inputIdx(t, ACCEL, T));
            odelta(t) = sol(inputIdx(t, STEER, T));
        }
        return {true, oa, odelta, ox, oy, oyaw};
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::tuple<std::vector<double>, std::vector<double>,
               Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd,
               Eigen::MatrixXd>
    linear_mpc_control(const Eigen::MatrixXd& ref_path,
                       const Eigen::Vector4d& x0,
                       std::vector<double> oa, std::vector<double> od) {
        if (oa.empty()) oa.assign(config_.TK, 0.0);
        if (od.empty()) od.assign(config_.TK, 0.0);

        Eigen::MatrixXd pp = predict_motion(x0, oa, od);
        auto [ok, ma, md, ox, oy, oyaw] = mpc_prob_solve(ref_path, pp, x0);

        if (!ok) return {oa, od, {},{},{}, pp};

        std::vector<double> oa_out(ma.data(),     ma.data()     + ma.size());
        std::vector<double> od_out(md.data(),     md.data()     + md.size());
        return {oa_out, od_out, ox, oy, oyaw, pp};
    }

    // ─────────────────────────────────────────────────────────────────────────
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (waypoints_.cols() == 0) return;

        double yaw = yawFromQuaternion(msg->pose.pose.orientation);
        State s;
        s.x   = msg->pose.pose.position.x;
        s.y   = msg->pose.pose.position.y;
        s.v   = std::hypot(msg->twist.twist.linear.x, msg->twist.twist.linear.y);
        s.yaw = yaw;

        Eigen::MatrixXd ref = calc_ref_trajectory(s);
        Eigen::Vector4d x0; x0 << s.x, s.y, s.v, s.yaw;

        auto [new_oa, new_od, ox, oy, oyaw, _] =
            linear_mpc_control(ref, x0, oa_, odelta_);
        oa_ = new_oa; odelta_ = new_od;

        double steer = odelta_.empty() ? 0.0 : odelta_[0];
        double speed = oa_.empty() ? s.v
            : std::clamp(s.v + oa_[0] * config_.DTK,
                         config_.MIN_SPEED, config_.MAX_SPEED);

        ackermann_msgs::msg::AckermannDriveStamped drv;
        drv.header.stamp = this->now();
        drv.drive.steering_angle = steer;
        drv.drive.speed = speed;
        drive_pub_->publish(drv);

        if (ox.size() > 0) publish_viz(ox, oy);
    }

    void publish_viz(const Eigen::VectorXd& ox, const Eigen::VectorXd& oy) {
        visualization_msgs::msg::MarkerArray ma;
        for (int i = 0; i < ox.size(); ++i) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "map"; m.header.stamp = this->now();
            m.ns = "pred"; m.id = i;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = ox(i); m.pose.position.y = oy(i);
            m.pose.position.z = 0.05;
            m.scale.x = m.scale.y = m.scale.z = 0.1;
            m.color.r=0.0; m.color.g=1.0; m.color.b=0.5; m.color.a=0.9;
            ma.markers.push_back(m);
        }
        viz_pub_->publish(ma);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPC>());
    rclcpp::shutdown();
    return 0;
}