/**
 * Racing MPC Node — acados SQP-RTI (SIM VERSION)
 * ================================================
 * Pose+Velocity: /ego_racecar/odom (sim ground truth)
 * Output:        /drive
 *
 * Features:
 * - Nonlinear bicycle model via acados SQP-RTI (~1-3ms solve)
 * - Curvature-based variable speed: fast straights, slow corners
 * - Yaw normalization per horizon step (prevents cost blowup)
 * - Solve time on /mpc_solve_time_ms
 */

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/float64.hpp"
#include "rclcpp/rclcpp.hpp"

#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_f1tenth_racing.h"

static constexpr int    N         = 20;
static constexpr double DT        = 0.05;
static constexpr int    NX        = 4;
static constexpr int    NU        = 2;
static constexpr int    NP        = 4;
static constexpr double V_MIN_CMD = 0.5;

struct Waypoint { double x, y, yaw, speed; };

static double normalize_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

static double quat_to_yaw(double qx, double qy, double qz, double qw) {
    return std::atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz));
}

class RacingMPC : public rclcpp::Node {
public:
    RacingMPC() : Node("racing_mpc") {
        this->declare_parameter<std::string>("waypoints_file", "waypoints.csv");
        this->declare_parameter<double>("v_ref",          3.0);
        this->declare_parameter<double>("v_min_corner",   1.5);
        this->declare_parameter<double>("curvature_gain", 3.0);
        this->declare_parameter<bool>  ("reverse_waypoints", false);

        std::string wp_file   = this->get_parameter("waypoints_file").as_string();
        v_ref_                = this->get_parameter("v_ref").as_double();
        double v_min_corner   = this->get_parameter("v_min_corner").as_double();
        double curvature_gain = this->get_parameter("curvature_gain").as_double();
        bool   reverse        = this->get_parameter("reverse_waypoints").as_bool();

        load_waypoints(wp_file, reverse, v_min_corner, curvature_gain);

        // acados
        capsule_ = f1tenth_racing_acados_create_capsule();
        if (f1tenth_racing_acados_create(capsule_) != 0) {
            RCLCPP_FATAL(get_logger(), "acados solver init failed");
            rclcpp::shutdown(); return;
        }
        nlp_config_ = f1tenth_racing_acados_get_nlp_config(capsule_);
        nlp_dims_   = f1tenth_racing_acados_get_nlp_dims(capsule_);
        nlp_in_     = f1tenth_racing_acados_get_nlp_in(capsule_);
        nlp_out_    = f1tenth_racing_acados_get_nlp_out(capsule_);

        double x_init[NX] = {0}; double u_init[NU] = {0};
        for (int k = 0; k <= N; ++k)
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "x", x_init);
        for (int k = 0; k < N; ++k)
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", u_init);

        RCLCPP_INFO(get_logger(), "acados SQP-RTI ready | %zu waypoints | v_ref=%.1f",
                    waypoints_.size(), v_ref_);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10,
            std::bind(&RacingMPC::odom_callback, this, std::placeholders::_1));

        drive_pub_      = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        viz_pub_        = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mpc_pred_traj", 10);
        solve_time_pub_ = this->create_publisher<std_msgs::msg::Float64>("/mpc_solve_time_ms", 10);
    }

    ~RacingMPC() {
        if (capsule_) {
            f1tenth_racing_acados_free(capsule_);
            f1tenth_racing_acados_free_capsule(capsule_);
        }
    }

private:
    std::vector<Waypoint> waypoints_;
    int    closest_idx_ = 0;
    double v_ref_;

    f1tenth_racing_solver_capsule* capsule_  = nullptr;
    ocp_nlp_config* nlp_config_ = nullptr;
    ocp_nlp_dims*   nlp_dims_   = nullptr;
    ocp_nlp_in*     nlp_in_     = nullptr;
    ocp_nlp_out*    nlp_out_    = nullptr;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr solve_time_pub_;

    void load_waypoints(const std::string& path, bool reverse,
                        double v_min_corner, double curvature_gain) {
        std::ifstream f(path);
        if (!f.is_open()) {
            RCLCPP_FATAL(get_logger(), "Cannot open: %s", path.c_str());
            rclcpp::shutdown(); return;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line); std::string tok;
            Waypoint wp;
            std::getline(ss, tok, ','); wp.x   = std::stod(tok);
            std::getline(ss, tok, ','); wp.y   = std::stod(tok);
            std::getline(ss, tok, ','); wp.yaw = std::stod(tok);
            wp.speed = v_ref_;
            waypoints_.push_back(wp);
        }

        if (reverse) {
            std::reverse(waypoints_.begin(), waypoints_.end());
            for (auto& wp : waypoints_)
                wp.yaw = normalize_angle(wp.yaw + M_PI);
        }

        // Curvature-based speed profile
        int n = static_cast<int>(waypoints_.size());
        std::vector<double> speeds(n, v_ref_);

        for (int i = 1; i < n - 1; ++i) {
            double dyaw = normalize_angle(
                waypoints_[i+1].yaw - waypoints_[i-1].yaw);
            double dx = std::hypot(
                waypoints_[i+1].x - waypoints_[i-1].x,
                waypoints_[i+1].y - waypoints_[i-1].y);
            double curvature = std::abs(dyaw / (dx + 1e-6));
            double v_scale   = std::exp(-curvature_gain * curvature);
            speeds[i] = std::clamp(v_ref_ * v_scale, v_min_corner, v_ref_);
        }
        speeds[0] = speeds[1]; speeds[n-1] = speeds[n-2];

        // Smooth 3 passes
        for (int iter = 0; iter < 3; ++iter)
            for (int i = 1; i < n - 1; ++i)
                speeds[i] = 0.25*speeds[i-1] + 0.50*speeds[i] + 0.25*speeds[i+1];

        for (int i = 0; i < n; ++i) waypoints_[i].speed = speeds[i];

        double v_lo = *std::min_element(speeds.begin(), speeds.end());
        double v_hi = *std::max_element(speeds.begin(), speeds.end());
        RCLCPP_INFO(get_logger(), "Speed profile: min=%.2f max=%.2f m/s", v_lo, v_hi);
    }

    int find_closest(double cx, double cy) {
        int n = static_cast<int>(waypoints_.size());
        int best = closest_idx_; double best_dist = 1e9;
        for (int i = -5; i < 60; ++i) {
            int idx = (closest_idx_ + i + n) % n;
            double d = std::hypot(cx - waypoints_[idx].x, cy - waypoints_[idx].y);
            if (d < best_dist) { best_dist = d; best = idx; }
        }
        return best;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (waypoints_.empty()) return;

        double cx     = msg->pose.pose.position.x;
        double cy     = msg->pose.pose.position.y;
        double ctheta = quat_to_yaw(
            msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
        double cv = std::hypot(msg->twist.twist.linear.x,
                               msg->twist.twist.linear.y);

        closest_idx_ = find_closest(cx, cy);
        int n = static_cast<int>(waypoints_.size());

        double x0[NX] = {cx, cy, ctheta, cv};
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_,
                                      0, "lbx", x0);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_,
                                      0, "ubx", x0);

        for (int k = 0; k <= N; ++k) {
            int idx = (closest_idx_ + k + 1) % n;
            double yaw_ref = waypoints_[idx].yaw;
            double diff = yaw_ref - ctheta;
            while (diff >  M_PI) diff -= 2.0 * M_PI;
            while (diff < -M_PI) diff += 2.0 * M_PI;
            yaw_ref = ctheta + diff;
            double p[NP] = {
                waypoints_[idx].x,
                waypoints_[idx].y,
                yaw_ref,
                waypoints_[idx].speed   // curvature-adapted
            };
            f1tenth_racing_acados_update_params(capsule_, k, p, NP);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        int status = f1tenth_racing_acados_solve(capsule_);
        double solve_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        std_msgs::msg::Float64 st; st.data = solve_ms;
        solve_time_pub_->publish(st);

        if (status != 0 && status != 2)
            RCLCPP_WARN(get_logger(), "acados status=%d solve=%.2fms", status, solve_ms);

        double u_opt[NU];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u_opt);

        double delta_cmd = u_opt[0];
        double accel_cmd = u_opt[1];
        double v_ref_now = waypoints_[closest_idx_].speed;
        double speed_cmd = std::clamp(v_ref_now + accel_cmd * DT * 3.0, V_MIN_CMD, 6.0);

        ackermann_msgs::msg::AckermannDriveStamped drv;
        drv.header.stamp         = this->now();
        drv.header.frame_id      = "ego_racecar/base_link";
        drv.drive.steering_angle = delta_cmd;
        drv.drive.speed          = speed_cmd;
        drive_pub_->publish(drv);

        publish_viz();
    }

    void publish_viz() {
        visualization_msgs::msg::MarkerArray ma;
        for (int k = 0; k <= N; ++k) {
            double x_pred[NX];
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "x", x_pred);
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "map"; m.header.stamp = this->now();
            m.ns = "racing_pred"; m.id = k;
            m.type   = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = x_pred[0]; m.pose.position.y = x_pred[1];
            m.pose.position.z = 0.08;
            m.scale.x = m.scale.y = m.scale.z = 0.12;
            double v_norm = std::clamp(x_pred[3] / 6.0, 0.0, 1.0);
            m.color.r = v_norm; m.color.g = 1.0 - v_norm;
            m.color.b = 0.0;    m.color.a = 0.9;
            ma.markers.push_back(m);
        }
        viz_pub_->publish(ma);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RacingMPC>());
    rclcpp::shutdown();
    return 0;
}