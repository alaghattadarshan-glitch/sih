#include "strapdown_ins.hpp"

namespace nav_core {

StrapdownINS::StrapdownINS(double g_val)
    : g_val_(g_val), prev_a_enu_(0, 0, 0), is_initialized_(false), last_time_(0.0) {}

void StrapdownINS::initialize(
    double initial_time,
    double lat_deg, double lon_deg, double alt_m,
    const Vec3& initial_vel_enu,
    const Quat& initial_quat,
    const Vec3& initial_accel_bias,
    const Vec3& initial_gyro_bias
) {
    state_.timestamp = initial_time;
    state_.origin_lat_deg = lat_deg;
    state_.origin_lon_deg = lon_deg;
    state_.origin_alt_m = alt_m;
    state_.pos_enu = Vec3(0, 0, 0);
    state_.vel_enu = initial_vel_enu;
    state_.q_b2n = initial_quat.normalized();
    state_.accel_bias = initial_accel_bias;
    state_.gyro_bias = initial_gyro_bias;
    state_.status = NAV_STATUS_GOOD;
    state_.confidence = 1.0;

    prev_a_enu_ = Vec3(0, 0, 0);
    last_time_ = initial_time;
    is_initialized_ = true;
}

NavState StrapdownINS::update(double timestamp, const Vec3& raw_accel, const Vec3& raw_gyro) {
    if (!is_initialized_) {
        return state_;
    }

    double dt = timestamp - last_time_;
    if (dt <= 0.0) {
        return state_;
    }

    // 1. Correct sensor biases
    Vec3 omega_corr = raw_gyro - state_.gyro_bias;
    Vec3 f_corr = raw_accel - state_.accel_bias;

    // 2. Quaternion attitude integration
    Vec3 rot_vec = omega_corr * dt;
    Quat dq = Quat::fromRotationVector(rot_vec);
    Quat q_new = Quat::multiply(state_.q_b2n, dq);

    // 3. Specific force rotation to ENU frame
    Mat3 R_b2n = q_new.toRotationMatrix();
    Vec3 f_enu = R_b2n * f_corr;

    // 4. Subtract gravity in ENU frame [0, 0, g]
    Vec3 a_enu = f_enu - Vec3(0.0, 0.0, g_val_);

    // 5. Trapezoidal velocity & position integration
    Vec3 v_old = state_.vel_enu;
    Vec3 a_avg = (prev_a_enu_ + a_enu) * 0.5;
    Vec3 v_new = v_old + a_avg * dt;

    Vec3 v_avg = (v_old + v_new) * 0.5;
    Vec3 p_new = state_.pos_enu + v_avg * dt;

    // Update state
    prev_a_enu_ = a_enu;
    state_.timestamp = timestamp;
    state_.pos_enu = p_new;
    state_.vel_enu = v_new;
    state_.q_b2n = q_new;
    last_time_ = timestamp;

    return state_;
}

void StrapdownINS::reset() {
    is_initialized_ = false;
    last_time_ = 0.0;
    prev_a_enu_ = Vec3(0, 0, 0);
    state_ = NavState();
}

} // namespace nav_core
