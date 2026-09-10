#ifndef NAV_STRAPDOWN_INS_HPP
#define NAV_STRAPDOWN_INS_HPP

#include "nav_state.hpp"

namespace nav_core {

class StrapdownINS {
public:
    StrapdownINS(double g_val = 9.80665);

    void initialize(
        double initial_time,
        double lat_deg, double lon_deg, double alt_m,
        const Vec3& initial_vel_enu,
        const Quat& initial_quat,
        const Vec3& initial_accel_bias = Vec3(0, 0, 0),
        const Vec3& initial_gyro_bias = Vec3(0, 0, 0)
    );

    NavState update(double timestamp, const Vec3& raw_accel, const Vec3& raw_gyro);

    const NavState& getState() const { return state_; }
    void setState(const NavState& s) { state_ = s; }
    bool isInitialized() const { return is_initialized_; }
    void reset();

    double getGravity() const { return g_val_; }
    const Vec3& getPrevAEnu() const { return prev_a_enu_; }
    void setPrevAEnu(const Vec3& a) { prev_a_enu_ = a; }

private:
    double g_val_;
    NavState state_;
    Vec3 prev_a_enu_;
    bool is_initialized_;
    double last_time_;
};

} // namespace nav_core

#endif // NAV_STRAPDOWN_INS_HPP
