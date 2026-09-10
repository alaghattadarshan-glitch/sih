#ifndef NAV_STATE_HPP
#define NAV_STATE_HPP

#include "math_types.hpp"
#include "quaternion_math.hpp"

namespace nav_core {

enum NavigationStatus {
    NAV_STATUS_GOOD = 0,
    NAV_STATUS_DEGRADED = 1,
    NAV_STATUS_OUTAGE = 2,
    NAV_STATUS_RECOVERING = 3
};

struct NavState {
    double timestamp;
    Vec3 pos_enu;          // [East, North, Up] in meters
    Vec3 vel_enu;          // [V_E, V_N, V_U] in m/s
    Quat q_b2n;            // Attitude quaternion body -> ENU
    Vec3 accel_bias;       // Accelerometer bias [bx, by, bz] in m/s^2
    Vec3 gyro_bias;        // Gyroscope bias [bx, by, bz] in rad/s

    double origin_lat_deg;
    double origin_lon_deg;
    double origin_alt_m;

    int status;            // NavigationStatus enum
    double confidence;     // Estimated position accuracy / confidence radius in meters

    NavState()
        : timestamp(0.0),
          pos_enu(0, 0, 0),
          vel_enu(0, 0, 0),
          q_b2n(1, 0, 0, 0),
          accel_bias(0, 0, 0),
          gyro_bias(0, 0, 0),
          origin_lat_deg(0.0),
          origin_lon_deg(0.0),
          origin_alt_m(0.0),
          status(NAV_STATUS_GOOD),
          confidence(1.0) {}

    Mat3 getRotationMatrix() const {
        return q_b2n.toRotationMatrix();
    }

    double getHeadingDeg() const {
        return q_b2n.getHeadingDeg();
    }

    double getSpeed() const {
        return std::sqrt(vel_enu.x * vel_enu.x + vel_enu.y * vel_enu.y);
    }
};

} // namespace nav_core

#endif // NAV_STATE_HPP
