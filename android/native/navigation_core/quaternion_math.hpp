#ifndef NAV_QUATERNION_MATH_HPP
#define NAV_QUATERNION_MATH_HPP

#include "math_types.hpp"
#include <cmath>

namespace nav_core {

struct Quat {
    double w, x, y, z;

    Quat() : w(1.0), x(0.0), y(0.0), z(0.0) {}
    Quat(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}

    double norm() const {
        return std::sqrt(w * w + x * x + y * y + z * z);
    }

    Quat normalized() const {
        double n = norm();
        if (n > 1e-12) {
            return Quat(w / n, x / n, y / n, z / n);
        }
        return Quat(1.0, 0.0, 0.0, 0.0);
    }

    void normalize() {
        *this = normalized();
    }

    Quat conjugate() const {
        return Quat(w, -x, -y, -z);
    }

    static Quat multiply(const Quat& q1, const Quat& q2) {
        return Quat(
            q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        ).normalized();
    }

    Mat3 toRotationMatrix() const {
        Mat3 R;
        double w2 = w * w, x2 = x * x, y2 = y * y, z2 = z * z;
        double xy = x * y, xz = x * z, yz = y * z;
        double wx = w * x, wy = w * y, wz = w * z;

        R.data[0][0] = 1.0 - 2.0 * (y2 + z2);
        R.data[0][1] = 2.0 * (xy - wz);
        R.data[0][2] = 2.0 * (xz + wy);

        R.data[1][0] = 2.0 * (xy + wz);
        R.data[1][1] = 1.0 - 2.0 * (x2 + z2);
        R.data[1][2] = 2.0 * (yz - wx);

        R.data[2][0] = 2.0 * (xz - wy);
        R.data[2][1] = 2.0 * (yz + wx);
        R.data[2][2] = 1.0 - 2.0 * (x2 + y2);

        return R;
    }

    static Quat fromRotationVector(const Vec3& rot_vec) {
        double angle = rot_vec.norm();
        if (angle < 1e-10) {
            return Quat(1.0, 0.5 * rot_vec.x, 0.5 * rot_vec.y, 0.5 * rot_vec.z).normalized();
        }
        double half_angle = 0.5 * angle;
        double sin_half = std::sin(half_angle) / angle;
        return Quat(
            std::cos(half_angle),
            rot_vec.x * sin_half,
            rot_vec.y * sin_half,
            rot_vec.z * sin_half
        ).normalized();
    }

    double getHeadingRad() const {
        Mat3 R = toRotationMatrix();
        // Yaw / Heading in ENU: atan2(R[1][0], R[0][0])
        return std::atan2(R.data[1][0], R.data[0][0]);
    }

    double getHeadingDeg() const {
        double deg = getHeadingRad() * 180.0 / M_PI;
        if (deg < 0.0) deg += 360.0;
        return deg;
    }
};

} // namespace nav_core

#endif // NAV_QUATERNION_MATH_HPP
