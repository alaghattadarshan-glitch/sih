#ifndef NAV_MATH_TYPES_HPP
#define NAV_MATH_TYPES_HPP

#include <cmath>
#include <cstring>
#include <algorithm>

namespace nav_core {

// 2-Element Vector
struct Vec2 {
    double x, y;

    Vec2() : x(0.0), y(0.0) {}
    Vec2(double x_, double y_) : x(x_), y(y_) {}

    double& operator[](int idx) { return (idx == 0) ? x : y; }
    const double& operator[](int idx) const { return (idx == 0) ? x : y; }

    Vec2 operator+(const Vec2& o) const { return Vec2(x + o.x, y + o.y); }
    Vec2 operator-(const Vec2& o) const { return Vec2(x - o.x, y - o.y); }
    Vec2 operator-() const { return Vec2(-x, -y); }
    Vec2 operator*(double s) const { return Vec2(x * s, y * s); }
    Vec2 operator/(double s) const { return Vec2(x / s, y / s); }

    double dot(const Vec2& o) const { return x * o.x + y * o.y; }
    double norm() const { return std::sqrt(x * x + y * y); }
};

// 2x2 Matrix
struct Mat2 {
    double data[2][2];

    Mat2() { setZero(); }

    static Mat2 Zero() {
        Mat2 m;
        m.setZero();
        return m;
    }

    static Mat2 Identity() {
        Mat2 m;
        m.setZero();
        m.data[0][0] = 1.0;
        m.data[1][1] = 1.0;
        return m;
    }

    void setZero() {
        std::memset(data, 0, sizeof(data));
    }

    double* operator[](int r) { return data[r]; }
    const double* operator[](int r) const { return data[r]; }

    Mat2 operator+(const Mat2& o) const {
        Mat2 r;
        r.data[0][0] = data[0][0] + o.data[0][0];
        r.data[0][1] = data[0][1] + o.data[0][1];
        r.data[1][0] = data[1][0] + o.data[1][0];
        r.data[1][1] = data[1][1] + o.data[1][1];
        return r;
    }

    Vec2 operator*(const Vec2& v) const {
        return Vec2(
            data[0][0] * v.x + data[0][1] * v.y,
            data[1][0] * v.x + data[1][1] * v.y
        );
    }

    double determinant() const {
        return data[0][0] * data[1][1] - data[0][1] * data[1][0];
    }

    Mat2 inverse() const {
        double det = determinant();
        if (std::abs(det) < 1e-15) {
            return Mat2::Identity();
        }
        double invdet = 1.0 / det;
        Mat2 minv;
        minv.data[0][0] = data[1][1] * invdet;
        minv.data[0][1] = -data[0][1] * invdet;
        minv.data[1][0] = -data[1][0] * invdet;
        minv.data[1][1] = data[0][0] * invdet;
        return minv;
    }
};

// 3-Element Vector
struct Vec3 {

    double x, y, z;

    Vec3() : x(0.0), y(0.0), z(0.0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    double& operator[](int idx) {
        if (idx == 0) return x;
        if (idx == 1) return y;
        return z;
    }

    const double& operator[](int idx) const {
        if (idx == 0) return x;
        if (idx == 1) return y;
        return z;
    }

    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }


    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }

    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return Vec3(
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x
        );
    }

    double norm() const { return std::sqrt(x * x + y * y + z * z); }
    double squaredNorm() const { return x * x + y * y + z * z; }

    Vec3 normalized() const {
        double n = norm();
        if (n > 1e-12) {
            return Vec3(x / n, y / n, z / n);
        }
        return Vec3(0, 0, 0);
    }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

// 3x3 Matrix
struct Mat3 {
    double data[3][3];

    Mat3() { setZero(); }

    static Mat3 Zero() {
        Mat3 m;
        m.setZero();
        return m;
    }

    static Mat3 Identity() {
        Mat3 m;
        m.setZero();
        m.data[0][0] = 1.0;
        m.data[1][1] = 1.0;
        m.data[2][2] = 1.0;
        return m;
    }

    void setZero() {
        std::memset(data, 0, sizeof(data));
    }

    double* operator[](int r) { return data[r]; }
    const double* operator[](int r) const { return data[r]; }

    Mat3 operator+(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.data[i][j] = data[i][j] + o.data[i][j];
        return r;
    }

    Mat3 operator-(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.data[i][j] = data[i][j] - o.data[i][j];
        return r;
    }

    Mat3 operator*(double s) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.data[i][j] = data[i][j] * s;
        return r;
    }

    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) {
                    sum += data[i][k] * o.data[k][j];
                }
                r.data[i][j] = sum;
            }
        }
        return r;
    }

    Vec3 operator*(const Vec3& v) const {
        return Vec3(
            data[0][0] * v.x + data[0][1] * v.y + data[0][2] * v.z,
            data[1][0] * v.x + data[1][1] * v.y + data[1][2] * v.z,
            data[2][0] * v.x + data[2][1] * v.y + data[2][2] * v.z
        );
    }

    Mat3 transpose() const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.data[i][j] = data[j][i];
        return r;
    }

    double determinant() const {
        return data[0][0] * (data[1][1] * data[2][2] - data[1][2] * data[2][1])
             - data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0])
             + data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);
    }

    Mat3 inverse() const {
        double det = determinant();
        if (std::abs(det) < 1e-15) {
            return Mat3::Identity();
        }
        double invdet = 1.0 / det;
        Mat3 minv;
        minv.data[0][0] = (data[1][1] * data[2][2] - data[1][2] * data[2][1]) * invdet;
        minv.data[0][1] = (data[0][2] * data[2][1] - data[0][1] * data[2][2]) * invdet;
        minv.data[0][2] = (data[0][1] * data[1][2] - data[0][2] * data[1][1]) * invdet;
        minv.data[1][0] = (data[1][2] * data[2][0] - data[1][0] * data[2][2]) * invdet;
        minv.data[1][1] = (data[0][0] * data[2][2] - data[0][2] * data[2][0]) * invdet;
        minv.data[1][2] = (data[0][2] * data[1][0] - data[0][0] * data[1][2]) * invdet;
        minv.data[2][0] = (data[1][0] * data[2][1] - data[1][1] * data[2][0]) * invdet;
        minv.data[2][1] = (data[0][1] * data[2][0] - data[0][0] * data[2][1]) * invdet;
        minv.data[2][2] = (data[0][0] * data[1][1] - data[0][1] * data[1][0]) * invdet;
        return minv;
    }
};

inline Mat3 skew_symmetric(const Vec3& v) {
    Mat3 m;
    m.data[0][0] = 0.0;   m.data[0][1] = -v.z;  m.data[0][2] = v.y;
    m.data[1][0] = v.z;   m.data[1][1] = 0.0;   m.data[1][2] = -v.x;
    m.data[2][0] = -v.y;  m.data[2][1] = v.x;   m.data[2][2] = 0.0;
    return m;
}

// 15-Element Error State Vector
struct Vec15 {
    double data[15];

    Vec15() { setZero(); }

    void setZero() {
        std::memset(data, 0, sizeof(data));
    }

    double& operator[](int idx) { return data[idx]; }
    const double& operator[](int idx) const { return data[idx]; }

    Vec3 getPos() const { return Vec3(data[0], data[1], data[2]); }
    Vec3 getVel() const { return Vec3(data[3], data[4], data[5]); }
    Vec3 getAtt() const { return Vec3(data[6], data[7], data[8]); }
    Vec3 getAccelBias() const { return Vec3(data[9], data[10], data[11]); }
    Vec3 getGyroBias() const { return Vec3(data[12], data[13], data[14]); }
};

// 15x15 Matrix for ESKF Covariance & Transitions
struct Mat15 {
    double data[15][15];

    Mat15() { setZero(); }

    static Mat15 Zero() {
        Mat15 m;
        m.setZero();
        return m;
    }

    static Mat15 Identity() {
        Mat15 m;
        m.setZero();
        for (int i = 0; i < 15; ++i) m.data[i][i] = 1.0;
        return m;
    }

    void setZero() {
        std::memset(data, 0, sizeof(data));
    }

    double* operator[](int r) { return data[r]; }
    const double* operator[](int r) const { return data[r]; }

    Mat15 operator+(const Mat15& o) const {
        Mat15 r;
        for (int i = 0; i < 15; ++i)
            for (int j = 0; j < 15; ++j)
                r.data[i][j] = data[i][j] + o.data[i][j];
        return r;
    }

    Mat15 operator-(const Mat15& o) const {
        Mat15 r;
        for (int i = 0; i < 15; ++i)
            for (int j = 0; j < 15; ++j)
                r.data[i][j] = data[i][j] - o.data[i][j];
        return r;
    }

    Mat15 operator*(double s) const {
        Mat15 r;
        for (int i = 0; i < 15; ++i)
            for (int j = 0; j < 15; ++j)
                r.data[i][j] = data[i][j] * s;
        return r;
    }

    Mat15 operator*(const Mat15& o) const {
        Mat15 r;
        for (int i = 0; i < 15; ++i) {
            for (int j = 0; j < 15; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 15; ++k) {
                    sum += data[i][k] * o.data[k][j];
                }
                r.data[i][j] = sum;
            }
        }
        return r;
    }

    Vec15 operator*(const Vec15& v) const {
        Vec15 r;
        for (int i = 0; i < 15; ++i) {
            double sum = 0.0;
            for (int j = 0; j < 15; ++j) {
                sum += data[i][j] * v.data[j];
            }
            r.data[i] = sum;
        }
        return r;
    }

    Mat15 transpose() const {
        Mat15 r;
        for (int i = 0; i < 15; ++i)
            for (int j = 0; j < 15; ++j)
                r.data[i][j] = data[j][i];
        return r;
    }

    void symmetrize() {
        for (int i = 0; i < 15; ++i) {
            for (int j = i + 1; j < 15; ++j) {
                double avg = 0.5 * (data[i][j] + data[j][i]);
                data[i][j] = avg;
                data[j][i] = avg;
            }
        }
    }
};

} // namespace nav_core

#endif // NAV_MATH_TYPES_HPP
