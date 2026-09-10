#ifndef NAV_ZUPT_DETECTOR_HPP
#define NAV_ZUPT_DETECTOR_HPP

#include "math_types.hpp"
#include <cmath>
#include <algorithm>

namespace nav_core {

enum ZUPTState {
    ZUPT_MOVING = 0,
    ZUPT_STATIONARY_CANDIDATE = 1,
    ZUPT_STATIONARY = 2,
    ZUPT_LEAVING_STATIONARY = 3
};

struct ZUPTDetectorConfig {
    double accel_norm_tol;
    double gyro_norm_threshold;
    double accel_var_threshold;
    double gyro_var_threshold;
    int persistence_samples;
    int window_size_samples;
    double g_val;
    int leaving_hysteresis_samples;

    ZUPTDetectorConfig()
        : accel_norm_tol(0.6),
          gyro_norm_threshold(0.08),
          accel_var_threshold(0.05),
          gyro_var_threshold(0.005),
          persistence_samples(30),
          window_size_samples(15),
          g_val(9.80665),
          leaving_hysteresis_samples(3) {}
};

class ZUPTDetector {
public:
    static constexpr int MAX_WINDOW_CAPACITY = 64;

    ZUPTDetector(const ZUPTDetectorConfig& cfg = ZUPTDetectorConfig());

    ZUPTState update(double timestamp, double ax, double ay, double az, double gx, double gy, double gz);

    bool isStationary() const { return state_ == ZUPT_STATIONARY; }
    ZUPTState getState() const { return state_; }
    int getPersistenceCounter() const { return persistence_counter_; }
    void reset();

    // Diagnostics
    double getLastAccelNorm() const { return last_accel_norm_; }
    double getLastGyroNorm() const { return last_gyro_norm_; }
    double getLastAccelVar() const { return last_accel_var_; }
    double getLastGyroVar() const { return last_gyro_var_; }

private:
    ZUPTDetectorConfig config_;
    ZUPTState state_;
    int persistence_counter_;
    int leaving_counter_;

    // Zero-allocation circular buffers for variance calculation
    double accel_buffer_[MAX_WINDOW_CAPACITY];
    double gyro_buffer_[MAX_WINDOW_CAPACITY];
    int head_;
    int count_;

    double last_accel_norm_;
    double last_gyro_norm_;
    double last_accel_var_;
    double last_gyro_var_;
};

} // namespace nav_core

#endif // NAV_ZUPT_DETECTOR_HPP
