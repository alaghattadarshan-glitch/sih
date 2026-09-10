#include "zupt_detector.hpp"

namespace nav_core {

ZUPTDetector::ZUPTDetector(const ZUPTDetectorConfig& cfg)
    : config_(cfg),
      state_(ZUPT_MOVING),
      persistence_counter_(0),
      leaving_counter_(0),
      head_(0),
      count_(0),
      last_accel_norm_(0.0),
      last_gyro_norm_(0.0),
      last_accel_var_(0.0),
      last_gyro_var_(0.0) {
    for (int i = 0; i < MAX_WINDOW_CAPACITY; ++i) {
        accel_buffer_[i] = 0.0;
        gyro_buffer_[i] = 0.0;
    }
}

ZUPTState ZUPTDetector::update(
    double timestamp,
    double ax, double ay, double az,
    double gx, double gy, double gz
) {
    (void)timestamp;
    double a_norm = std::sqrt(ax * ax + ay * ay + az * az);
    double g_norm = std::sqrt(gx * gx + gy * gy + gz * gz);

    accel_buffer_[head_] = a_norm;
    gyro_buffer_[head_] = g_norm;
    head_ = (head_ + 1) % MAX_WINDOW_CAPACITY;
    if (count_ < config_.window_size_samples) {
        count_++;
    }

    last_accel_norm_ = a_norm;
    last_gyro_norm_ = g_norm;

    // Calculate variance over sliding window
    int win = std::min(count_, config_.window_size_samples);
    if (win >= 3) {
        double sum_a = 0.0, sum_g = 0.0;
        for (int i = 0; i < win; ++i) {
            int idx = (head_ - 1 - i + MAX_WINDOW_CAPACITY) % MAX_WINDOW_CAPACITY;
            sum_a += accel_buffer_[idx];
            sum_g += gyro_buffer_[idx];
        }
        double mean_a = sum_a / win;
        double mean_g = sum_g / win;

        double sq_diff_a = 0.0, sq_diff_g = 0.0;
        for (int i = 0; i < win; ++i) {
            int idx = (head_ - 1 - i + MAX_WINDOW_CAPACITY) % MAX_WINDOW_CAPACITY;
            double da = accel_buffer_[idx] - mean_a;
            double dg = gyro_buffer_[idx] - mean_g;
            sq_diff_a += da * da;
            sq_diff_g += dg * dg;
        }
        last_accel_var_ = sq_diff_a / win;
        last_gyro_var_ = sq_diff_g / win;
    } else {
        last_accel_var_ = 0.0;
        last_gyro_var_ = 0.0;
    }

    double gravity_diff = std::abs(a_norm - config_.g_val);
    bool c_accel_norm = gravity_diff <= config_.accel_norm_tol;
    bool c_gyro_norm = g_norm <= config_.gyro_norm_threshold;
    bool c_accel_var = last_accel_var_ <= config_.accel_var_threshold;
    bool c_gyro_var = last_gyro_var_ <= config_.gyro_var_threshold;

    bool is_candidate = c_accel_norm && c_gyro_norm && c_accel_var && c_gyro_var;

    // State machine transitions with hysteresis
    if (state_ == ZUPT_MOVING) {
        if (is_candidate) {
            state_ = ZUPT_STATIONARY_CANDIDATE;
            persistence_counter_ = 1;
        } else {
            persistence_counter_ = 0;
        }
    } else if (state_ == ZUPT_STATIONARY_CANDIDATE) {
        if (is_candidate) {
            persistence_counter_++;
            if (persistence_counter_ >= config_.persistence_samples) {
                state_ = ZUPT_STATIONARY;
            }
        } else {
            state_ = ZUPT_MOVING;
            persistence_counter_ = 0;
        }
    } else if (state_ == ZUPT_STATIONARY) {
        if (is_candidate) {
            leaving_counter_ = 0;
        } else {
            bool severe_accel = gravity_diff > (2.0 * config_.accel_norm_tol);
            bool severe_gyro = g_norm > (2.0 * config_.gyro_norm_threshold);
            if (severe_accel || severe_gyro) {
                state_ = ZUPT_MOVING;
                persistence_counter_ = 0;
                leaving_counter_ = 0;
            } else {
                state_ = ZUPT_LEAVING_STATIONARY;
                leaving_counter_ = 1;
            }
        }
    } else if (state_ == ZUPT_LEAVING_STATIONARY) {
        if (is_candidate) {
            state_ = ZUPT_STATIONARY;
            leaving_counter_ = 0;
        } else {
            leaving_counter_++;
            if (leaving_counter_ >= config_.leaving_hysteresis_samples) {
                state_ = ZUPT_MOVING;
                persistence_counter_ = 0;
                leaving_counter_ = 0;
            }
        }
    }

    return state_;
}

void ZUPTDetector::reset() {
    state_ = ZUPT_MOVING;
    persistence_counter_ = 0;
    leaving_counter_ = 0;
    head_ = 0;
    count_ = 0;
    last_accel_norm_ = 0.0;
    last_gyro_norm_ = 0.0;
    last_accel_var_ = 0.0;
    last_gyro_var_ = 0.0;
}

} // namespace nav_core
