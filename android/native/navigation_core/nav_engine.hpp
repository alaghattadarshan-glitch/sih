#ifndef NAV_ENGINE_HPP
#define NAV_ENGINE_HPP

#include "math_types.hpp"
#include "quaternion_math.hpp"
#include "nav_state.hpp"
#include "eskf_core.hpp"
#include "outage_detector.hpp"
#include "zupt_detector.hpp"


namespace nav_core {

struct IMUSample {
    double timestamp;
    Vec3 accel;
    Vec3 gyro;
};

class NavEngine {
public:
    static constexpr int IMU_BUFFER_CAPACITY = 200;
    static constexpr int AI_WINDOW_SIZE = 100;

    NavEngine(const ESKFConfig& eskf_cfg = ESKFConfig(), double g_val = 9.80665);

    void initialize(
        double initial_time,
        double lat_deg, double lon_deg, double alt_m,
        const Vec3& initial_vel_enu,
        const Quat& initial_quat,
        const Vec3& initial_accel_bias = Vec3(0, 0, 0),
        const Vec3& initial_gyro_bias = Vec3(0, 0, 0)
    );

    NavState processIMU(double timestamp, const Vec3& accel, const Vec3& gyro);

    NavState processGNSS(
        double timestamp,
        double lat_deg, double lon_deg, double alt_m,
        double h_acc_m = -1.0,
        double v_acc_m = -1.0,
        double hdop = -1.0,
        double cn0 = -1.0
    );

    struct AIResult {
        NavState state;
        bool accepted;
        double mahalanobis_dist;
    };

    AIResult processAIDisplacement(
        const Vec3& ref_pos_enu,
        const Vec3& delta_p_ai,
        const Vec3& r_ai_std = Vec3(1.5, 1.5, 3.0)
    );

    struct NHCResult {
        NavState state;
        bool accepted;
        double mahalanobis_dist;
    };

    NHCResult processNHC(
        double sigma_y = 0.1,
        double sigma_z = 0.1,
        double gate_threshold = 4.0
    );

    struct ZUPTResult {
        NavState state;
        bool accepted;
        double mahalanobis_dist;
    };

    ZUPTResult processZUPT(
        double sigma_zupt = 0.01,
        double gate_threshold = 4.0
    );

    void setMotionConstraints(bool enable_nhc, bool enable_zupt) {
        enable_nhc_ = enable_nhc;
        enable_zupt_ = enable_zupt;
    }

    ZUPTState getZUPTState() const { return zupt_detector_.getState(); }
    bool isStationary() const { return zupt_detector_.isStationary(); }

    // Extracts normalized/raw features for the latest 100 IMU samples [100, 8]
    int extractAIWindowFeatures(float* out_buffer, int max_floats) const;

    NavState getState() const;
    const Mat15& getCovariance() const { return eskf_.getCovariance(); }
    NavigationStatus getNavigationStatus() const { return outage_detector_.getStatus(); }

    bool isInitialized() const { return is_initialized_; }
    void reset();

    // Coordinate Conversion
    Vec3 geodeticToENU(double lat_deg, double lon_deg, double alt_m) const;
    void enuToGeodetic(const Vec3& enu, double& lat_deg, double& lon_deg, double& alt_m) const;

private:
    ESKFCore eskf_;
    OutageDetector outage_detector_;
    ZUPTDetector zupt_detector_;
    bool is_initialized_;
    bool enable_nhc_;
    bool enable_zupt_;


    double origin_lat_rad_;
    double origin_lon_rad_;
    double origin_alt_m_;

    // Preallocated Circular Buffer for IMU samples
    IMUSample imu_buffer_[IMU_BUFFER_CAPACITY];
    int imu_head_;
    int imu_count_;

    // Reference position at window start
    Vec3 window_start_pos_enu_;
    double window_start_time_;
};

} // namespace nav_core

#endif // NAV_ENGINE_HPP
