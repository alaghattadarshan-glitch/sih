#ifndef NAV_ESKF_CORE_HPP
#define NAV_ESKF_CORE_HPP

#include "math_types.hpp"
#include "quaternion_math.hpp"
#include "nav_state.hpp"
#include "strapdown_ins.hpp"

namespace nav_core {

struct ESKFConfig {
    double accel_noise_std;
    double gyro_noise_std;
    double accel_bias_rw_std;
    double gyro_bias_rw_std;

    double pos_std_init;
    double vel_std_init;
    double att_std_init;
    double accel_bias_std_init;
    double gyro_bias_std_init;

    double gnss_pos_std_default;
    double ai_gate_threshold;

    ESKFConfig()
        : accel_noise_std(0.05),
          gyro_noise_std(0.005),
          accel_bias_rw_std(0.0001),
          gyro_bias_rw_std(0.00001),
          pos_std_init(1.0),
          vel_std_init(0.1),
          att_std_init(0.01),
          accel_bias_std_init(0.05),
          gyro_bias_std_init(0.005),
          gnss_pos_std_default(2.5),
          ai_gate_threshold(4.0) {}
};

class ESKFCore {
public:
    ESKFCore(const ESKFConfig& cfg = ESKFConfig(), double g_val = 9.80665);

    void initialize(
        double initial_time,
        double lat_deg, double lon_deg, double alt_m,
        const Vec3& initial_vel_enu,
        const Quat& initial_quat,
        const Vec3& initial_accel_bias = Vec3(0, 0, 0),
        const Vec3& initial_gyro_bias = Vec3(0, 0, 0)
    );

    NavState predict(double timestamp, const Vec3& raw_accel, const Vec3& raw_gyro);

    NavState updateGNSS(
        const Vec3& gnss_pos_enu,
        double h_acc_m = -1.0,
        double v_acc_m = -1.0
    );

    struct AIUpdateResult {
        NavState state;
        bool accepted;
        double mahalanobis_dist;
    };

    AIUpdateResult updateAIDisplacement(
        const Vec3& ref_pos_enu,
        const Vec3& delta_p_ai,
        const Vec3& r_ai_std = Vec3(1.5, 1.5, 3.0)
    );

    struct NHCUpdateResult {
        NavState state;
        bool accepted;
        double mahalanobis_dist;
    };

    NHCUpdateResult updateNHC(
        double sigma_y = 0.1,
        double sigma_z = 0.1,
        double gate_threshold = 4.0
    );

    struct ZUPTUpdateResult {
        NavState state;
        bool accepted;
        double mahalanobis_dist;
    };

    ZUPTUpdateResult updateZUPT(
        double sigma_zupt = 0.01,
        double gate_threshold = 4.0
    );

    const NavState& getState() const { return ins_.getState(); }

    const Mat15& getCovariance() const { return P_; }
    bool isInitialized() const { return is_initialized_ && ins_.isInitialized(); }
    void reset();

    StrapdownINS& getINS() { return ins_; }

private:
    ESKFConfig config_;
    StrapdownINS ins_;
    Mat15 P_;
    bool is_initialized_;

    void buildErrorDynamics(
        const Mat3& R_b2n,
        const Vec3& f_body,
        double dt,
        Mat15& Phi,
        Mat15& Qd
    );
};

} // namespace nav_core

#endif // NAV_ESKF_CORE_HPP
