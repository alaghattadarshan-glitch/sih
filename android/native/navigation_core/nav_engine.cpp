#include "nav_engine.hpp"
#include <cmath>

namespace nav_core {

static constexpr double WGS84_A = 6378137.0;
static constexpr double WGS84_F = 1.0 / 298.257223563;
static constexpr double WGS84_E2 = WGS84_F * (2.0 - WGS84_F);

NavEngine::NavEngine(const ESKFConfig& eskf_cfg, double g_val)
    : eskf_(eskf_cfg, g_val),
      outage_detector_(),
      zupt_detector_(),
      is_initialized_(false),
      enable_nhc_(false),
      enable_zupt_(false),
      origin_lat_rad_(0.0),
      origin_lon_rad_(0.0),
      origin_alt_m_(0.0),
      imu_head_(0),
      imu_count_(0),
      window_start_pos_enu_(0, 0, 0),
      window_start_time_(0.0) {}

void NavEngine::initialize(
    double initial_time,
    double lat_deg, double lon_deg, double alt_m,
    const Vec3& initial_vel_enu,
    const Quat& initial_quat,
    const Vec3& initial_accel_bias,
    const Vec3& initial_gyro_bias
) {
    origin_lat_rad_ = lat_deg * M_PI / 180.0;
    origin_lon_rad_ = lon_deg * M_PI / 180.0;
    origin_alt_m_ = alt_m;

    eskf_.initialize(
        initial_time, lat_deg, lon_deg, alt_m,
        initial_vel_enu, initial_quat,
        initial_accel_bias, initial_gyro_bias
    );

    outage_detector_.reset();
    zupt_detector_.reset();
    imu_head_ = 0;
    imu_count_ = 0;
    window_start_pos_enu_ = Vec3(0, 0, 0);
    window_start_time_ = initial_time;
    is_initialized_ = true;
}


Vec3 NavEngine::geodeticToENU(double lat_deg, double lon_deg, double alt_m) const {
    double lat_r = lat_deg * M_PI / 180.0;
    double lon_r = lon_deg * M_PI / 180.0;

    // Origin ECEF
    double sin_lat0 = std::sin(origin_lat_rad_);
    double cos_lat0 = std::cos(origin_lat_rad_);
    double sin_lon0 = std::sin(origin_lon_rad_);
    double cos_lon0 = std::cos(origin_lon_rad_);

    double N0 = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat0 * sin_lat0);
    double x0 = (N0 + origin_alt_m_) * cos_lat0 * cos_lon0;
    double y0 = (N0 + origin_alt_m_) * cos_lat0 * sin_lon0;
    double z0 = (N0 * (1.0 - WGS84_E2) + origin_alt_m_) * sin_lat0;

    // Point ECEF
    double sin_lat = std::sin(lat_r);
    double cos_lat = std::cos(lat_r);
    double sin_lon = std::sin(lon_r);
    double cos_lon = std::cos(lon_r);

    double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
    double x = (N + alt_m) * cos_lat * cos_lon;
    double y = (N + alt_m) * cos_lat * sin_lon;
    double z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat;

    double dx = x - x0;
    double dy = y - y0;
    double dz = z - z0;

    // ECEF to ENU rotation
    double e = -sin_lon0 * dx + cos_lon0 * dy;
    double n = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz;
    double u = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz;

    return Vec3(e, n, u);
}

void NavEngine::enuToGeodetic(const Vec3& enu, double& lat_deg, double& lon_deg, double& alt_m) const {
    double sin_lat0 = std::sin(origin_lat_rad_);
    double cos_lat0 = std::cos(origin_lat_rad_);
    double sin_lon0 = std::sin(origin_lon_rad_);
    double cos_lon0 = std::cos(origin_lon_rad_);

    double N0 = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat0 * sin_lat0);
    double x0 = (N0 + origin_alt_m_) * cos_lat0 * cos_lon0;
    double y0 = (N0 + origin_alt_m_) * cos_lat0 * sin_lon0;
    double z0 = (N0 * (1.0 - WGS84_E2) + origin_alt_m_) * sin_lat0;

    // ENU to ECEF
    double dx = -sin_lon0 * enu.x - sin_lat0 * cos_lon0 * enu.y + cos_lat0 * cos_lon0 * enu.z;
    double dy = cos_lon0 * enu.x - sin_lat0 * sin_lon0 * enu.y + cos_lat0 * sin_lon0 * enu.z;
    double dz = cos_lat0 * enu.y + sin_lat0 * enu.z;

    double x = x0 + dx;
    double y = y0 + dy;
    double z = z0 + dz;

    // ECEF to Geodetic (Bowring's algorithm)
    double p = std::sqrt(x * x + y * y);
    double theta = std::atan2(z * WGS84_A, p * (WGS84_A * (1.0 - WGS84_F)));
    double e_prime_sq = (WGS84_A * WGS84_A - (WGS84_A * (1.0 - WGS84_F)) * (WGS84_A * (1.0 - WGS84_F))) /
                        ((WGS84_A * (1.0 - WGS84_F)) * (WGS84_A * (1.0 - WGS84_F)));

    double lat = std::atan2(
        z + e_prime_sq * (WGS84_A * (1.0 - WGS84_F)) * std::pow(std::sin(theta), 3),
        p - WGS84_E2 * WGS84_A * std::pow(std::cos(theta), 3)
    );
    double lon = std::atan2(y, x);
    double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * std::sin(lat) * std::sin(lat));
    double h = p / std::cos(lat) - N;

    lat_deg = lat * 180.0 / M_PI;
    lon_deg = lon * 180.0 / M_PI;
    alt_m = h;
}

NavState NavEngine::processIMU(double timestamp, const Vec3& accel, const Vec3& gyro) {
    if (!is_initialized_) return NavState();

    // 1. Update stationary ZUPT detector
    zupt_detector_.update(timestamp, accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z);

    // 2. Store in circular buffer (zero-allocation)
    imu_buffer_[imu_head_].timestamp = timestamp;
    imu_buffer_[imu_head_].accel = accel;
    imu_buffer_[imu_head_].gyro = gyro;
    imu_head_ = (imu_head_ + 1) % IMU_BUFFER_CAPACITY;
    if (imu_count_ < IMU_BUFFER_CAPACITY) imu_count_++;

    // 3. ESKF Nominal Propagation and Prediction
    NavState state = eskf_.predict(timestamp, accel, gyro);
    state.status = outage_detector_.getStatus();

    // 4. If in OUTAGE, apply motion constraints if enabled
    if (state.status == NAV_STATUS_OUTAGE) {
        if (enable_nhc_ && !zupt_detector_.isStationary()) {
            ESKFCore::NHCUpdateResult nhc_res = eskf_.updateNHC();
            if (nhc_res.accepted) {
                state = nhc_res.state;
            }
        }
        if (enable_zupt_ && zupt_detector_.isStationary()) {
            ESKFCore::ZUPTUpdateResult zupt_res = eskf_.updateZUPT();
            if (zupt_res.accepted) {
                state = zupt_res.state;
            }
        }
    }

    // Calculate position confidence radius (sqrt of horizontal pos trace)
    const Mat15& P = eskf_.getCovariance();
    state.confidence = std::sqrt(std::max(0.0, P.data[0][0] + P.data[1][1]));

    return state;
}

NavState NavEngine::processGNSS(
    double timestamp,
    double lat_deg, double lon_deg, double alt_m,
    double h_acc_m, double v_acc_m,
    double hdop, double cn0
) {
    if (!is_initialized_) return NavState();

    bool has_fix = (lat_deg != 0.0 || lon_deg != 0.0);
    NavigationStatus status = outage_detector_.processObservation(has_fix, timestamp, h_acc_m, hdop, cn0);

    if (status != NAV_STATUS_OUTAGE && has_fix) {
        Vec3 gnss_enu = geodeticToENU(lat_deg, lon_deg, alt_m);
        NavState state = eskf_.updateGNSS(gnss_enu, h_acc_m, v_acc_m);
        state.status = status;
        const Mat15& P = eskf_.getCovariance();
        state.confidence = std::sqrt(std::max(0.0, P.data[0][0] + P.data[1][1]));
        return state;
    }

    NavState state = eskf_.getState();
    state.status = status;
    const Mat15& P = eskf_.getCovariance();
    state.confidence = std::sqrt(std::max(0.0, P.data[0][0] + P.data[1][1]));
    return state;
}

NavEngine::AIResult NavEngine::processAIDisplacement(
    const Vec3& ref_pos_enu,
    const Vec3& delta_p_ai,
    const Vec3& r_ai_std
) {
    AIResult res;
    if (!is_initialized_) {
        res.state = NavState();
        res.accepted = false;
        res.mahalanobis_dist = 0.0;
        return res;
    }

    ESKFCore::AIUpdateResult eskf_res = eskf_.updateAIDisplacement(ref_pos_enu, delta_p_ai, r_ai_std);
    res.state = eskf_res.state;
    res.accepted = eskf_res.accepted;
    res.mahalanobis_dist = eskf_res.mahalanobis_dist;
    res.state.status = outage_detector_.getStatus();

    const Mat15& P = eskf_.getCovariance();
    res.state.confidence = std::sqrt(std::max(0.0, P.data[0][0] + P.data[1][1]));

    return res;
}

NavEngine::NHCResult NavEngine::processNHC(
    double sigma_y,
    double sigma_z,
    double gate_threshold
) {
    NHCResult res;
    if (!is_initialized_) {
        res.state = NavState();
        res.accepted = false;
        res.mahalanobis_dist = 0.0;
        return res;
    }

    ESKFCore::NHCUpdateResult eskf_res = eskf_.updateNHC(sigma_y, sigma_z, gate_threshold);
    res.state = eskf_res.state;
    res.accepted = eskf_res.accepted;
    res.mahalanobis_dist = eskf_res.mahalanobis_dist;
    res.state.status = outage_detector_.getStatus();

    const Mat15& P = eskf_.getCovariance();
    res.state.confidence = std::sqrt(std::max(0.0, P.data[0][0] + P.data[1][1]));

    return res;
}

NavEngine::ZUPTResult NavEngine::processZUPT(
    double sigma_zupt,
    double gate_threshold
) {
    ZUPTResult res;
    if (!is_initialized_) {
        res.state = NavState();
        res.accepted = false;
        res.mahalanobis_dist = 0.0;
        return res;
    }

    ESKFCore::ZUPTUpdateResult eskf_res = eskf_.updateZUPT(sigma_zupt, gate_threshold);
    res.state = eskf_res.state;
    res.accepted = eskf_res.accepted;
    res.mahalanobis_dist = eskf_res.mahalanobis_dist;
    res.state.status = outage_detector_.getStatus();

    const Mat15& P = eskf_.getCovariance();
    res.state.confidence = std::sqrt(std::max(0.0, P.data[0][0] + P.data[1][1]));

    return res;
}

int NavEngine::extractAIWindowFeatures(float* out_buffer, int max_floats) const {
    if (imu_count_ < AI_WINDOW_SIZE || max_floats < (AI_WINDOW_SIZE * 8)) {
        return 0; // Not enough samples
    }

    // Extract the chronological 100 samples ending at the latest sample
    int start_idx = (imu_head_ - AI_WINDOW_SIZE + IMU_BUFFER_CAPACITY) % IMU_BUFFER_CAPACITY;

    for (int i = 0; i < AI_WINDOW_SIZE; ++i) {
        int idx = (start_idx + i) % IMU_BUFFER_CAPACITY;
        const IMUSample& s = imu_buffer_[idx];

        double ax = s.accel.x, ay = s.accel.y, az = s.accel.z;
        double gx = s.gyro.x, gy = s.gyro.y, gz = s.gyro.z;
        double a_norm = std::sqrt(ax * ax + ay * ay + az * az);
        double g_norm = std::sqrt(gx * gx + gy * gy + gz * gz);

        int base = i * 8;
        out_buffer[base + 0] = static_cast<float>(ax);
        out_buffer[base + 1] = static_cast<float>(ay);
        out_buffer[base + 2] = static_cast<float>(az);
        out_buffer[base + 3] = static_cast<float>(gx);
        out_buffer[base + 4] = static_cast<float>(gy);
        out_buffer[base + 5] = static_cast<float>(gz);
        out_buffer[base + 6] = static_cast<float>(a_norm);
        out_buffer[base + 7] = static_cast<float>(g_norm);
    }

    return AI_WINDOW_SIZE;
}

NavState NavEngine::getState() const {
    NavState s = eskf_.getState();
    s.status = outage_detector_.getStatus();
    const Mat15& P = eskf_.getCovariance();
    s.confidence = std::sqrt(std::max(0.0, P.data[0][0] + P.data[1][1]));
    return s;
}

void NavEngine::reset() {
    is_initialized_ = false;
    eskf_.reset();
    outage_detector_.reset();
    zupt_detector_.reset();
    imu_head_ = 0;
    imu_count_ = 0;
}

} // namespace nav_core

