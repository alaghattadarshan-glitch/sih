#include "nav_core_c_api.h"
#include "nav_engine.hpp"

using namespace nav_core;

extern "C" {

NAV_EXPORT NavEngineHandle nav_core_create(double g_val) {
    ESKFConfig cfg;
    NavEngine* engine = new NavEngine(cfg, g_val > 0.0 ? g_val : 9.80665);
    return static_cast<NavEngineHandle>(engine);
}

NAV_EXPORT void nav_core_destroy(NavEngineHandle handle) {
    if (handle) {
        NavEngine* engine = static_cast<NavEngine*>(handle);
        delete engine;
    }
}

NAV_EXPORT int nav_core_initialize(
    NavEngineHandle handle,
    double initial_time,
    double lat_deg, double lon_deg, double alt_m,
    double v_e, double v_n, double v_u,
    double q_w, double q_x, double q_y, double q_z,
    double ab_x, double ab_y, double ab_z,
    double gb_x, double gb_y, double gb_z
) {
    if (!handle) return -1;
    NavEngine* engine = static_cast<NavEngine*>(handle);

    Vec3 v0(v_e, v_n, v_u);
    Quat q0(q_w, q_x, q_y, q_z);
    Vec3 ab0(ab_x, ab_y, ab_z);
    Vec3 gb0(gb_x, gb_y, gb_z);

    engine->initialize(initial_time, lat_deg, lon_deg, alt_m, v0, q0, ab0, gb0);
    return 0;
}

NAV_EXPORT int nav_core_process_imu(
    NavEngineHandle handle,
    double timestamp,
    double ax, double ay, double az,
    double gx, double gy, double gz
) {
    if (!handle) return -1;
    NavEngine* engine = static_cast<NavEngine*>(handle);

    Vec3 accel(ax, ay, az);
    Vec3 gyro(gx, gy, gz);
    engine->processIMU(timestamp, accel, gyro);
    return 0;
}

NAV_EXPORT int nav_core_process_gnss(
    NavEngineHandle handle,
    double timestamp,
    double lat, double lon, double alt,
    double h_acc, double v_acc,
    double hdop, double cn0
) {
    if (!handle) return -1;
    NavEngine* engine = static_cast<NavEngine*>(handle);

    engine->processGNSS(timestamp, lat, lon, alt, h_acc, v_acc, hdop, cn0);
    return 0;
}

NAV_EXPORT int nav_core_process_ai_displacement(
    NavEngineHandle handle,
    double ref_e, double ref_n, double ref_u,
    double d_e, double d_n, double d_u,
    double r_e, double r_n, double r_u,
    int* out_accepted,
    double* out_mahalanobis
) {
    if (!handle) return -1;
    NavEngine* engine = static_cast<NavEngine*>(handle);

    Vec3 ref_pos(ref_e, ref_n, ref_u);
    Vec3 delta_ai(d_e, d_n, d_u);
    Vec3 r_std(r_e > 0.0 ? r_e : 1.5, r_n > 0.0 ? r_n : 1.5, r_u > 0.0 ? r_u : 3.0);

    NavEngine::AIResult res = engine->processAIDisplacement(ref_pos, delta_ai, r_std);
    if (out_accepted) *out_accepted = res.accepted ? 1 : 0;
    if (out_mahalanobis) *out_mahalanobis = res.mahalanobis_dist;
    return 0;
}

NAV_EXPORT int nav_core_process_nhc(
    NavEngineHandle handle,
    double sigma_y,
    double sigma_z,
    double gate_threshold,
    int* out_accepted,
    double* out_mahalanobis
) {
    if (!handle) return -1;
    NavEngine* engine = static_cast<NavEngine*>(handle);

    NavEngine::NHCResult res = engine->processNHC(
        sigma_y > 0.0 ? sigma_y : 0.1,
        sigma_z > 0.0 ? sigma_z : 0.1,
        gate_threshold > 0.0 ? gate_threshold : 4.0
    );
    if (out_accepted) *out_accepted = res.accepted ? 1 : 0;
    if (out_mahalanobis) *out_mahalanobis = res.mahalanobis_dist;
    return 0;
}

NAV_EXPORT int nav_core_process_zupt(
    NavEngineHandle handle,
    double sigma_zupt,
    double gate_threshold,
    int* out_accepted,
    double* out_mahalanobis
) {
    if (!handle) return -1;
    NavEngine* engine = static_cast<NavEngine*>(handle);

    NavEngine::ZUPTResult res = engine->processZUPT(
        sigma_zupt > 0.0 ? sigma_zupt : 0.01,
        gate_threshold > 0.0 ? gate_threshold : 4.0
    );
    if (out_accepted) *out_accepted = res.accepted ? 1 : 0;
    if (out_mahalanobis) *out_mahalanobis = res.mahalanobis_dist;
    return 0;
}

NAV_EXPORT int nav_core_set_motion_constraints(
    NavEngineHandle handle,
    int enable_nhc,
    int enable_zupt
) {
    if (!handle) return -1;
    NavEngine* engine = static_cast<NavEngine*>(handle);
    engine->setMotionConstraints(enable_nhc != 0, enable_zupt != 0);
    return 0;
}

NAV_EXPORT int nav_core_get_zupt_state(
    NavEngineHandle handle,
    int* out_zupt_state,
    int* out_is_stationary
) {
    if (!handle) return -1;
    NavEngine* engine = static_cast<NavEngine*>(handle);
    if (out_zupt_state) *out_zupt_state = static_cast<int>(engine->getZUPTState());
    if (out_is_stationary) *out_is_stationary = engine->isStationary() ? 1 : 0;
    return 0;
}

NAV_EXPORT int nav_core_get_state(

    NavEngineHandle handle,
    double* out_timestamp,
    double* out_lat, double* out_lon, double* out_alt,
    double* out_pos_enu,
    double* out_vel_enu,
    double* out_quat,
    double* out_heading,
    int* out_status,
    double* out_confidence
) {
    if (!handle) return -1;
    NavEngine* engine = static_cast<NavEngine*>(handle);

    NavState state = engine->getState();

    if (out_timestamp) *out_timestamp = state.timestamp;

    if (out_lat && out_lon && out_alt) {
        double lat = 0, lon = 0, alt = 0;
        engine->enuToGeodetic(state.pos_enu, lat, lon, alt);
        *out_lat = lat;
        *out_lon = lon;
        *out_alt = alt;
    }

    if (out_pos_enu) {
        out_pos_enu[0] = state.pos_enu.x;
        out_pos_enu[1] = state.pos_enu.y;
        out_pos_enu[2] = state.pos_enu.z;
    }

    if (out_vel_enu) {
        out_vel_enu[0] = state.vel_enu.x;
        out_vel_enu[1] = state.vel_enu.y;
        out_vel_enu[2] = state.vel_enu.z;
    }

    if (out_quat) {
        out_quat[0] = state.q_b2n.w;
        out_quat[1] = state.q_b2n.x;
        out_quat[2] = state.q_b2n.y;
        out_quat[3] = state.q_b2n.z;
    }

    if (out_heading) *out_heading = state.getHeadingDeg();
    if (out_status) *out_status = state.status;
    if (out_confidence) *out_confidence = state.confidence;

    return 0;
}

NAV_EXPORT int nav_core_extract_ai_features(
    NavEngineHandle handle,
    float* out_buffer,
    int max_floats
) {
    if (!handle || !out_buffer) return 0;
    NavEngine* engine = static_cast<NavEngine*>(handle);
    return engine->extractAIWindowFeatures(out_buffer, max_floats);
}

NAV_EXPORT int nav_core_reset(NavEngineHandle handle) {
    if (!handle) return -1;
    NavEngine* engine = static_cast<NavEngine*>(handle);
    engine->reset();
    return 0;
}

} // extern "C"
