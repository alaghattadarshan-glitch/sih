#ifndef NAV_CORE_C_API_H
#define NAV_CORE_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define NAV_EXPORT __declspec(dllexport)
#else
#define NAV_EXPORT __attribute__((visibility("default")))
#endif

typedef void* NavEngineHandle;

NAV_EXPORT NavEngineHandle nav_core_create(double g_val);

NAV_EXPORT void nav_core_destroy(NavEngineHandle handle);

NAV_EXPORT int nav_core_initialize(
    NavEngineHandle handle,
    double initial_time,
    double lat_deg, double lon_deg, double alt_m,
    double v_e, double v_n, double v_u,
    double q_w, double q_x, double q_y, double q_z,
    double ab_x, double ab_y, double ab_z,
    double gb_x, double gb_y, double gb_z
);

NAV_EXPORT int nav_core_process_imu(
    NavEngineHandle handle,
    double timestamp,
    double ax, double ay, double az,
    double gx, double gy, double gz
);

NAV_EXPORT int nav_core_process_gnss(
    NavEngineHandle handle,
    double timestamp,
    double lat, double lon, double alt,
    double h_acc, double v_acc,
    double hdop, double cn0
);

NAV_EXPORT int nav_core_process_ai_displacement(
    NavEngineHandle handle,
    double ref_e, double ref_n, double ref_u,
    double d_e, double d_n, double d_u,
    double r_e, double r_n, double r_u,
    int* out_accepted,
    double* out_mahalanobis
);

NAV_EXPORT int nav_core_process_nhc(
    NavEngineHandle handle,
    double sigma_y,
    double sigma_z,
    double gate_threshold,
    int* out_accepted,
    double* out_mahalanobis
);

NAV_EXPORT int nav_core_process_zupt(
    NavEngineHandle handle,
    double sigma_zupt,
    double gate_threshold,
    int* out_accepted,
    double* out_mahalanobis
);

NAV_EXPORT int nav_core_set_motion_constraints(
    NavEngineHandle handle,
    int enable_nhc,
    int enable_zupt
);

NAV_EXPORT int nav_core_get_zupt_state(
    NavEngineHandle handle,
    int* out_zupt_state,
    int* out_is_stationary
);

NAV_EXPORT int nav_core_get_state(

    NavEngineHandle handle,
    double* out_timestamp,
    double* out_lat, double* out_lon, double* out_alt,
    double* out_pos_enu,  // 3 doubles: [E, N, U]
    double* out_vel_enu,  // 3 doubles: [Ve, Vn, Vu]
    double* out_quat,     // 4 doubles: [qw, qx, qy, qz]
    double* out_heading,  // 1 double: heading deg
    int* out_status,      // 1 int: status enum
    double* out_confidence // 1 double: confidence radius m
);

NAV_EXPORT int nav_core_extract_ai_features(
    NavEngineHandle handle,
    float* out_buffer,
    int max_floats
);

NAV_EXPORT int nav_core_reset(NavEngineHandle handle);

#ifdef __cplusplus
}
#endif

#endif // NAV_CORE_C_API_H
