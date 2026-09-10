#include <jni.h>
#include <string>
#include "nav_core_c_api.h"

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_sih26168_navigation_core_NativeNavCore_nativeCreate(JNIEnv *env, jobject thiz, jdouble g_val) {
    return reinterpret_cast<jlong>(nav_core_create(g_val));
}

JNIEXPORT void JNICALL
Java_com_sih26168_navigation_core_NativeNavCore_nativeDestroy(JNIEnv *env, jobject thiz, jlong handle) {
    nav_core_destroy(reinterpret_cast<NavEngineHandle>(handle));
}

JNIEXPORT jint JNICALL
Java_com_sih26168_navigation_core_NativeNavCore_nativeInitialize(
    JNIEnv *env, jobject thiz, jlong handle,
    jdouble initial_time,
    jdouble lat_deg, jdouble lon_deg, jdouble alt_m,
    jdouble v_e, jdouble v_n, jdouble v_u,
    jdouble q_w, jdouble q_x, jdouble q_y, jdouble q_z,
    jdouble ab_x, jdouble ab_y, jdouble ab_z,
    jdouble gb_x, jdouble gb_y, jdouble gb_z
) {
    return nav_core_initialize(
        reinterpret_cast<NavEngineHandle>(handle),
        initial_time, lat_deg, lon_deg, alt_m,
        v_e, v_n, v_u,
        q_w, q_x, q_y, q_z,
        ab_x, ab_y, ab_z,
        gb_x, gb_y, gb_z
    );
}

JNIEXPORT jint JNICALL
Java_com_sih26168_navigation_core_NativeNavCore_nativeProcessIMU(
    JNIEnv *env, jobject thiz, jlong handle,
    jdouble timestamp,
    jdouble ax, jdouble ay, jdouble az,
    jdouble gx, jdouble gy, jdouble gz
) {
    return nav_core_process_imu(
        reinterpret_cast<NavEngineHandle>(handle),
        timestamp, ax, ay, az, gx, gy, gz
    );
}

JNIEXPORT jint JNICALL
Java_com_sih26168_navigation_core_NativeNavCore_nativeProcessGNSS(
    JNIEnv *env, jobject thiz, jlong handle,
    jdouble timestamp,
    jdouble lat, jdouble lon, jdouble alt,
    jdouble h_acc, jdouble v_acc,
    jdouble hdop, jdouble cn0
) {
    return nav_core_process_gnss(
        reinterpret_cast<NavEngineHandle>(handle),
        timestamp, lat, lon, alt,
        h_acc, v_acc, hdop, cn0
    );
}

JNIEXPORT jint JNICALL
Java_com_sih26168_navigation_core_NativeNavCore_nativeProcessAIDisplacement(
    JNIEnv *env, jobject thiz, jlong handle,
    jdouble ref_e, jdouble ref_n, jdouble ref_u,
    jdouble d_e, jdouble d_n, jdouble d_u,
    jdouble r_e, jdouble r_n, jdouble r_u,
    jintArray out_accepted,
    jdoubleArray out_mahalanobis
) {
    int accepted = 0;
    double mahalanobis = 0.0;

    int res = nav_core_process_ai_displacement(
        reinterpret_cast<NavEngineHandle>(handle),
        ref_e, ref_n, ref_u,
        d_e, d_n, d_u,
        r_e, r_n, r_u,
        &accepted,
        &mahalanobis
    );

    if (out_accepted != nullptr) {
        jint acc_val = accepted;
        env->SetIntArrayRegion(out_accepted, 0, 1, &acc_val);
    }
    if (out_mahalanobis != nullptr) {
        jdouble mah_val = mahalanobis;
        env->SetDoubleArrayRegion(out_mahalanobis, 0, 1, &mah_val);
    }

    return res;
}

JNIEXPORT jint JNICALL
Java_com_sih26168_navigation_core_NativeNavCore_nativeGetState(
    JNIEnv *env, jobject thiz, jlong handle,
    jdoubleArray out_doubles, // [timestamp, lat, lon, alt, east, north, up, ve, vn, vu, qw, qx, qy, qz, heading, confidence]
    jintArray out_ints        // [status]
) {
    double timestamp = 0, lat = 0, lon = 0, alt = 0;
    double pos_enu[3] = {0, 0, 0};
    double vel_enu[3] = {0, 0, 0};
    double quat[4] = {1, 0, 0, 0};
    double heading = 0;
    int status = 0;
    double confidence = 0;

    int res = nav_core_get_state(
        reinterpret_cast<NavEngineHandle>(handle),
        &timestamp, &lat, &lon, &alt,
        pos_enu, vel_enu, quat, &heading,
        &status, &confidence
    );

    if (res == 0 && out_doubles != nullptr) {
        double d_buf[16] = {
            timestamp, lat, lon, alt,
            pos_enu[0], pos_enu[1], pos_enu[2],
            vel_enu[0], vel_enu[1], vel_enu[2],
            quat[0], quat[1], quat[2], quat[3],
            heading, confidence
        };
        env->SetDoubleArrayRegion(out_doubles, 0, 16, d_buf);
    }

    if (res == 0 && out_ints != nullptr) {
        jint s_val = status;
        env->SetIntArrayRegion(out_ints, 0, 1, &s_val);
    }

    return res;
}

JNIEXPORT jint JNICALL
Java_com_sih26168_navigation_core_NativeNavCore_nativeExtractAIFeatures(
    JNIEnv *env, jobject thiz, jlong handle,
    jfloatArray out_buffer,
    jint max_floats
) {
    if (out_buffer == nullptr) return 0;
    jfloat *buf = env->GetFloatArrayElements(out_buffer, nullptr);
    int count = nav_core_extract_ai_features(
        reinterpret_cast<NavEngineHandle>(handle),
        buf, max_floats
    );
    env->ReleaseFloatArrayElements(out_buffer, buf, 0);
    return count;
}

JNIEXPORT jint JNICALL
Java_com_sih26168_navigation_core_NativeNavCore_nativeReset(JNIEnv *env, jobject thiz, jlong handle) {
    return nav_core_reset(reinterpret_cast<NavEngineHandle>(handle));
}

} // extern "C"
