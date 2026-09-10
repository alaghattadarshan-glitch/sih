package com.sih26168.navigation.logging

import org.json.JSONObject

data class SessionMetadata(
    val sessionId: String,
    val appVersion: String = "1.0.0",
    val modelVersion: String = "1.0.0",
    val deviceManufacturer: String = "Unknown",
    val deviceModel: String = "Unknown",
    val androidVersion: String = "Unknown",
    val abi: String = "arm64-v8a",
    val mountingPreset: String = "PRESET_FLAT_PORTRAIT",
    val requestedImuHz: Double = 100.0,
    val startTimeUtc: String = "",
    var endTimeUtc: String = "",
    var totalImuSamples: Long = 0,
    var totalGnssFixes: Long = 0,
    var calibrationStatus: String = "UNINITIALIZED",
    var gyroBiasEstimated: List<Double> = listOf(0.0, 0.0, 0.0),
    var accelBiasEstimated: List<Double> = listOf(0.0, 0.0, 0.0),
    val testDescription: String = "Real-world sensor drive/walk validation session"
) {
    fun toJson(): String {
        val json = JSONObject()
        json.put("session_id", sessionId)
        json.put("app_version", appVersion)
        json.put("model_version", modelVersion)

        val devObj = JSONObject()
        devObj.put("manufacturer", deviceManufacturer)
        devObj.put("model", deviceModel)
        devObj.put("android_version", androidVersion)
        devObj.put("abi", abi)
        json.put("device_info", devObj)

        json.put("mounting_preset", mountingPreset)
        json.put("requested_imu_hz", requestedImuHz)
        json.put("start_time_utc", startTimeUtc)
        json.put("end_time_utc", endTimeUtc)
        json.put("total_imu_samples", totalImuSamples)
        json.put("total_gnss_fixes", totalGnssFixes)
        json.put("calibration_status", calibrationStatus)
        json.put("gyro_bias_estimated", gyroBiasEstimated)
        json.put("accel_bias_estimated", accelBiasEstimated)
        json.put("test_description", testDescription)

        return json.toString(2)
    }
}
