package com.sih26168.navigation.calibration

enum class CalibrationStatus {
    IDLE,
    CALIBRATING,
    SUCCESS,
    FAILED_MOTION_DETECTED,
    FAILED_TIMEOUT
}

data class CalibrationResult(
    val status: CalibrationStatus,
    val sampleCount: Int,
    val accelBias: DoubleArray,      // [bx, by, bz] in m/s^2
    val gyroBias: DoubleArray,       // [bx, by, bz] in rad/s
    val initialQuat: DoubleArray,    // [qw, qx, qy, qz]
    val accelVariance: Double,
    val gyroVariance: Double,
    val message: String
)

class LiveCalibrationManager(
    private val requiredDurationSec: Double = 3.0,
    private val targetSampleRateHz: Double = 100.0,
    private val maxAccelVar: Double = 0.05,
    private val maxGyroVar: Double = 0.005
) {
    private val requiredSamples: Int = (requiredDurationSec * targetSampleRateHz).toInt()

    private val accelXList = mutableListOf<Double>()
    private val accelYList = mutableListOf<Double>()
    private val accelZList = mutableListOf<Double>()

    private val gyroXList = mutableListOf<Double>()
    private val gyroYList = mutableListOf<Double>()
    private val gyroZList = mutableListOf<Double>()

    var status: CalibrationStatus = CalibrationStatus.IDLE
        private set

    val progressPercent: Int
        get() = if (requiredSamples > 0) ((accelXList.size * 100) / requiredSamples).coerceAtMost(100) else 0

    fun start() {
        reset()
        status = CalibrationStatus.CALIBRATING
    }

    fun reset() {
        accelXList.clear(); accelYList.clear(); accelZList.clear()
        gyroXList.clear(); gyroYList.clear(); gyroZList.clear()
        status = CalibrationStatus.IDLE
    }

    /**
     * Add an IMU sample during live calibration.
     * @return CalibrationResult with updated status
     */
    fun addSample(
        ax: Double, ay: Double, az: Double,
        gx: Double, gy: Double, gz: Double
    ): CalibrationResult {
        if (status != CalibrationStatus.CALIBRATING) {
            return CalibrationResult(
                status, accelXList.size,
                DoubleArray(3), DoubleArray(3), doubleArrayOf(1.0, 0.0, 0.0, 0.0),
                0.0, 0.0, "Calibration not active"
            )
        }

        accelXList.add(ax); accelYList.add(ay); accelZList.add(az)
        gyroXList.add(gx); gyroYList.add(gy); gyroZList.add(gz)

        if (accelXList.size < requiredSamples) {
            return CalibrationResult(
                CalibrationStatus.CALIBRATING, accelXList.size,
                DoubleArray(3), DoubleArray(3), doubleArrayOf(1.0, 0.0, 0.0, 0.0),
                0.0, 0.0, "Collecting samples: ${progressPercent}%"
            )
        }

        // Evaluate Stationarity
        val (meanAx, varAx) = computeMeanAndVariance(accelXList)
        val (meanAy, varAy) = computeMeanAndVariance(accelYList)
        val (meanAz, varAz) = computeMeanAndVariance(accelZList)

        val (meanGx, varGx) = computeMeanAndVariance(gyroXList)
        val (meanGy, varGy) = computeMeanAndVariance(gyroYList)
        val (meanGz, varGz) = computeMeanAndVariance(gyroZList)

        val totalAccelVar = (varAx + varAy + varAz) / 3.0
        val totalGyroVar = (varGx + varGy + varGz) / 3.0

        val accelNorm = Math.sqrt(meanAx * meanAx + meanAy * meanAy + meanAz * meanAz)

        // Stationarity Checks
        if (totalAccelVar > maxAccelVar || totalGyroVar > maxGyroVar || Math.abs(accelNorm - 9.80665) > 1.5) {
            status = CalibrationStatus.FAILED_MOTION_DETECTED
            return CalibrationResult(
                status, accelXList.size,
                DoubleArray(3), DoubleArray(3), doubleArrayOf(1.0, 0.0, 0.0, 0.0),
                totalAccelVar, totalGyroVar,
                "Motion detected during calibration (acc_var=${String.format("%.4f", totalAccelVar)}, gyro_var=${String.format("%.4f", totalGyroVar)}). Please keep device stationary."
            )
        }

        // Gyroscope bias = mean stationary gyro readings
        val gyroBias = doubleArrayOf(meanGx, meanGy, meanGz)

        // Deriving Initial Attitude from Gravity Vector in body frame (+z Up)
        // Unit gravity direction: [meanAx, meanAy, meanAz] / accelNorm
        val gxNorm = meanAx / accelNorm
        val gyNorm = meanAy / accelNorm
        val gzNorm = meanAz / accelNorm

        val initialRoll = Math.atan2(-gyNorm, gzNorm)
        val initialPitch = Math.atan2(gxNorm, Math.sqrt(gyNorm * gyNorm + gzNorm * gzNorm))
        val initialYaw = 0.0 // Default 0 rad (East in math convention, North in geographic)

        val q = eulerToQuat(initialRoll, initialPitch, initialYaw)

        // Accelerometer bias = measured specific force minus gravity projection
        val accelBias = doubleArrayOf(0.0, 0.0, 0.0) // Zero default or residual bias

        status = CalibrationStatus.SUCCESS
        return CalibrationResult(
            status, accelXList.size,
            accelBias, gyroBias, q,
            totalAccelVar, totalGyroVar,
            "Stationary calibration successful (gyro_bias=[${String.format("%.4f", meanGx)}, ${String.format("%.4f", meanGy)}, ${String.format("%.4f", meanGz)}])"
        )
    }

    private fun computeMeanAndVariance(values: List<Double>): Pair<Double, Double> {
        if (values.isEmpty()) return Pair(0.0, 0.0)
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return Pair(mean, variance)
    }

    private fun eulerToQuat(roll: Double, pitch: Double, yaw: Double): DoubleArray {
        val cr = Math.cos(roll * 0.5); val sr = Math.sin(roll * 0.5)
        val cp = Math.cos(pitch * 0.5); val sp = Math.sin(pitch * 0.5)
        val cy = Math.cos(yaw * 0.5); val sy = Math.sin(yaw * 0.5)

        val qw = cr * cp * cy + sr * sp * sy
        val qx = sr * cp * cy - cr * sp * sy
        val qy = cr * sp * cy + sr * cp * sy
        val qz = cr * cp * sy - sr * sp * cy

        val norm = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        return doubleArrayOf(qw / norm, qx / norm, qy / norm, qz / norm)
    }
}
