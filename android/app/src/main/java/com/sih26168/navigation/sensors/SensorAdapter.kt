package com.sih26168.navigation.sensors

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import com.sih26168.navigation.core.NativeNavCore

class SensorAdapter(
    private val navCore: NativeNavCore,
    var mountingTransform: MountingTransform = MountingTransform()
) : SensorEventListener, LocationListener {

    private var lastImuTimestampSec: Double = -1.0
    private var lastAccX = 0.0
    private var lastAccY = 0.0
    private var lastAccZ = 9.80665
    private var hasAccel = false

    // Quality & Frequency Monitoring
    var imuSampleCount: Long = 0
        private set
    var gnssFixCount: Long = 0
        private set
    var duplicateTimestampCount: Long = 0
        private set
    var timestampGapCount: Long = 0
        private set

    private var sampleWindowStartSec: Double = 0.0
    private var samplesInCurrentWindow: Int = 0
    var measuredImuRateHz: Double = 0.0
        private set

    // Software GNSS Outage Simulation Toggle
    var isOutageSimulated: Boolean = false

    fun register(sensorManager: SensorManager) {
        val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        // Request 100 Hz (10,000 microseconds)
        accel?.let { sensorManager.registerListener(this, it, 10000) }
        gyro?.let { sensorManager.registerListener(this, it, 10000) }
    }

    fun unregister(sensorManager: SensorManager) {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        // Single nanosecond -> second conversion
        val timestampSec = event.timestamp / 1_000_000_000.0

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            val (bx, by, bz) = mountingTransform.transform(
                event.values[0].toDouble(),
                event.values[1].toDouble(),
                event.values[2].toDouble()
            )
            lastAccX = bx
            lastAccY = by
            lastAccZ = bz
            hasAccel = true
        } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            if (!hasAccel) return

            val (gx, gy, gz) = mountingTransform.transform(
                event.values[0].toDouble(),
                event.values[1].toDouble(),
                event.values[2].toDouble()
            )

            if (lastImuTimestampSec > 0.0) {
                val dt = timestampSec - lastImuTimestampSec
                if (dt <= 0.0) {
                    duplicateTimestampCount++
                    return // Reject non-monotonic or duplicate timestamp
                }
                if (dt > 0.05) {
                    timestampGapCount++ // Significant sensor gap (>50ms)
                }
            }

            lastImuTimestampSec = timestampSec
            imuSampleCount++
            samplesInCurrentWindow++

            // Live rate tracking
            if (sampleWindowStartSec == 0.0) {
                sampleWindowStartSec = timestampSec
            } else if (timestampSec - sampleWindowStartSec >= 1.0) {
                val elapsed = timestampSec - sampleWindowStartSec
                measuredImuRateHz = samplesInCurrentWindow / elapsed
                samplesInCurrentWindow = 0
                sampleWindowStartSec = timestampSec
            }

            navCore.processIMU(
                timestampSec,
                lastAccX, lastAccY, lastAccZ,
                gx, gy, gz
            )
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onLocationChanged(location: Location) {
        val timestampSec = location.time / 1000.0

        // If software outage simulation is active, suppress GNSS update from filter
        if (isOutageSimulated) {
            // Send missing fix (0, 0, 0) to notify outage detector
            navCore.processGNSS(timestampSec, 0.0, 0.0, 0.0)
            return
        }

        val lat = location.latitude
        val lon = location.longitude
        val alt = location.altitude
        val hAcc = if (location.hasAccuracy()) location.accuracy.toDouble() else -1.0

        gnssFixCount++
        navCore.processGNSS(
            timestampSec,
            lat, lon, alt,
            hAcc = hAcc,
            vAcc = if (location.hasVerticalAccuracy()) location.verticalAccuracyMeters.toDouble() else -1.0
        )
    }
}
