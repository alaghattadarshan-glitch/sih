package com.sih26168.navigation.logging

import com.sih26168.navigation.core.NavState
import java.io.BufferedWriter
import java.io.Closeable
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue

class RealSensorRecorder(
    private val baseDir: File,
    private val metadata: SessionMetadata
) : Closeable {

    private val sessionDir: File = File(baseDir, metadata.sessionId).apply { mkdirs() }
    private val imuWriter: BufferedWriter = BufferedWriter(FileWriter(File(sessionDir, "imu.csv")))
    private val gnssWriter: BufferedWriter = BufferedWriter(FileWriter(File(sessionDir, "gnss.csv")))
    private val navWriter: BufferedWriter = BufferedWriter(FileWriter(File(sessionDir, "navigation_state.csv")))

    private val imuQueue = ConcurrentLinkedQueue<String>()
    private val gnssQueue = ConcurrentLinkedQueue<String>()
    private val navQueue = ConcurrentLinkedQueue<String>()

    private var isRecording = false

    init {
        // Write headers
        imuWriter.write("timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")
        gnssWriter.write("timestamp,latitude,longitude,altitude,horizontal_accuracy,speed,bearing\n")
        navWriter.write("timestamp,pos_east,pos_north,pos_up,vel_east,vel_north,vel_up,heading_deg,mode,confidence_m,ai_accepted,ai_mahalanobis\n")

        flushAll()
        isRecording = true
    }

    fun recordImu(timestampSec: Double, ax: Double, ay: Double, az: Double, gx: Double, gy: Double, gz: Double) {
        if (!isRecording) return
        val line = String.format(Locale.US, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", timestampSec, ax, ay, az, gx, gy, gz)
        imuQueue.add(line)
        metadata.totalImuSamples++
        if (imuQueue.size >= 50) flushImu()
    }

    fun recordGnss(timestampSec: Double, lat: Double, lon: Double, alt: Double, hAcc: Double, speed: Double, bearing: Double) {
        if (!isRecording) return
        val line = String.format(Locale.US, "%.4f,%.8f,%.8f,%.2f,%.2f,%.2f,%.2f\n", timestampSec, lat, lon, alt, hAcc, speed, bearing)
        gnssQueue.add(line)
        metadata.totalGnssFixes++
        if (gnssQueue.size >= 10) flushGnss()
    }

    fun recordNavState(state: NavState, aiAccepted: Boolean = false, aiMahalanobis: Double = 0.0) {
        if (!isRecording) return
        val line = String.format(
            Locale.US,
            "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%s,%.2f,%d,%.4f\n",
            state.timestamp, state.east, state.north, state.up,
            state.velEast, state.velNorth, state.velUp, state.headingDeg,
            state.mode.name, state.confidenceMeters,
            if (aiAccepted) 1 else 0, aiMahalanobis
        )
        navQueue.add(line)
        if (navQueue.size >= 20) flushNav()
    }

    @Synchronized
    private fun flushImu() {
        while (true) {
            val item = imuQueue.poll() ?: break
            imuWriter.write(item)
        }
        imuWriter.flush()
    }

    @Synchronized
    private fun flushGnss() {
        while (true) {
            val item = gnssQueue.poll() ?: break
            gnssWriter.write(item)
        }
        gnssWriter.flush()
    }

    @Synchronized
    private fun flushNav() {
        while (true) {
            val item = navQueue.poll() ?: break
            navWriter.write(item)
        }
        navWriter.flush()
    }

    @Synchronized
    fun flushAll() {
        flushImu()
        flushGnss()
        flushNav()
    }

    override fun close() {
        isRecording = false
        flushAll()
        imuWriter.close()
        gnssWriter.close()
        navWriter.close()

        // Write completed metadata JSON
        val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }
        metadata.endTimeUtc = sdf.format(Date())

        val metaFile = File(sessionDir, "session_metadata.json")
        metaFile.writeText(metadata.toJson())
    }
}
