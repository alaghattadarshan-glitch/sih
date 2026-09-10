package com.sih26168.navigation.logging

import com.sih26168.navigation.core.NavState
import java.io.BufferedWriter
import java.io.Closeable
import java.io.File
import java.io.FileWriter
import java.util.concurrent.ConcurrentLinkedQueue

class OfflineLogger(
    private val outputFile: File
) : Closeable {

    private val queue = ConcurrentLinkedQueue<String>()
    private var writer: BufferedWriter? = null
    private var isLogging = false

    init {
        outputFile.parentFile?.mkdirs()
        writer = BufferedWriter(FileWriter(outputFile, false))
        writeHeader()
        isLogging = true
    }

    private fun writeHeader() {
        val header = "timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z," +
                "gnss_lat,gnss_lon,gnss_alt,gnss_hacc," +
                "pos_east,pos_north,pos_up,vel_east,vel_north,vel_up,heading_deg," +
                "nav_mode,ai_de,ai_dn,ai_du,ai_accepted,ai_mahalanobis,confidence_m\n"
        writer?.write(header)
        writer?.flush()
    }

    fun logSample(
        timestamp: Double,
        ax: Double, ay: Double, az: Double,
        gx: Double, gy: Double, gz: Double,
        gnssLat: Double = 0.0, gnssLon: Double = 0.0, gnssAlt: Double = 0.0, gnssHacc: Double = -1.0,
        state: NavState? = null,
        aiDe: Double = 0.0, aiDn: Double = 0.0, aiDu: Double = 0.0,
        aiAccepted: Boolean = false, aiMahalanobis: Double = 0.0
    ) {
        if (!isLogging) return

        val pE = state?.east ?: 0.0
        val pN = state?.north ?: 0.0
        val pU = state?.up ?: 0.0
        val vE = state?.velEast ?: 0.0
        val vN = state?.velNorth ?: 0.0
        val vU = state?.velUp ?: 0.0
        val heading = state?.headingDeg ?: 0.0
        val mode = state?.mode?.name ?: "UNKNOWN"
        val conf = state?.confidenceMeters ?: 0.0

        val line = String.format(
            "%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f," +
            "%.8f,%.8f,%.2f,%.2f," +
            "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f," +
            "%s,%.4f,%.4f,%.4f,%d,%.4f,%.2f\n",
            timestamp, ax, ay, az, gx, gy, gz,
            gnssLat, gnssLon, gnssAlt, gnssHacc,
            pE, pN, pU, vE, vN, vU, heading,
            mode, aiDe, aiDn, aiDu, if (aiAccepted) 1 else 0, aiMahalanobis, conf
        )

        queue.add(line)
        if (queue.size >= 50) {
            flushQueue()
        }
    }

    @Synchronized
    fun flushQueue() {
        val w = writer ?: return
        while (true) {
            val item = queue.poll() ?: break
            w.write(item)
        }
        w.flush()
    }

    override fun close() {
        isLogging = false
        flushQueue()
        writer?.close()
        writer = null
    }
}
