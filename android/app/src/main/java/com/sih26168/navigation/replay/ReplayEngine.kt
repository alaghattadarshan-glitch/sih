package com.sih26168.navigation.replay

import com.sih26168.navigation.core.NavState
import com.sih26168.navigation.core.NativeNavCore
import com.sih26168.navigation.inference.OnnxInferenceEngine
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader

class ReplayEngine(
    private val navCore: NativeNavCore,
    private val aiEngine: OnnxInferenceEngine? = null
) {
    data class ReplayStats(
        val totalImuSamples: Int,
        val totalGnssFixes: Int,
        val totalAiUpdates: Int,
        val acceptedAiUpdates: Int,
        val rejectedAiUpdates: Int
    )

    fun replayCsv(inputStream: InputStream, onStateUpdate: ((NavState) -> Unit)? = null): ReplayStats {
        val reader = BufferedReader(InputStreamReader(inputStream))
        var header = reader.readLine()?.split(",") ?: return ReplayStats(0, 0, 0, 0, 0)
        val colMap = header.mapIndexed { idx, name -> name.trim() to idx }.toMap()

        var imuCount = 0
        var gnssCount = 0
        var aiCount = 0
        var aiAccepted = 0
        var aiRejected = 0

        val featureBuffer = FloatArray(800)
        var lastAiInferenceTime = 0.0

        var line: String? = reader.readLine()
        while (line != null) {
            val tokens = line.split(",")
            if (tokens.size >= header.size) {
                val t = tokens[colMap["timestamp"] ?: 0].toDoubleOrNull() ?: 0.0
                val ax = tokens[colMap["accel_x"] ?: 1].toDoubleOrNull() ?: 0.0
                val ay = tokens[colMap["accel_y"] ?: 2].toDoubleOrNull() ?: 0.0
                val az = tokens[colMap["accel_z"] ?: 3].toDoubleOrNull() ?: 9.80665
                val gx = tokens[colMap["gyro_x"] ?: 4].toDoubleOrNull() ?: 0.0
                val gy = tokens[colMap["gyro_y"] ?: 5].toDoubleOrNull() ?: 0.0
                val gz = tokens[colMap["gyro_z"] ?: 6].toDoubleOrNull() ?: 0.0

                navCore.processIMU(t, ax, ay, az, gx, gy, gz)
                imuCount++

                // Check GNSS
                if (colMap.containsKey("lat") && colMap.containsKey("lon")) {
                    val lat = tokens[colMap["lat"]!!].toDoubleOrNull()
                    val lon = tokens[colMap["lon"]!!].toDoubleOrNull()
                    val alt = tokens[colMap["alt"] ?: 0].toDoubleOrNull() ?: 0.0
                    val hAcc = tokens[colMap["h_acc"] ?: 0].toDoubleOrNull() ?: 2.5
                    if (lat != null && lon != null && (lat != 0.0 || lon != 0.0)) {
                        navCore.processGNSS(t, lat, lon, alt, hAcc)
                        gnssCount++
                    }
                }

                // Periodic 1-second AI displacement inference during outage
                val state = navCore.getNavigationState()
                if (state != null && aiEngine != null) {
                    if (t - lastAiInferenceTime >= 1.0) {
                        val extracted = navCore.extractAIFeatures(featureBuffer)
                        if (extracted == 100) {
                            val disp = aiEngine.predictDisplacement(featureBuffer)
                            val (accepted, mah) = navCore.processAIDisplacement(
                                state.east - disp[0], state.north - disp[1], state.up - disp[2],
                                disp[0].toDouble(), disp[1].toDouble(), disp[2].toDouble()
                            )
                            aiCount++
                            if (accepted) aiAccepted++ else aiRejected++
                            lastAiInferenceTime = t
                        }
                    }
                    onStateUpdate?.invoke(state)
                }
            }
            line = reader.readLine()
        }

        reader.close()
        return ReplayStats(imuCount, gnssCount, aiCount, aiAccepted, aiRejected)
    }
}
