package com.sih26168.navigation.ui

import android.content.Context
import android.graphics.Color
import android.hardware.SensorManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.sih26168.navigation.R
import com.sih26168.navigation.calibration.CalibrationStatus
import com.sih26168.navigation.calibration.LiveCalibrationManager
import com.sih26168.navigation.core.NativeNavCore
import com.sih26168.navigation.core.NavigationMode
import com.sih26168.navigation.logging.OfflineLogger
import com.sih26168.navigation.sensors.SensorAdapter
import kotlinx.coroutines.*
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var navCore: NativeNavCore
    private lateinit var sensorAdapter: SensorAdapter
    private val calibrationManager = LiveCalibrationManager()
    private var logger: OfflineLogger? = null

    private lateinit var tvNavMode: TextView
    private lateinit var tvImuRate: TextView
    private lateinit var tvCoords: TextView
    private lateinit var tvMotion: TextView
    private lateinit var trajectoryView: TrajectoryView
    private lateinit var btnCalibrate: Button
    private lateinit var btnToggleOutage: Button

    private val uiScope = CoroutineScope(Dispatchers.Main + Job())
    private var isSimulatedOutageActive = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvNavMode = findViewById(R.id.tvNavMode)
        tvImuRate = findViewById(R.id.tvImuRate)
        tvCoords = findViewById(R.id.tvCoords)
        tvMotion = findViewById(R.id.tvMotion)
        trajectoryView = findViewById(R.id.trajectoryView)
        btnCalibrate = findViewById(R.id.btnCalibrate)
        btnToggleOutage = findViewById(R.id.btnToggleOutage)

        navCore = NativeNavCore()
        sensorAdapter = SensorAdapter(navCore)

        // Initialize default reference origin
        navCore.initialize(
            initialTime = System.currentTimeMillis() / 1000.0,
            latDeg = 12.9716, lonDeg = 77.5946, altM = 920.0,
            velEast = 0.0, velNorth = 0.0, velUp = 0.0,
            qw = 1.0, qx = 0.0, qy = 0.0, qz = 0.0
        )

        // Initialize Offline Logger in app external files directory
        val logFile = File(getExternalFilesDir(null), "live_session_${System.currentTimeMillis()}.csv")
        logger = OfflineLogger(logFile)

        setupListeners()
        startSensorStreaming()
        startUiUpdateLoop()
    }

    private fun setupListeners() {
        btnCalibrate.setOnClickListener {
            calibrationManager.start()
            Toast.makeText(this, "Keep device stationary for 3 seconds...", Toast.LENGTH_SHORT).show()
        }

        btnToggleOutage.setOnClickListener {
            isSimulatedOutageActive = !isSimulatedOutageActive
            sensorAdapter.isOutageSimulated = isSimulatedOutageActive
            if (isSimulatedOutageActive) {
                btnToggleOutage.text = "Restore GNSS"
                btnToggleOutage.setBackgroundColor(Color.parseColor("#388E3C"))
            } else {
                btnToggleOutage.text = "Simulate Outage"
                btnToggleOutage.setBackgroundColor(Color.parseColor("#D32F2F"))
            }
        }
    }

    private fun startSensorStreaming() {
        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sensorAdapter.register(sensorManager)
    }

    private fun startUiUpdateLoop() {
        uiScope.launch {
            while (isActive) {
                delay(100) // 10 Hz UI refresh

                val state = navCore.getNavigationState() ?: continue

                // Update Badges & Text
                tvNavMode.text = state.mode.name
                when (state.mode) {
                    NavigationMode.GOOD -> tvNavMode.setBackgroundColor(Color.parseColor("#2E7D32"))
                    NavigationMode.DEGRADED -> tvNavMode.setBackgroundColor(Color.parseColor("#F57C00"))
                    NavigationMode.OUTAGE -> tvNavMode.setBackgroundColor(Color.parseColor("#C62828"))
                    NavigationMode.RECOVERING -> tvNavMode.setBackgroundColor(Color.parseColor("#0288D1"))
                }

                tvImuRate.text = String.format(
                    "IMU: %.1f Hz (Samples: %d) | Gaps: %d",
                    sensorAdapter.measuredImuRateHz,
                    sensorAdapter.imuSampleCount,
                    sensorAdapter.timestampGapCount
                )

                tvCoords.text = String.format("%.6f° N\n%.6f° E\nAlt: %.1f m", state.latitude, state.longitude, state.altitude)
                tvMotion.text = String.format("Speed: %.1f m/s\nHeading: %.1f°\nConf: ±%.1f m", state.speed, state.headingDeg, state.confidenceMeters)

                // Update 2D Trajectory
                trajectoryView.addPoint(state.east, state.north, isGnss = false)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        uiScope.cancel()
        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sensorAdapter.unregister(sensorManager)
        logger?.close()
        navCore.close()
    }
}
