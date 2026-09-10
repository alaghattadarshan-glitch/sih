package com.sih26168.navigation.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.SensorManager
import android.location.LocationManager
import android.os.Build
import android.os.IBinder
import com.sih26168.navigation.core.NavState
import com.sih26168.navigation.core.NativeNavCore
import com.sih26168.navigation.inference.OnnxInferenceEngine
import com.sih26168.navigation.sensors.SensorAdapter
import kotlinx.coroutines.*

class NavigationService : Service() {

    private lateinit var navCore: NativeNavCore
    private var aiEngine: OnnxInferenceEngine? = null
    private lateinit var sensorAdapter: SensorAdapter
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private var isRunning = false

    override fun onCreate() {
        super.onCreate()
        navCore = NativeNavCore()
        sensorAdapter = SensorAdapter(navCore)

        // Load AI model from assets if present
        try {
            assets.open("model.onnx").use { input ->
                val bytes = input.readBytes()
                aiEngine = OnnxInferenceEngine(bytes)
            }
        } catch (e: Exception) {
            // Model loading fallback
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundServiceNotification()
        startNavigation()
        return START_STICKY
    }

    private fun startNavigation() {
        if (isRunning) return
        isRunning = true

        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sensorAdapter.register(sensorManager)

        // Periodic AI pseudo-measurement background job (1 Hz during outages)
        serviceScope.launch {
            val featureBuffer = FloatArray(800)
            while (isActive && isRunning) {
                delay(1000)
                val state = navCore.getNavigationState() ?: continue
                if (state.mode == com.sih26168.navigation.core.NavigationMode.OUTAGE && aiEngine != null) {
                    val count = navCore.extractAIFeatures(featureBuffer)
                    if (count == 100) {
                        val disp = aiEngine!!.predictDisplacement(featureBuffer)
                        navCore.processAIDisplacement(
                            state.east - disp[0], state.north - disp[1], state.up - disp[2],
                            disp[0].toDouble(), disp[1].toDouble(), disp[2].toDouble()
                        )
                    }
                }
            }
        }
    }

    private fun startForegroundServiceNotification() {
        val channelId = "navigation_channel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "Dead Reckoning Navigation Service",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }

        val notification = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, channelId)
                .setContentTitle("Dead Reckoning Active")
                .setContentText("100 Hz IMU + AI Sensor Fusion running")
                .setSmallIcon(android.R.drawable.ic_menu_compass)
                .build()
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
                .setContentTitle("Dead Reckoning Active")
                .setContentText("100 Hz IMU + AI Sensor Fusion running")
                .setSmallIcon(android.R.drawable.ic_menu_compass)
                .build()
        }

        startForeground(1, notification)
    }

    override fun onDestroy() {
        super.onDestroy()
        isRunning = false
        serviceScope.cancel()
        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sensorAdapter.unregister(sensorManager)
        aiEngine?.close()
        navCore.close()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
