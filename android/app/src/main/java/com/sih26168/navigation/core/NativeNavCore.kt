package com.sih26168.navigation.core

import java.io.Closeable

class NativeNavCore(gVal: Double = 9.80665) : Closeable {

    private var nativeHandle: Long = 0

    init {
        try {
            System.loadLibrary("navigation_core")
        } catch (e: UnsatisfiedLinkError) {
            // Handled when running on JVM without NDK library
        }
        nativeHandle = nativeCreate(gVal)
    }

    val isCreated: Boolean
        get() = nativeHandle != 0L

    fun initialize(
        initialTime: Double,
        latDeg: Double, lonDeg: Double, altM: Double,
        velEast: Double, velNorth: Double, velUp: Double,
        qw: Double, qx: Double, qy: Double, qz: Double,
        abx: Double = 0.0, aby: Double = 0.0, abz: Double = 0.0,
        gbx: Double = 0.0, gby: Double = 0.0, gbz: Double = 0.0
    ): Boolean {
        if (nativeHandle == 0L) return false
        val res = nativeInitialize(
            nativeHandle, initialTime, latDeg, lonDeg, altM,
            velEast, velNorth, velUp,
            qw, qx, qy, qz,
            abx, aby, abz,
            gbx, gby, gbz
        )
        return res == 0
    }

    fun processIMU(
        timestamp: Double,
        ax: Double, ay: Double, az: Double,
        gx: Double, gy: Double, gz: Double
    ): Boolean {
        if (nativeHandle == 0L) return false
        return nativeProcessIMU(nativeHandle, timestamp, ax, ay, az, gx, gy, gz) == 0
    }

    fun processGNSS(
        timestamp: Double,
        lat: Double, lon: Double, alt: Double,
        hAcc: Double = -1.0, vAcc: Double = -1.0,
        hdop: Double = -1.0, cn0: Double = -1.0
    ): Boolean {
        if (nativeHandle == 0L) return false
        return nativeProcessGNSS(nativeHandle, timestamp, lat, lon, alt, hAcc, vAcc, hdop, cn0) == 0
    }

    fun processAIDisplacement(
        refEast: Double, refNorth: Double, refUp: Double,
        deltaEast: Double, deltaNorth: Double, deltaUp: Double,
        rEast: Double = 1.5, rNorth: Double = 1.5, rUp: Double = 3.0
    ): Pair<Boolean, Double> {
        if (nativeHandle == 0L) return Pair(false, 0.0)
        val acceptedArr = IntArray(1)
        val mahalanobisArr = DoubleArray(1)

        nativeProcessAIDisplacement(
            nativeHandle,
            refEast, refNorth, refUp,
            deltaEast, deltaNorth, deltaUp,
            rEast, rNorth, rUp,
            acceptedArr,
            mahalanobisArr
        )
        return Pair(acceptedArr[0] == 1, mahalanobisArr[0])
    }

    fun getNavigationState(): NavState? {
        if (nativeHandle == 0L) return null
        val dBuf = DoubleArray(16)
        val iBuf = IntArray(1)

        val res = nativeGetState(nativeHandle, dBuf, iBuf)
        if (res != 0) return null

        return NavState(
            timestamp = dBuf[0],
            latitude = dBuf[1],
            longitude = dBuf[2],
            altitude = dBuf[3],
            east = dBuf[4],
            north = dBuf[5],
            up = dBuf[6],
            velEast = dBuf[7],
            velNorth = dBuf[8],
            velUp = dBuf[9],
            qw = dBuf[10],
            qx = dBuf[11],
            qy = dBuf[12],
            qz = dBuf[13],
            headingDeg = dBuf[14],
            confidenceMeters = dBuf[15],
            mode = NavigationMode.fromInt(iBuf[0])
        )
    }

    fun extractAIFeatures(outBuffer: FloatArray): Int {
        if (nativeHandle == 0L) return 0
        return nativeExtractAIFeatures(nativeHandle, outBuffer, outBuffer.size)
    }

    fun reset(): Boolean {
        if (nativeHandle == 0L) return false
        return nativeReset(nativeHandle) == 0
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0L
        }
    }

    // Native JNI functions
    private external fun nativeCreate(gVal: Double): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeInitialize(
        handle: Long,
        initialTime: Double,
        latDeg: Double, lonDeg: Double, altM: Double,
        vE: Double, vN: Double, vU: Double,
        qw: Double, qx: Double, qy: Double, qz: Double,
        abx: Double, aby: Double, abz: Double,
        gbx: Double, gby: Double, gbz: Double
    ): Int

    private external fun nativeProcessIMU(
        handle: Long,
        timestamp: Double,
        ax: Double, ay: Double, az: Double,
        gx: Double, gy: Double, gz: Double
    ): Int

    private external fun nativeProcessGNSS(
        handle: Long,
        timestamp: Double,
        lat: Double, lon: Double, alt: Double,
        hAcc: Double, vAcc: Double,
        hdop: Double, cn0: Double
    ): Int

    private external fun nativeProcessAIDisplacement(
        handle: Long,
        refE: Double, refN: Double, refU: Double,
        dE: Double, dN: Double, dU: Double,
        rE: Double, rN: Double, rU: Double,
        outAccepted: IntArray,
        outMahalanobis: DoubleArray
    ): Int

    private external fun nativeGetState(
        handle: Long,
        outDoubles: DoubleArray,
        outInts: IntArray
    ): Int

    private external fun nativeExtractAIFeatures(
        handle: Long,
        outBuffer: FloatArray,
        maxFloats: Int
    ): Int

    private external fun nativeReset(handle: Long): Int
}
