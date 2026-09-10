package com.sih26168.navigation.core

enum class NavigationMode {
    GOOD,
    DEGRADED,
    OUTAGE,
    RECOVERING;

    companion object {
        fun fromInt(value: Int): NavigationMode = when (value) {
            0 -> GOOD
            1 -> DEGRADED
            2 -> OUTAGE
            3 -> RECOVERING
            else -> GOOD
        }
    }
}

data class NavState(
    val timestamp: Double,
    val latitude: Double,
    val longitude: Double,
    val altitude: Double,
    val east: Double,
    val north: Double,
    val up: Double,
    val velEast: Double,
    val velNorth: Double,
    val velUp: Double,
    val qw: Double,
    val qx: Double,
    val qy: Double,
    val qz: Double,
    val headingDeg: Double,
    val mode: NavigationMode,
    val confidenceMeters: Double
) {
    val speed: Double
        get() = Math.sqrt(velEast * velEast + velNorth * velNorth)
}
