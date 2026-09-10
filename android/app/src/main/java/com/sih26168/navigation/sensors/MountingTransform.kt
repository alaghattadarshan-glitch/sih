package com.sih26168.navigation.sensors

/**
 * Configurable Device-to-Body Coordinate Frame Transformation.
 *
 * Canonical Project Body Frame Convention:
 *   +x : Forward (Vehicle driving direction)
 *   +y : Left (Vehicle left door direction)
 *   +z : Up (Perpendicular to road towards roof)
 *
 * Standard Android Sensor Frame Convention:
 *   +x : Right (Short edge to the right)
 *   +y : Up (Long edge towards top of screen)
 *   +z : Out of screen (Perpendicular out towards user)
 */
class MountingTransform(
    private val rMatrix: DoubleArray = PRESET_FLAT_PORTRAIT
) {
    init {
        require(rMatrix.size == 9) { "Rotation matrix must contain exactly 9 elements (3x3)." }
    }

    /**
     * Rotate raw 3D Android sensor vector [sx, sy, sz] into canonical vehicle body frame [bx, by, bz].
     */
    fun transform(sx: Double, sy: Double, sz: Double): Triple<Double, Double, Double> {
        val bx = rMatrix[0] * sx + rMatrix[1] * sy + rMatrix[2] * sz
        val by = rMatrix[3] * sx + rMatrix[4] * sy + rMatrix[5] * sz
        val bz = rMatrix[6] * sx + rMatrix[7] * sy + rMatrix[8] * sz
        return Triple(bx, by, bz)
    }

    companion object {
        /**
         * Preset: Phone lying flat on console/seat, top pointing Forward.
         * Phone +y = Body +x, Phone -x = Body +y, Phone +z = Body +z.
         */
        val PRESET_FLAT_PORTRAIT = doubleArrayOf(
            0.0,  1.0,  0.0,
           -1.0,  0.0,  0.0,
            0.0,  0.0,  1.0
        )

        /**
         * Preset: Phone mounted on windshield / dashboard vertically, facing driver.
         * Phone +z = Body +x (screen faces back, back camera faces forward),
         * Phone -x = Body +y, Phone +y = Body +z.
         */
        val PRESET_WINDSHIELD_VERTICAL = doubleArrayOf(
            0.0,  0.0,  1.0,
           -1.0,  0.0,  0.0,
            0.0,  1.0,  0.0
        )

        /**
         * Preset: Phone lying flat in landscape mode (top pointing Left).
         * Phone +x = Body +x, Phone +y = Body +y, Phone +z = Body +z.
         */
        val PRESET_FLAT_LANDSCAPE = doubleArrayOf(
            1.0,  0.0,  0.0,
            0.0,  1.0,  0.0,
            0.0,  0.0,  1.0
        )

        /**
         * Create custom rotation matrix from Euler angles (in degrees: roll, pitch, yaw).
         */
        fun fromEulerDeg(rollDeg: Double, pitchDeg: Double, yawDeg: Double): MountingTransform {
            val r = Math.toRadians(rollDeg)
            val p = Math.toRadians(pitchDeg)
            val y = Math.toRadians(yawDeg)

            val cr = Math.cos(r); val sr = Math.sin(r)
            val cp = Math.cos(p); val sp = Math.sin(p)
            val cy = Math.cos(y); val sy = Math.sin(y)

            val mat = doubleArrayOf(
                cp * cy,  cp * sy, -sp,
                sr * sp * cy - cr * sy,  sr * sp * sy + cr * cy,  sr * cp,
                cr * sp * cy + sr * sy,  cr * sp * sy - sr * cy,  cr * cp
            )
            return MountingTransform(mat)
        }
    }
}
