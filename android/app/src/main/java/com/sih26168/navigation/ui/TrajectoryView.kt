package com.sih26168.navigation.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

class TrajectoryView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val pathPaint = Paint().apply {
        color = Color.CYAN
        strokeWidth = 5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val gnssPaint = Paint().apply {
        color = Color.YELLOW
        strokeWidth = 3f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val currentPosPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val gridPaint = Paint().apply {
        color = Color.DKGRAY
        strokeWidth = 1f
        style = Paint.Style.STROKE
    }

    private val deadReckonPoints = mutableListOf<Pair<Float, Float>>() // East, North
    private val gnssPoints = mutableListOf<Pair<Float, Float>>()

    fun addPoint(east: Double, north: Double, isGnss: Boolean = false) {
        val pt = Pair(east.toFloat(), north.toFloat())
        if (isGnss) {
            gnssPoints.add(pt)
            if (gnssPoints.size > 2000) gnssPoints.removeAt(0)
        } else {
            deadReckonPoints.add(pt)
            if (deadReckonPoints.size > 5000) deadReckonPoints.removeAt(0)
        }
        postInvalidate()
    }

    fun clear() {
        deadReckonPoints.clear()
        gnssPoints.clear()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.rgb(18, 22, 28)) // Dark dashboard background

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0 || deadReckonPoints.isEmpty()) {
            // Draw crosshairs
            canvas.drawLine(w / 2, 0f, w / 2, h, gridPaint)
            canvas.drawLine(0f, h / 2, w, h / 2, gridPaint)
            return
        }

        // Compute bounding box
        var minE = Float.MAX_VALUE; var maxE = -Float.MAX_VALUE
        var minN = Float.MAX_VALUE; var maxN = -Float.MAX_VALUE

        for (p in deadReckonPoints) {
            if (p.first < minE) minE = p.first
            if (p.first > maxE) maxE = p.first
            if (p.second < minN) minN = p.second
            if (p.second > maxN) maxN = p.second
        }

        val rangeE = (maxE - minE).coerceAtLeast(20f)
        val rangeN = (maxN - minN).coerceAtLeast(20f)
        val span = maxOf(rangeE, rangeN) * 1.2f

        val centerE = (minE + maxE) / 2f
        val centerN = (minN + maxN) / 2f

        val scale = minOf(w, h) / span

        fun toScreenX(east: Float): Float = w / 2f + (east - centerE) * scale
        fun toScreenY(north: Float): Float = h / 2f - (north - centerN) * scale // Y is inverted

        // Draw grid
        canvas.drawLine(w / 2, 0f, w / 2, h, gridPaint)
        canvas.drawLine(0f, h / 2, w, h / 2, gridPaint)

        // Draw GNSS Path
        if (gnssPoints.size > 1) {
            val gPath = Path()
            gPath.moveTo(toScreenX(gnssPoints[0].first), toScreenY(gnssPoints[0].second))
            for (i in 1 until gnssPoints.size) {
                gPath.lineTo(toScreenX(gnssPoints[i].first), toScreenY(gnssPoints[i].second))
            }
            canvas.drawPath(gPath, gnssPaint)
        }

        // Draw Dead Reckoned Path
        if (deadReckonPoints.size > 1) {
            val drPath = Path()
            drPath.moveTo(toScreenX(deadReckonPoints[0].first), toScreenY(deadReckonPoints[0].second))
            for (i in 1 until deadReckonPoints.size) {
                drPath.lineTo(toScreenX(deadReckonPoints[i].first), toScreenY(deadReckonPoints[i].second))
            }
            canvas.drawPath(drPath, pathPaint)
        }

        // Draw current position marker
        val lastPt = deadReckonPoints.last()
        canvas.drawCircle(toScreenX(lastPt.first), toScreenY(lastPt.second), 10f, currentPosPaint)
    }
}
