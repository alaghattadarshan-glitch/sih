package com.sih26168.navigation.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.nio.FloatBuffer

class OnnxInferenceEngine(
    modelBytes: ByteArray
) : Closeable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // Frozen Training Normalization Statistics (Version 1.0.0)
    private val featureMean = floatArrayOf(
        0.01783962f, -0.01026197f, 9.83632946f,
        0.00101370f, -0.00098612f, -0.00218364f,
        9.84495258f, 0.01163642f
    )

    private val featureStd = floatArrayOf(
        0.40961212f, 0.05008770f, 0.04994510f,
        0.00501095f, 0.00499533f, 0.01469043f,
        0.05331007f, 0.01171466f
    )

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
        }
        session = env.createSession(modelBytes, options)
    }

    /**
     * Run TCN inference on a 100x8 raw IMU feature buffer.
     * @param rawFeatures 800 floats: 100 samples x 8 features [accel_xyz, gyro_xyz, norm_acc, norm_gyro]
     * @return FloatArray of size 3: [deltaEast, deltaNorth, deltaUp] in meters
     */
    fun predictDisplacement(rawFeatures: FloatArray): FloatArray {
        if (rawFeatures.size < 800) {
            return floatArrayOf(0f, 0f, 0f)
        }

        // Apply frozen normalization in-place on normalized buffer
        val normBuffer = FloatBuffer.allocate(800)
        for (i in 0 until 100) {
            for (f in 0 until 8) {
                val idx = i * 8 + f
                val normalized = (rawFeatures[idx] - featureMean[f]) / featureStd[f]
                normBuffer.put(normalized)
            }
        }
        normBuffer.rewind()

        val shape = longArrayOf(1, 100, 8)
        val tensor = OnnxTensor.createTensor(env, normBuffer, shape)

        val inputs = mapOf("imu_features" to tensor)
        val results = session.run(inputs)
        val outputTensor = results[0] as OnnxTensor
        val outputBuffer = outputTensor.floatBuffer

        val displacement = FloatArray(3)
        outputBuffer.get(displacement)

        tensor.close()
        results.close()

        return displacement
    }

    override fun close() {
        session.close()
        env.close()
    }
}
