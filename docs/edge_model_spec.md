# Edge Model Specification & Production Contract

## 1. Model Overview

The Edge AI Displacement Predictor is a Temporal Convolutional Network (TCN) trained to predict 3D positional displacement over a sliding 1-second causal IMU window during GPS-denied intervals.

- **Model Type**: Temporal Convolutional Network (`TCNDriftModel`)
- **Execution Target**: ONNX Runtime Mobile / Android NDK C++
- **Checkpoint Source**: `results/ml_training/best_drift_model.pt`
- **Export Destination**: `results/edge_model/model.onnx`

---

## 2. Input / Output Tensor Specification

### Input Tensor
- **Tensor Name**: `imu_features`
- **Data Type**: `float32`
- **Shape**: `[batch, 100, 8]`
  - `batch`: Dynamic batch size (typically 1 during live streaming)
  - `100`: 100 consecutive IMU samples @ 100 Hz (1.0 second duration)
  - `8`: Feature dimension per timestep

#### Feature Ordering (Strict Contract)
1. `accel_x` — Specific force X (body forward, $\text{m/s}^2$)
2. `accel_y` — Specific force Y (body left, $\text{m/s}^2$)
3. `accel_z` — Specific force Z (body up, $\text{m/s}^2$)
4. `gyro_x` — Angular velocity X (body roll, $\text{rad/s}$)
5. `gyro_y` — Angular velocity Y (body pitch, $\text{rad/s}$)
6. `gyro_z` — Angular velocity Z (body yaw, $\text{rad/s}$)
7. `accel_norm` — Euclidean norm $\sqrt{a_x^2 + a_y^2 + a_z^2}$ ($\text{m/s}^2$)
8. `gyro_norm` — Euclidean norm $\sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$ ($\text{rad/s}$)

### Output Tensor
- **Tensor Name**: `predicted_displacement`
- **Data Type**: `float32`
- **Shape**: `[batch, 3]`
- **Meaning**: 3D displacement vector in local East-North-Up (ENU) frame over the 1.0-second window:
  - Output Index 0: $\Delta East$ ($\text{meters}$)
  - Output Index 1: $\Delta North$ ($\text{meters}$)
  - Output Index 2: $\Delta Up$ ($\text{meters}$)

---

## 3. Frozen Normalization Contract

The edge engine must normalize each raw input feature vector $\mathbf{x} \in \mathbb{R}^8$ using the **exact frozen training statistics** before feeding the tensor into the neural network:

$$\hat{x}_i = \frac{x_i - \mu_i}{\sigma_i}$$

### Frozen Training Statistics (Version 1.0.0)
| Feature Index | Feature Name | Mean ($\mu$) | Std ($\sigma$) | Unit |
|---|---|---|---|---|
| 0 | `accel_x` | `0.01783962` | `0.40961212` | $\text{m/s}^2$ |
| 1 | `accel_y` | `-0.01026197` | `0.05008770` | $\text{m/s}^2$ |
| 2 | `accel_z` | `9.83632946` | `0.04994510` | $\text{m/s}^2$ |
| 3 | `gyro_x` | `0.00101370` | `0.00501095` | $\text{rad/s}$ |
| 4 | `gyro_y` | `-0.00098612` | `0.00499533` | $\text{rad/s}$ |
| 5 | `gyro_z` | `-0.00218364` | `0.01469043` | $\text{rad/s}$ |
| 6 | `accel_norm` | `9.84495258` | `0.05331007` | $\text{m/s}^2$ |
| 7 | `gyro_norm` | `0.01163642` | `0.01171466` | $\text{rad/s}$ |

> [!IMPORTANT]
> The edge engine must **never** recalculate mean or variance on runtime test data. All edge deployments must load `normalization.json` directly.
