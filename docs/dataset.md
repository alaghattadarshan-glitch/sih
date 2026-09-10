# Dataset Specification & Data Management

## Data Organization

- `data/raw/`: Original uncompressed sensor datasets (IMU logs, GNSS NMEA/RTCM, ground truth reference trajectories).
- `data/processed/`: Synchronized, calibrated, and windowed tensors ready for model training and evaluation.
- `data/sample/`: Small sample sensor captures for integration testing and unit testing.

## Coordinate Systems & Units Standardization

All spatial data within the repository adheres to strict unit conventions:

| Quantity | Unit | Representation / Boundaries |
|----------|------|-----------------------------|
| Latitude / Longitude | Degrees | `[-90.0, +90.0]`, `[-180.0, +180.0]` at API boundaries |
| Internal Angles | Radians | Trigonometric equations |
| Height / Altitude | Meters | WGS84 Ellipsoidal Height |
| Local Position (ENU) | Meters | East, North, Up relative to reference origin |
| Velocity | m/s | East, North, Up components |
| Acceleration | m/s² | Accel X, Y, Z sensor axes |

## Data Schemas

### IMU Sensor Schema
| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `timestamp` | float64 | seconds | Epoch timestamp (UTC) |
| `accel_x` | float64 | m/s^2 | Accelerometer X-axis |
| `accel_y` | float64 | m/s^2 | Accelerometer Y-axis |
| `accel_z` | float64 | m/s^2 | Accelerometer Z-axis |
| `gyro_x` | float64 | rad/s | Gyroscope X-axis |
| `gyro_y` | float64 | rad/s | Gyroscope Y-axis |
| `gyro_z` | float64 | rad/s | Gyroscope Z-axis |

### GNSS Schema
| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `timestamp` | float64 | seconds | Epoch timestamp |
| `latitude` | float64 | degrees | WGS84 Latitude |
| `longitude` | float64 | degrees | WGS84 Longitude |
| `altitude` | float64 | meters | WGS84 Ellipsoidal Height |
| `accuracy` | float64 | meters | Horizontal Accuracy Estimate |

## ML Window Dataset & Target Schemas

### Feature Window Schema `[N_windows, 100, 8]`
Each sample window spans 1.0 second (100 samples at 100Hz) with 0.5 second stride:

| Feature Column | Type | Unit | Description |
|----------------|------|------|-------------|
| `accel_x` | float32 | m/s² | Body acceleration X |
| `accel_y` | float32 | m/s² | Body acceleration Y |
| `accel_z` | float32 | m/s² | Body acceleration Z |
| `gyro_x` | float32 | rad/s | Angular rate X |
| `gyro_y` | float32 | rad/s | Angular rate Y |
| `gyro_z` | float32 | rad/s | Angular rate Z |
| `accel_norm` | float32 | m/s² | Euclidean magnitude $||\mathbf{a}||_2$ |
| `gyro_norm` | float32 | rad/s | Euclidean magnitude $||\boldsymbol{\omega}||_2$ |

### Supervised Motion Target Schema `[N_windows, 3]`
Used ONLY for training and evaluation. NEVER passed into live navigation input.

| Target Column | Type | Unit | Description |
|---------------|------|------|-------------|
| `delta_p_east` | float32 | meters | True East displacement $\Delta p_{east}$ over window |
| `delta_p_north` | float32 | meters | True North displacement $\Delta p_{north}$ over window |
| `delta_p_up` | float32 | meters | True Up displacement $\Delta p_{up}$ over window |

## Trajectory-Aware Splitting & Anti-Leakage Protocol

> [!IMPORTANT]
> Overlapping temporal windows sliced from the same continuous motion trajectory share strong correlations. Random window shuffling creates severe data leakage between train and test sets.
> All splits must be partitioned strictly by session/trajectory ID:
> - `Train Set`: Trajectory sessions 02, 03, 05 (60%)
> - `Val Set`: Trajectory session 01 (20%)
> - `Test Set`: Trajectory session 04 (20%)
> Zero trajectory IDs overlap across splits ($ID_{train} \cap ID_{val} \cap ID_{test} = \emptyset$).

## Real-Data Ingestion & Adapter Layer (Step 10)

To transition from synthetic-only validation to real smartphone and vehicular recordings, a provider-independent adapter layer isolates the navigation core from dataset-specific quirks, column names, units, and mounting orientations.

```
Raw Real Data (CSV / IO-VNBD / Sensor Logger)
               │
               ▼
      [ Dataset Adapter ] ──► Unit Normalization (s/ms/μs/ns, g, deg/s)
               │          ──► Axis Transformation (R_device->body ∈ SO(3))
               ▼
  [ Timestamp & Quality Audit ] ──► Gap / Non-monotonic / NaN Detection
               │
               ▼
  [ Initial Calibration Check ] ──► Stationary Variance & Bias Audit
               │
               ▼
   Canonical Observations (IMUObservation, GNSSObservation, GroundTruthObservation)
               │
               ▼
   Navigation Fusion Pipeline (INS / ESKF / AI / Map Matching)
```

### Canonical Units

| Quantity | Canonical Unit | Supported Input Units |
|---|---|---|
| Timestamps | Seconds ($s$) | `seconds`, `milliseconds`, `microseconds`, `nanoseconds` |
| Accelerometer | $\text{m/s}^2$ | $\text{m/s}^2$, $g$ ($1g = 9.80665\,\text{m/s}^2$) |
| Gyroscope | $\text{rad/s}$ | $\text{rad/s}$, $\text{deg/s}$ ($1^\circ/\text{s} = \pi/180\,\text{rad/s}$) |
| GNSS Coordinates | Decimal Degrees | Decimal degrees (WGS84 ellipsoid) |
| Altitude / Height | Meters ($m$) | Ellipsoidal height above WGS84 datum |

### Sensor Axis & Body Frame Convention

The project's canonical vehicle body frame adheres to:
- **X-Axis ($+x$)**: Forward (longitudinal direction of travel)
- **Y-Axis ($+y$)**: Left (lateral direction)
- **Z-Axis ($+z$)**: Up (vertical / normal to road plane)

Real smartphone recordings mounted in arbitrary orientations are mapped to the canonical body frame via an orthogonal signed-permutation matrix $\mathbf{R} \in SO(3)$:
$$\begin{bmatrix} a_x \\ a_y \\ a_z \end{bmatrix}_{body} = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & s_z \end{bmatrix} \mathbf{P} \begin{bmatrix} a_{x'} \\ a_{y'} \\ a_{z'} \end{bmatrix}_{device}$$
where $\mathbf{P}$ is a valid 3x3 permutation matrix and $s_i \in \{+1, -1\}$ such that $\det(\mathbf{R}) = +1$ (proper rotation preserving right-handedness) or verified orthogonal transform.

### Dataset Quality & Diagnostic Reports

Before running navigation estimators, `src/evaluation/dataset_quality.py` validates recordings:
1. **Timestamp Health**: Monotonicity verification, duplicate timestamp detection, sampling interval statistics (median $\Delta t$, min/max $\Delta t$, frequency).
2. **Data Integrity**: NaN/Inf sample detection and column validation.
3. **Usable Overlap**: Temporal overlap between high-rate IMU and GNSS/ground-truth streams.
4. **Initial Static Calibration Check**: Evaluates an explicitly configured stationary window ($t_{start} \to t_{start} + \Delta t$). Confirms sensor variance $\sigma_a \le \sigma_{a,max}$ and $\sigma_\omega \le \sigma_{\omega,max}$ before computing initial accelerometer and gyroscope zero-bias offsets. If motion is detected, a non-fatal `WARNING` is generated and uncalibrated raw data proceeds safely.

### IO-VNBD Benchmark Schema Standby

The `IOVNBDAdapter` (`src/data/adapters/io_vnbd.py`) and config `config/datasets/io_vnbd.yaml` implement exact schema compatibility for the IO-VNBD dataset. When IO-VNBD files are provided in `data/raw/io_vnbd/`, the adapter automatically parses the synchronized session streams. When absent, the adapter operates in validated standby mode and provides clear configuration requirements without fabricating benchmark data.

## Multi-Session Architecture & Session Isolation (Step 11)

To evaluate navigation performance and AI model generalization across independent driving sessions, the system implements a strict multi-session abstraction:

- **`DatasetSession`**: Encapsulates a single, continuous trajectory recording with its own IMU, GNSS, ground-truth streams, local geodetic origin, and quality audit report.
- **Strict Session Isolation**: Independent drives are never concatenated into a single trajectory. Each session maintains independent timestamps, initial orientation, and filter state.
- **Source Classification**: Every session is explicitly categorized into `REAL`, `SYNTHETIC`, `FIXTURE`, or `UNKNOWN`.
- **Leave-One-Session-Out (LOSO) Partitioning**: `MultiSessionDatasetBuilder` enforces strict trajectory-level partitioning for model training ($ID_{train} \cap ID_{val} \cap ID_{test} = \emptyset$). Zero feature window shuffling across sessions is permitted.
- **Domain-Shift Analysis**: Statistical comparisons (accelerometer mean/std, gyroscope mean/std, acceleration norm, sampling interval, displacement target distributions) evaluate distribution shifts between training and unseen test drives.



