# Algorithms Specification

## SIH26168 Intelligent Dead Reckoning Algorithms

### Implemented Algorithms

#### 1. Geodetic Coordinate Transformations (WGS84)
- **WGS84 Reference Ellipsoid Model**: Centralized parameters ($a = 6378137.0\text{m}$, $1/f = 298.257223563$, $e^2$, $e'^2$, $N(\phi)$).
- **LLH to ECEF**: Closed-form conversion of geodetic latitude, longitude, and height into Cartesian $X, Y, Z$.
- **ECEF to LLH**: Bowring's method with high-precision iterative refinement ($\text{tol} < 10^{-12}\text{ rad}$) handling polar singularities and high altitudes.
- **ECEF to Local ENU**: Local tangent plane transformation using reference origin rotation matrix $R_{ECEF \to ENU}$.
- **ENU to ECEF**: Inverse orthogonal rotation $R^T$ and origin translation.

#### 2. IMU Static Calibration & Filtering
- **Static Bias Estimation**: Estimates static accelerometer bias $\mathbf{b}_a = \bar{\mathbf{f}}_b - [0, 0, g]^T$ and gyroscope bias $\mathbf{b}_g = \bar{\boldsymbol{\omega}}_b$ from stationary observations.
- **Butterworth Low-Pass Filtering**: Zero-phase digital filtering suppressing high-frequency sensor vibration noise without smoothing away true motion dynamics.

#### 3. Quaternion Attitude Propagation & Kinematics
- **Body $\rightarrow$ ENU Rotation Convention**: Unit quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]$ transforms vectors from vehicle body frame into local ENU navigation frame ($\mathbf{v}_{ENU} = R_b^n \mathbf{v}_{body}$).
- **Attitude Propagation**:
  $$\Delta \mathbf{q} = \left[ \cos\frac{|\boldsymbol{\omega}_b|\Delta t}{2}, \frac{\boldsymbol{\omega}_b}{|\boldsymbol{\omega}_b|} \sin\frac{|\boldsymbol{\omega}_b|\Delta t}{2} \right]^T$$
  $$\mathbf{q}_{k+1} = \text{normalize}(\mathbf{q}_k \otimes \Delta \mathbf{q})$$

#### 4. Strapdown INS Mechanization Engine
- **Specific Force Transformation**: $\mathbf{f}_{ENU} = R_b^n(q_{k+1}) \cdot (\mathbf{a}_{raw} - \mathbf{b}_a)$
- **Gravity Compensation**: Net linear acceleration $\mathbf{a}_{ENU} = \mathbf{f}_{ENU} - [0, 0, g]^T$
- **Trapezoidal Integration**:
  $$\mathbf{v}_{k+1} = \mathbf{v}_k + \frac{1}{2}(\mathbf{a}_k + \mathbf{a}_{k+1}) \Delta t$$
  $$\mathbf{p}_{k+1} = \mathbf{p}_k + \frac{1}{2}(\mathbf{v}_k + \mathbf{v}_{k+1}) \Delta t$$

#### 5. 15-State Error-State Kalman Filter (ESKF) GNSS/INS Fusion
- **15-State Error Vector**:
  $$\delta \mathbf{x} = [\delta \mathbf{p}^T_{1\times 3}, \delta \mathbf{v}^T_{1\times 3}, \delta \boldsymbol{\theta}^T_{1\times 3}, \delta \mathbf{b}_{a}^T{}_{1\times 3}, \delta \mathbf{b}_{g}^T{}_{1\times 3}]^T \in \mathbb{R}^{15}$$
- **Continuous Error Dynamics**:
  $$\dot{\delta\mathbf{x}} = \mathbf{F} \delta\mathbf{x} + \mathbf{G} \mathbf{n}$$
  where $\mathbf{F}_{3:6, 6:9} = -[\mathbf{f}_{enu} \times]$, $\mathbf{F}_{3:6, 9:12} = R_b^n$, $\mathbf{F}_{6:9, 12:15} = -R_b^n$.
- **Discretization & Covariance Prediction**:
  $$\mathbf{\Phi} = \mathbf{I}_{15} + \mathbf{F} \Delta t + \frac{1}{2} \mathbf{F}^2 \Delta t^2, \quad \mathbf{P}_{k+1} = \mathbf{\Phi} \mathbf{P}_k \mathbf{\Phi}^T + \mathbf{Q}_d$$
- **GNSS Position Measurement Update**:
  $$\mathbf{y} = \mathbf{p}_{GNSS\_ENU} - \mathbf{p}_{INS\_nominal}, \quad \mathbf{S} = \mathbf{H} \mathbf{P} \mathbf{H}^T + \mathbf{R}, \quad \mathbf{K} = \mathbf{P} \mathbf{H}^T \mathbf{S}^{-1}$$
- **Error Injection & Reset**:
  $$\mathbf{p} \leftarrow \mathbf{p} + \delta \mathbf{p}, \quad \mathbf{v} \leftarrow \mathbf{v} + \delta \mathbf{v}, \quad \mathbf{q} \leftarrow \mathbf{q} \otimes \delta \mathbf{q}, \quad \mathbf{b}_a \leftarrow \mathbf{b}_a + \delta \mathbf{b}_a, \quad \mathbf{b}_g \leftarrow \mathbf{b}_g + \delta \mathbf{b}_g$$
  $$\delta \mathbf{x} \leftarrow \mathbf{0}_{15}$$

#### 6. GNSS Outage & Quality Degradation State Machine
- **Monitoring Indicators**: GNSS observation availability, timestamp gap ($\Delta t > \Delta t_{max}$), horizontal accuracy threshold ($a_{horiz} \ge a_{outage}$), HDOP ($HDOP \ge HDOP_{outage}$), and C/N0 ($C/N_0 \le C/N_{0, outage}$).
- **State Classifications**:
  - `GOOD`: All metrics satisfy optimal operational thresholds.
  - `DEGRADED`: Signal degraded (e.g. urban canyonmultipath); horizontal accuracy or HDOP exceeds degraded thresholds.
  - `OUTAGE`: Complete loss of fix or severe signal degradation.
  - `RECOVERING`: Transition state after returning from outage to prevent false lock-on.
- **Hysteresis Persistence Logic**: Candidate status transitions require $N_{persistence}$ consecutive matching samples to switch state, avoiding state flickering from isolated noisy samples.

#### 7. IMU Windowing & Supervised Drift Target Generation
- **Sliding Window Generation**: Slices high-rate ($100\text{Hz}$) IMU data into temporal windows of size $L$ (e.g. 100 samples = 1.0s) with stride $S$ (e.g. 50 samples = 0.5s). Preserves strict temporal ordering.
- **Feature Extraction**: Extract raw accelerations ($\mathbf{a}$), gyroscopes ($\boldsymbol{\omega}$), acceleration norm ($||\mathbf{a}||_2$), angular rate norm ($||\boldsymbol{\omega}||_2$), relative window time ($t - t_{start}$).
- **Supervised Ground-Truth Target Generation**:
  - **Displacement Target**: $\Delta \mathbf{p} = \mathbf{p}_{true}(t_{end}) - \mathbf{p}_{true}(t_{start})$ in local ENU frame.
  - **Velocity Target**: $\mathbf{v}_{true}(t_{end})$ in local ENU m/s.
- **Trajectory-Aware Splitting**: Shuffles and splits datasets strictly by session/trajectory ID ($ID_{train} \cap ID_{val} \cap ID_{test} = \emptyset$), eliminating temporal data leakage across overlapping windows.

#### 9. AI Pseudo-Measurement ESKF Integration & Innovation Gating
- **AI Position Pseudo-Measurement Model**:
  $$\mathbf{z}_{AI} = \mathbf{p}_{ref} + \Delta \mathbf{p}_{AI}$$
  where $\mathbf{p}_{ref} = \mathbf{p}_{ins}(t_{start})$ is the nominal INS position captured at window start, and $\Delta \mathbf{p}_{AI} = [\Delta p_{east}, \Delta p_{north}, \Delta p_{up}]$ is the AI displacement prediction over the 1.0s window.
- **Innovation Vector**:
  $$\mathbf{y}_{AI} = \mathbf{z}_{AI} - \mathbf{p}_{ins}(t_{end}) = (\mathbf{p}_{ref} + \Delta \mathbf{p}_{AI}) - \mathbf{p}_{ins}(t_{end})$$
- **Measurement Matrix & Noise Covariance**:
  $$\mathbf{H}_{AI} = [\mathbf{I}_3 \quad \mathbf{0}_{3 \times 12}], \quad \mathbf{R}_{AI} = \text{diag}(\sigma_{E}^2, \sigma_{N}^2, \sigma_{U}^2)$$
  where $\sigma_E = \sigma_N = 1.5\text{m}, \sigma_U = 3.0\text{m}$.
- **Mahalanobis Innovation Gating Statistic**:
  $$\mathbf{S}_{AI} = \mathbf{H}_{AI} \mathbf{P} \mathbf{H}_{AI}^T + \mathbf{R}_{AI}, \quad d_M = \sqrt{\mathbf{y}_{AI}^T \mathbf{S}_{AI}^{-1} \mathbf{y}_{AI}}$$
  - If $d_M > \text{gate\_threshold}$ ($4.0\sigma$), the update is rejected to prevent unphysical model predictions from corrupting state estimates.
  - If $d_M \le \text{gate\_threshold}$, Joseph-form covariance update and 15-state error injection are executed.

#### 10. AI-ESKF Failure Injection & Safety Robustness Engine (Step 8.1)
- **Why AI Predictions Can Fail**: Deep learning inertial drift models (e.g. 1D CNN / TCN) may emit unphysical or biased predictions due to out-of-distribution motion dynamics, sensor anomalies, unmodeled vibrations, or model inference artifacts.
- **Why Innovation Gating is Required**: Unconstrained insertion of erroneous AI measurements into an ESKF corrupts nominal navigation state estimates $\mathbf{x}$, distorts error covariance $\mathbf{P}$, and leads to catastrophic filter divergence.
- **Failure-Injection Modes**:
  1. `Mode 1: Normal`: Uncorrupted trained DriftPredictor output.
  2. `Mode 2: Small Bias`: Constant $+1\text{m}$ East / $+1\text{m}$ North displacement bias.
  3. `Mode 3: Moderate Bias`: Constant $+5\text{m}$ East / $+5\text{m}$ North displacement bias.
  4. `Mode 4: Large Outlier`: Transient extreme $+50\text{m}$ East / $-50\text{m}$ North displacement jumps.
  5. `Mode 5: Random Outliers`: Deterministic random corruption (15% of AI updates corrupted with $\pm 50\text{m}$ noise).
  6. `Mode 6: Model Disabled`: Reference baseline with AI updates disabled.
- **Mahalanobis Innovation Gating & Fallback Mechanism**:
  - Gating decision: $d_M = \sqrt{\mathbf{y}_{AI}^T \mathbf{S}_{AI}^{-1} \mathbf{y}_{AI}} > 4.0\sigma \implies \text{REJECT}$.
  - On `REJECT`: Filter skips measurement correction step entirely ($\mathbf{K}=\mathbf{0}$, $\delta \mathbf{x} = \mathbf{0}$). Nominal INS propagation continues uninterrupted without state corruption.
  - Error covariance $\mathbf{P}$ remains positive-definite and symmetric.
- **Post-Failure GNSS Recovery**: Upon GNSS signal restoration following an outage with rejected AI predictions, standard GNSS position updates re-anchor the filter cleanly without divergence.

#### 11. Road Network Representation & Map Matching Layer (Step 9)
- **Road Network Abstraction**: Provider-independent geometric representation in local metric ENU coordinates comprising `RoadNode` and `RoadSegment` with polyline vertices, tangent heading $\psi_{road}$, segment length, and functional classification (`RoadClass`).
- **Multi-Hypothesis Candidate Search**: For current estimated position $\mathbf{p}_{est}$, extracts all road segments within spatial search radius $R_{search} = 50.0\text{m}$.
- **Perpendicular Projection & Tangents**: Computes closest point $\mathbf{p}_{proj}$, orthogonal distance $d_\perp$, along-track cumulative distance, and tangent azimuth $\psi_{road}$ using piecewise polyline projections.
- **Candidate Scoring Formulation**:
  $$\text{score} = w_{dist} s_{dist} + w_{head} s_{head} + w_{trans} s_{trans} + w_{motion} s_{motion}$$
  where:
  - $s_{dist} = \exp\left(-\frac{d_\perp^2}{2 \sigma_{dist}^2}\right)$ with $\sigma_{dist} = 10.0\text{m}, w_{dist} = 0.40$
  - $s_{head} = \exp\left(-\frac{\Delta \psi^2}{2 \sigma_{head}^2}\right)$ with $\Delta \psi = |\text{wrap}_{180}(\psi_{veh} - \psi_{road})|, \sigma_{head} = 25.0^\circ, w_{head} = 0.35$
  - $s_{trans} \in [0.3, 1.0]$ enforcing topological continuity with previous matched segment ($w_{trans} = 0.15$)
  - $s_{motion} = \exp\left(-\frac{v_{lat}^2}{2 \sigma_v^2}\right)$ enforcing non-holonomic road vehicle constraints ($w_{motion} = 0.10$)
- **Candidate Disambiguation**: Candidates with $\Delta \psi > 60.0^\circ$ (e.g. crossing roads) or confidence $< 0.25$ are rejected.
- **Soft Map Pseudo-Measurement ESKF Update**:
  $$\mathbf{z}_{map} = \mathbf{p}_{proj}, \quad \mathbf{H}_{map} = [\mathbf{I}_3 \quad \mathbf{0}_{3 \times 12}]$$
  $$\mathbf{R}_{map} = \text{diag}\left(\left(\frac{\sigma_E}{c}\right)^2, \left(\frac{\sigma_N}{c}\right)^2, \sigma_U^2\right), \quad c = \max(0.05, \text{confidence})$$
- **Mahalanobis Innovation Gating & Fallback**:
  $$\mathbf{S}_{map} = \mathbf{H}_{map} \mathbf{P} \mathbf{H}_{map}^T + \mathbf{R}_{map}, \quad d_M = \sqrt{\mathbf{y}_{map}^T \mathbf{S}_{map}^{-1} \mathbf{y}_{map}}$$
  - If $d_M \le 4.0\sigma$: Joseph-form error covariance update and state error injection are executed.
  - If $d_M > 4.0\sigma$: update is rejected without state mutation, falling back to nominal INS+AI propagation.

### Planned Algorithm Roadmap

1. **Android Deployment & Native Sensor Pipeline (Step 10)**
   - Android NDK / Kotlin wrapper for high-rate IMU streaming and edge inference.



