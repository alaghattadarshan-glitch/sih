# Step 16.1: Diagnostic and Root-Cause Analysis of Multi-Constraint Fusion

**Status**: COMPLETED  
**Evaluation Target**: Vehicle Motion Constraints (NHC + Safe ZUPT + AI Drift Inference)  
**Classification**: **YELLOW** (High mathematical confidence on deterministic/synthetic harness; real Android hardware remains unverified)  
**Provenance**: SYNTHETIC_DATA (Software Harness / Host Execution)

---

## 1. Executive Summary & 5-Way Ablation Benchmark

During the initial Step 16 evaluation on a 120-second multi-phase trajectory featuring a 70-second GNSS outage (30.0s to 100.0s), severe anomalies were observed where adding motion constraints caused position error to diverge up to **9,920.90 m RMSE**.

Step 16.1 isolated, proved mathematically, and resolved three independent root causes across the synchronization, frame transformation, and measurement gating pipelines:

1. **GNSS Outage Synchronization Flaw**: High-rate 100 Hz IMU samples without a GNSS fix were treated as instantaneous dropouts, locking the filter into a persistent false `OUTAGE` state.
2. **ESKF Frame Transformation & Jacobian Inconsistency**: The Non-Holonomic Constraint (NHC) Jacobian $H_{\text{NHC}}$ and attitude error injection previously mixed body-frame and navigation-frame conventions without applying the transformation $R_{b2n}^T$, producing positive-feedback yaw instability during vehicle turning.
3. **ZUPT Mahalanobis Gating & Innovation Lockout**: Accumulated velocity drift prior to stationary red-light stops exceeded the default 4.0-sigma gating threshold, causing 100% of true zero-velocity measurements to be rejected.

### 5-Way Ablation Comparison (Before vs. After Step 16.1 Fixes)

| Navigation Mode | Previous Outage RMSE | **Corrected Outage RMSE** | Previous Final Error | **Corrected Final Error** | **Drift %** | **Vel RMSE** | **NHC Acc %** | **ZUPT Acc %** | **AI Acc %** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A: ESKF only** | 1007.69 m | **216.15 m** | 2035.00 m | **413.60 m** | 84.00% | 6.95 m/s | 0.0% | 0.0% | 0.0% |
| **B: ESKF + AI** | 5245.78 m | **560.32 m** | 10930.31 m | **836.53 m** | 169.89% | 15.65 m/s | 0.0% | 0.0% | 66.2% |
| **C: ESKF + NHC** | 3240.44 m | **261.74 m** | 8217.14 m | **654.93 m** | 133.01% | 14.82 m/s | 90.68% | 0.0% | 0.0% |
| **D: ESKF + AI + NHC** | 970.05 m | **139.98 m** | 2616.90 m | **225.25 m** | **45.75%** | 8.12 m/s | 90.68% | 0.0% | 21.6% |
| **E: ESKF + AI + NHC + ZUPT** | 9920.90 m | **116.47 m** | 20678.08 m | **192.01 m** | **38.99%** | **5.41 m/s** | 90.68% | 0.85% | 32.4% |

**Key Takeaway**: The fully integrated multi-constraint pipeline (**Configuration E**) achieves the lowest outage RMSE (**116.47 m**, a **46.1% reduction** relative to the unconstrained ESKF baseline) and reduces total outage drift from 84.00% down to **38.99%**.

---

## 2. Independent Root-Cause Investigations

### A. AI Integration & Heading Frame Alignment
- **Observation**: AI alone (Configuration B: 560.32 m) exhibited higher RMSE than ESKF alone (Configuration A: 216.15 m).
- **Mathematical Cause**: The trained Temporal Convolutional Network (TCN) takes body-frame IMU features $[a_x, a_y, a_z, \omega_x, \omega_y, \omega_z]$ as input and directly predicts local ENU displacement targets $[\Delta p_{\text{East}}, \Delta p_{\text{North}}, \Delta p_{\text{Up}}]$. Because the synthetic training trajectories were oriented predominantly along the East corridor (Heading $\approx 0^\circ$), the model learned a strong positive bias mapping forward body acceleration to $+\Delta p_{\text{East}}$.
- **Turn Dynamics**: When the vehicle makes a $90^\circ$ right turn to the South during the outage, unconstrained AI predictions pull the estimate Eastward. When fused with NHC (Configuration D), NHC strictly clamps cross-track velocity ($v_{\text{body}, y} = 0$), while Mahalanobis gating rejects contradictory orthogonal AI innovations, allowing AI to provide along-track velocity scaling.

### B. NHC Frame Inconsistency & Attitude Feedback Loop
- **Observation**: NHC alone previously diverged to 3,240.44 m RMSE.
- **Mathematical Cause**: In the 15-state ESKF, the attitude error $\delta\theta$ is defined in the **navigation frame** (ENU):
  $$\delta \dot{\mathbf{v}}_{\text{ENU}} = -[\mathbf{f}_{\text{ENU}} \times] \delta\theta_{\text{nav}} + R_{b2n} \delta\mathbf{b}_a$$
  The body-frame velocity perturbation is:
  $$\delta \mathbf{v}_{\text{body}} = R_{b2n}^T \delta \mathbf{v}_{\text{ENU}} + [\mathbf{v}_{\text{body}} \times] R_{b2n}^T \delta\theta_{\text{nav}}$$
  Previously, $H_{\text{NHC}}$ computed the attitude sensitivity directly as $[\mathbf{v}_{\text{body}} \times]$ without rotating by $R_{b2n}^T$. Furthermore, attitude injection right-multiplied $q_{\text{nom}} \otimes \delta q$ instead of left-multiplying $\delta q \otimes q_{\text{nom}}$.
- **Effect**: At Heading $0^\circ$ ($R_{b2n} \approx I$), the error was latent. During the $90^\circ$ turn, the sensitivity applied attitude corrections along inverted axes, creating a positive-feedback loop that drove yaw error to $-37.6^\circ$ and exploded velocity error.
- **Resolution**:
  $$H_{\text{NHC}} = \begin{bmatrix} R_{b2n}[:, 1]^T & [v_z, 0, -v_x] R_{b2n}^T & \mathbf{0}_{1\times 6} \\ R_{b2n}[:, 2]^T & [-v_y, v_x, 0] R_{b2n}^T & \mathbf{0}_{1\times 6} \end{bmatrix}$$
  Attitude error injection corrected to: $\mathbf{q}_{\text{corr}} = \delta\mathbf{q} \otimes \mathbf{q}_{\text{nom}}$.

### C. ZUPT Gating Lockout
- **Observation**: Previous configuration E diverged to 9,920.90 m with 0.0% ZUPT acceptance.
- **Mathematical Cause**: During the 75s–95s stationary stop, velocity error had grown to $\approx 6\text{ m/s}$. The innovation covariance $S = P_{vv} + R_{\text{ZUPT}} \approx 1.0\text{ m}^2/\text{s}^2$. The Mahalanobis distance was:
  $$d_M = \sqrt{(-6)^T S^{-1} (-6)} \approx 6.0 > 4.0\text{ (gate threshold)}$$
  The conservative 4.0-sigma gate rejected the first stationary update, locking the filter out of all subsequent zero-velocity corrections.
- **Resolution**: Calibrated `zupt_gate_threshold` to allow stationary updates when the physics-based stationary detector confidently asserts `is_stationary = True`.

---

## 3. Mathematical Validation & Jacobian Verification

A 100-state deterministic Monte Carlo perturbation test compared the analytical Jacobians against central finite differences across random orientations ($\text{Roll}, \text{Pitch} \in [-\pi/4, \pi/4]$, $\text{Yaw} \in [-\pi, \pi]$) and velocities ($\mathbf{v} \in [-20, 20]\text{ m/s}$):

- **NHC Analytical vs. Finite Difference**:
  - Max Absolute Error: $6.71 \times 10^{-9}$
  - Mean Absolute Error: $2.49 \times 10^{-10}$
  - RMSE: $5.83 \times 10^{-10}$
- **ZUPT Analytical vs. Finite Difference**:
  - Max Absolute Error: $1.03 \times 10^{-9}$
  - Mean Absolute Error: $3.42 \times 10^{-11}$
  - RMSE: $1.64 \times 10^{-10}$

### Known Heading Invariance Verification ($v_{\text{body}} = [10.0, 2.0, 1.0]\text{ m/s}$)

| Cardinal Heading | $v_{\text{body}}$ Before Update | $v_{\text{body}}$ After Update | Forward Velocity ($v_x$) Preserved | Lateral/Vertical Attenuation |
| :--- | :--- | :--- | :--- | :--- |
| **East ($0^\circ$)** | $[10.0, 2.0, 1.0]$ | $[10.08, 0.66, 0.33]$ | Yes ($10.08\text{ m/s}$) | $-67.0\%$ |
| **North ($90^\circ$)** | $[10.0, 2.0, 1.0]$ | $[10.08, 0.66, 0.33]$ | Yes ($10.08\text{ m/s}$) | $-67.0\%$ |
| **West ($180^\circ$)** | $[10.0, 2.0, 1.0]$ | $[10.08, 0.66, 0.33]$ | Yes ($10.08\text{ m/s}$) | $-67.0\%$ |
| **South ($270^\circ$)** | $[10.0, 2.0, 1.0]$ | $[10.08, 0.66, 0.33]$ | Yes ($10.08\text{ m/s}$) | $-67.0\%$ |

---

## 4. Covariance Matrix Diagnostics

Covariance matrix conditioning and positive-definiteness were recorded across propagation and measurement cycles:

| Timeline Event | Trace($P$) | Min Eigenvalue | Max Eigenvalue | Condition Number | Matrix Symmetry | Positive Definite |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Initialization** | $1.200 \times 10^{-1}$ | $1.000 \times 10^{-6}$ | $1.000 \times 10^{-2}$ | $1.000 \times 10^{4}$ | **True** | **True** |
| **5s INS Propagation** | $3.418 \times 10^{1}$ | $9.999 \times 10^{-7}$ | $2.845 \times 10^{1}$ | $2.845 \times 10^{7}$ | **True** | **True** |
| **After NHC Update** | $3.391 \times 10^{1}$ | $9.999 \times 10^{-7}$ | $2.845 \times 10^{1}$ | $2.845 \times 10^{7}$ | **True** | **True** |
| **After ZUPT Update** | $3.342 \times 10^{1}$ | $9.999 \times 10^{-7}$ | $2.845 \times 10^{1}$ | $2.845 \times 10^{7}$ | **True** | **True** |

---

## 5. Python vs. C++ Native Parity

Native C++ navigation core implementation in `android/native/navigation_core/` was synchronized and verified against Python:

- **NHC Correction Parity**: Max velocity difference = $0.0\text{ m/s}$, Mahalanobis difference = $0.0$, Acceptance match = $100.0\%$.
- **ZUPT Correction Parity**: Max velocity difference = $0.0\text{ m/s}$, Mahalanobis difference = $0.0$, Acceptance match = $100.0\%$.

---

## 6. Computational Performance Benchmark

Measured on Host Development Machine (`Darwin arm64`):

| Pipeline Stage | Mean Execution Latency | Source Tag |
| :--- | :--- | :--- |
| **NHC Constraint Update** | $74.9\ \mu\text{s}$ | `SOFTWARE_HARNESS / HOST` |
| **ZUPT Signal Detection** | $16.4\ \mu\text{s}$ | `SOFTWARE_HARNESS / HOST` |
| **ZUPT Measurement Update** | $64.5\ \mu\text{s}$ | `SOFTWARE_HARNESS / HOST` |
| **Complete 100 Hz Loop** | $171.3\ \mu\text{s}$ | `SOFTWARE_HARNESS / HOST` |

*Note: Real physical Android latency on Qualcomm Snapdragon / Tensor silicon remains unmeasured until physical device testing.*

---

## 7. Artifact Manifest & Verification Status

### Generated Artifacts (`results/step16_1_debug/`)
- `reproduction.json`: 5-way ablation comparison
- `nhc_math_validation.json`: 100-state Jacobian perturbation and cardinal heading test results
- `zupt_validation.json`: Stationary and multi-phase transition metrics
- `ai_integration_debug.json`: TCN feature and frame target analysis
- `covariance_diagnostics.json`: Eigenvalue and condition number traces
- `parameter_sensitivity.json`: 25-point parameter grid across $\sigma_{\text{NHC}}$ and gating thresholds
- Plots:
  - `nhc_known_orientation.png`
  - `nhc_velocity_correction.png`
  - `zupt_detector_signals.png`
  - `zupt_state.png`
  - `constraint_failure_timeline.png`
  - `ai_reference_alignment.png`
  - `covariance_trace.png`

### Test Suite Execution
- **Total Tests Run**: **209 / 209 passing**
- **Regressions**: 0
- **Failures**: 0

### External Device & Dataset Status
- **Physical Android Device**: `NOT VERIFIED` (Requires ADB connection to physical Android testbed)
- **IO-VNBD Dataset**: `NOT AVAILABLE` (Synthetic and real logged phone formats used for offline evaluation)
