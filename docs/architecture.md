# System Architecture

## SIH26168 — AI-ML based Intelligent Dead Reckoning System

### System Overview

The Intelligent Dead Reckoning System combines Inertial Navigation Systems (INS), Extended Kalman Filtering (EKF), and Deep Learning to maintain accurate positioning during GNSS outages (tunnels, urban canyons, indoor environments).

```
IMU Sensors (100Hz)                          GNSS Receiver (1Hz)
        │                                            │
        ▼                                            ▼
[ Strapdown INS Mechanization ] ◄──────────┐  [ ECEF -> ENU Transform ]
  (Attitude, Velocity, Pos)                │         │
        │                                  │         ▼
        │ (Nominal State)                  │   [ GNSS Position Fix ]
        ▼                                  │         │
[ 15-State Error-State EKF ]               │         │
  (Prediction: 100Hz)                      │         │
        │                                  │         │
        ├─── (GNSS Available) ─────────────┼─────────┘
        │      (Measurement Update &       │  (State Correction &
        │       Error-State Injection)     │   Bias Reset)
        │                                  │
        └─── (GNSS Outage) ────────────────┘
               (Pure INS Propagation &
                Uncertainty Growth)
```

### Coordinate Frames & Spatial Representation Strategy

A central requirement for inertial navigation systems is selecting an appropriate coordinate frame.

#### Why the System Uses a Local Cartesian Frame (ENU)

- **GNSS Source**: GNSS receivers output global geographic coordinates ($\text{Latitude}, \text{Longitude}, \text{Height}$).
- **Inertial & Dead Reckoning Nature**: Accelerometer and gyroscope measurements operate naturally in flat Euclidean space over local navigation scales ($\text{Meters}$ displacement, $\text{m/s}$ velocity, $\text{m/s}^2$ acceleration).
- **Curvature Avoidance**: Performing high-rate ($100\text{Hz}+$) strapdown numerical integration directly on spherical/ellipsoidal coordinates introduces complex trigonometric overhead and singularities.

#### Data Conversion Pipeline

```
GNSS Position (LLH)
        │
        ▼
   [ ECEF Frame ] (Earth-Centered Earth-Fixed XYZ)
        │
        ▼
   [ Local ENU Frame ] (East-North-Up Tangent Plane)
        │
        ▼
[ High-Rate Inertial Navigation & EKF Fusion / AI Dead Reckoning ]
        │
        ▼
   [ Updated Local ENU Position ]
        │
        ▼
   [ ECEF Frame ]
        │
        ▼
   [ Output LLH Coordinates for UI & Map Display ]
```

### Key Components

1. **Sensor Processing & Synchronization**: Ingests raw IMU (100Hz+) and GNSS (1Hz) signals, applying time alignment, unit conversion, and sensor bias calibration.
2. **Coordinate Transformations**: Converts global GNSS LLH coordinates into local ENU Cartesian coordinates using centralized WGS84 ellipsoid model.
3. **Inertial Navigation & Dead Reckoning**: Mechanizes strapdown INS equations to estimate position, velocity, and orientation updates.
4. **15-State Error-State Kalman Filter (ESKF)**: Estimates small errors in position, velocity, attitude, accel bias, and gyro bias ($\delta\mathbf{x} \in \mathbb{R}^{15}$) to correct nominal INS states when GNSS is available and track uncertainty during outages.
5. **AI/ML Error Prediction Module**: Uses neural networks (TCN / LSTM) to estimate motion displacement, step length, or INS drift corrections during GNSS outages.
6. **Map Matching**: Constrains trajectory solutions onto road networks or indoor floor plans.
7. **Outage Detection**: Identifies GNSS signal degradation and seamlessly transitions navigation state estimation to AI-assisted DR.

### Current Development Stage
> [!NOTE]
> Step 11 Complete: Multi-Session Benchmark & TCN Generalization framework implemented. Includes `DatasetSession` abstraction, session discovery, multi-session quality control, strict Leave-One-Session-Out trajectory isolation, standalone TCN generalization testing on held-out sessions (+34.4% to +48.0% improvement over classical baseline), multi-duration outage benchmarks, domain shift distribution auditing, 8 consolidated visualization plots, and 169 passing tests.



### Data Flow & Training/Deployment Separation

```
TRAINING PIPELINE (Offline Only):
High-Rate IMU (100Hz) ──► Windowing ──► Feature Extractor ──► PyTorch IMUWindowDataset
                                                                      │
Ground Truth Trajectory ──► Target Builder (Δp = p_end - p_start) ────┘
                                                                      │
Trajectory IDs ──────────► Trajectory-Aware Splitter (Train/Val/Test) ┘
                                                                      │
Train Split Statistics ──► Feature Normalization (mean, std) ─────────┤
                                                                      ▼
                                                          [ 1D CNN / TCN Model ]
                                                          (MSE Loss + Adam)
                                                                      │
                                                                      ▼
                                                          [ Checkpoint Export ]
                                                          (best_drift_model.pt)

DEPLOYMENT PIPELINE (Online Real-Time):
Sensors (IMU 100Hz, GNSS 1Hz)
         │
         ▼
[ GNSS Outage Detector ]
         │
         ├─── (GNSS AVAILABLE: GOOD / RECOVERING) ──────────────────────────┐
         │      Run ESKF GNSS Measurement Update                             │
         │      Reset Window Reference Position                              │
         │                                                                   ▼
         └─── (GNSS OUTAGE) ──────────────────────────┐             [ Navigation State ]
                Causal IMU Window Buffer (100Hz)      │             (Pos, Vel, Att, Biases)
                DriftPredictor (TCN Inference)        │                      ▲
                Δp_AI = [dE, dN, dU]                  │                      │
                z_AI = p_ref + Δp_AI                  │                      │
                Mahalanobis Innovation Gating (d_M) ──┤                      │
                (Accepted -> ESKF AI Joseph Update)   │                      │
                                                      ▼                      │
                [ Map Candidate Generation ] ─────────┤                      │
                (Spatial search in local ENU network) │                      │
                                                      ▼                      │
                [ Multi-Hypothesis Candidate Scoring ]┤                      │
                (Distance, Heading, Motion, Temporal) │                      │
                                                      ▼                      │
                [ Soft Map Pseudo-Measurement ] ──────┤                      │
                z_map = p_proj, R_map(confidence)     │                      │
                                                      ▼                      │
                [ Map Mahalanobis Gating (d_M <= 4σ) ]┴──────────────────────┘
                (Accepted -> Soft ESKF Map Correction)
                (Rejected -> Continue INS + AI State)
```

