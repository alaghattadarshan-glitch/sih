# Edge Navigation Architecture — Android Deployment

## 1. System Overview

The SIH26168 Edge Navigation Architecture enables real-time, low-latency, GPS-denied dead reckoning on Android smartphones with low computational overhead and zero dynamic heap allocation in the high-rate IMU loop.

```
+-------------------------------------------------------------------------+
|                          Android Sensors Layer                          |
|   Hardware IMU (100 Hz FIFO)           GNSS Location Provider (1 Hz)    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  Sensor Adapter & Monotonic Clock Sync                  |
|   • Android nanoseconds to seconds conversion                           |
|   • Jitter & duplicate timestamp rejection                              |
|   • Sensor frame to body frame alignment (+x Fwd, +y Left, +z Up)      |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                       C++ Native Navigation Core                        |
|                                                                         |
|   +--------------------------+        +-----------------------------+   |
|   |   100 Hz Strapdown INS   |        |    15-State ESKF Engine     |   |
|   | • Quaternion propagation |------->| • Error covariance (P)      |   |
|   | • Specific force / ENU   |        | • Discrete transition (Φ)   |   |
|   | • Zero-allocation loop   |        | • Joseph covariance update  |   |
|   +--------------------------+        +-----------------------------+   |
|                 ^                                    ^                  |
|                 | (Nominal State)                    | (Pseudo-Meas)    |
|                 v                                    v                  |
|   +--------------------------+        +-----------------------------+   |
|   |  GNSS Outage Detector    |        | AI Pseudo-Meas Fusion Engine|   |
|   | • Accuracy / Gap check   |        | • Mahalanobis Gating (4.0σ) |   |
|   | • GOOD / OUTAGE / RECOV  |        | • State Error Injection     |   |
|   +--------------------------+        +-----------------------------+   |
+------------------------------------+------------------------------------+
                                     |
                  +------------------+------------------+
                  | (During Outage: Causal 1s Window)   |
                  v                                     v
+------------------------------------+  +---------------------------------+
|     AI Inference Engine (ONNX)     |  |       Map Matching Layer        |
| • 100 IMU samples @ 100 Hz         |  | • Soft road constraint          |
| • Frozen training normalization    |  | • Disabled without valid map    |
| • Displacement: [ΔE, ΔN, ΔU] (m)   |  +---------------------------------+
+------------------------------------+
                  |
                  v
+-------------------------------------------------------------------------+
|                        Navigation State Output                          |
|   Timestamp | Lat/Lon/Alt | Position ENU | Velocity | Heading | Conf   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   Kotlin Navigation Service & UI                        |
|   Decoupled async state consumer (10-20 Hz UI refresh, non-blocking)    |
+-------------------------------------------------------------------------+
```

---

## 2. Key Architectural Guarantees

### A. 100 Hz Thread Decoupling
- The C++ Navigation Core executes on a dedicated high-priority native sensor thread.
- Android UI rendering and database writes run asynchronously on the main/worker threads and **never block** the $100\text{ Hz}$ IMU integration loop.

### B. Zero Dynamic Heap Allocation in Critical Loop
- All state vectors ($15\times 1$), covariance matrices ($15\times 15$), system dynamics matrices ($\boldsymbol{\Phi}, \mathbf{Q}_d$), and sliding window circular buffers are preallocated during initialization.
- Step execution incurs zero `malloc` / `new` calls during IMU and GNSS processing.

### C. Strict Causal AI Sliding Window
- The AI inference buffer strictly ingests past and present IMU samples $[t - 1.0\text{s}, t]$.
- No future observations or ground-truth references enter the model window.

### D. Multi-Tier Outage Safety & Gating
- GNSS Outage Detector tracks signal degradation using horizontal accuracy ($>5.0\text{m}$ degraded, $>15.0\text{m}$ outage) and latency ($>2.0\text{s}$ outage).
- AI displacement updates are validated by a $4.0\sigma$ Mahalanobis distance test ($\chi^2_3$ gating) before filter injection, protecting the navigation filter against neural network outliers.
- Seamless GNSS recovery without covariance collapse or position discontinuities.
