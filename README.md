# SIH-26168 — AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-221%20Passing-brightgreen.svg)]()
[![Classification](https://img.shields.io/badge/System%20Classification-YELLOW-yellow.svg)]()
[![Provenance](https://img.shields.io/badge/Data%20Provenance-SYNTHETIC%20%2F%20OFFLINE%20REPLAY-orange.svg)]()

> **Project:** SIH-26168  
> **Classification:** **YELLOW** (Software Validated | Physical Vehicle Trials Pending)  
> **Core Innovation:** Fusing 100 Hz strapdown Inertial Navigation (INS), 15-State Error-State Kalman Filter (ESKF), Temporal Convolutional Network (TCN) AI displacement estimation, Non-Holonomic Constraints (NHC), Safe Zero-Velocity Updates (ZUPT), and Road Map Matching to suppress dead-reckoning drift during complete GNSS outages.

---

## ⚡ Quick Start (Run in 2 Minutes)

### 1. Prerequisites
- **Python:** Version 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14
- **Operating System:** Linux, macOS, or Windows
- **Internet Connection:** Only required for the initial `pip install` (the web prototype runs 100% offline).

### 2. Clone the Repository
```bash
git clone https://github.com/alaghattadarshan-glitch/sih.git
cd sih
```

### 3. Setup Virtual Environment & Install Dependencies
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 4. Launch Interactive Demonstration Dashboard
```bash
python scripts/run_demo.py --port 8080
```

### 5. Open Web UI
Open your browser and navigate to:
```text
http://127.0.0.1:8080
```

### 6. Run Complete Test Suite (Optional)
```bash
pytest -v
```
All **221 / 221** unit, integration, and regression tests will pass.

---

## 🎮 Interactive Demonstration Features

When you launch the dashboard at `http://127.0.0.1:8080`, you have access to:

1. **✨ Run Full SIH Demo (1-Click Automation):**
   - Click the pulsing purple button to execute an automated 60-second guided presentation:
     $$\text{GNSS Nominal} \longrightarrow \text{GNSS Outage (30s)} \longrightarrow \text{Dead Reckoning Active} \longrightarrow \text{GNSS Recovery} \longrightarrow \text{Mission Summary}$$
2. **Primary Visual ENU Trajectory Map:**
   - Real-time tangent plane (ENU) trajectory auto-scaled to the active path.
   - `━━ Ground Truth`: Reference path (dashed light slate).
   - `━━ Full Stack Navigation`: AI + NHC + ZUPT + Map trajectory (glowing cyan).
   - `━━ Pure INS Baseline`: Unconstrained inertial divergence (red).
   - `🚗 Current Vehicle`: Vehicle heading and position marker.
3. **View Mode Switcher:**
   - `[ 🎯 Simple Demo View ]`: Streamlined layout for judges focusing on the Map, Outage Banner, Current Operation explanation, and Error Graph.
   - `[ 🔬 Technical View ]`: Full telemetry inspection including 15-state ESKF covariance trace $\text{Tr}(P)$, 100 Hz IMU acceleration/gyroscope, GNSS satellite dilution, and real-time operational logs.
4. **"What the System Is Doing" (Current Operation) Panel:**
   - Plain-English live operational feedback explaining which sensors and filter algorithms are actively engaged.
5. **Real-Time Error Graph with Shaded Outage Window:**
   - Shaded band marking the GNSS outage region (`[30s to 60s]`), visually proving that Full Stack error stays clamped well below the 10% drift target while Pure INS diverges exponentially.
6. **Step 17 Master Benchmark Matrix:**
   - Interactive table browsing 150 evaluated scenarios (5 trajectories $\times$ 5 outage durations $\times$ 6 filter configurations).

---

## 🧠 System Architecture

```
GNSS Data (1 Hz)                IMU Sensors (100 Hz Accel & Gyro)
       │                                     │
       ▼                                     ▼
[ GNSS Outage Detector ]             [ Sensor Calibration ]
       │                                     │
       ├─────────────────────────┐           ▼
       │                         │   [ Strapdown INS Mechanization ]
       ▼                         │           │
[ 15-State ESKF Engine ] ◄───────┼───────────┤
       ▲                         │           │
       │ (During GNSS Outage)    │           │
       ├── [ TCN AI Model ] ◄────┴───────────┤
       ├── [ NHC Kinematic Constraint ] ─────┤
       ├── [ Safe ZUPT Detector ] ───────────┘
       └── [ Road Map Matching ]
                 │
                 ▼
       [ Navigation Output ] (Position, Velocity, Orientation)
```

---

## 📊 Summary Benchmark Performance (Step 17 Final Matrix)

Ablation evaluation during a 30-second GNSS outage in Mixed Urban Driving (482 m travelled):

| Configuration | Outage 2D RMSE (m) | Final Error (m) | Drift % | SIH Target (<10%) |
| :--- | :---: | :---: | :---: | :---: |
| **Pure INS Baseline** | 42.60 m | 89.45 m | 18.55% | ❌ FAIL |
| **ESKF (Inertial Only)** | 41.80 m | 87.20 m | 18.08% | ❌ FAIL |
| **ESKF + AI (TCN)** | 18.40 m | 32.10 m | 6.66% | ✅ PASS |
| **ESKF + NHC** | 12.15 m | 19.80 m | 4.11% | ✅ PASS |
| **ESKF + AI + NHC + ZUPT** | 5.20 m | 8.40 m | 1.74% | ✅ PASS |
| **Full Stack (+ Map Matching)** | **2.85 m** | **4.12 m** | **0.85%** | **✅ PASS** |

---

## 📁 Repository Directory Structure

```
sih/
├── README.md                      # Quick start, system architecture & benchmark summary
├── requirements.txt               # Declared Python dependencies
├── pyproject.toml                 # Package configuration & pytest settings
├── .gitignore                     # Git exclusions
│
├── src/                           # Primary Python navigation engine
│   ├── coordinate_transforms/     # WGS84, ECEF, Local ENU geodetic transforms
│   ├── calibration/               # IMU bias, scale factor, and temperature calibration
│   ├── navigation/                # Strapdown INS, 15-state ESKF, quaternions, NHC, ZUPT
│   ├── ml/                        # TCN neural network architecture, datasets, inference
│   ├── outage_detection/          # GNSS outage detector with persistence hysteresis
│   ├── map_matching/              # Road network geometry & projection matching
│   ├── demo/                      # Demonstration server, session manager, REST API
│   └── evaluation/                # Benchmark metrics, multi-session evaluators
│
├── web/                           # 100% Offline interactive dashboard frontend
│   ├── index.html                 # Single-page application markup
│   ├── styles.css                 # Dark-mode design system & animations
│   └── app.js                     # Dynamic Canvas map, error graph & telemetry engine
│
├── android/                       # Mobile edge deployment assets
│   ├── native/navigation_core/    # Native C++ navigation core (Eigen-free C++17)
│   └── jni/                       # JNI bridge headers for Android integration
│
├── checkpoints/                   # Trained PyTorch model weights (best_drift_model.pt)
├── results/                       # Step 1-18.1 evaluation JSONs, plots, and ONNX models
├── scripts/                       # CLI runners (run_demo.py, run_final_benchmark.py, etc.)
├── tests/                         # Complete 221-test automated test suite
└── docs/                          # Detailed engineering milestone documentation
```

---

## 🔒 Data Honesty & Provenance Statement

In strict compliance with Smart India Hackathon standards:
- **Demonstration Data:** All current live demonstration trajectories are generated from validated synthetic kinematic vehicle simulations (`DEMO DATA: SYNTHETIC / OFFLINE REPLAY`).
- **Physical Vehicle Validation:** Marked **NOT VERIFIED**. Hardware trials on physical Android devices with real automotive sensor logging are scheduled for physical field testing.
- **IO-VNBD Dataset:** Marked **UNAVAILABLE**. Adapters and pipeline compatibility are validated; execution on physical raw IO-VNBD sequences is pending raw dataset availability.
- **Map Matching:** Disclosed as **DEMO MAP / SYNTHETIC ROAD NETWORK**.
