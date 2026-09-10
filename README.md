# SIH26168 — AI-ML based Intelligent Dead Reckoning System for Seamless Navigation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Development-Foundation_Stage-orange.svg)]()

> **Notice:** This repository is currently at the foundation stage. Navigation algorithms and AI models will be implemented incrementally.

---

## 1. SIH26168 Problem Overview

In modern navigation systems, Satellite Navigation (GNSS/GPS) provides absolute position positioning. However, GNSS signals are prone to complete outages or severe multipath degradation in challenging environments such as:
- Urban canyons surrounded by skyscrapers
- Tunnels and underpasses
- Dense foliage and forest canopies
- Indoor facilities, basements, and parking structures

When GNSS signals fail, conventional navigation systems quickly lose position accuracy or freeze completely.

## 2. What the Project Solves

The **SIH26168 Intelligent Dead Reckoning System** addresses GNSS signal loss by developing an AI/ML-assisted strapdown Dead Reckoning (DR) framework. By fusing high-rate Inertial Measurement Unit (IMU) sensor readings (accelerometer and gyroscope) with machine learning models and Extended Kalman Filtering (EKF), the system continuously predicts vehicle/pedestrian position, velocity, and orientation during GNSS outages, delivering seamless positioning without spatial jumps.

## 3. High-Level Architecture

```
GNSS Data + IMU Sensors (Accel/Gyro)
         │
         ▼
[ Sensor Processing & Synchronization ]
         │
         ├───► [ Strapdown INS Mechanization ] ───┐
         │                                       ▼
         ├───► [ AI/ML Motion & Drift Model ] ───► [ EKF Sensor Fusion ] ──► Navigation State
         │                                       ▲
         └───► [ GNSS Outage Detection ] ────────┘
```

## 4. Current Development Stage

**Foundation Stage (Step 1)**:
- Modular project structure established
- Configuration file schema (`config/config.yaml`) created
- Documentation and architectural blueprints defined
- Unit test framework configured (`pytest`)

Navigation algorithms (INS, EKF, PDR), machine learning models, map matching, and mobile deployment will be added step-by-step in subsequent iterations.

## 5. Project Directory Structure

```
sih26168/
│
├── README.md                  # Project overview and documentation
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Packaging & pytest configuration
├── .gitignore                 # Version control exclusions
│
├── config/
│   └── config.yaml            # Central configuration file
│
├── data/                      # Data storage directories
│   ├── raw/                   # Raw sensor datasets
│   ├── processed/             # Preprocessed tensors and logs
│   └── sample/                # Sample data for integration testing
│
├── src/                       # Source code package root
│   ├── __init__.py
│   │
│   ├── data/                  # Loaders, synchronization, preprocessing
│   │   ├── __init__.py
│   │   ├── loaders/
│   │   ├── synchronization/
│   │   └── preprocessing/
│   │
│   ├── calibration/           # Sensor bias & scale factor calibration
│   │   └── __init__.py
│   │
│   ├── coordinate_transforms/ # ECEF, ENU, LLH, NED spatial transforms
│   │   └── __init__.py
│   │
│   ├── navigation/            # Inertial Navigation & Dead Reckoning
│   │   ├── __init__.py
│   │   ├── imu/               # IMU signal models & noise density
│   │   ├── ins/               # Strapdown INS mechanization
│   │   ├── dead_reckoning/    # PDR & vehicle DR logic
│   │   ├── ekf/               # Extended Kalman Filter
│   │   └── state/             # Navigation state representations
│   │
│   ├── ml/                    # Machine Learning sub-system
│   │   ├── __init__.py
│   │   ├── datasets/          # PyTorch datasets
│   │   ├── models/            # Neural network architectures
│   │   ├── training/          # Training scripts & loss functions
│   │   ├── inference/         # Model inference engines
│   │   └── evaluation/        # ML performance metrics
│   │
│   ├── map_matching/          # Road network & indoor map matching
│   │   └── __init__.py
│   │
│   ├── outage_detection/      # GNSS signal degradation detection
│   │   └── __init__.py
│   │
│   └── evaluation/            # Trajectory analysis & metrics
│       └── __init__.py
│
├── scripts/                   # Helper CLI scripts
├── tests/                     # Automated test suite
├── notebooks/                 # R&D Jupyter notebooks
├── results/                   # Evaluation results & plots
├── dashboard/                 # Real-time monitoring UI
├── android/                   # Mobile/Edge application code
│
└── docs/                      # Technical documentation
    ├── architecture.md        # System architecture details
    ├── algorithms.md          # Algorithm roadmap & specifications
    ├── dataset.md             # Data schemas and format specifications
    ├── experiments.md         # Experiment tracking log
    └── sih_demo.md            # SIH finale demonstration guide
```

## 6. Installation Instructions

### Prerequisites
- Python 3.9 or higher

### Setup

1. **Clone repository and navigate into project folder:**
   ```bash
   cd sih26168
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install as an editable package:
   ```bash
   pip install -e .
   ```

## 7. Running Tests

Run the test suite using `pytest`:

```bash
pytest
```

To run with verbose output:
```bash
pytest -v
```
