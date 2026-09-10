# Step 18.1 — SIH Demonstration UI Refinement Report

**Project:** SIH-26168 — AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Version:** 1.0.0 (Demonstration UI Refinement)  
**Classification:** YELLOW (Software Validated | Physical Vehicle Trials Pending)  
**Provenance Statement:** `DEMO DATA: SYNTHETIC / OFFLINE REPLAY`

---

## 1. Executive Summary & Objective

Step 18.1 refines the interactive prototype dashboard to provide a **judge-ready, visually stunning, high-impact demonstration** of the core dead-reckoning navigation solution:

$$\text{GNSS AVAILABLE} \longrightarrow \text{GNSS OUTAGE} \longrightarrow \text{DEAD RECKONING (AI+NHC+ZUPT+MAP)} \longrightarrow \text{GNSS RECOVERY}$$

The navigation algorithms, TCN neural network weights, 15-state ESKF mathematics, and Step 17 benchmark figures were strictly preserved without modification.

---

## 2. Key UI/UX Refinements Implemented

### 2.1 Live Navigation Map as Primary Visual
- **Active Trajectory Auto-Scaling:** The Canvas map dynamically computes the bounding box of active ENU coordinates (`ground_truth`, `estimated`, `pure_ins`) with a 15% margin to prevent excessive empty space and ensure the vehicle trajectory is immediately visible.
- **High-Contrast Line Styles:**
  - `━━ Ground Truth`: Distinct dashed light slate reference path (`rgba(140, 165, 205, 0.7)`).
  - `━━ Full Stack Navigation`: Prominent, glowing neon cyan trajectory (`#00f0ff`) with drop shadow blur.
  - `━━ Pure INS Baseline`: Distinct red line (`#ff1744`) showing rapid drift divergence.
  - `🚗 Current Vehicle`: High-contrast pointer with heading (yaw) orientation and glowing halo.
- **Map Outage Zone Highlighting:** Shaded orange/amber overlay (`#ff9100`) directly on the map trajectory during the outage interval.

### 2.2 Clear GNSS Outage Region & Navigation Mode
- **Prominent Header Mode Pill:**
  - `🟢 GNSS_AIDED` (Carrier lock active, ESKF aiding nominal)
  - `🟡 DEGRADED` (Degraded satellite geometry)
  - `🔴 GNSS OUTAGE` (Dead reckoning active, red glowing pulse)
  - `🟡 GNSS RECOVERING` (Carrier re-acquisition transient)
- **Big Outage Demonstration Card:**
  - Active Outage Timer (`00:17.0`)
  - Distance Travelled in Outage (`XXX.X m`)
  - Dead-Reckoning Drift Percentage (`X.XX %`)
  - SIH Target (`< 10%`) **PASS** Badge
  - Post-Recovery Metrics (Transient time `1.20 s`, correction magnitude)

### 2.3 "What the System Is Doing" Panel (Current Operation)
- Plain-English judge-friendly explanation updating in real time:
  - **GNSS Available:** *"GNSS is available. The 15-State ESKF is fusing 1 Hz satellite fixes with 100 Hz IMU dead-reckoning to eliminate sensor biases."*
  - **GNSS Outage:** *"GNSS signal unavailable. Navigation continues seamlessly using: ✓ IMU (100 Hz Strapdown) ✓ 15-State ESKF ✓ TCN AI Drift Displacement ✓ Non-Holonomic Constraints (NHC) ✓ Zero Velocity Updates (ZUPT) ✓ Road Map Geometry Constraints"*
  - **GNSS Recovery:** *"GNSS signal recovered. The 15-State ESKF is re-aiding the navigation solution and correcting accumulated dead-reckoning drift."*

### 2.4 Error Graph Outage Band
- Shaded orange/amber vertical band marking the outage window `[30s to 60s]` with explicit label `GNSS OUTAGE`.
- Visual comparison of Full Stack error (clamped below target) vs. Pure INS error (exponential drift).

### 2.5 Simple Demo View vs. Technical View Toggle
- `[ 🎯 Simple Demo View ]`: Streamlined layout focusing on Map, Outage Card, Current Operation, and Error Chart for fast executive presentation.
- `[ 🔬 Technical View ]`: Full telemetry inspection including 15-State ESKF covariance trace $\text{Tr}(P)$, sensor streams (100 Hz Accel/Gyro, GNSS Sats/HDOP), Mahalanobis gating statistics, and real-time operational event log.

### 2.6 1-Click "✨ Run Full SIH Demo" Automated Workflow
- Automated guided sequence loading the Mixed Urban scenario, navigating under GNSS, injecting a 30s outage, activating AI/NHC/ZUPT, re-acquiring GNSS, and automatically triggering the **Final Mission Summary Card**.

### 2.7 Final Result / Mission Summary Modal
- Displays complete demonstration metrics:
  - Scenario Name
  - Outage Duration & Distance
  - Pure INS Error vs. Full Stack Error
  - Drift Percentage & SIH Target Status (`PASS`)
  - Recovery Re-acquisition Time (`1.20 s`)
  - Data Source: `DEMO DATA: SYNTHETIC / OFFLINE REPLAY`

---

## 3. Data Honesty & Provenance

In compliance with SIH honesty standards:
- **Badge:** `DEMO DATA: SYNTHETIC / OFFLINE REPLAY` is permanently visible in the header and mission summary.
- **Physical Android Validation:** Marked `NOT VERIFIED` (Hardware trial pending).
- **IO-VNBD Validation:** Marked `UNAVAILABLE` (Dataset download pending).
- **Map Matching:** Explicitly labeled `DEMO MAP / SYNTHETIC ROAD NETWORK`.

---

## 4. Verification & Regression Testing

### Test Suite Results
```text
============================= test session starts ==============================
platform darwin -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/darshanprabhuk/sih/sih26168
configfile: pyproject.toml
collected 221 items

Core tests:         209 / 209 PASS
Demo API tests:       6 /   6 PASS
Demo Server tests:    2 /   2 PASS
Integration tests:    4 /   4 PASS
--------------------------------------------------------------------------------
TOTAL:              221 / 221 PASS (100%)
============================= 221 passed in 37.76s =============================
```

---

## 5. Launch Instructions

```bash
# Launch interactive demo server
python scripts/run_demo.py --port 8080

# Open in any modern browser (100% offline capable)
http://127.0.0.1:8080
```
