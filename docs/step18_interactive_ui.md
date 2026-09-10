# Step 18 — Interactive SIH Prototype UI & Python Engine Integration

**Module**: SIH-26168 Interactive Navigation Demonstration Dashboard  
**Status**: Step 18 Complete | Judge-Ready Interactive Prototype Connected to Python Engine  
**System Readiness Classification**: **YELLOW** (Software & Synthetic Pipelines Fully Validated; Physical Android Vehicle Trials Pending)  
**Evidence Provenance**: `SYNTHETIC_DATA` / `SOFTWARE_FIXTURE` / `OFFLINE_LOG_REPLAY`  
**Test Suite Status**: **221 / 221 PASSING** (0 regressions)

---

## 1. Architectural Overview

The interactive prototype UI transforms the validated navigation algorithms into an accessible, judge-demonstrable local web dashboard without modifying or duplicating the core navigation algorithms.

```
       +-------------------------------------------------------------+
       |                  INTERACTIVE WEB DASHBOARD                  |
       |  (HTML5 / CSS3 / Vanilla JS / Offline 2D Canvas Engines)    |
       |                                                             |
       |  - 2D Tangent Plane Trajectory Canvas (Follow / Pan / Zoom) |
       |  - Real-Time Position Error & Drift % Dynamic Charts        |
       |  - System Pipeline Flow Indicator Ribbons                   |
       |  - Live Geodetic, ENU & Kinematic Telemetry Panels          |
       |  - Motion Constraint Activity (NHC / ZUPT) Monitors         |
       |  - AI Mahalanobis Innovation Gating Diagnostics             |
       |  - Step 17 Master Benchmark Matrix Filterable Table         |
       +------------------------------+------------------------------+
                                      |
                      REST API Polling Loop (~15 Hz)
                                      |
                                      v
       +-------------------------------------------------------------+
       |                 PYTHON DEMO BACKEND SERVER                  |
       |  (src/demo/server.py, api.py, session_manager.py)           |
       |                                                             |
       |  - ThreadingHTTPServer on 127.0.0.1:8080                    |
       |  - Thread-Safe NavigationSessionManager                     |
       |  - Background ReplayController & Guided Demo Sequence       |
       +------------------------------+------------------------------+
                                      |
                           Direct In-Memory Calls
                                      |
                                      v
       +-------------------------------------------------------------+
       |           AUTHORITATIVE PYTHON NAVIGATION ENGINE            |
       |  (src/navigation/pipeline.py: EndToEndNavigationPipeline)   |
       |                                                             |
       |  - Strapdown INS Nominal Kinematics                         |
       |  - 15-State Error-State Kalman Filter (ESKF)                |
       |  - Online GNSS Outage Detector Finite State Machine         |
       |  - Causal Sliding IMU Buffer & PyTorch TCN Drift Predictor  |
       |  - Non-Holonomic Motion Constraints (NHC)                   |
       |  - Multi-Condition Safe Zero-Velocity Updates (ZUPT)        |
       |  - Soft Road Network Map Matching Constraints               |
       +-------------------------------------------------------------+
```

---

## 2. Directory & Component Structure

```text
web/
├── index.html                  # Semantic, accessible, dark-mode single page dashboard
├── styles.css                  # Aerospace-grade dark UI styling with responsive CSS Grid/Flexbox
└── app.js                      # High-performance Vanilla JS engine (Canvas map & chart renderers)

src/demo/
├── __init__.py                 # Module exports
├── session_manager.py          # Real-time session state, dataset streams, and telemetry serializer
├── replay_controller.py        # Asynchronous ticker loop and automated presentation worker
├── api.py                      # REST API routing and request dispatchers
└── server.py                   # Embedded ThreadingHTTPServer serving static files and /api/*

scripts/
└── run_demo.py                 # CLI launcher script with configurable port and scenarios

tests/
├── test_demo_api.py            # Unit tests for all REST endpoints and configuration handlers
└── test_demo_integration.py    # End-to-end integration test over live HTTP socket connection

results/demo/
├── demo_session_summary.json   # Session completion metrics
├── demo_events.json            # Operational event log
└── demo_metrics.json           # Telemetry snapshot
```

---

## 3. Key UI Features & Panels

### A. Provenance & Mode Identification
* **Provenance Badge**: Explicitly declares `DEMO DATA: SYNTHETIC / OFFLINE REPLAY` on all screens to guarantee transparency.
* **Navigation Mode Pill**: Dynamically transitions between:
  * 🟢 `GNSS_AIDED`: Nominal GNSS carrier position aiding.
  * 🔴 `GNSS_OUTAGE`: Dead reckoning active with motion constraints and AI.
  * 🟡 `RECOVERING`: Re-acquisition transient tracking.

### B. System Pipeline Flow Ribbon
An interactive visual flow ribbon indicating live operational states for:
`[ IMU 100Hz ]` ➔ `[ CALIB ]` ➔ `[ INS ]` ➔ `[ 15-STATE ESKF ]` ➔ `[ OUTAGE DET ]` ➔ `[ NHC ]` ➔ `[ ZUPT ]` ➔ `[ TCN AI ]` ➔ `[ MAP MATCH ]` ➔ `[ NAV OUTPUT ]`.

### C. Offline 2D Canvas Trajectory Map
* Plots Local Tangent Plane (ENU) metric coordinates in real time.
* Displays Ground Truth path (gray dashed), Pure INS drift shadow (red), and Full Stack estimated path (cyan).
* Dynamic vehicle marker triangular pointer with geographic heading orientation and pulse halo.
* Toolbar with `Follow Vehicle`, `Fit Trajectory`, and `Reset Zoom`.

### D. Outage & Recovery Demonstration Banner
* Real-time outage stopwatch timer (`00:17.4`).
* Live calculated outage distance and dead-reckoning drift percentage.
* Instantaneous SIH target check badge (`PASS` when drift $<10\%$, `FAIL` when $>10\%$).

### E. Real-Time Telemetry & Sensor Panels
* Speed (km/h & m/s), Heading (degrees and cardinal compass), 2D Position Error (m), Uncertainty $\sqrt{\text{Tr}(P)}$.
* Geodetic Coordinates (WGS84 Lat/Lon/Alt), Local ENU Coordinates, 3D Velocity Vector.
* Live 100 Hz IMU Accelerometer ($a_x, a_y, a_z$), Gyroscope ($\omega_x, \omega_y, \omega_z$), and GNSS Satellite / HDOP diagnostics.

### F. Step 17 Master Benchmark Matrix Tab
* Interactive, filterable table loading all 150 evaluations from `results/final_benchmark/benchmark_summary.json`.
* Filters by Trajectory Scenario (Mixed Urban, Straight, Turning, Stop & Go, Accel/Decel) and Outage Duration (5s, 10s, 20s, 30s, 60s).
* Shows comparative 2D RMSE, final horizontal error, drift %, and official SIH compliance badges.

---

## 4. Launching the Prototype

To launch the local prototype, execute:

```bash
python scripts/run_demo.py --port 8080
```

### CLI Arguments:
* `--port`: Port number to bind (default: `8080`).
* `--host`: Host interface to bind (default: `127.0.0.1`).
* `--scenario`: Initial scenario (`mixed_urban`, `straight`, `turning`, `stop_and_go`, `accel_decel`).
* `--outage-duration`: Outage duration in seconds (default: `30.0`).
* `--auto-open`: Automatically open default web browser.

### Browser URL:
```text
http://127.0.0.1:8080
```

---

## 5. Demonstration Workflow (Judge Walkthrough)

1. Open `http://127.0.0.1:8080`.
2. Click **`✨ Run Full SIH Demo`** or **`▶ Start`**.
3. **Phase 1 (t = 0–30s)**: Observe vehicle moving in 🟢 `GNSS_AIDED` mode with sub-meter accuracy.
4. **Phase 2 (t = 30–60s)**: Click **`🚨 Simulate Outage`** (or let automated demo trigger it):
   * Mode flips to 🔴 `GNSS_OUTAGE`.
   * Outage banner glows red and timer counts active dead reckoning duration.
   * Pure INS begins drifting quadratically (visible as red path divergence).
   * Full Stack (TCN AI + NHC + ZUPT + Map) restrains cross-track drift (visible as cyan path staying aligned).
5. **Phase 3 (t = 60–80s)**: Click **`📡 Restore GNSS`**:
   * Mode transitions to 🟡 `RECOVERING` as carrier lock re-engages.
   * Filter smoothly converges back to 🟢 `GNSS_AIDED` without numerical divergence.
6. **Phase 4**: Switch to the **`📊 Step 17 Benchmark Matrix`** tab to inspect the comprehensive 150-scenario benchmark database.

---

## 6. Verification and Regression Testing

The test suite includes 8 new unit and integration tests for the UI backend, with **zero regressions**:

* **New UI Backend Tests (`tests/test_demo_api.py`)**:
  * `test_api_status_endpoint`: PASS
  * `test_api_telemetry_endpoint`: PASS
  * `test_api_control_flow`: PASS
  * `test_api_outage_and_restore`: PASS
  * `test_api_config_update`: PASS
  * `test_api_benchmark_endpoint`: PASS
* **New Network Integration Tests (`tests/test_demo_integration.py`)**:
  * `test_http_static_assets_serving`: PASS
  * `test_http_live_navigation_cycle`: PASS
* **Total Workspace Test Baseline**: **221 / 221 PASSING**.

---

## 7. Data Provenance & Known Limitations

1. **Synthetic Trajectories**: Generated using high-rate (100 Hz) kinematic simulation with realistic sensor bias drifts, colored noise, and gravity vectors.
2. **Android Log Replay**: Compatible with logged sensor streams; physical in-vehicle dynamic mounting calibration is marked **`NOT VERIFIED`**.
3. **IO-VNBD Dataset**: **`"IO-VNBD raw data unavailable — adapter validated, benchmark execution pending."`**
4. **System Classification**: **`YELLOW`** (All software components, mathematical filters, neural network inference, and UI integrations are verified; physical field vehicle trials remain pending).
