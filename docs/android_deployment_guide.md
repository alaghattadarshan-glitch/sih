# Android Physical Device Deployment & Live Sensor Field Manual

## 1. Prerequisites & Environment Setup

### Required Toolchain
- **Android Studio**: Iguana (2023.2.1) or newer
- **Android Gradle Plugin (AGP)**: 8.3.0
- **Gradle Version**: 8.4+
- **Android SDK**: Compile SDK 34, Minimum SDK 26 (Android 8.0 Oreo)
- **Android NDK**: 25.2.9519653 or newer
- **CMake**: 3.22.1+
- **Target ABI**: `arm64-v8a` (Modern 64-bit ARM smartphones)

---

## 2. Building & Installing the APK via ADB

### Step 1: Connect Physical Device
Enable **Developer Options** and **USB Debugging** on the target Android device:
```bash
# Verify device connection
adb devices -l
```

### Step 2: Build Native & Java/Kotlin Packages
```bash
cd android
./gradlew assembleDebug
```
*Generated APK path*: `android/app/build/outputs/apk/debug/app-debug.apk`

### Step 3: Install APK onto Target Smartphone
```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Step 4: Grant Permissions
```bash
adb shell pm grant com.sih26168.navigation android.permission.ACCESS_FINE_LOCATION
adb shell pm grant com.sih26168.navigation android.permission.ACCESS_COARSE_LOCATION
```

---

## 3. Physical Mounting & Sensor Alignment Protocol

Canonical body frame is defined as:
- **+x**: Forward (Vehicle longitudinal driving axis)
- **+y**: Left (Vehicle lateral axis out left door)
- **+z**: Up (Perpendicular to road towards sky)

### Supported Mounting Presets
1. **Flat Portrait (Default)**: Phone placed flat on center console or passenger seat with top edge pointing toward windshield.
2. **Windshield Vertical**: Phone attached to windshield suction mount in vertical portrait facing the driver.
3. **Flat Landscape**: Phone placed flat with top edge facing left door.
4. **Custom Euler Alignment**: Configure arbitrary roll, pitch, yaw offsets via `MountingTransform.fromEulerDeg(roll, pitch, yaw)`.

---

## 4. Live Startup Calibration Protocol

1. Securely mount the device in the vehicle holder or place flat on stationary surface.
2. Launch the application and press **`Calibrate`**.
3. **Keep the vehicle and smartphone completely stationary for 3.0 seconds**.
   - The engine measures accelerometer gravity alignment and initial gyroscope biases ($\mathbf{b}_g$).
   - If vibration or motion is detected ($\text{Var}(\mathbf{a}) > 0.05\text{ m/s}^2$ or $\text{Var}(\boldsymbol{\omega}) > 0.005\text{ rad/s}$), the UI returns `CALIBRATION_FAILED_MOTION_DETECTED` and prompts retry.
4. Upon `CALIBRATION_SUCCESS`, navigation transitions to `GNSS_AIDED`.

---

## 5. Field Testing Protocols

### Test A: Stationary Baseline (5 minutes)
- Verify zero drift runaway; position uncertainty circle should remain $<1.5\text{ m}$.

### Test B: Straight-Line Road Segment
- Verify forward velocity tracking and consistency between GNSS speed and S-INS integration.

### Test C: 90-Degree Turns & Roundabouts
- Verify gyroscope integration and yaw heading responsiveness without lag or gimbal lock.

### Test D: Simulated GPS Outage (Underpass / Tunnel Simulation)
1. While driving at steady speed, press **`Simulate Outage`** on the validation dashboard.
2. Observe navigation state transition: `GNSS_AIDED` $\to$ `GNSS_OUTAGE`.
3. AI model outputs 1-second displacement pseudo-measurements $[\Delta E, \Delta N, \Delta U]$; ESKF updates position with Mahalanobis innovation gating.
4. After 20–30 seconds, press **`Restore GNSS`**.
5. Observe smooth transition: `GNSS_OUTAGE` $\to$ `RECOVERING` $\to$ `GNSS_AIDED` without state discontinuities.

---

## 6. Extracting Offline Diagnostic Logs

Logs are written asynchronously to the application data directory:
```bash
# Pull live session CSV to host machine for offline evaluation
adb pull /sdcard/Android/data/com.sih26168.navigation/files/ results/android_device/
```
Then replay through the Python reference pipeline:
```bash
python scripts/replay_edge_session.py --input results/android_device/live_session_*.csv
```
