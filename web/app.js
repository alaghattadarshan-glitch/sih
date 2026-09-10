/**
 * SIH-26168 Interactive Navigation Prototype Engine (Vanilla JS)
 * Step 18.1 UI Refinement: High-Impact Visuals, Dynamic Map Scaling,
 * Clear Outage Regions, Plain-English Operation Panel, 100% Offline.
 */

// Application State
let appState = {
  currentTab: 'live',
  viewMode: 'simple', // 'simple' | 'technical'
  followVehicle: true,
  mapScale: 0.55, // pixels per meter
  mapCenter: { x: 0, y: 0 },
  isDragging: false,
  dragStart: { x: 0, y: 0 },
  hasAutoFitted: false,
  features: {
    enable_ai: true,
    enable_nhc: true,
    enable_zupt: true,
    enable_map: true
  },
  benchmarkData: null,
  lastCompletedState: false
};

// Polling interval handle
let telemetryTimer = null;

// Initialize on window load
window.addEventListener('DOMContentLoaded', () => {
  setViewMode('simple');
  initMapInteractions();
  startTelemetryLoop();
  loadBenchmarkData();
});

// -----------------------------------------------------------------------------
// View Mode Switcher (Simple Demo View vs Technical View)
// -----------------------------------------------------------------------------
function setViewMode(mode) {
  appState.viewMode = mode;
  const btnSimple = document.getElementById('btn-view-simple');
  const btnTech = document.getElementById('btn-view-tech');

  if (mode === 'simple') {
    document.body.classList.add('view-mode-simple');
    if (btnSimple) btnSimple.classList.add('active');
    if (btnTech) btnTech.classList.remove('active');
  } else {
    document.body.classList.remove('view-mode-simple');
    if (btnSimple) btnSimple.classList.remove('active');
    if (btnTech) btnTech.classList.add('active');
  }
}

// -----------------------------------------------------------------------------
// Tab Switching
// -----------------------------------------------------------------------------
function switchTab(tabName) {
  appState.currentTab = tabName;
  const liveTab = document.getElementById('view-live');
  const benchTab = document.getElementById('view-benchmark');
  const liveBtn = document.getElementById('tab-live-btn');
  const benchBtn = document.getElementById('tab-bench-btn');

  if (tabName === 'live') {
    liveTab.classList.remove('hidden');
    benchTab.classList.add('hidden');
    liveBtn.classList.add('active');
    benchBtn.classList.remove('active');
  } else {
    liveTab.classList.add('hidden');
    benchTab.classList.remove('hidden');
    liveBtn.classList.remove('active');
    benchBtn.classList.add('active');
    renderBenchmarkTable();
  }
}

// -----------------------------------------------------------------------------
// Telemetry Polling Loop
// -----------------------------------------------------------------------------
function startTelemetryLoop() {
  if (telemetryTimer) clearInterval(telemetryTimer);
  telemetryTimer = setInterval(async () => {
    try {
      const resp = await fetch('/api/telemetry');
      if (resp.ok) {
        const data = await resp.json();
        updateUI(data);
      }
    } catch (err) {
      console.warn('Telemetry fetch error:', err);
    }
  }, 65); // ~15 Hz update rate for silky smooth animations
}

// -----------------------------------------------------------------------------
// UI Updates from Backend Telemetry
// -----------------------------------------------------------------------------
function updateUI(data) {
  if (!data) return;

  const mode = data.navigation_mode || 'GNSS_AIDED';

  // 1. Navigation Mode Pill (Header)
  const modePill = document.getElementById('nav-mode-pill');
  const modeText = document.getElementById('nav-mode-text');
  modeText.textContent = mode;
  modePill.className = 'mode-pill ' + (
    mode === 'GNSS_OUTAGE' ? 'mode-gnss-outage' :
    mode === 'RECOVERING' ? 'mode-recovering' :
    mode === 'DEGRADED' ? 'mode-degraded' : 'mode-gnss-aided'
  );

  // 2. Big Outage Demonstration Card
  const outCard = document.getElementById('outage-card');
  const outIcon = document.getElementById('outage-icon-sym');
  const outTitle = document.getElementById('outage-state-title');
  const outDesc = document.getElementById('outage-state-desc');
  const outTimer = document.getElementById('outage-timer-box');
  const sihBadge = document.getElementById('val-sih-badge');

  if (mode === 'GNSS_OUTAGE') {
    outCard.className = 'card outage-banner active-outage';
    outIcon.textContent = '🔴';
    outTitle.textContent = '🔴 GNSS SIGNAL LOST — DEAD RECKONING ACTIVE';
    outDesc.textContent = 'Navigation continues seamlessly without satellite fix using IMU + TCN AI + Kinematic Constraints';
    outTimer.textContent = formatDuration(data.error_metrics.outage_timer_s);
    outTimer.style.color = '#ff1744';
  } else if (mode === 'RECOVERING') {
    outCard.className = 'card outage-banner recovering-outage';
    outIcon.textContent = '🟡';
    outTitle.textContent = '🟡 GNSS SIGNAL RECOVERED — RE-AIDING FILTER';
    const recCorr = data.error_metrics.recovery_correction_m ? data.error_metrics.recovery_correction_m.toFixed(2) + 'm' : 'evaluating';
    outDesc.textContent = `Carrier tracking re-engaged. Position bias correction: ${recCorr}`;
    outTimer.textContent = formatDuration(data.error_metrics.outage_timer_s);
    outTimer.style.color = '#ff9100';
  } else {
    outCard.className = 'card outage-banner';
    outIcon.textContent = '🛰️';
    outTitle.textContent = '🟢 GNSS SIGNAL NOMINAL';
    outDesc.textContent = 'Full carrier lock active — 15-State ESKF nominal position aiding active';
    outTimer.textContent = '00:00.0';
    outTimer.style.color = '#ffffff';
  }

  document.getElementById('val-out-dist').textContent = `${data.error_metrics.outage_distance_m.toFixed(1)} m`;
  const driftVal = data.error_metrics.drift_percentage;
  document.getElementById('val-out-drift').textContent = `${driftVal.toFixed(2)} %`;

  if (data.error_metrics.sih_target_pass) {
    sihBadge.textContent = 'PASS';
    sihBadge.className = 'val pass-badge';
  } else {
    sihBadge.textContent = 'FAIL';
    sihBadge.className = 'val fail-badge';
  }

  const insErrEl = document.getElementById('val-ins-error');
  if (insErrEl) insErrEl.textContent = `${data.error_metrics.current_ins_error_m.toFixed(2)} m`;

  // 3. "What the System Is Doing" Panel (Current Operation)
  updateOperationPanel(mode, data.subsystems, data.diagnostics);

  // 4. Fast Telemetry Readouts
  document.getElementById('sim-clock').textContent = `${data.session_state.current_time_s.toFixed(1)}s`;
  document.getElementById('sim-scenario-lbl').textContent = (data.session_state.scenario_name || 'Mixed Urban').replace('_', ' ').toUpperCase();
  document.getElementById('tel-speed').innerHTML = `${data.motion.speed_kmh.toFixed(1)} <small>km/h</small>`;
  document.getElementById('tel-speed-mps').textContent = `${data.motion.speed_mps.toFixed(2)} m/s`;
  document.getElementById('tel-heading').textContent = `${data.motion.heading_deg.toFixed(1)}°`;
  document.getElementById('tel-heading-cardinal').textContent = getCardinalHeading(data.motion.heading_deg);

  document.getElementById('tel-err-2d').innerHTML = `${data.error_metrics.current_2d_error_m.toFixed(2)} <small>m</small>`;
  document.getElementById('tel-err-ins').textContent = `INS: ${data.error_metrics.current_ins_error_m.toFixed(2)} m`;
  document.getElementById('tel-uncert').innerHTML = `±${data.position.uncertainty_m.toFixed(2)} <small>m</small>`;
  document.getElementById('tel-cov-trace').textContent = `Tr(P): ${data.diagnostics.trace_P ? data.diagnostics.trace_P.toFixed(2) : '0.0'}`;

  // Coords & Sensors
  const enuEl = document.getElementById('tel-enu');
  if (enuEl) enuEl.textContent = `E: ${data.position.east_m >= 0 ? '+' : ''}${data.position.east_m.toFixed(2)} m | N: ${data.position.north_m >= 0 ? '+' : ''}${data.position.north_m.toFixed(2)} m | U: ${data.position.up_m >= 0 ? '+' : ''}${data.position.up_m.toFixed(2)} m`;
  const llhEl = document.getElementById('tel-llh');
  if (llhEl) llhEl.textContent = `Lat: ${data.position.latitude_deg.toFixed(6)}° | Lon: ${data.position.longitude_deg.toFixed(6)}° | Alt: ${data.position.altitude_m.toFixed(1)} m`;
  const velEl = document.getElementById('tel-vel');
  if (velEl) velEl.textContent = `Ve: ${data.motion.vel_east_mps.toFixed(2)} m/s | Vn: ${data.motion.vel_north_mps.toFixed(2)} m/s | Vu: ${data.motion.vel_up_mps.toFixed(2)} m/s`;

  const a = data.sensor_stream.imu_accel;
  const g = data.sensor_stream.imu_gyro;
  const accelEl = document.getElementById('sens-accel');
  if (accelEl) accelEl.textContent = `X: ${a[0]>=0?'+':''}${a[0].toFixed(2)} | Y: ${a[1]>=0?'+':''}${a[1].toFixed(2)} | Z: ${a[2]>=0?'+':''}${a[2].toFixed(2)}`;
  const gyroEl = document.getElementById('sens-gyro');
  if (gyroEl) gyroEl.textContent = `X: ${g[0]>=0?'+':''}${g[0].toFixed(3)} | Y: ${g[1]>=0?'+':''}${g[1].toFixed(3)} | Z: ${g[2]>=0?'+':''}${g[2].toFixed(3)}`;
  const gnssEl = document.getElementById('sens-gnss');
  if (gnssEl) gnssEl.textContent = `Status: ${data.sensor_stream.gnss_status} | Sats: ${data.sensor_stream.gnss_satellites} | HDOP: ${data.sensor_stream.gnss_hdop.toFixed(1)} | Acc: ±${data.sensor_stream.gnss_accuracy_m.toFixed(1)}m`;

  // 5. Diagnostics Readouts
  const aiDiag = data.diagnostics.ai || {};
  document.getElementById('diag-ai-acc').textContent = aiDiag.accepted_count || 0;
  document.getElementById('diag-ai-rej').textContent = aiDiag.rejected_count || 0;

  const motDiag = data.diagnostics.motion_constraints || {};
  document.getElementById('diag-nhc-acc').textContent = motDiag.nhc_accepted_count || 0;
  document.getElementById('diag-zupt-acc').textContent = motDiag.zupt_accepted_count || 0;
  
  const statEl = document.getElementById('diag-stationary');
  if (statEl) {
    statEl.textContent = motDiag.is_stationary ? '🛑 STATIONARY (ZUPT ACTIVE)' : '🚗 MOVING';
    statEl.style.color = motDiag.is_stationary ? '#00e676' : '#8e9fb8';
  }

  // 6. Subsystem Badges & Flow Ribbon
  updateSubsystems(data.subsystems, mode);

  // 7. Event Log
  updateEventLog(data.events);

  // 8. Render Canvas Map & Error Chart
  renderMap(data.trajectories, data.position, data.motion.heading_deg, data.outage_config);
  renderErrorChart(data.trajectories, data.session_state.current_time_s, mode, data.outage_config);

  // 9. Mission Summary Completion Trigger
  if (data.session_state.is_completed && !appState.lastCompletedState) {
    showMissionSummary(data.mission_summary);
  }
  appState.lastCompletedState = data.session_state.is_completed;
}

// -----------------------------------------------------------------------------
// "What the System Is Doing" Dynamic Text Generator
// -----------------------------------------------------------------------------
function updateOperationPanel(mode, subs, diag) {
  const opCard = document.getElementById('operation-card');
  const opBadge = document.getElementById('op-mode-badge');
  const opBody = document.getElementById('op-body-text');

  if (!opCard || !opBadge || !opBody) return;

  opBadge.textContent = mode;

  if (mode === 'GNSS_OUTAGE') {
    opCard.className = 'card operation-card mode-outage-state';
    opBadge.style.color = '#ff1744';
    opBadge.style.borderColor = 'rgba(255, 23, 68, 0.4)';
    opBadge.style.background = 'rgba(255, 23, 68, 0.15)';

    opBody.innerHTML = `
      <p class="op-primary-msg">
        <strong class="text-red">GNSS signal unavailable.</strong> Navigation continues in Dead-Reckoning mode using:
      </p>
      <div class="active-subsystems-list">
        <span class="op-tag outage-active">✓ IMU (100 Hz Strapdown)</span>
        <span class="op-tag outage-active">✓ 15-State ESKF Propagation</span>
        <span class="op-tag outage-active">✓ TCN AI Drift Displacement</span>
        <span class="op-tag outage-active">✓ Non-Holonomic Constraints (NHC)</span>
        <span class="op-tag outage-active">✓ Zero Velocity Updates (ZUPT)</span>
        <span class="op-tag outage-active">✓ Road Map Geometry Constraints</span>
      </div>
    `;
  } else if (mode === 'RECOVERING') {
    opCard.className = 'card operation-card mode-rec-state';
    opBadge.style.color = '#ff9100';
    opBadge.style.borderColor = 'rgba(255, 145, 0, 0.4)';
    opBadge.style.background = 'rgba(255, 145, 0, 0.15)';

    opBody.innerHTML = `
      <p class="op-primary-msg">
        <strong class="text-amber">GNSS signal recovered.</strong> The 15-State ESKF is re-aiding the navigation solution and correcting accumulated dead-reckoning drift.
      </p>
      <div class="active-subsystems-list">
        <span class="op-tag active">✓ IMU 100Hz</span>
        <span class="op-tag active">✓ ESKF Carrier Re-acquisition</span>
        <span class="op-tag active">✓ Bias Covariance Convergence</span>
        <span class="op-tag">TCN AI (Standby)</span>
      </div>
    `;
  } else {
    opCard.className = 'card operation-card';
    opBadge.style.color = '#00f0ff';
    opBadge.style.borderColor = 'rgba(0, 240, 255, 0.3)';
    opBadge.style.background = 'rgba(0, 240, 255, 0.15)';

    opBody.innerHTML = `
      <p class="op-primary-msg">
        <strong class="text-green">GNSS is available.</strong> The 15-State ESKF is using 1 Hz GNSS satellite positions to continuously estimate and correct inertial sensor errors.
      </p>
      <div class="active-subsystems-list">
        <span class="op-tag active">✓ IMU 100Hz</span>
        <span class="op-tag active">✓ INS Mechanization</span>
        <span class="op-tag active">✓ 15-State ESKF Aiding</span>
        <span class="op-tag">TCN AI (Standby)</span>
        <span class="op-tag">NHC (Standby)</span>
        <span class="op-tag">ZUPT (Standby)</span>
        <span class="op-tag">Map Matching (Standby)</span>
      </div>
    `;
  }
}

function updateSubsystems(subs, mode) {
  if (!subs) return;
  const subKeys = ['gnss', 'imu', 'ins', 'eskf', 'ai', 'nhc', 'zupt', 'map'];
  for (const k of subKeys) {
    const elBadge = document.getElementById(`sub-${k === 'map' ? 'map' : k}`);
    const elText = document.getElementById(`st-${k === 'map' ? 'map' : k}`);
    const pipeNode = document.getElementById(`pipe-${k === 'map' ? 'map' : k}`);
    const val = subs[k === 'map' ? 'map_matching' : k] || 'INACTIVE';

    if (elText) elText.textContent = val;
    if (elBadge) {
      elBadge.className = 'subsystem-badge ' + (
        val === 'ACTIVE' ? 'active' :
        val === 'OUTAGE' ? 'outage' :
        val === 'RECOVERING' ? 'recovering' : ''
      );
    }
    if (pipeNode) {
      pipeNode.className = 'pipe-node ' + (
        val === 'ACTIVE' ? 'active' :
        val === 'OUTAGE' ? 'outage' : ''
      );
    }
  }

  const outPipe = document.getElementById('pipe-outage');
  if (outPipe) {
    outPipe.className = 'pipe-node ' + (mode === 'GNSS_OUTAGE' ? 'outage' : (mode === 'RECOVERING' ? 'active' : ''));
  }
}

function updateEventLog(events) {
  if (!events || events.length === 0) return;
  const logContainer = document.getElementById('event-log');
  if (!logContainer) return;
  let html = '';
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    html += `<div class="log-entry log-${ev.type}">[${ev.wall_time} / ${ev.timestamp.toFixed(1)}s] ${ev.message}</div>`;
  }
  logContainer.innerHTML = html;
}

// -----------------------------------------------------------------------------
// Dynamic Map Renderer with Active Trajectory Auto-Scaling
// -----------------------------------------------------------------------------
function renderMap(traj, currentPos, headingDeg, outageConfig) {
  const canvas = document.getElementById('map-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  // Clear background
  ctx.fillStyle = '#030509';
  ctx.fillRect(0, 0, w, h);

  if (!traj) return;

  const gt = traj.ground_truth || [];
  const est = traj.estimated || [];
  const ins = traj.pure_ins || [];

  // Dynamic Trajectory Bounding Box Calculation
  if (gt.length > 5 && !appState.hasAutoFitted) {
    let minE = Infinity, maxE = -Infinity, minN = Infinity, maxN = -Infinity;
    for (const p of gt) {
      if (p[0] < minE) minE = p[0];
      if (p[0] > maxE) maxE = p[0];
      if (p[1] < minN) minN = p[1];
      if (p[1] > maxN) maxN = p[1];
    }
    const spanE = Math.max(80, maxE - minE);
    const spanN = Math.max(80, maxN - minN);
    const pad = 60;
    const scaleX = (w - pad * 2) / spanE;
    const scaleY = (h - pad * 2) / spanN;
    appState.mapScale = Math.min(scaleX, scaleY);
    appState.mapCenter.x = (minE + maxE) / 2;
    appState.mapCenter.y = (minN + maxN) / 2;
    appState.hasAutoFitted = true;
  }

  // Auto-follow vehicle
  if (appState.followVehicle && currentPos) {
    appState.mapCenter.x = currentPos.east_m;
    appState.mapCenter.y = currentPos.north_m;
  }

  const scale = appState.mapScale;
  const cx = w / 2;
  const cy = h / 2;

  // World ENU (East, North) to Canvas (x, y)
  const toScreen = (e, n) => ({
    x: cx + (e - appState.mapCenter.x) * scale,
    y: cy - (n - appState.mapCenter.y) * scale // Invert Y
  });

  // 1. Draw Grid Lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  const gridSize = 100;
  const minE = appState.mapCenter.x - (cx / scale);
  const maxE = appState.mapCenter.x + (cx / scale);
  const minN = appState.mapCenter.y - (cy / scale);
  const maxN = appState.mapCenter.y + (cy / scale);

  const startGridE = Math.floor(minE / gridSize) * gridSize;
  const startGridN = Math.floor(minN / gridSize) * gridSize;

  for (let e = startGridE; e <= maxE; e += gridSize) {
    const p1 = toScreen(e, minN);
    const p2 = toScreen(e, maxN);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }
  for (let n = startGridN; n <= maxN; n += gridSize) {
    const p1 = toScreen(minE, n);
    const p2 = toScreen(maxE, n);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }

  // 2. Origin Marker
  const p0 = toScreen(0, 0);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
  ctx.beginPath();
  ctx.arc(p0.x, p0.y, 4, 0, Math.PI * 2);
  ctx.fill();

  // 3. Draw Ground Truth Path (Clean dashed light slate)
  if (gt.length > 1) {
    ctx.strokeStyle = 'rgba(140, 165, 205, 0.7)';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    for (let i = 0; i < gt.length; i++) {
      const pt = toScreen(gt[i][0], gt[i][1]);
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 4. Draw Pure INS Baseline (Vibrant Red)
  if (ins.length > 1) {
    ctx.strokeStyle = 'rgba(255, 23, 68, 0.85)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < ins.length; i++) {
      const pt = toScreen(ins[i][0], ins[i][1]);
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();
  }

  // 5. Draw Full-Stack Filter Estimated Path (Glowing Cyan)
  if (est.length > 1) {
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 3.5;
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    for (let i = 0; i < est.length; i++) {
      const pt = toScreen(est[i][0], est[i][1]);
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0; // reset
  }

  // 6. Draw Outage Highlight Region on Map Trajectory
  if (traj.time_series && traj.time_series.length > 1 && outageConfig) {
    const startT = outageConfig.start_s || 30.0;
    const endT = outageConfig.end_s || 60.0;

    let inOutage = false;
    ctx.strokeStyle = '#ff9100';
    ctx.lineWidth = 6;
    ctx.shadowColor = '#ff9100';
    ctx.shadowBlur = 12;

    for (let i = 0; i < traj.time_series.length; i++) {
      const t = traj.time_series[i];
      if (t >= startT && t <= endT) {
        const pt = toScreen(est[i][0], est[i][1]);
        if (!inOutage) {
          ctx.beginPath();
          ctx.moveTo(pt.x, pt.y);
          inOutage = true;
        } else {
          ctx.lineTo(pt.x, pt.y);
        }
      } else if (inOutage) {
        ctx.stroke();
        inOutage = false;
      }
    }
    if (inOutage) ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // 7. Draw Current Vehicle Marker with Heading
  if (currentPos) {
    const vPos = toScreen(currentPos.east_m, currentPos.north_m);
    const yawRad = (headingDeg * Math.PI) / 180; // 0=North, 90=East

    ctx.save();
    ctx.translate(vPos.x, vPos.y);
    ctx.rotate(yawRad);

    // Glowing outer halo
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.7)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 0, 16, 0, Math.PI * 2);
    ctx.stroke();

    // Inner vehicle triangle pointer
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(0, -14); // Tip
    ctx.lineTo(8, 10);
    ctx.lineTo(0, 5);
    ctx.lineTo(-8, 10);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }
}

// -----------------------------------------------------------------------------
// Canvas Error Chart Renderer with Outage Region
// -----------------------------------------------------------------------------
function renderErrorChart(traj, currentT, mode, outageConfig) {
  const canvas = document.getElementById('error-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#030509';
  ctx.fillRect(0, 0, w, h);

  if (!traj || !traj.time_series || traj.time_series.length < 2) return;

  const times = traj.time_series;
  const err2d = traj.error_2d_series;
  const errIns = traj.error_ins_series;

  const padLeft = 42;
  const padBottom = 22;
  const padTop = 18;
  const padRight = 14;

  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;

  const maxT = Math.max(60, times[times.length - 1]);
  let maxErr = 10.0;
  for (const e of err2d) if (e > maxErr) maxErr = e;
  for (const e of errIns) if (e > maxErr) maxErr = e;
  maxErr = Math.ceil(maxErr * 1.15);

  const startT = outageConfig ? outageConfig.start_s : 30.0;
  const endT = outageConfig ? outageConfig.end_s : 60.0;

  // 1. Shaded GNSS Outage Region Band
  const outX1 = padLeft + (startT / maxT) * chartW;
  const outX2 = padLeft + (endT / maxT) * chartW;
  const bandW = Math.max(0, outX2 - outX1);

  ctx.fillStyle = 'rgba(255, 145, 0, 0.15)';
  ctx.fillRect(outX1, padTop, bandW, chartH);

  ctx.strokeStyle = 'rgba(255, 145, 0, 0.4)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(outX1, padTop);
  ctx.lineTo(outX1, h - padBottom);
  ctx.moveTo(outX2, padTop);
  ctx.lineTo(outX2, h - padBottom);
  ctx.stroke();
  ctx.setLineDash([]);

  // Label inside Outage Region
  ctx.fillStyle = 'rgba(255, 145, 0, 0.85)';
  ctx.font = '700 9px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText('GNSS OUTAGE', outX1 + bandW / 2, padTop + 14);

  // 2. Axes & Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop);
  ctx.lineTo(padLeft, h - padBottom);
  ctx.lineTo(w - padRight, h - padBottom);
  ctx.stroke();

  ctx.fillStyle = '#7a8ba8';
  ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'right';
  ctx.fillText(`${maxErr.toFixed(0)}m`, padLeft - 6, padTop + 8);
  ctx.fillText('0m', padLeft - 6, h - padBottom);

  ctx.textAlign = 'center';
  ctx.fillText('0s', padLeft, h - 6);
  ctx.fillText(`${maxT.toFixed(0)}s`, w - padRight, h - 6);

  // 3. Plot Pure INS Baseline Error (Red)
  ctx.strokeStyle = '#ff1744';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < times.length; i++) {
    const x = padLeft + (times[i] / maxT) * chartW;
    const y = (h - padBottom) - (errIns[i] / maxErr) * chartH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // 4. Plot Full Stack 2D Error (Cyan)
  ctx.strokeStyle = '#00f0ff';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < times.length; i++) {
    const x = padLeft + (times[i] / maxT) * chartW;
    const y = (h - padBottom) - (err2d[i] / maxErr) * chartH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// -----------------------------------------------------------------------------
// Interactive Map Navigation Controls
// -----------------------------------------------------------------------------
function initMapInteractions() {
  const canvas = document.getElementById('map-canvas');
  if (!canvas) return;

  canvas.addEventListener('mousedown', (e) => {
    appState.isDragging = true;
    appState.dragStart = { x: e.clientX, y: e.clientY };
    appState.followVehicle = false;
    const btn = document.getElementById('btn-follow');
    if (btn) {
      btn.textContent = 'Follow: OFF';
      btn.classList.remove('active');
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!appState.isDragging) return;
    const dx = (e.clientX - appState.dragStart.x) / appState.mapScale;
    const dy = (e.clientY - appState.dragStart.y) / appState.mapScale;
    appState.mapCenter.x -= dx;
    appState.mapCenter.y += dy;
    appState.dragStart = { x: e.clientX, y: e.clientY };
  });

  window.addEventListener('mouseup', () => {
    appState.isDragging = false;
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    appState.mapScale = Math.max(0.05, Math.min(4.0, appState.mapScale * zoomFactor));
  });
}

function toggleFollow() {
  appState.followVehicle = !appState.followVehicle;
  const btn = document.getElementById('btn-follow');
  if (btn) {
    btn.textContent = `Follow: ${appState.followVehicle ? 'ON' : 'OFF'}`;
    if (appState.followVehicle) btn.classList.add('active');
    else btn.classList.remove('active');
  }
}

function fitMap() {
  appState.followVehicle = false;
  appState.hasAutoFitted = false;
  const btn = document.getElementById('btn-follow');
  if (btn) {
    btn.textContent = 'Follow: OFF';
    btn.classList.remove('active');
  }
}

function resetMap() {
  appState.followVehicle = true;
  appState.hasAutoFitted = false;
  const btn = document.getElementById('btn-follow');
  if (btn) {
    btn.textContent = 'Follow: ON';
    btn.classList.add('active');
  }
}

// -----------------------------------------------------------------------------
// Mission Summary Modal Controls
// -----------------------------------------------------------------------------
function showMissionSummary(summary) {
  if (!summary) return;
  const modal = document.getElementById('mission-summary-modal');
  if (!modal) return;

  document.getElementById('sum-drift').textContent = `${summary.drift_percentage.toFixed(2)} %`;
  const badge = document.getElementById('sum-sih-badge');
  if (summary.sih_target_pass) {
    badge.textContent = 'PASS';
    badge.className = 'sh-badge pass';
  } else {
    badge.textContent = 'FAIL';
    badge.className = 'sh-badge fail';
  }

  document.getElementById('sum-recovery').textContent = summary.recovery_status;
  document.getElementById('sum-scenario').textContent = summary.scenario_name;
  document.getElementById('sum-outage-dur').textContent = `${summary.outage_duration_s.toFixed(1)} s`;
  document.getElementById('sum-outage-dist').textContent = `${summary.outage_distance_m.toFixed(1)} m`;
  document.getElementById('sum-ins-err').textContent = `${summary.pure_ins_error_m.toFixed(2)} m`;
  document.getElementById('sum-fs-err').textContent = `${summary.full_stack_error_m.toFixed(2)} m`;
  document.getElementById('sum-rec-time').textContent = `${summary.recovery_time_s.toFixed(2)} s`;
  document.getElementById('sum-data-src').textContent = summary.data_source;

  modal.classList.remove('hidden');
}

function closeMissionSummary() {
  const modal = document.getElementById('mission-summary-modal');
  if (modal) modal.classList.add('hidden');
}

function runDemoAgain() {
  closeMissionSummary();
  sendControl('demo/run');
}

// -----------------------------------------------------------------------------
// Benchmark Matrix Tab Loader
// -----------------------------------------------------------------------------
async function loadBenchmarkData() {
  try {
    const resp = await fetch('/api/benchmark');
    if (resp.ok) {
      appState.benchmarkData = await resp.json();
    }
  } catch (err) {
    console.warn('Failed to load benchmark data:', err);
  }
}

function renderBenchmarkTable() {
  if (!appState.benchmarkData || !appState.benchmarkData.benchmark) return;
  const tbody = document.getElementById('bench-table-body');
  if (!tbody) return;

  const results = appState.benchmarkData.benchmark.results || [];
  const filterScen = document.getElementById('bench-filter-scen').value;
  const filterDur = document.getElementById('bench-filter-dur').value;

  let html = '';
  for (const r of results) {
    if (filterScen !== 'ALL' && r.scenario_type !== filterScen) continue;
    if (filterDur !== 'ALL' && Math.abs(r.outage_duration_s - parseFloat(filterDur)) > 0.1) continue;

    const statusCls = r.sih_target_status === 'PASS' ? 'status-pill-pass' :
                      r.sih_target_status === 'WARN' ? 'status-pill-warn' : 'status-pill-fail';

    html += `
      <tr>
        <td><strong>${r.scenario_type.replace('_', ' ').toUpperCase()}</strong></td>
        <td>${r.config_name}</td>
        <td>${r.outage_duration_s}s</td>
        <td>${r.outage_distance_m.toFixed(1)}m</td>
        <td>${r.outage_rmse_2d_m.toFixed(2)}m</td>
        <td>${r.final_horizontal_error_m.toFixed(2)}m</td>
        <td><strong>${r.drift_percentage.toFixed(2)}%</strong></td>
        <td><span class="${statusCls}">${r.sih_target_status}</span></td>
      </tr>
    `;
  }
  tbody.innerHTML = html || '<tr><td colspan="8" style="text-align:center;color:#8a99b5;">No matching results found.</td></tr>';
}

// -----------------------------------------------------------------------------
// Control Actions & API Dispatch
// -----------------------------------------------------------------------------
async function sendControl(action, payload = {}) {
  try {
    await fetch(`/api/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (err) {
    console.error(`Control action '${action}' failed:`, err);
  }
}

async function updateConfig() {
  const scenario = document.getElementById('sel-scenario').value;
  const duration = parseFloat(document.getElementById('sel-duration').value);
  const speed = parseFloat(document.getElementById('sel-speed').value);

  await sendControl('config', {
    scenario: scenario,
    outage_duration: duration,
    playback_speed: speed
  });
}

async function toggleFeature(featName) {
  appState.features[featName] = !appState.features[featName];
  const btn = document.getElementById('tog-' + featName.replace('enable_', ''));
  if (appState.features[featName]) {
    if (btn) btn.classList.add('active');
  } else {
    if (btn) btn.classList.remove('active');
  }

  const payload = {};
  payload[featName] = appState.features[featName];
  await sendControl('config', payload);
}

// -----------------------------------------------------------------------------
// Helper Utilities
// -----------------------------------------------------------------------------
function formatDuration(sec) {
  if (!sec || isNaN(sec)) return '00:00.0';
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m.toString().padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

function getCardinalHeading(deg) {
  const cardinals = ['East', 'North-East', 'North', 'North-West', 'West', 'South-West', 'South', 'South-East'];
  const idx = Math.round(deg / 45) % 8;
  return cardinals[idx];
}
