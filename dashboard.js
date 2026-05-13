/* ============================================================
   FaceMask Detection System v2.0 — Dashboard JS
   Real-time stats, chart, log, stream management
   ============================================================ */

'use strict';

// ── STATE ─────────────────────────────────────────────────────────
const STATE = {
  sessionStart:    Date.now(),
  with_mask:       0,
  without_mask:    0,
  mask_incorrect:  0,
  alerts:          0,
  fps:             0,
  alertsEnabled:   true,
  streamPaused:    false,
  prevWith:        0,
  prevWithout:     0,
  prevIncorrect:   0,
  prevAlerts:      0,
  complianceHistory: [],   // [{time, rate}]
  logEntries:      [],
};

const MAX_HISTORY  = 60;   // seconds of chart data
const POLL_INTERVAL = 1500; // ms between API polls

// ── CHART ─────────────────────────────────────────────────────────
let chartCtx  = null;
let chartData = [];

function initChart() {
  const canvas = document.getElementById('compChart');
  if (!canvas) return;
  chartCtx = canvas.getContext('2d');
  renderChart();
}

function renderChart() {
  if (!chartCtx) return;
  const canvas = chartCtx.canvas;
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = 140;

  chartCtx.clearRect(0, 0, W, H);

  const data = STATE.complianceHistory;

  // Background grid
  chartCtx.strokeStyle = 'rgba(0,255,136,0.06)';
  chartCtx.lineWidth   = 1;
  [0, 25, 50, 75, 100].forEach(y => {
    const py = H - (y / 100) * H;
    chartCtx.beginPath();
    chartCtx.moveTo(0, py);
    chartCtx.lineTo(W, py);
    chartCtx.stroke();
    // Labels
    chartCtx.fillStyle = 'rgba(90,122,152,0.7)';
    chartCtx.font = '9px IBM Plex Mono';
    chartCtx.fillText(y + '%', 2, py - 3);
  });

  if (data.length < 2) return;

  // Threshold lines
  [80, 50].forEach((thresh, i) => {
    const py = H - (thresh / 100) * H;
    chartCtx.setLineDash([4, 4]);
    chartCtx.strokeStyle = i === 0 ? 'rgba(0,255,136,0.25)' : 'rgba(255,204,0,0.25)';
    chartCtx.lineWidth = 1;
    chartCtx.beginPath();
    chartCtx.moveTo(0, py);
    chartCtx.lineTo(W, py);
    chartCtx.stroke();
    chartCtx.setLineDash([]);
  });

  // Line
  chartCtx.beginPath();
  data.forEach((d, i) => {
    const x = (i / (MAX_HISTORY - 1)) * W;
    const y = H - (d.rate / 100) * H;
    i === 0 ? chartCtx.moveTo(x, y) : chartCtx.lineTo(x, y);
  });

  // Gradient fill
  const grad = chartCtx.createLinearGradient(0, 0, 0, H);
  const lastRate = data[data.length - 1]?.rate ?? 100;
  const col = lastRate >= 80 ? '0,255,136' : lastRate >= 50 ? '255,204,0' : '255,59,92';
  grad.addColorStop(0,   `rgba(${col},0.3)`);
  grad.addColorStop(0.7, `rgba(${col},0.05)`);
  grad.addColorStop(1,   `rgba(${col},0)`);

  // Draw fill
  chartCtx.lineTo((( (data.length-1) / (MAX_HISTORY - 1)) * W), H);
  chartCtx.lineTo(0, H);
  chartCtx.closePath();
  chartCtx.fillStyle = grad;
  chartCtx.fill();

  // Re-draw line on top
  chartCtx.beginPath();
  data.forEach((d, i) => {
    const x = (i / (MAX_HISTORY - 1)) * W;
    const y = H - (d.rate / 100) * H;
    i === 0 ? chartCtx.moveTo(x, y) : chartCtx.lineTo(x, y);
  });
  chartCtx.strokeStyle = `rgba(${col},0.9)`;
  chartCtx.lineWidth   = 2;
  chartCtx.lineJoin    = 'round';
  chartCtx.stroke();

  // Current point dot
  const lastX = ((data.length - 1) / (MAX_HISTORY - 1)) * W;
  const lastY = H - (lastRate / 100) * H;
  chartCtx.beginPath();
  chartCtx.arc(lastX, lastY, 4, 0, Math.PI * 2);
  chartCtx.fillStyle = `rgb(${col})`;
  chartCtx.fill();
}

// ── UPTIME ────────────────────────────────────────────────────────
function updateUptime() {
  const delta = Date.now() - STATE.sessionStart;
  const h = Math.floor(delta / 3_600_000);
  const m = Math.floor((delta % 3_600_000) / 60_000);
  const s = Math.floor((delta % 60_000) / 1_000);
  const el = document.getElementById('uptime');
  if (el) el.textContent = `${pad(h)}:${pad(m)}:${pad(s)}`;
}
function pad(n) { return String(n).padStart(2, '0'); }

// ── STATS UPDATE ──────────────────────────────────────────────────
function updateStats(data) {
  // Update counts
  STATE.prevWith      = STATE.with_mask;
  STATE.prevWithout   = STATE.without_mask;
  STATE.prevIncorrect = STATE.mask_incorrect;
  STATE.prevAlerts    = STATE.alerts;

  if (data) {
    STATE.with_mask      = data.with_mask      || STATE.with_mask;
    STATE.without_mask   = data.without_mask   || STATE.without_mask;
    STATE.mask_incorrect = data.mask_incorrect || STATE.mask_incorrect;
    STATE.alerts         = data.alerts         || STATE.alerts;
    STATE.fps            = data.fps            || STATE.fps;
  }

  const total = STATE.with_mask + STATE.without_mask + STATE.mask_incorrect;
  const rate  = total > 0 ? (STATE.with_mask / total) * 100 : 100;

  // DOM updates
  setText('statWith',      STATE.with_mask);
  setText('statWithout',   STATE.without_mask);
  setText('statIncorrect', STATE.mask_incorrect);
  setText('statAlerts',    STATE.alerts);

  // Trends
  setText('trendWith',      `+${STATE.with_mask      - STATE.prevWith}`);
  setText('trendWithout',   `+${STATE.without_mask   - STATE.prevWithout}`);
  setText('trendIncorrect', `+${STATE.mask_incorrect  - STATE.prevIncorrect}`);
  setText('trendAlerts',    `+${STATE.alerts          - STATE.prevAlerts}`);

  // Compliance bar
  const fill = document.getElementById('compFill');
  const pct  = document.getElementById('compPct');
  if (fill) {
    fill.style.width = rate.toFixed(1) + '%';
    const col = rate >= 80 ? '#00ff88' : rate >= 50 ? '#ffcc00' : '#ff3b5c';
    fill.style.background = `linear-gradient(90deg, ${col}88, ${col})`;
  }
  if (pct) {
    pct.textContent = rate.toFixed(1) + '%';
    pct.style.color = rate >= 80 ? 'var(--green)' : rate >= 50 ? 'var(--yellow)' : 'var(--red)';
  }

  // FPS
  const fpsBadge = document.getElementById('fpsBadge');
  if (fpsBadge) fpsBadge.textContent = STATE.fps.toFixed(1) + ' FPS';

  // Distribution bars
  if (total > 0) {
    setWidth('distWith',     (STATE.with_mask      / total * 100).toFixed(1));
    setWidth('distWithout',  (STATE.without_mask   / total * 100).toFixed(1));
    setWidth('distIncorrect',(STATE.mask_incorrect  / total * 100).toFixed(1));
    setText('distWithPct',      (STATE.with_mask      / total * 100).toFixed(0) + '%');
    setText('distWithoutPct',   (STATE.without_mask   / total * 100).toFixed(0) + '%');
    setText('distIncorrectPct', (STATE.mask_incorrect  / total * 100).toFixed(0) + '%');
  }

  // Chart history
  STATE.complianceHistory.push({ time: Date.now(), rate });
  if (STATE.complianceHistory.length > MAX_HISTORY) {
    STATE.complianceHistory.shift();
  }
  renderChart();
}

// ── EVENT LOG ─────────────────────────────────────────────────────
function addLogEntry(label, confidence = null) {
  const list = document.getElementById('logList');
  if (!list) return;

  // Remove empty state
  const empty = list.querySelector('.log-empty');
  if (empty) empty.remove();

  const now  = new Date();
  const time = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  const labelText = {
    'with_mask':      'MASK ON',
    'without_mask':   'NO MASK — ALERT',
    'mask_incorrect': 'INCORRECT MASK',
  }[label] || label.toUpperCase();

  const entry = document.createElement('div');
  entry.className = `log-entry ${label}`;
  entry.innerHTML = `
    <div class="log-dot"></div>
    <span class="log-time">${time}</span>
    <span class="log-label">${labelText}</span>
    ${confidence !== null ? `<span class="log-conf">${(confidence*100).toFixed(0)}%</span>` : ''}
  `;

  list.insertBefore(entry, list.firstChild);

  // Keep max 100 entries
  while (list.children.length > 100) {
    list.removeChild(list.lastChild);
  }

  STATE.logEntries.unshift({ time, label, confidence });
}

function clearLog() {
  const list = document.getElementById('logList');
  if (list) list.innerHTML = '<div class="log-empty">Log cleared</div>';
  STATE.logEntries = [];
}

// ── STREAM MANAGEMENT ─────────────────────────────────────────────
function handleStreamError() {
  const img     = document.getElementById('streamImg');
  const offline = document.getElementById('feedOffline');
  if (img)     img.style.display     = 'none';
  if (offline) offline.classList.add('visible');
  // Start demo mode
  startDemoMode();
}

function handleStreamLoaded() {
  const offline = document.getElementById('feedOffline');
  if (offline) offline.classList.remove('visible');
}

let demoInterval = null;
let demoStats    = { with_mask:0, without_mask:0, mask_incorrect:0, alerts:0 };

function startDemoMode() {
  if (demoInterval) return;
  console.log('[SIMRAN] Demo mode: simulating detection events');

  demoInterval = setInterval(() => {
    if (STATE.streamPaused) return;

    // Simulate detections
    const rand = Math.random();
    let label;
    if (rand < 0.72)      label = 'with_mask';
    else if (rand < 0.90) label = 'without_mask';
    else                   label = 'mask_incorrect';

    const conf = 0.85 + Math.random() * 0.14;

    // Update state
    demoStats[label]++;
    if (label !== 'with_mask') demoStats.alerts++;
    demoStats.fps = 24 + Math.random() * 10;

    updateStats(demoStats);
    addLogEntry(label, conf);

    // Flash alert indicator
    if (label !== 'with_mask' && STATE.alertsEnabled) {
      flashAlert();
    }

  }, 1200);
}

function stopDemoMode() {
  if (demoInterval) {
    clearInterval(demoInterval);
    demoInterval = null;
  }
}

function flashAlert() {
  document.body.style.boxShadow = 'inset 0 0 30px rgba(255,59,92,0.15)';
  setTimeout(() => { document.body.style.boxShadow = ''; }, 400);
}

// ── API POLLING ───────────────────────────────────────────────────
async function pollStats() {
  try {
    const res = await fetch('/api/stats', { signal: AbortSignal.timeout(2000) });
    if (!res.ok) throw new Error('API not available');
    const data = await res.json();
    updateStats(data);
    if (data.recent_detections) {
      data.recent_detections.forEach(d => addLogEntry(d.label, d.confidence));
    }
  } catch {
    // Server not running — demo mode handles it
  }
}

// ── CONTROLS ──────────────────────────────────────────────────────
function initControls() {
  // Screenshot
  const screenshotBtn = document.getElementById('screenshotBtn');
  if (screenshotBtn) {
    screenshotBtn.addEventListener('click', () => {
      fetch('/api/screenshot', { method: 'POST' })
        .then(() => showToast('Screenshot saved!'))
        .catch(() => showToast('Screenshot (server not running)'));
    });
  }

  // Alerts toggle
  const alertsBtn = document.getElementById('alertsBtn');
  if (alertsBtn) {
    alertsBtn.addEventListener('click', () => {
      STATE.alertsEnabled = !STATE.alertsEnabled;
      const span = document.getElementById('alertStatus');
      if (span) {
        span.textContent = STATE.alertsEnabled ? 'ON' : 'OFF';
        alertsBtn.style.color = STATE.alertsEnabled ? 'var(--red)' : 'var(--muted2)';
      }
      fetch('/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alerts_enabled: STATE.alertsEnabled })
      }).catch(() => {});
    });
  }

  // Pause stream
  const pauseBtn = document.getElementById('pauseBtn');
  if (pauseBtn) {
    pauseBtn.addEventListener('click', () => {
      STATE.streamPaused = !STATE.streamPaused;
      pauseBtn.textContent = STATE.streamPaused ? '▶ Resume' : '⏸ Pause';
      const img = document.getElementById('streamImg');
      if (img) {
        if (STATE.streamPaused) {
          img.src = '';
        } else {
          img.src = '/api/stream?' + Date.now();
        }
      }
    });
  }

  // Fullscreen
  const fsBtn = document.getElementById('fullscreenBtn');
  if (fsBtn) {
    fsBtn.addEventListener('click', () => {
      const wrap = document.getElementById('feedWrap');
      if (!document.fullscreenElement) {
        wrap?.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen();
      }
    });
  }

  // Export log
  const exportBtn = document.getElementById('exportBtn');
  if (exportBtn) {
    exportBtn.addEventListener('click', exportLog);
  }
}

function exportLog() {
  const rows  = [['time','label','confidence']];
  STATE.logEntries.forEach(e => rows.push([e.time, e.label, e.confidence || '']));
  const csv   = rows.map(r => r.join(',')).join('\n');
  const blob  = new Blob([csv], { type: 'text/csv' });
  const a     = document.createElement('a');
  a.href      = URL.createObjectURL(blob);
  a.download  = `facemask_log_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
  showToast('Log exported as CSV');
}

// ── TOAST ─────────────────────────────────────────────────────────
function showToast(msg) {
  const existing = document.querySelector('.fmd-toast');
  if (existing) existing.remove();
  const t = document.createElement('div');
  t.className = 'fmd-toast';
  t.textContent = msg;
  t.style.cssText = `
    position:fixed;bottom:24px;right:24px;z-index:9999;
    background:#0d1520;border:1px solid rgba(0,255,136,0.3);
    color:#00ff88;font-family:'IBM Plex Mono',monospace;font-size:12px;
    padding:10px 18px;border-radius:4px;
    box-shadow:0 4px 20px rgba(0,0,0,0.5);
    animation:toastIn .3s ease;letter-spacing:.04em;
  `;
  document.head.insertAdjacentHTML('beforeend',
    `<style>@keyframes toastIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}</style>`);
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

// ── UTILS ─────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function setWidth(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = pct + '%';
}

// ── INIT ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  initControls();

  // Uptime ticker
  setInterval(updateUptime, 1000);

  // API polling (falls back to demo mode)
  pollStats();
  setInterval(pollStats, POLL_INTERVAL);

  // Resize chart on window resize
  window.addEventListener('resize', renderChart);

  // Seed chart with initial flat line
  for (let i = 0; i < 10; i++) {
    STATE.complianceHistory.push({ time: Date.now() - (10-i)*1000, rate: 100 });
  }
  renderChart();
});
