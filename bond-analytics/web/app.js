// Bond price comparison chart. No date-adapter library is vendored, so the x
// axis is a plain linear scale of millisecond timestamps with a tick/tooltip
// callback that formats them back into dates -- avoids needing to bundle a
// second library (date-fns/luxon) just for the axis.

const PALETTE_LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
const PALETTE_DARK  = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const FALLBACK_COLOR = '#898781'; // muted ink -- used past the 8th series

const RANGES = [
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '6M', days: 182 },
  { label: '1Y', days: 365 },
  { label: '5Y', days: 1825 },
  { label: 'All', days: null },
];

const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
const themeAttr = document.documentElement.getAttribute('data-theme');
const isDark = themeAttr ? themeAttr === 'dark' : prefersDark;
const PALETTE = isDark ? PALETTE_DARK : PALETTE_LIGHT;

const statusEl = document.getElementById('status');
const bondListEl = document.getElementById('bondList');
const rangeButtonsEl = document.getElementById('rangeButtons');
const refreshBtn = document.getElementById('refreshBtn');
const canvas = document.getElementById('chart');

let chart = null;
let bonds = [];        // [{cusip, label, color, history, stale, error, visible}]
let activeRangeIdx = RANGES.length - 1; // default "All"

function colorFor(index) {
  return index < PALETTE.length ? PALETTE[index] : FALLBACK_COLOR;
}

// "YYYY-MM-DD" parsed via `new Date(str)` is treated as UTC midnight, then
// formatted in the browser's local time -- for any timezone west of UTC that
// silently shifts the displayed date back a day. Parsing the parts directly
// into the local-time Date constructor avoids that.
function parseDateLocal(dateStr) {
  const [y, m, d] = dateStr.split('-').map(Number);
  return new Date(y, m - 1, d);
}

function abbreviateIssuer(issuer) {
  if (!issuer) return '?';
  return issuer === 'United States Treasury' ? 'US Treasury' : issuer;
}

function formatCoupon(coupon) {
  // Number(...).toString() drops trailing zeros (4.625 -> "4.625", 5 -> "5"),
  // matching how the server formats it elsewhere without a second format string.
  return coupon != null ? Number(coupon).toString() : '?';
}

function setStatus(text, isError) {
  statusEl.textContent = text;
  statusEl.classList.toggle('error', !!isError);
}

async function loadData() {
  setStatus('Fetching bond price history from Webull…');
  const [watchlistResp, historyResp] = await Promise.all([
    fetch('/api/watchlist').then(r => r.json()),
    fetch('/api/history').then(r => r.json()),
  ]);

  const priorVisibility = new Map(bonds.map(b => [b.cusip, b.visible]));

  bonds = watchlistResp.cusips.map((cusip, i) => {
    const entry = historyResp[cusip] || { history: [], stale: true, error: 'not returned' };
    return {
      cusip,
      label: entry.label || cusip,
      issuer: entry.issuer || null,
      coupon: entry.coupon != null ? entry.coupon : null,
      maturity: entry.maturity || null,
      color: colorFor(i),
      history: entry.history || [],
      stale: !!entry.stale,
      error: entry.error || null,
      visible: priorVisibility.has(cusip) ? priorVisibility.get(cusip) : true,
    };
  });

  const errCount = bonds.filter(b => b.error).length;
  const now = new Date().toLocaleTimeString();
  setStatus(
    errCount ? `Updated ${now} -- ${errCount} of ${bonds.length} bond(s) using cached/stale data.` : `Updated ${now}.`,
    errCount > 0
  );
}

function renderBondList() {
  bondListEl.innerHTML = '';
  bonds.forEach((bond, i) => {
    const row = document.createElement('label');
    row.className = 'bond-row';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = bond.visible;
    checkbox.style.setProperty('--swatch', bond.color);
    checkbox.addEventListener('change', () => {
      bond.visible = checkbox.checked;
      updateChart();
    });

    const textWrap = document.createElement('div');

    const line1 = document.createElement('div');
    line1.className = 'bond-line1';
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = bond.color;
    const labelText = document.createElement('span');
    labelText.className = 'bond-label';
    labelText.textContent = `${abbreviateIssuer(bond.issuer)} ${formatCoupon(bond.coupon)}%`;
    line1.appendChild(swatch);
    line1.appendChild(labelText);

    const maturityText = document.createElement('div');
    maturityText.className = 'bond-maturity';
    maturityText.textContent = `Maturity: ${bond.maturity ? formatDate(parseDateLocal(bond.maturity).getTime()) : '?'}`;

    const cusipText = document.createElement('div');
    cusipText.className = 'bond-cusip';
    cusipText.textContent = bond.cusip;

    textWrap.appendChild(line1);
    textWrap.appendChild(maturityText);
    textWrap.appendChild(cusipText);

    if (bond.error) {
      const warn = document.createElement('div');
      warn.className = 'bond-warn';
      warn.textContent = bond.stale && bond.history.length
        ? `Live fetch failed, showing cached data (${bond.error})`
        : `No data: ${bond.error}`;
      textWrap.appendChild(warn);
    } else if (bond.history.length <= 1) {
      const sub = document.createElement('div');
      sub.className = 'bond-sub';
      sub.textContent = 'Little or no trading history yet (newly issued/reopened)';
      textWrap.appendChild(sub);
    }

    row.appendChild(checkbox);
    row.appendChild(textWrap);
    bondListEl.appendChild(row);
  });
}

function renderRangeButtons() {
  rangeButtonsEl.innerHTML = '';
  RANGES.forEach((range, i) => {
    const btn = document.createElement('button');
    btn.textContent = range.label;
    btn.className = i === activeRangeIdx ? 'active' : '';
    btn.addEventListener('click', () => {
      activeRangeIdx = i;
      renderRangeButtons();
      updateChart();
    });
    rangeButtonsEl.appendChild(btn);
  });
}

function filteredPoints(history) {
  const days = RANGES[activeRangeIdx].days;
  if (!days || !history.length) return history;
  const latest = parseDateLocal(history[history.length - 1].date).getTime();
  const cutoff = latest - days * 86400000;
  return history.filter(p => parseDateLocal(p.date).getTime() >= cutoff);
}

function buildDatasets() {
  return bonds
    .filter(b => b.visible && b.history.length)
    .map(b => ({
      label: b.label,
      cusip: b.cusip,
      data: filteredPoints(b.history).map(p => ({ x: parseDateLocal(p.date).getTime(), y: p.price })),
      borderColor: b.color,
      backgroundColor: b.color,
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHitRadius: 8,
      tension: 0,
      spanGaps: true,
    }));
}

function formatDate(ms) {
  return new Date(ms).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

function updateChart() {
  const datasets = buildDatasets();
  if (chart) {
    chart.data.datasets = datasets;
    chart.update();
    return;
  }

  chart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: true },
      scales: {
        x: {
          type: 'linear',
          ticks: { callback: formatDate, color: cssVar('--text-muted'), maxTicksLimit: 8 },
          grid: { color: cssVar('--gridline') },
          border: { color: cssVar('--baseline') },
        },
        y: {
          ticks: { color: cssVar('--text-muted') },
          grid: { color: cssVar('--gridline') },
          border: { color: cssVar('--baseline') },
          title: { display: true, text: 'Price', color: cssVar('--text-secondary') },
        },
      },
      plugins: {
        legend: { display: false }, // checkboxes in the side panel are the legend
        tooltip: {
          callbacks: {
            title: items => (items.length ? formatDate(items[0].parsed.x) : ''),
            label: ctx => `${ctx.dataset.cusip}: ${ctx.parsed.y}`,
          },
        },
      },
    },
  });
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

async function refresh() {
  refreshBtn.disabled = true;
  try {
    await loadData();
    renderBondList();
    updateChart();
  } catch (e) {
    setStatus(`Failed to load: ${e.message}`, true);
  } finally {
    refreshBtn.disabled = false;
  }
}

refreshBtn.addEventListener('click', refresh);
renderRangeButtons();
refresh();
