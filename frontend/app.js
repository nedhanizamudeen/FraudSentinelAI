const API_BASE = window.location.origin;
const chartRegistry = {};
const sessionState = {
  lastBatch: null,
  health: null,
};

Chart.defaults.color = '#A88B93';
Chart.defaults.borderColor = 'rgba(78,40,56,0.45)';
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 10;

function $(id) { return document.getElementById(id); }
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }
function pct(value) { return `${(Number(value || 0) * 100).toFixed(1)}%`; }
function money(value) { return `$${Number(value || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`; }
function titleCase(value) { return String(value || '').toLowerCase().replace(/\b\w/g, c => c.toUpperCase()); }

function updateClock() {
  $('topbarTime').textContent = new Date().toLocaleTimeString('en-IN', {
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
}
setInterval(updateClock, 1000);
updateClock();

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  $(`page-${name}`).classList.add('active');
  document.querySelector(`.nav-item[data-page="${name}"]`).classList.add('active');
  const labels = {
    dashboard: 'Dashboard',
    single: 'Single Analysis',
    csv: 'CSV Batch Analysis',
    about: 'About the System'
  };
  $('breadcrumb').textContent = labels[name] || name;
  if (window.innerWidth <= 900) $('sidebar').classList.remove('open');
}

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', (e) => {
    e.preventDefault();
    showPage(item.dataset.page);
  });
});
$('menuToggle').addEventListener('click', () => $('sidebar').classList.toggle('open'));

async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    sessionState.health = data;
    $('kpiApiStatus').textContent = data.status.toUpperCase();
    $('kpiApiSub').textContent = data.service;
    const ready = Object.values(data.artifacts || {}).filter(Boolean).length;
    const total = Object.keys(data.artifacts || {}).length;
    $('kpiArtifacts').textContent = `${ready}/${total}`;
    $('kpiArtifactsSub').textContent = ready === total ? 'All inference artifacts found' : 'Some artifacts missing';
    $('apiStatusLabel').textContent = data.status === 'ok' ? 'API Connected' : 'API Degraded';
    $('apiDot').classList.toggle('degraded', data.status !== 'ok');
  } catch (err) {
    $('kpiApiStatus').textContent = 'OFFLINE';
    $('kpiApiSub').textContent = 'Health check failed';
    $('kpiArtifacts').textContent = '0/3';
    $('kpiArtifactsSub').textContent = 'Backend unreachable';
    $('apiStatusLabel').textContent = 'API Offline';
    $('apiDot').classList.add('degraded');
  }
}

function renderChart(id, config) {
  if (chartRegistry[id]) chartRegistry[id].destroy();
  const ctx = $(id).getContext('2d');
  chartRegistry[id] = new Chart(ctx, config);
}

function countBy(rows, key) {
  return rows.reduce((acc, row) => {
    const value = row[key] || 'UNKNOWN';
    acc[value] = (acc[value] || 0) + 1;
    return acc;
  }, {});
}

function distribution(values, step = 0.1) {
  const bins = [];
  for (let start = 0; start < 1; start += step) {
    const end = Math.min(start + step, 1);
    const label = `${Math.round(start * 100)}-${Math.round(end * 100)}`;
    bins.push({ label, count: 0, start, end });
  }
  values.forEach(v => {
    const value = Math.max(0, Math.min(0.999999, Number(v || 0)));
    const idx = Math.min(Math.floor(value / step), bins.length - 1);
    bins[idx].count += 1;
  });
  return bins;
}

function sumByCategory(rows, categoryKey, valueKey) {
  return rows.reduce((acc, row) => {
    const key = row[categoryKey] || 'UNKNOWN';
    acc[key] = (acc[key] || 0) + Number(row[valueKey] || 0);
    return acc;
  }, {});
}

function updateDashboardFromBatch(batch) {
  const rows = batch.rows || [];
  const summary = batch.summary || {};
  $('kpiRowsScored').textContent = Number(batch.rows_scored || 0).toLocaleString();
  $('kpiFraudCount').textContent = Number(summary.fraud_count || 0).toLocaleString();
  $('kpiImpact').textContent = money(summary.total_estimated_financial_impact_usd || 0);

  const recent = [
    `Average adjusted risk score: ${pct(summary.average_adjusted_risk_score)}`,
    `Average fraud probability: ${pct(summary.average_fraud_probability)}`,
    `Most common risk level: ${summary.average_risk_level || 'UNKNOWN'}`,
    `Invalid rows skipped: ${batch.invalid_rows || 0}`
  ];
  $('recentStats').innerHTML = recent.map(item => `<div class="recent-item static">${item}</div>`).join('');

  const verdictCounts = countBy(rows, 'FinalVerdict');
  renderChart('dashboardVerdictChart', {
    type: 'bar',
    data: {
      labels: Object.keys(verdictCounts),
      datasets: [{ label: 'Rows', data: Object.values(verdictCounts), backgroundColor: 'rgba(200,48,80,0.45)', borderColor: 'rgba(232,64,112,0.9)', borderWidth: 1 }]
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
  });

  const riskCounts = countBy(rows, 'AdjustedRiskLevel');
  renderChart('dashboardRiskChart', {
    type: 'doughnut',
    data: {
      labels: Object.keys(riskCounts),
      datasets: [{ data: Object.values(riskCounts), backgroundColor: ['#48CC88', '#E8C87A', '#FF8C40', '#E84070'], borderColor: '#2C1820', borderWidth: 2 }]
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
  });
}

$('singleForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  hide($('singleError'));
  show($('singleLoading'));
  hide($('singleEmpty'));
  hide($('singleResult'));

  const formData = new FormData(e.target);
  const payload = Object.fromEntries(formData.entries());
  Object.keys(payload).forEach(key => {
    if (payload[key] === '') delete payload[key];
  });
  if (payload.amount !== undefined) payload.amount = Number(payload.amount);
  if (payload.transaction_hour !== undefined) payload.transaction_hour = Number(payload.transaction_hour);

  try {
    const res = await fetch(`${API_BASE}/predict-single`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');

    $('singleFinalVerdict').textContent = data.FinalVerdict || '—';
    $('singleRiskLevel').textContent = data.AdjustedRiskLevel || '—';
    $('singleRiskScore').textContent = pct(data.AdjustedRiskScore);
    $('singleFraudProb').textContent = pct(data.FraudProbability);
    $('singleRecommendation').textContent = data.Recommendation || '—';
    $('singleLoss').textContent = money(data.EstimatedLoss_USD);
    $('singleExecutiveSummary').textContent = data.ExecutiveSummary || 'No executive summary returned.';
    $('singleExplanation').textContent = data.LLM_Explanation || 'No explanation returned.';
    $('singleCounterfactual').textContent = data.CounterfactualSummary || 'No counterfactual generated for this verdict.';
    show($('singleResult'));
  } catch (err) {
    $('singleError').textContent = err.message;
    show($('singleError'));
    show($('singleEmpty'));
  } finally {
    hide($('singleLoading'));
  }
});

$('csvForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  hide($('csvError'));
  hide($('csvResults'));
  show($('csvLoading'));

  const file = $('csvFile').files[0];
  if (!file) {
    $('csvError').textContent = 'Please select a CSV file.';
    show($('csvError'));
    hide($('csvLoading'));
    return;
  }

  const fd = new FormData();
  fd.append('file', file);
  fd.append('mode', 'rich');

  try {
    const res = await fetch(`${API_BASE}/predict-csv`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'CSV prediction failed');

    sessionState.lastBatch = data;
    renderBatchResults(data);
    updateDashboardFromBatch(data);
    show($('csvResults'));
  } catch (err) {
    $('csvError').textContent = err.message;
    show($('csvError'));
  } finally {
    hide($('csvLoading'));
  }
});

function renderBatchResults(batch) {
  const summary = batch.summary || {};
  const rows = batch.rows || [];

  $('csvRowsScored').textContent = Number(batch.rows_scored || 0).toLocaleString();
  $('csvFraudCount').textContent = Number(summary.fraud_count || 0).toLocaleString();
  $('csvLegitCount').textContent = Number(summary.legitimate_count || 0).toLocaleString();
  $('csvAvgFraudProb').textContent = pct(summary.average_fraud_probability);
  $('csvAvgRiskScore').textContent = pct(summary.average_adjusted_risk_score);
  $('csvTotalImpact').textContent = money(summary.total_estimated_financial_impact_usd);
  $('csvExecutiveSummary').textContent = summary.executive_summary || 'No summary returned.';

  if (batch.output_csv) {
    $('downloadCsvBtn').dataset.path = batch.output_csv;
    show($('downloadCsvBtn'));
  } else {
    hide($('downloadCsvBtn'));
  }

  const verdictCounts = countBy(rows, 'FinalVerdict');
  renderChart('csvVerdictChart', {
    type: 'bar',
    data: { labels: Object.keys(verdictCounts), datasets: [{ label: 'Rows', data: Object.values(verdictCounts), backgroundColor: 'rgba(232,64,112,0.48)', borderColor: 'rgba(232,64,112,0.9)', borderWidth: 1 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
  });

  const riskCounts = countBy(rows, 'AdjustedRiskLevel');
  renderChart('csvRiskLevelChart', {
    type: 'pie',
    data: { labels: Object.keys(riskCounts), datasets: [{ data: Object.values(riskCounts), backgroundColor: ['#48CC88', '#E8C87A', '#FF8C40', '#E84070'], borderColor: '#2C1820', borderWidth: 2 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
  });

  const riskScoreBins = distribution(rows.map(r => Number(r.AdjustedRiskScore || 0)), 0.1);
  renderChart('csvRiskScoreChart', {
    type: 'bar',
    data: { labels: riskScoreBins.map(b => b.label), datasets: [{ label: 'Rows', data: riskScoreBins.map(b => b.count), backgroundColor: 'rgba(56,176,204,0.4)', borderColor: 'rgba(56,176,204,0.9)', borderWidth: 1 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
  });

  const fraudProbBins = distribution(rows.map(r => Number(r.FraudProbability || 0)), 0.1);
  renderChart('csvFraudProbChart', {
    type: 'line',
    data: { labels: fraudProbBins.map(b => b.label), datasets: [{ label: 'Rows', data: fraudProbBins.map(b => b.count), tension: 0.25, fill: true, backgroundColor: 'rgba(232,200,122,0.18)', borderColor: 'rgba(232,200,122,0.95)', borderWidth: 2, pointRadius: 3 }] },
    options: { responsive: true, maintainAspectRatio: false }
  });

  const lossByVerdict = sumByCategory(rows, 'FinalVerdict', 'EstimatedLoss_USD');
  renderChart('csvEstimatedLossChart', {
    type: 'bar',
    data: { labels: Object.keys(lossByVerdict), datasets: [{ label: 'Estimated Loss (USD)', data: Object.values(lossByVerdict).map(v => Number(v.toFixed(2))), backgroundColor: 'rgba(255,140,64,0.42)', borderColor: 'rgba(255,140,64,0.9)', borderWidth: 1 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
  });

  renderResultsTable(rows);
}

function renderResultsTable(rows) {
  const preview = rows.slice(0, 25);
  const columns = [
    'RowNo',
    'TransactionID',
    'TransactionAmt',
    'FinalVerdict',
    'AdjustedRiskLevel',
    'AdjustedRiskScore',
    'FraudProbability',
    'EstimatedLoss_USD',
    'Recommendation',
    'ExecutiveSummary',
    'LLM_Explanation',
    'CounterfactualSummary'
  ].filter(col => preview.some(row => Object.prototype.hasOwnProperty.call(row, col)));

  $('resultsTable').querySelector('thead').innerHTML =
    `<tr>${columns.map(c => `<th>${c}</th>`).join('')}</tr>`;

  $('resultsTable').querySelector('tbody').innerHTML = preview.map(row => `
    <tr>${columns.map(c => `<td>${formatCell(c, row[c])}</td>`).join('')}</tr>
  `).join('');
}

function formatCell(column, value) {
  if (value === null || value === undefined || value === '') return '—';
  if (['AdjustedRiskScore', 'FraudProbability'].includes(column)) return pct(value);
  if (['EstimatedLoss_USD', 'TransactionAmt'].includes(column)) return money(value);
  return String(value);
}

$('downloadCsvBtn').addEventListener('click', () => {
  const filePath = $('downloadCsvBtn').dataset.path;
  if (!filePath) return;

  const encodedPath = encodeURIComponent(filePath);
  window.open(`${API_BASE}/download-output?path=${encodedPath}`, '_blank');
});

fetchHealth();