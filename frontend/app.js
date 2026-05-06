// ===== CONFIG =====
const API_BASE = 'http://localhost:5000';

// ===== TEST CASES (from testing.py) =====
const TEST_CASES = [
  {
    name: "Test #1: Typical (Conf 0.69)",
    data: `Predicted Class  : Typical (0)\nConfidence       : 0.6882\nDecision Path    : NC <= 1508.9295 AND ADD > 15.7500 AND DM <= 3564.2007 AND NC <= 1498.5457 AND ADD > 29.5000 AND ADD <= 63.0000 AND DM > 1056.7697 AND DM > 1142.8244 AND SUB <= 61.0000 AND ADD > 32.5000\nDomain Severity  : {'Number Series': 0.0, 'Addition vs. Subtraction Asymmetry': 0.2914, 'Overall Arithmetic Fluency': 0.0095, 'Multi-Digit Addition and Subtraction': 0.0, 'Digit-Dot Matching': 0.0051, 'Symbolic vs. Non-Symbolic Processing Difference': 0.0648, 'Basic vs. Complex Arithmetic Contrast': 0.2403, 'Overall Processing Efficiency': 0.1608, 'Processing-Fluency Integration': 0.1744, 'Single-Digit Subtraction': 0.0011, 'Single-Digit Addition': 0.0157, 'Number Comparison': 0.0370}\nTask Importance  : NC: 0.10, DM: 0.03, ADD: 0.03, SUB: 0.01, NP: 0.14, SN: 0.06, AF: 0.01, BC: 0.21, AS: 0.26, PF: 0.15`
  },
  {
    name: "Test #5: At-Risk (Conf 0.62)",
    data: `Predicted Class  : At-Risk (1)\nConfidence       : 0.6190\nDecision Path    : NC <= 1508.9295 AND ADD > 15.7500 AND DM <= 3564.2007 AND NC <= 1498.5457 AND ADD > 29.5000 AND ADD <= 63.0000 AND DM > 1056.7697 AND DM > 1142.8244 AND SUB <= 61.0000 AND ADD <= 32.5000\nDomain Severity  : {'Number Series': 0.0, 'Addition vs. Subtraction Asymmetry': 0.4508, 'Overall Arithmetic Fluency': 0.0194, 'Multi-Digit Addition and Subtraction': 0.0, 'Digit-Dot Matching': 0.0033, 'Symbolic vs. Non-Symbolic Processing Difference': 0.2026, 'Basic vs. Complex Arithmetic Contrast': 0.1595, 'Overall Processing Efficiency': 0.0399, 'Processing-Fluency Integration': 0.0643, 'Single-Digit Subtraction': 0.0015, 'Single-Digit Addition': 0.0289, 'Number Comparison': 0.0297}\nTask Importance  : NC: 0.08, DM: 0.02, ADD: 0.06, SUB: 0.01, NP: 0.04, SN: 0.18, AF: 0.02, BC: 0.14, AS: 0.40, PF: 0.06`
  },
  {
    name: "Test #8: At-Risk (Conf 0.80)",
    data: `Predicted Class  : At-Risk (1)\nConfidence       : 0.8000\nDecision Path    : NC <= 1508.9295 AND ADD > 15.7500 AND DM <= 3564.2007 AND NC <= 1498.5457 AND ADD <= 29.5000 AND NS > 6.5000 AND SUB <= 40.5000 AND DM > 1909.3360\nDomain Severity  : {'Number Series': 0.0078, 'Addition vs. Subtraction Asymmetry': 0.0298, 'Overall Arithmetic Fluency': 0.1284, 'Multi-Digit Addition and Subtraction': 0.0, 'Digit-Dot Matching': 0.0321, 'Symbolic vs. Non-Symbolic Processing Difference': 0.2088, 'Basic vs. Complex Arithmetic Contrast': 0.1107, 'Overall Processing Efficiency': 0.2681, 'Processing-Fluency Integration': 0.1621, 'Single-Digit Subtraction': 0.0054, 'Single-Digit Addition': 0.0200, 'Number Comparison': 0.0269}\nTask Importance  : NC: 0.07, DM: 0.07, NS: 0.01, ADD: 0.04, SUB: 0.01, NP: 0.24, SN: 0.18, AF: 0.11, BC: 0.10, AS: 0.03, PF: 0.14`
  },
  {
    name: "Test #9: Typical (Conf 0.89)",
    data: `Predicted Class  : Typical (0)\nConfidence       : 0.8889\nDecision Path    : NC <= 1508.9295 AND ADD > 15.7500 AND DM <= 3564.2007 AND NC > 1498.5457\nDomain Severity  : {'Number Series': 0.0, 'Addition vs. Subtraction Asymmetry': 0.1040, 'Overall Arithmetic Fluency': 0.1221, 'Multi-Digit Addition and Subtraction': 0.0, 'Digit-Dot Matching': 0.0030, 'Symbolic vs. Non-Symbolic Processing Difference': 0.0528, 'Basic vs. Complex Arithmetic Contrast': 0.2127, 'Overall Processing Efficiency': 0.2405, 'Processing-Fluency Integration': 0.2125, 'Single-Digit Subtraction': 0.0, 'Single-Digit Addition': 0.0020, 'Number Comparison': 0.0504}\nTask Importance  : NC: 0.14, DM: 0.02, ADD: 0.01, NP: 0.21, SN: 0.05, AF: 0.11, BC: 0.19, AS: 0.09, PF: 0.19`
  }
];

// ===== DOM REFS =====
const $ = (s) => document.querySelector(s);
const navLinks = document.querySelectorAll('.nav-link');
const selectEl = $('#test-case-select');
const diagInput = $('#diagnostic-input');
const btnGenerate = $('#btn-generate');
const btnClearInput = $('#btn-clear-input');
const progressGen = $('#progress-generate');
const outputGen = $('#output-generate');
const retestDiag = $('#retest-diagnostic');
const retestPrev = $('#retest-previous');
const btnRetest = $('#btn-retest');
const btnClearRetest = $('#btn-clear-retest');
const progressRet = $('#progress-retest');
const outputRet = $('#output-retest');

// ===== INIT =====
TEST_CASES.forEach((tc, i) => {
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = tc.name;
  selectEl.appendChild(opt);
});

selectEl.addEventListener('change', () => {
  const idx = selectEl.value;
  if (idx !== '') diagInput.value = TEST_CASES[idx].data;
});

// Nav switching
navLinks.forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    navLinks.forEach(l => l.classList.remove('active'));
    link.classList.add('active');
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(link.dataset.section);
    if (target) target.classList.add('active');
  });
});

btnClearInput.addEventListener('click', () => { diagInput.value = ''; selectEl.value = ''; outputGen.innerHTML = ''; outputGen.classList.add('hidden'); });
btnClearRetest.addEventListener('click', () => { retestDiag.value = ''; retestPrev.value = ''; outputRet.innerHTML = ''; outputRet.classList.add('hidden'); });

// ===== SERVER STATUS =====
async function checkServer() {
  const dot = $('#server-status .status-dot');
  const label = $('#server-status .status-label');
  try {
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 4000);
    // Use OPTIONS on an actual endpoint — Flask-CORS handles it without 404
    await fetch(API_BASE + '/generate_module', { method: 'OPTIONS', signal: ctrl.signal });
    dot.className = 'status-dot online';
    label.textContent = 'Server Online';
  } catch {
    dot.className = 'status-dot offline';
    label.textContent = 'Server Offline';
  }
}
checkServer();
setInterval(checkServer, 60000); // check every 60s, not 15s

// ===== TIMER HELPER =====
function startTimer(timerEl, fillEl) {
  let seconds = 0;
  const maxSec = 90;
  const id = setInterval(() => {
    seconds++;
    timerEl.textContent = seconds + 's';
    fillEl.style.width = Math.min((seconds / maxSec) * 100, 100) + '%';
  }, 1000);
  return { stop: () => clearInterval(id) };
}

// ===== GENERATE MODULE =====
btnGenerate.addEventListener('click', async () => {
  const input = diagInput.value.trim();
  if (!input) return alert('Please enter diagnostic data.');

  btnGenerate.disabled = true;
  progressGen.classList.remove('hidden');
  outputGen.innerHTML = '';
  outputGen.classList.add('hidden');

  const timer = startTimer($('#progress-timer'), $('#progress-fill'));

  try {
    const res = await fetch(API_BASE + '/generate_module', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ diagnostic_data: input }),
      signal: AbortSignal.timeout(120000)
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || 'Request failed');
    renderModuleOutput(json);
  } catch (err) {
    outputGen.innerHTML = errorHTML(err.message);
    outputGen.classList.remove('hidden');
  } finally {
    timer.stop();
    progressGen.classList.add('hidden');
    btnGenerate.disabled = false;
  }
});

// ===== GENERATE RETEST =====
btnRetest.addEventListener('click', async () => {
  const diag = retestDiag.value.trim();
  if (!diag) return alert('Please enter diagnostic data.');

  let prev = [];
  const prevRaw = retestPrev.value.trim();
  if (prevRaw) {
    try { prev = JSON.parse(prevRaw); } catch { return alert('Previous questions must be valid JSON.'); }
  }

  btnRetest.disabled = true;
  progressRet.classList.remove('hidden');
  outputRet.innerHTML = '';
  outputRet.classList.add('hidden');

  const timer = startTimer($('#progress-timer-retest'), $('#progress-fill-retest'));

  try {
    const res = await fetch(API_BASE + '/generate_retest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ diagnostic_data: diag, previous_questions: prev }),
      signal: AbortSignal.timeout(120000)
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || 'Request failed');
    renderRetestOutput(json);
  } catch (err) {
    outputRet.innerHTML = errorHTML(err.message);
    outputRet.classList.remove('hidden');
  } finally {
    timer.stop();
    progressRet.classList.add('hidden');
    btnRetest.disabled = false;
  }
});

// ===== ERROR HTML =====
function errorHTML(msg) {
  return `<div class="error-banner"><span class="error-icon">⚠️</span><div class="error-msg"><strong>Error</strong><br>${escHTML(msg)}</div></div>`;
}

function escHTML(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ===== RENDER: MODULE OUTPUT =====
function renderModuleOutput(data) {
  const status = data.status || 'Unknown';
  const isRisk = status.toLowerCase().includes('at-risk');
  const modules = data.diagnostic_modules || [];
  const formative = data.formative_assessment || [];
  const meta = data._meta_validation_report || {};
  const interp = data.decision_path_interpretation || '';
  const summary = data.overall_summary || '';
  const rationale = data.decision_path_rationale || '';

  let html = '';

  // Summary strip
  html += `<div class="summary-strip">
    <div class="summary-stat"><div class="stat-label">Classification</div><div class="stat-value ${isRisk ? 'status-at-risk' : 'status-typical'}">${escHTML(status)}</div></div>
    <div class="summary-stat"><div class="stat-label">Modules Generated</div><div class="stat-value">${modules.length}</div></div>
    <div class="summary-stat"><div class="stat-label">Formative Items</div><div class="stat-value">${formative.length}</div></div>
    <div class="summary-stat"><div class="stat-label">Success Rate</div><div class="stat-value">${meta.success_rate != null ? (meta.success_rate * 100).toFixed(0) + '%' : '—'}</div></div>
  </div>`;

  // Interpretation
  if (interp || summary || rationale) {
    html += `<div class="interpretation-panel">`;
    if (summary) html += `<h3>🧠 Overall Summary</h3><p>${escHTML(summary)}</p>`;
    if (rationale) html += `<h3 style="margin-top:16px">🔍 Decision Rationale</h3><p>${escHTML(rationale)}</p>`;
    if (interp) html += `<h3 style="margin-top:16px">📊 Clinical Interpretation</h3><p>${escHTML(interp)}</p>`;
    html += `</div>`;
  }

  // Modules
  modules.forEach((mod, i) => {
    const we = mod.worked_example || {};
    const ps = mod.practice_set || [];
    const ce = mod.conceptual_explanation || '';
    html += `<div class="module-card" style="animation-delay:${i * 0.1}s">
      <div class="module-header" onclick="toggleModule(this)">
        <div class="module-title-group">
          <div class="module-number">${i + 1}</div>
          <div><div class="module-title">${escHTML(mod.module_title || 'Module ' + (i + 1))}</div>
          <div class="module-domain">${escHTML(mod.domain || '')}</div></div>
        </div>
        <span class="module-toggle">▼</span>
      </div>
      <div class="module-body">`;

    // Worked example
    if (we.problem) {
      html += `<div class="module-section"><div class="module-section-title">Worked Example</div>
        <div class="worked-example"><div class="we-problem">${escHTML(we.problem)} = ${escHTML(String(we.answer ?? ''))}</div>
        <div class="we-steps">${escHTML(we.explanation || '')}</div></div></div>`;
    }

    // Conceptual explanation
    if (ce) {
      html += `<div class="module-section"><div class="module-section-title">Conceptual Explanation</div>
        <div class="concept-box">${escHTML(ce)}</div></div>`;
    }

    // Practice set
    if (ps.length) {
      html += `<div class="module-section"><div class="module-section-title">Practice Set (${ps.length} items)</div><div class="practice-grid">`;
      ps.forEach(p => {
        html += `<div class="practice-item">
          <div class="pi-problem">${escHTML(p.problem || '')}</div>
          <div class="pi-answer">Answer: ${escHTML(String(p.expected_answer ?? ''))}</div>
          <div class="pi-hint">💡 ${escHTML(p.hint || '')}</div>
        </div>`;
      });
      html += `</div></div>`;
    }

    html += `</div></div>`;
  });

  // Formative assessment
  if (formative.length) {
    html += `<div class="module-card"><div class="module-header" onclick="toggleModule(this)">
      <div class="module-title-group"><div class="module-number">📝</div><div><div class="module-title">Formative Assessment</div>
      <div class="module-domain">${formative.length} questions</div></div></div>
      <span class="module-toggle">▼</span></div>
      <div class="module-body"><div class="formative-grid">`;
    formative.forEach(f => {
      const choices = f.choices || [];
      const correct = String(f.expected_answer ?? '');
      html += `<div class="formative-item"><div class="fi-q">${escHTML(f.question || f.problem || '')}</div><ul class="fi-choices">`;
      choices.forEach(c => {
        html += `<li class="${String(c) === correct ? 'correct' : ''}">${escHTML(String(c))}</li>`;
      });
      html += `</ul>`;
      if (f.explanation) html += `<div class="fi-explanation">${escHTML(f.explanation)}</div>`;
      html += `</div>`;
    });
    html += `</div></div></div>`;
  }

  // Validation report
  html += renderValidation(meta);

  // Raw JSON toggle
  html += `<div class="json-toggle-wrap"><button class="btn btn-ghost btn-sm" onclick="toggleRawJSON(this)">Show Raw JSON</button>
    <div class="json-raw" style="display:none"><pre>${escHTML(JSON.stringify(data, null, 2))}</pre></div></div>`;

  outputGen.innerHTML = html;
  outputGen.classList.remove('hidden');
}

// ===== RENDER: RETEST =====
function renderRetestOutput(data) {
  const questions = data.retest_questions || [];
  const meta = data._meta_validation_report || {};

  let html = `<div class="summary-strip">
    <div class="summary-stat"><div class="stat-label">Questions Generated</div><div class="stat-value">${questions.length}</div></div>
    <div class="summary-stat"><div class="stat-label">Pruned</div><div class="stat-value">${meta.counts?.pruned ?? 0}</div></div>
    <div class="summary-stat"><div class="stat-label">Math Errors</div><div class="stat-value">${(meta.math_errors || []).length}</div></div>
  </div>`;

  html += `<div class="retest-list">`;
  questions.forEach((q, i) => {
    html += `<div class="retest-item" style="animation-delay:${i * 0.08}s">
      <div class="ri-num">Question ${i + 1}</div>
      <div class="ri-problem">${escHTML(q.problem || '')}</div>
      <div class="ri-answer">Answer: ${escHTML(String(q.expected_answer ?? ''))}</div>
      <div class="ri-hint">💡 ${escHTML(q.hint || '')}</div>
    </div>`;
  });
  html += `</div>`;

  html += renderValidation(meta);

  html += `<div class="json-toggle-wrap"><button class="btn btn-ghost btn-sm" onclick="toggleRawJSON(this)">Show Raw JSON</button>
    <div class="json-raw" style="display:none"><pre>${escHTML(JSON.stringify(data, null, 2))}</pre></div></div>`;

  outputRet.innerHTML = html;
  outputRet.classList.remove('hidden');
}

// ===== RENDER: VALIDATION =====
function renderValidation(meta) {
  if (!meta || !meta.counts) return '';
  const c = meta.counts;
  const mErr = meta.math_errors || [];
  const pErr = meta.pedagogy_errors || [];
  const sErr = meta.schema_errors || [];
  const warns = meta.warnings || [];

  let html = `<div class="validation-panel"><h3>🛡️ Validation Report</h3><div class="val-grid">
    <div class="val-cell good"><div class="vc-label">Returned</div><div class="vc-value">${c.returned ?? '—'}</div></div>
    <div class="val-cell ${(c.pruned || 0) > 0 ? 'warn' : 'good'}"><div class="vc-label">Pruned</div><div class="vc-value">${c.pruned ?? 0}</div></div>
    <div class="val-cell ${mErr.length > 0 ? 'bad' : 'good'}"><div class="vc-label">Math Errors</div><div class="vc-value">${mErr.length}</div></div>
    <div class="val-cell ${pErr.length > 0 ? 'bad' : 'good'}"><div class="vc-label">Pedagogy Errors</div><div class="vc-value">${pErr.length}</div></div>
    <div class="val-cell ${sErr.length > 0 ? 'bad' : 'good'}"><div class="vc-label">Schema Errors</div><div class="vc-value">${sErr.length}</div></div>
  </div>`;

  const allIssues = [...mErr.map(e => '🔢 ' + e), ...pErr.map(e => '📖 ' + e), ...sErr.map(e => '📄 ' + e), ...warns.map(e => '⚠️ ' + e)];
  if (allIssues.length) {
    html += `<div class="val-errors"><ul class="val-error-list">`;
    allIssues.forEach(e => html += `<li>${escHTML(e)}</li>`);
    html += `</ul></div>`;
  }
  html += `</div>`;
  return html;
}

// ===== TOGGLE HELPERS =====
function toggleModule(header) {
  const body = header.nextElementSibling;
  const icon = header.querySelector('.module-toggle');
  body.classList.toggle('open');
  icon.classList.toggle('open');
}

function toggleRawJSON(btn) {
  const wrap = btn.nextElementSibling;
  const visible = wrap.style.display !== 'none';
  wrap.style.display = visible ? 'none' : 'block';
  btn.textContent = visible ? 'Show Raw JSON' : 'Hide Raw JSON';
}
