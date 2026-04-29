"""Agent Evaluation Harness — Databricks App."""
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Agent Evaluation Harness")
executor = ThreadPoolExecutor(max_workers=2)

# In-memory store for eval results (per-session)
eval_store = {}


class EvalRequest(BaseModel):
    agent_type: str  # customer_support | document_processing


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


@app.post("/api/evaluate")
async def evaluate(req: EvalRequest):
    """Start evaluation — runs in background."""
    from server.evaluator import run_evaluation
    eval_id = f"{req.agent_type}_{id(req)}"
    eval_store[req.agent_type] = {"status": "running", "progress": 0}

    def _run():
        try:
            result = run_evaluation(req.agent_type)
            eval_store[req.agent_type] = {"status": "completed", "result": result}
        except Exception as e:
            eval_store[req.agent_type] = {"status": "error", "error": str(e)}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run)
    return {"status": "started", "agent_type": req.agent_type}


@app.get("/api/status/{agent_type}")
async def get_status(agent_type: str):
    """Poll evaluation status."""
    return eval_store.get(agent_type, {"status": "idle"})


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════
# SINGLE-PAGE HTML UI
# ═══════════════════════════════════════════════════════════════

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Evaluation Harness</title>
<style>
  :root {
    --bg: #faf9f5; --card: #fff; --border: #e5e2d9; --text: #1a1a2e;
    --muted: #666; --accent: #FF3621; --green: #0e8a6c; --red: #dc2626;
    --yellow: #b47209; --blue: #0055d4;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
  .container { max-width: 1000px; margin: 0 auto; padding: 24px; }

  /* Header */
  .header { text-align: center; padding: 32px 0 24px; border-bottom: 2px solid var(--border); margin-bottom: 32px; }
  .header h1 { font-size: 1.8rem; font-weight: 700; }
  .header h1 span { color: var(--accent); }
  .header p { color: var(--muted); margin-top: 4px; }

  /* Agent Cards */
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 32px; }
  .card { background: var(--card); border: 2px solid var(--border); border-radius: 12px; padding: 24px; transition: border-color .2s; cursor: pointer; }
  .card:hover, .card.selected { border-color: var(--accent); }
  .card.selected { box-shadow: 0 0 0 3px rgba(255,54,33,.15); }
  .card h3 { font-size: 1.1rem; margin-bottom: 6px; }
  .card .type-badge { display: inline-block; font-size: .7rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; text-transform: uppercase; }
  .badge-single { background: rgba(0,85,212,.1); color: var(--blue); border: 1px solid rgba(0,85,212,.2); }
  .badge-multi { background: rgba(168,85,247,.1); color: #7c3aed; border: 1px solid rgba(168,85,247,.2); }
  .card p { color: var(--muted); font-size: .85rem; margin-top: 8px; }
  .card .stats { display: flex; gap: 16px; margin-top: 12px; font-size: .8rem; }
  .card .stat { color: var(--muted); }
  .card .stat strong { color: var(--text); }

  /* Run Button */
  .run-section { text-align: center; margin: 24px 0; }
  .btn { background: var(--accent); color: #fff; border: none; padding: 12px 32px; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: opacity .2s; }
  .btn:hover { opacity: .9; }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  .btn-outline { background: transparent; color: var(--accent); border: 2px solid var(--accent); }

  /* Progress */
  .progress { display: none; text-align: center; padding: 32px; }
  .progress.show { display: block; }
  .spinner { width: 40px; height: 40px; border: 4px solid var(--border); border-top: 4px solid var(--accent); border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 16px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Results */
  .results { display: none; }
  .results.show { display: block; }

  /* KPI */
  .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .kpi { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; text-align: center; }
  .kpi .value { font-size: 2rem; font-weight: 800; }
  .kpi .label { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
  .kpi.pass .value { color: var(--green); }
  .kpi.fail .value { color: var(--red); }
  .kpi.warn .value { color: var(--yellow); }

  /* Scorer Table */
  .scorer-table { width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 24px; }
  .scorer-table th { background: #f5f3ee; font-size: .75rem; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); padding: 10px 14px; text-align: left; }
  .scorer-table td { padding: 10px 14px; border-top: 1px solid var(--border); font-size: .88rem; }
  .scorer-table tr:hover { background: rgba(0,0,0,.02); }
  .pass-badge { background: rgba(14,138,108,.1); color: var(--green); padding: 2px 10px; border-radius: 10px; font-weight: 600; font-size: .78rem; }
  .fail-badge { background: rgba(220,38,38,.1); color: var(--red); padding: 2px 10px; border-radius: 10px; font-weight: 600; font-size: .78rem; }
  .rate-bar { display: inline-block; height: 6px; border-radius: 3px; background: var(--green); }

  /* Detail Table */
  .detail-table { width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; font-size: .82rem; }
  .detail-table th { background: #f5f3ee; font-size: .7rem; text-transform: uppercase; color: var(--muted); padding: 8px 10px; text-align: left; }
  .detail-table td { padding: 8px 10px; border-top: 1px solid var(--border); }
  .detail-table td.pass { color: var(--green); }
  .detail-table td.fail { color: var(--red); font-weight: 600; }

  h2 { font-size: 1.2rem; margin-bottom: 12px; }
  .section { margin-bottom: 32px; }
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>🔬 Agent <span>Evaluation Harness</span></h1>
    <p>Configuration-driven evaluation for AI agents on Databricks</p>
  </div>

  <!-- Agent Selection -->
  <div class="cards">
    <div class="card selected" onclick="selectAgent('customer_support')" id="card-cs">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <h3>🛒 Customer Support Agent</h3>
        <span class="type-badge badge-single">Single Agent</span>
      </div>
      <p>RAG retrieval + tool calls (order lookup, returns, KB search). Tests product inquiries, order tracking, refunds, and adversarial attacks.</p>
      <div class="stats">
        <div class="stat"><strong>10</strong> test cases</div>
        <div class="stat"><strong>4</strong> scorers</div>
        <div class="stat"><strong>4</strong> adversarial</div>
      </div>
    </div>
    <div class="card" onclick="selectAgent('document_processing')" id="card-dp">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <h3>📄 Document Processing Pipeline</h3>
        <span class="type-badge badge-multi">Multi-Agent</span>
      </div>
      <p>5 sub-agents: extraction → classification → validation → compliance → approval. 4-layer evaluation framework.</p>
      <div class="stats">
        <div class="stat"><strong>5</strong> test cases</div>
        <div class="stat"><strong>7</strong> scorers</div>
        <div class="stat"><strong>4</strong> layers</div>
      </div>
    </div>
  </div>

  <!-- Run Button -->
  <div class="run-section">
    <button class="btn" id="run-btn" onclick="runEval()">▶ Run Evaluation</button>
    <p style="color:var(--muted);font-size:.8rem;margin-top:8px" id="run-hint">Select an agent above, then click Run</p>
  </div>

  <!-- Progress -->
  <div class="progress" id="progress">
    <div class="spinner"></div>
    <p style="font-weight:600" id="progress-text">Running evaluation...</p>
    <p style="color:var(--muted);font-size:.85rem" id="progress-detail">Sending test cases to the agent and scoring responses</p>
  </div>

  <!-- Results -->
  <div class="results" id="results">

    <!-- KPIs -->
    <div class="kpi-row" id="kpi-row"></div>

    <!-- Scorer Breakdown -->
    <div class="section">
      <h2>Scorer Breakdown</h2>
      <table class="scorer-table" id="scorer-table">
        <thead><tr><th>Scorer</th><th>Layer</th><th>Pass</th><th>Fail</th><th>Rate</th><th>Status</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <!-- Per Test Case -->
    <div class="section">
      <h2>Test Case Details</h2>
      <table class="detail-table" id="detail-table">
        <thead><tr><th>#</th><th>Category</th><th>Input</th><th>Latency</th><th>Scores</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

  </div>

</div>

<script>
let selectedAgent = 'customer_support';

function selectAgent(type) {
  selectedAgent = type;
  document.querySelectorAll('.card').forEach(c => c.classList.remove('selected'));
  document.getElementById(type === 'customer_support' ? 'card-cs' : 'card-dp').classList.add('selected');
  document.getElementById('results').classList.remove('show');
}

async function runEval() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Running...';
  document.getElementById('progress').classList.add('show');
  document.getElementById('results').classList.remove('show');
  document.getElementById('progress-text').textContent = `Evaluating ${selectedAgent.replace('_', ' ')} agent...`;

  try {
    await fetch('/api/evaluate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({agent_type: selectedAgent})
    });

    // Poll for completion
    let done = false;
    while (!done) {
      await new Promise(r => setTimeout(r, 3000));
      const resp = await fetch(`/api/status/${selectedAgent}`);
      const data = await resp.json();

      if (data.status === 'completed') {
        done = true;
        renderResults(data.result);
      } else if (data.status === 'error') {
        done = true;
        alert('Error: ' + data.error);
      } else {
        document.getElementById('progress-detail').textContent = 'Agent is processing test cases...';
      }
    }
  } catch (e) {
    alert('Error: ' + e.message);
  }

  btn.disabled = false;
  btn.textContent = '▶ Run Evaluation';
  document.getElementById('progress').classList.remove('show');
}

function renderResults(data) {
  // KPIs
  const kpiClass = data.pass_rate >= 90 ? 'pass' : data.pass_rate >= 70 ? 'warn' : 'fail';
  document.getElementById('kpi-row').innerHTML = `
    <div class="kpi ${kpiClass}"><div class="value">${data.pass_rate}%</div><div class="label">Pass Rate</div></div>
    <div class="kpi"><div class="value">${data.total_test_cases}</div><div class="label">Test Cases</div></div>
    <div class="kpi pass"><div class="value">${data.passed}</div><div class="label">Passed</div></div>
    <div class="kpi ${data.failed > 0 ? 'fail' : 'pass'}"><div class="value">${data.failed}</div><div class="label">Failed</div></div>
  `;

  // Scorer table
  const tbody = document.querySelector('#scorer-table tbody');
  tbody.innerHTML = '';
  for (const [name, stats] of Object.entries(data.scorer_stats)) {
    const rate = Math.round(stats.passed / stats.total * 100);
    const badge = rate >= 90 ? 'pass-badge' : 'fail-badge';
    tbody.innerHTML += `<tr>
      <td><strong>${name}</strong></td>
      <td style="color:var(--muted);font-size:.78rem">${stats.layer || ''}</td>
      <td>${stats.passed}</td>
      <td>${stats.total - stats.passed}</td>
      <td><span class="rate-bar" style="width:${rate}px;background:${rate>=90?'var(--green)':'var(--red)'}"></span> ${rate}%</td>
      <td><span class="${badge}">${rate >= 90 ? 'PASS' : 'FAIL'}</span></td>
    </tr>`;
  }

  // Detail table
  const dtbody = document.querySelector('#detail-table tbody');
  dtbody.innerHTML = '';
  for (const r of data.results) {
    const allPassed = r.scores.every(s => s.passed);
    const scoreIcons = r.scores.map(s => s.passed ? '✅' : '❌').join(' ');
    dtbody.innerHTML += `<tr>
      <td>${r.test_case}</td>
      <td>${r.category || ''}</td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${r.inputs_preview}</td>
      <td>${r.latency}s</td>
      <td>${scoreIcons}</td>
    </tr>`;
  }

  document.getElementById('results').classList.add('show');
}
</script>
</body>
</html>"""
