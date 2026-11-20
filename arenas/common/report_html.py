"""HTML interativo para análise visual de um relatório estatístico.

O relatório é autocontido (dados embutidos). Os gráficos usam Plotly via CDN;
se a CDN não estiver disponível, a interface mostra os dados em tabela e um
aviso.
"""
from __future__ import annotations

import json
from html import escape


def render_html(payload: dict, *, title: str, subtitle: str) -> str:
    """Gera o HTML a partir do dicionário do relatório."""
    data = json.dumps(payload, ensure_ascii=False)
    data = data.replace("</", "<\\/")
    return (HTML_TEMPLATE
            .replace("__TITLE__", escape(title))
            .replace("__SUBTITLE__", escape(subtitle))
            .replace("__DATA__", data))


HTML_TEMPLATE = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    --bg: #f5f7fb;
    --card: #ffffff;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #e2e8f0;
    --accent: #2563eb;
    --green: #0f9d58;
    --green-bg: #dcfce7;
    --green-ink: #166534;
    --amber: #d97706;
    --red: #dc2626;
    --radius: 16px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    color: var(--ink);
    background: linear-gradient(180deg, #eef4ff 0, #f8fafc 360px, var(--bg) 100%);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .wrap { max-width: 1440px; margin: 0 auto; padding: 30px 22px 80px; }
  .hero { padding: 18px 2px 8px; }
  .eyebrow { color: var(--accent); font-size: .78rem; font-weight: 700;
             letter-spacing: .12em; text-transform: uppercase; }
  h1 { margin: 8px 0 6px; font-size: clamp(1.7rem, 3.2vw, 2.5rem);
       letter-spacing: -.03em; }
  .subtitle { margin: 0 0 14px; color: var(--muted); font-size: 1rem; }
  .meta { display: flex; flex-wrap: wrap; gap: 8px; }
  .pill { background: #fff; border: 1px solid var(--line); border-radius: 999px;
          padding: 5px 11px; font-size: .82rem; color: #334155; }
  .pill b { color: var(--ink); }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 14px; margin: 24px 0; }
  .kpi { background: var(--card); border: 1px solid var(--line);
         border-radius: var(--radius); padding: 16px 18px;
         box-shadow: 0 10px 30px rgba(15, 23, 42, .05); }
  .kpi .label { color: var(--muted); font-size: .8rem; text-transform: uppercase;
                letter-spacing: .06em; }
  .kpi .value { margin-top: 6px; font-size: 1.55rem; font-weight: 750;
                letter-spacing: -.02em; }
  .kpi .hint { margin-top: 3px; color: var(--muted); font-size: .82rem; }
  .card { background: var(--card); border: 1px solid var(--line);
          border-radius: var(--radius); padding: 18px;
          box-shadow: 0 10px 30px rgba(15, 23, 42, .05); }
  .controls { display: flex; flex-wrap: wrap; align-items: flex-start;
              justify-content: space-between; gap: 16px; margin-bottom: 18px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
  .chip { display: inline-flex; align-items: center; gap: 7px; cursor: pointer;
          background: #fff; border: 1px solid var(--line); border-radius: 999px;
          padding: 7px 11px; font-size: .88rem; user-select: none; }
  .chip input { accent-color: var(--accent); }
  .actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  button { border: 1px solid var(--line); background: #fff; color: var(--ink);
           border-radius: 10px; padding: 8px 12px; font: inherit; cursor: pointer; }
  button:hover { border-color: #cbd5e1; background: #f8fafc; }
  button.primary { background: var(--accent); border-color: var(--accent);
                   color: #fff; }
  button.primary:hover { background: #1d4ed8; }
  .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 18px; margin: 0 0 18px; }
  @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  .card h2 { margin: 0 0 10px; font-size: 1.03rem; letter-spacing: -.01em; }
  .plot { min-height: 360px; }
  .plot .empty { display: grid; place-items: center; height: 320px;
                 color: var(--muted); font-size: .95rem; }
  .table-wrap { overflow: auto; max-height: 560px; border: 1px solid var(--line);
                border-radius: 12px; }
  table { border-collapse: collapse; width: 100%; font-size: .88rem; }
  th, td { padding: 9px 10px; border-bottom: 1px solid var(--line);
           text-align: left; white-space: nowrap; }
  th { position: sticky; top: 0; z-index: 1; background: #f8fafc;
       cursor: pointer; user-select: none; }
  th:hover { color: var(--accent); }
  tbody tr:hover { background: #f8fbff; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
           font-size: .78rem; font-weight: 650; }
  .badge.yes { background: var(--green-bg); color: var(--green-ink); }
  .badge.no { background: #f1f5f9; color: #475569; }
  .muted { color: var(--muted); }
  .warn { background: #fffbeb; border-color: #fde68a; }
  .warn h2 { color: #92400e; }
  .warn ul { margin: 8px 0 0; padding-left: 20px; }
  .hidden { display: none !important; }
  .footer { margin-top: 26px; color: var(--muted); font-size: .82rem; }
  @media print {
    body { background: #fff; }
    .controls, button { display: none !important; }
    .card, .kpi { box-shadow: none; break-inside: avoid; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <div class="eyebrow">RL Arena · análise de experimentos</div>
    <h1>__TITLE__</h1>
    <p class="subtitle">__SUBTITLE__</p>
    <div class="meta" id="meta"></div>
  </header>

  <section class="kpis" id="kpis"></section>

  <section class="card controls">
    <div>
      <strong>Algoritmos</strong>
      <div class="chips" id="algorithm-filter"></div>
    </div>
    <div class="actions">
      <label class="chip"><input type="checkbox" id="only-significant">
        só comparações significativas</label>
      <button id="download-json">Baixar JSON</button>
      <button id="reset">Resetar</button>
    </div>
  </section>

  <section class="grid">
    <div class="card"><h2>Média e IC</h2><div id="chart-means" class="plot"></div></div>
    <div class="card"><h2>Distribuição por seed</h2><div id="chart-box" class="plot"></div></div>
    <div class="card"><h2>Diferença entre algoritmos</h2><div id="chart-diff" class="plot"></div></div>
    <div class="card"><h2>Efeito × evidência</h2><div id="chart-effect" class="plot"></div></div>
  </section>

  <section class="card" style="margin-bottom:18px">
    <h2>Comparações</h2>
    <div class="table-wrap">
      <table id="table-comparisons">
        <thead><tr>
          <th data-key="a">A</th>
          <th data-key="b">B</th>
          <th data-key="n" class="num">n</th>
          <th data-key="diff" class="num">Diferença</th>
          <th data-key="diff_ci_low" class="num">IC</th>
          <th data-key="method">Teste</th>
          <th data-key="p_value" class="num">p</th>
          <th data-key="p_adjusted" class="num">p corrigido</th>
          <th data-key="effect" class="num">Efeito</th>
          <th data-key="significant">Sig.</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <section class="card" style="margin-bottom:18px">
    <h2>Descritivas</h2>
    <div class="table-wrap">
      <table id="table-summaries">
        <thead><tr>
          <th>Algoritmo</th>
          <th class="num">n</th>
          <th class="num">Média</th>
          <th class="num">Mediana</th>
          <th class="num">Desvio</th>
          <th class="num">IQR</th>
          <th class="num">IC</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <section class="card warn hidden" id="warnings">
    <h2>Avisos</h2>
    <ul id="warning-list"></ul>
  </section>

  <p class="footer">
    Diferença = média(A) − média(B). Valores positivos favorecem A quando
    maior é melhor. Significância usa o p-valor corrigido e o alpha do relatório.
  </p>
</div>

<script>
const REPORT = __DATA__;
const PALETTE = ["#2563eb", "#0f9d58", "#d97706", "#dc2626", "#7c3aed",
                 "#0891b2", "#db2777", "#65a30d", "#475569", "#c2410c"];
const state = {
  selected: new Set(REPORT.summaries.map(s => s.algorithm)),
  onlySig: false,
  sortKey: "p_adjusted",
  sortDir: 1
};
const $ = id => document.getElementById(id);
const fmt = (value, digits = 3) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/d";
  if (!Number.isFinite(value)) return value > 0 ? "inf" : "-inf";
  return Number(value).toFixed(digits);
};
const pfmt = value => {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/d";
  if (value < 0.001) return Number(value).toExponential(2);
  return Number(value).toFixed(4);
};
const esc = value => String(value).replace(/[&<>"']/g, ch => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[ch]));
const colorFor = (algorithm, index) => PALETTE[index % PALETTE.length];
const plotConfig = { responsive: true, displaylogo: false,
                     modeBarButtonsToRemove: ["lasso2d", "select2d"] };
const baseLayout = (extra = {}) => Object.assign({
  margin: { l: 58, r: 22, t: 18, b: 70 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif",
          color: "#334155", size: 12 },
  hoverlabel: { bgcolor: "#0f172a", font: { color: "#fff" } },
  showlegend: false
}, extra);

function filteredSummaries() {
  return REPORT.summaries.filter(s => state.selected.has(s.algorithm));
}
function filteredComparisons() {
  return REPORT.comparisons.filter(c =>
    state.selected.has(c.a) && state.selected.has(c.b) &&
    (!state.onlySig || c.significant));
}
function filteredObservations() {
  return REPORT.observations.filter(o => state.selected.has(o.algorithm));
}

function renderMeta() {
  const setup = (REPORT.setups || []).join(", ");
  const scenarios = (REPORT.scenarios || []).join(", ");
  const levels = (REPORT.levels || []).join(", ");
  const pills = [
    `<span class="pill"><b>Entrada:</b> ${esc(REPORT.input_label || "—")}</span>`,
    `<span class="pill"><b>Comparação:</b> ${REPORT.paired ? "pareada por seed" : "independente"}</span>`,
    `<span class="pill"><b>Correção:</b> ${esc(REPORT.correction)}</span>`,
    `<span class="pill"><b>Alpha:</b> ${REPORT.alpha}</span>`,
    `<span class="pill"><b>IC:</b> ${Math.round(REPORT.confidence * 100)}%</span>`
  ];
  if (setup) pills.push(`<span class="pill"><b>Setup:</b> ${esc(setup)}</span>`);
  if (scenarios) pills.push(`<span class="pill"><b>Cenário:</b> ${esc(scenarios)}</span>`);
  if (levels) pills.push(`<span class="pill"><b>Nível:</b> ${esc(levels)}</span>`);
  $("meta").innerHTML = pills.join("");
}

function renderKpis() {
  const summaries = filteredSummaries();
  const observations = filteredObservations();
  const comparisons = filteredComparisons();
  const significant = REPORT.comparisons.filter(c => c.significant &&
    state.selected.has(c.a) && state.selected.has(c.b)).length;
  const best = summaries.reduce((acc, s) =>
    (!acc || s.mean > acc.mean) ? s : acc, null);
  const cards = [
    ["Algoritmos", summaries.length, "selecionados no filtro"],
    ["Observações", observations.length, "linhas de resultado"],
    ["Maior média", best ? fmt(best.mean) : "n/d",
      best ? best.algorithm : "sem dados"],
    ["Comparações significativas", significant,
      `de ${REPORT.comparisons.length} no relatório`]
  ];
  $("kpis").innerHTML = cards.map(([label, value, hint]) => `
    <div class="kpi">
      <div class="label">${esc(label)}</div>
      <div class="value">${esc(value)}</div>
      <div class="hint">${esc(hint)}</div>
    </div>`).join("");
}

function renderAlgorithmFilter() {
  const container = $("algorithm-filter");
  container.innerHTML = "";
  REPORT.summaries.forEach((summary, index) => {
    const label = document.createElement("label");
    label.className = "chip";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.selected.has(summary.algorithm);
    input.style.accentColor = colorFor(summary.algorithm, index);
    input.addEventListener("change", () => {
      if (input.checked) state.selected.add(summary.algorithm);
      else state.selected.delete(summary.algorithm);
      renderAll();
    });
    label.appendChild(input);
    label.appendChild(document.createTextNode(summary.algorithm));
    container.appendChild(label);
  });
}

function showEmpty(id, text) {
  const element = $(id);
  if (window.Plotly) Plotly.purge(element);
  element.innerHTML = `<div class="empty">${esc(text)}</div>`;
}

function draw(id, traces, layout) {
  const element = $(id);
  if (!window.Plotly) {
    element.innerHTML = `<div class="empty">Plotly não carregou (CDN offline). As tabelas continuam disponíveis.</div>`;
    return;
  }
  if (!traces || traces.length === 0) {
    showEmpty(id, "Sem dados para o filtro atual.");
    return;
  }
  // Purge + newPlot (em vez de react) garante que um filtro que esvaziou
  // o gráfico e depois voltou a ter dados redesenhe do zero.
  Plotly.purge(element);
  element.innerHTML = "";
  Plotly.newPlot(element, traces, baseLayout(layout), plotConfig);
}

function renderMeans() {
  const summaries = filteredSummaries();
  if (!summaries.length) return showEmpty("chart-means", "Selecione ao menos um algoritmo.");
  const trace = {
    type: "bar",
    x: summaries.map(s => s.algorithm),
    y: summaries.map(s => s.mean),
    marker: { color: summaries.map((s, i) => colorFor(s.algorithm, i)),
              line: { color: "#fff", width: 1 } },
    error_y: {
      type: "data", symmetric: false,
      array: summaries.map(s => Math.max(0, s.ci_high - s.mean)),
      arrayminus: summaries.map(s => Math.max(0, s.mean - s.ci_low)),
      color: "#334155", thickness: 1.3, width: 4
    },
    customdata: summaries.map(s => [s.n, s.ci_low, s.ci_high]),
    hovertemplate: "<b>%{x}</b><br>média=%{y:.3f}<br>" +
      "IC=[%{customdata[1]:.3f}, %{customdata[2]:.3f}]<br>n=%{customdata[0]}<extra></extra>"
  };
  draw("chart-means", [trace], {
    xaxis: { title: "algoritmo" },
    yaxis: { title: REPORT.metric, zeroline: false, gridcolor: "#eef2f7" }
  });
}

function renderBox() {
  const observations = filteredObservations();
  const algorithms = [...new Set(observations.map(o => o.algorithm))].sort();
  if (!algorithms.length) return showEmpty("chart-box", "Sem observações para o filtro atual.");
  const traces = algorithms.map((algorithm, index) => {
    const subset = observations.filter(o => o.algorithm === algorithm);
    const color = colorFor(algorithm, index);
    return {
      type: "box", name: algorithm,
      y: subset.map(o => o.value),
      text: subset.map(o => o.seed === null || o.seed === undefined ? "" : String(o.seed)),
      boxpoints: "all", jitter: 0.35, pointpos: 0,
      marker: { color: color, size: 7, opacity: 0.85 },
      line: { color: color },
      hovertemplate: `<b>${esc(algorithm)}</b><br>valor=%{y:.3f}` +
        `<br>seed=%{text}<extra></extra>`
    };
  });
  draw("chart-box", traces, {
    xaxis: { title: "" },
    yaxis: { title: REPORT.metric, gridcolor: "#eef2f7" },
    showlegend: false
  });
}

function renderDiff() {
  const comparisons = filteredComparisons();
  if (!comparisons.length) return showEmpty("chart-diff", "Sem comparações entre os algoritmos selecionados. Se o relatório usou --baseline, mantenha o baseline ligado.");
  const labels = comparisons.map(c => `${c.a} − ${c.b}`);
  const trace = {
    type: "bar", orientation: "h",
    x: comparisons.map(c => c.diff),
    y: labels,
    marker: { color: comparisons.map(c => c.significant ? "#0f9d58" : "#94a3b8") },
    error_x: {
      type: "data", symmetric: false,
      array: comparisons.map(c => Math.max(0, c.diff_ci_high - c.diff)),
      arrayminus: comparisons.map(c => Math.max(0, c.diff - c.diff_ci_low)),
      color: "#334155", thickness: 1.2, width: 4
    },
    customdata: comparisons.map(c => [c.p_adjusted, c.effect, c.effect_name]),
    hovertemplate: "<b>%{y}</b><br>diferença=%{x:.3f}<br>" +
      "p corrigido=%{customdata[0]:.4f}<br>" +
      "%{customdata[2]}=%{customdata[1]:.3f}<extra></extra>"
  };
  draw("chart-diff", [trace], {
    xaxis: { title: "média(A) − média(B)", zeroline: true,
             zerolinecolor: "#cbd5e1" },
    yaxis: { automargin: true },
    margin: { l: 150, r: 22, t: 18, b: 60 }
  });
}

function renderEffect() {
  const comparisons = filteredComparisons();
  if (!comparisons.length) return showEmpty("chart-effect", "Sem comparações entre os algoritmos selecionados. Se o relatório usou --baseline, mantenha o baseline ligado.");
  const finite = comparisons.map(c => Math.abs(c.effect)).filter(Number.isFinite);
  const cap = finite.length ? Math.max(1, ...finite) : 1;
  const effectX = c => Number.isFinite(c.effect)
    ? c.effect : Math.sign(c.effect || 1) * cap * 1.2;
  // Comparações com o mesmo efeito/p-valor ficariam exatamente sobrepostas.
  const offsets = comparisons.map((_, index) =>
    (index - (comparisons.length - 1) / 2) * cap * 0.06);
  const trace = {
    type: "scatter", mode: "markers+text",
    x: comparisons.map((c, index) => effectX(c) + offsets[index]),
    y: comparisons.map(c => Math.max(c.p_adjusted, 1e-12)),
    text: comparisons.map(c => `${c.a} × ${c.b}`),
    textposition: comparisons.map((_, index) =>
      index % 2 ? "bottom center" : "top center"),
    textfont: { size: 10, color: "#475569" },
    marker: {
      size: comparisons.map(c => 9 + Math.min(16, Math.abs(effectX(c)) * 3)),
      color: comparisons.map(c => c.significant ? "#0f9d58" : "#94a3b8"),
      line: { color: "#fff", width: 1 }
    },
    customdata: comparisons.map(c => [c.a, c.b, c.effect, c.effect_name, c.p_adjusted]),
    hovertemplate: "<b>%{customdata[0]} × %{customdata[1]}</b><br>" +
      "%{customdata[3]}=%{customdata[2]:.3f}<br>" +
      "p corrigido=%{customdata[4]:.4f}<extra></extra>"
  };
  draw("chart-effect", [trace], {
    xaxis: { title: "tamanho de efeito", zeroline: true,
             zerolinecolor: "#cbd5e1" },
    yaxis: { title: "p corrigido", type: "log", autorange: "reversed",
             gridcolor: "#eef2f7" }
  });
}

function sortComparisons(comparisons) {
  const key = state.sortKey;
  return comparisons.slice().sort((a, b) => {
    let left = a[key], right = b[key];
    if (key === "n") { left = a.paired ? a.n : a.n_a + a.n_b; right = b.paired ? b.n : b.n_a + b.n_b; }
    if (key === "significant") { left = a.significant ? 1 : 0; right = b.significant ? 1 : 0; }
    if (typeof left === "string" || typeof right === "string") {
      return state.sortDir * String(left).localeCompare(String(right));
    }
    return state.sortDir * ((left ?? 0) - (right ?? 0));
  });
}

function renderComparisonsTable() {
  const body = $("table-comparisons").querySelector("tbody");
  const comparisons = sortComparisons(filteredComparisons());
  if (!comparisons.length) {
    body.innerHTML = `<tr><td colspan="10" class="muted">Sem comparações entre os algoritmos selecionados. Se o relatório usou --baseline, mantenha o baseline ligado.</td></tr>`;
    return;
  }
  body.innerHTML = comparisons.map(c => {
    const n = c.paired ? `${c.n} pares` : `${c.n_a}/${c.n_b}`;
    const sig = c.significant
      ? '<span class="badge yes">sim</span>'
      : '<span class="badge no">não</span>';
    return `<tr>
      <td><b>${esc(c.a)}</b></td>
      <td>${esc(c.b)}</td>
      <td class="num">${esc(n)}</td>
      <td class="num">${fmt(c.diff)}</td>
      <td class="num">[${fmt(c.diff_ci_low)}, ${fmt(c.diff_ci_high)}]</td>
      <td>${esc(c.method)}</td>
      <td class="num">${pfmt(c.p_value)}</td>
      <td class="num"><b>${pfmt(c.p_adjusted)}</b></td>
      <td class="num">${esc(c.effect_name)}=${fmt(c.effect)}</td>
      <td>${sig}</td>
    </tr>`;
  }).join("");
}

function renderSummariesTable() {
  const body = $("table-summaries").querySelector("tbody");
  const summaries = filteredSummaries();
  if (!summaries.length) {
    body.innerHTML = `<tr><td colspan="7" class="muted">Sem algoritmos selecionados.</td></tr>`;
    return;
  }
  body.innerHTML = summaries.map(s => `<tr>
    <td><b>${esc(s.algorithm)}</b></td>
    <td class="num">${s.n}</td>
    <td class="num">${fmt(s.mean)}</td>
    <td class="num">${fmt(s.median)}</td>
    <td class="num">${fmt(s.std)}</td>
    <td class="num">${fmt(s.iqr)}</td>
    <td class="num">[${fmt(s.ci_low)}, ${fmt(s.ci_high)}]</td>
  </tr>`).join("");
}

function renderWarnings() {
  const warnings = REPORT.warnings || [];
  const section = $("warnings");
  if (!warnings.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");
  $("warning-list").innerHTML = warnings.map(w => `<li>${esc(w)}</li>`).join("");
}

function renderAll() {
  renderKpis();
  renderMeans();
  renderBox();
  renderDiff();
  renderEffect();
  renderComparisonsTable();
  renderSummariesTable();
}

function downloadJson() {
  const blob = new Blob([JSON.stringify(REPORT, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "report.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

document.querySelectorAll("#table-comparisons th[data-key]").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (state.sortKey === key) state.sortDir *= -1;
    else { state.sortKey = key; state.sortDir = 1; }
    renderComparisonsTable();
  });
});

$("only-significant").addEventListener("change", event => {
  state.onlySig = event.target.checked;
  renderAll();
});
$("download-json").addEventListener("click", downloadJson);
$("reset").addEventListener("click", () => {
  state.selected = new Set(REPORT.summaries.map(s => s.algorithm));
  state.onlySig = false;
  state.sortKey = "p_adjusted";
  state.sortDir = 1;
  $("only-significant").checked = false;
  renderAlgorithmFilter();
  renderAll();
});

renderMeta();
renderAlgorithmFilter();
renderWarnings();
renderAll();
</script>
</body>
</html>
"""
