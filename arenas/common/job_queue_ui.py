"""HTML/CSS/JS da interface web da fila de experimentos."""
from __future__ import annotations


HTML = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RL Arena · Fila de Experimentos</title>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --ink:#0f172a; --muted:#64748b;
          --line:#e2e8f0; --accent:#2563eb; --green:#0f9d58;
          --green-bg:#dcfce7; --green-ink:#166534; --amber:#d97706;
          --red:#dc2626; --red-bg:#fee2e2; --radius:16px; }
  * { box-sizing: border-box; }
  body { margin:0; background:linear-gradient(180deg,#eef4ff 0,#f8fafc 300px,var(--bg) 100%);
         color:var(--ink); font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:1500px; margin:0 auto; padding:26px 22px 70px; }
  .eyebrow { color:var(--accent); font-size:.78rem; font-weight:700;
             letter-spacing:.12em; text-transform:uppercase; }
  h1 { margin:8px 0 4px; font-size:clamp(1.6rem,3vw,2.3rem); letter-spacing:-.03em; }
  .muted { color:var(--muted); }
  .top { display:flex; flex-wrap:wrap; gap:16px; justify-content:space-between;
         align-items:flex-end; }
  .pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
  .pill { background:#fff; border:1px solid var(--line); border-radius:999px;
          padding:5px 11px; font-size:.82rem; }
  .pill b { color:var(--ink); }
  .actions { display:flex; gap:8px; flex-wrap:wrap; }
  button { border:1px solid var(--line); background:#fff; color:var(--ink);
           border-radius:10px; padding:8px 12px; font:inherit; cursor:pointer; }
  button:hover { background:#f8fafc; border-color:#cbd5e1; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.primary:hover { background:#1d4ed8; }
  button.danger { color:var(--red); border-color:#fecaca; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
          padding:18px; box-shadow:0 10px 30px rgba(15,23,42,.05); margin-top:18px; }
  .card h2 { margin:0 0 10px; font-size:1.03rem; }
  .grid { display:grid; grid-template-columns:repeat(12,1fr); gap:12px; }
  .grid .field { display:flex; flex-direction:column; gap:5px; }
  .field label { font-size:.78rem; color:var(--muted); font-weight:600; }
  input, textarea { width:100%; border:1px solid var(--line); border-radius:9px;
                    padding:8px 10px; font:inherit; background:#fff; }
  textarea { min-height:72px; resize:vertical; font-family:ui-monospace,monospace; font-size:.86rem; }
  .span-6 { grid-column:span 6; } .span-4 { grid-column:span 4; }
  .span-3 { grid-column:span 3; } .span-12 { grid-column:span 12; }
  @media(max-width:900px){ .span-6,.span-4,.span-3{grid-column:span 12;} }
  .row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  .status { display:inline-block; padding:3px 9px; border-radius:999px;
            font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.03em; }
  .s-queued { background:#e2e8f0; color:#334155; }
  .s-running { background:#dbeafe; color:#1d4ed8; }
  .s-done { background:var(--green-bg); color:var(--green-ink); }
  .s-failed { background:var(--red-bg); color:#991b1b; }
  .s-canceled { background:#f1f5f9; color:#475569; }
  .s-paused { background:#fef3c7; color:#92400e; }
  table { width:100%; border-collapse:collapse; font-size:.86rem; }
  th,td { padding:9px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
  th { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.05em; }
  tr:hover { background:#f8fbff; }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.8rem; }
  .cmd { max-width:420px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .job-actions { display:flex; flex-wrap:wrap; gap:6px; }
  .job-actions button { padding:4px 8px; font-size:.76rem; }
  .empty { color:var(--muted); padding:18px; text-align:center; }
  pre#log { margin:0; background:#0f172a; color:#dbeafe; padding:14px; border-radius:12px;
            min-height:220px; max-height:460px; overflow:auto; font-size:.8rem;
            font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; }
  .banner { display:none; margin-top:12px; padding:10px 12px; border-radius:10px;
            background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  .banner.show { display:block; }
  .hidden { display:none !important; }
  .artifacts { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:10px; }
  .artifact { border:1px solid var(--line); border-radius:10px; padding:10px; background:#fbfdff; }
  .artifact a { color:var(--accent); text-decoration:none; font-weight:600; }
  .artifact video, .artifact img { width:100%; border-radius:8px; margin-top:6px; background:#0f172a; }
  .artifact .kind { color:var(--muted); font-size:.75rem; text-transform:uppercase; }
  .footer { color:var(--muted); font-size:.8rem; margin-top:22px; }
  .tags { color:var(--muted); font-size:.76rem; }
  .tabs { display:flex; gap:6px; flex-wrap:wrap; margin-top:20px;
          border-bottom:1px solid var(--line); }
  .tabs button { border:1px solid transparent; border-bottom:0;
                 border-radius:10px 10px 0 0; background:transparent;
                 color:var(--muted); font-weight:600; }
  .tabs button.active { background:#fff; color:var(--ink);
                        border-color:var(--line); margin-bottom:-1px; }
  fieldset { border:1px solid var(--line); border-radius:12px; padding:14px;
             margin:0 0 14px; }
  legend { padding:0 8px; font-weight:700; font-size:.85rem; color:var(--accent); }
  .check { display:inline-flex; align-items:center; gap:6px; font-size:.86rem; }
  .check input { width:auto; }
  .warnings ul { margin:6px 0 0 18px; padding:0; }
  .code { margin:0; background:#0f172a; color:#dbeafe; padding:14px;
          border-radius:12px; max-height:520px; overflow:auto;
          white-space:pre-wrap; font-size:.78rem;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .report-frame { width:100%; height:840px; border:1px solid var(--line);
                  border-radius:12px; background:#fff; margin-top:12px; }
  .s-secondary { background:#fef3c7; color:#92400e; }
  .s-unknown { background:#e2e8f0; color:#334155; }
  .algo-arena { border-top:1px solid var(--line); padding-top:14px;
                margin-top:16px; }
  .algo-arena:first-child { border-top:0; padding-top:0; margin-top:0; }
  .algo-arena h3 { margin:0; font-size:1.05rem; }
  .algo-name { font-weight:700; }
  .badge { display:inline-block; padding:1px 7px; border-radius:999px;
           font-size:.7rem; font-weight:700; margin-left:6px;
           background:#dbeafe; color:#1d4ed8; }
  .doc-links { display:flex; flex-wrap:wrap; gap:6px; }
  .doc-links a { color:var(--accent); text-decoration:none; font-weight:600;
                 font-size:.82rem; border:1px solid var(--line);
                 border-radius:8px; padding:3px 8px; background:#fbfdff; }
  .plot { width:100%; min-height:340px; }
  .plot-sm { width:100%; min-height:260px; }
  .table-wrap { max-height:560px; overflow:auto; }
  .table-wrap table th { position:sticky; top:0; background:#fff; }
  .sortable { cursor:pointer; user-select:none; }
  .sortable:hover { color:var(--accent); }
  .bo-summary { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  .bo-summary .pill b { color:var(--ink); }
</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <div class="eyebrow">RL Arena · experimentos</div>
      <h1>Fila de Experimentos</h1>
      <div class="pills" id="status-pills"></div>
    </div>
    <div class="actions">
      <button id="toggle" class="primary">Pausar fila</button>
      <button id="refresh">Atualizar</button>
    </div>
  </header>

  <div id="banner" class="banner"></div>

  <nav class="tabs" role="tablist" aria-label="Seções da Fila">
    <button class="tab active" data-tab="queue" role="tab">Fila</button>
    <button class="tab" data-tab="algorithms" role="tab">Algoritmos</button>
    <button class="tab" data-tab="analysis" role="tab">Análise de resultados</button>
    <button class="tab" data-tab="bo" role="tab">BO</button>
    <button class="tab" data-tab="proposal" role="tab">Proposta de experimentos</button>
  </nav>

  <div id="tab-queue" class="tab-panel">
  <section class="card">
    <h2>Adicionar job</h2>
    <form id="add-form">
      <div class="grid">
        <div class="field span-4">
          <label for="name">Nome</label>
          <input id="name" name="name" placeholder="ppo-level3-30min" required>
        </div>
        <div class="field span-3">
          <label for="priority">Prioridade</label>
          <input id="priority" name="priority" type="number" value="0">
        </div>
        <div class="field span-3">
          <label for="max_runtime">Tempo máx. (s)</label>
          <input id="max_runtime" name="max_runtime" type="number" step="1" placeholder="opcional">
        </div>
        <div class="field span-2">
          <label for="cwd">cwd</label>
          <input id="cwd" name="cwd" value=".">
        </div>
        <div class="field span-12">
          <label for="command">Comando Python</label>
          <textarea id="command" name="command" required
            placeholder="python train.py --config configs/seek_coin/ppo-level3.yaml --set training.max_hours=0.5"></textarea>
        </div>
        <div class="field span-6">
          <label for="depends_on">Depende de (IDs separados por vírgula)</label>
          <input id="depends_on" name="depends_on" placeholder="job-...">
        </div>
        <div class="field span-6">
          <label for="artifacts">Artefatos (caminhos separados por vírgula)</label>
          <input id="artifacts" name="artifacts" placeholder="checkpoints/..., logs/...">
        </div>
      </div>
      <div class="row" style="margin-top:12px">
        <button type="submit" class="primary">Adicionar job</button>
        <span class="muted">O comando é validado; por padrão só Python do repositório.</span>
      </div>
    </form>
  </section>

  <section class="card">
    <h2>Enfileirar experimento</h2>
    <form id="experiment-form">
      <div class="row">
        <input id="experiment-path" name="experiment_path" style="flex:1;min-width:320px"
               placeholder="docs/experimentos/EXP-AAAAMMDD-slug.yaml" required>
        <button type="submit" class="primary">Enfileirar YAML de experimento</button>
      </div>
      <p class="muted" style="margin:8px 0 0">
        O YAML precisa ter <span class="mono">execution.command</span>.
        A fila mostra o YAML, logs, relatórios, vídeos e estatísticas associados.
      </p>
    </form>
  </section>

  <section class="card">
    <div class="row" style="justify-content:space-between">
      <h2>Fila</h2>
      <span class="muted" id="counts"></span>
    </div>
    <div style="overflow:auto">
      <table>
        <thead><tr>
          <th>Job</th><th>Status</th><th>Comando</th><th>Progresso</th><th>Ações</th>
        </tr></thead>
        <tbody id="job-rows"><tr><td colspan="5" class="empty">Carregando…</td></tr></tbody>
      </table>
    </div>
  </section>

  <section class="card">
    <div class="row" style="justify-content:space-between">
      <h2 id="log-title">Log</h2>
      <button id="log-refresh">Atualizar log</button>
    </div>
    <pre id="log">Selecione “Logs” em um job.</pre>
  </section>

  <section class="card hidden" id="experiment-panel">
    <div class="row" style="justify-content:space-between">
      <h2 id="experiment-title">Experimento</h2>
      <button id="experiment-close">Fechar</button>
    </div>
    <div class="row" style="justify-content:space-between;margin-bottom:10px">
      <div id="experiment-status" class="muted"></div>
      <button id="experiment-bo">Ver BO</button>
      <button id="experiment-sync" class="hidden">Atualizar status do YAML</button>
    </div>
    <div class="grid">
      <div class="field span-12">
        <label>YAML do experimento</label>
        <pre id="experiment-yaml" style="background:#0f172a;color:#dbeafe;padding:14px;border-radius:12px;max-height:420px;overflow:auto;white-space:pre-wrap"></pre>
      </div>
      <div class="field span-12">
        <label>Artefatos</label>
        <div id="artifact-list" class="artifacts"></div>
      </div>
    </div>
  </section>
  </div>

  <div id="tab-algorithms" class="tab-panel hidden">
    <section class="card">
      <div class="row" style="justify-content:space-between">
        <h2>Algoritmos por arena</h2>
        <a id="algorithms-guide" class="muted" href="/docs/docs/algoritmos.md#como-escolher"
           target="_blank" rel="noopener">Como escolher →</a>
      </div>
      <p class="muted" style="margin-top:0">
        Lista derivada do <span class="mono">--alg</span> aceito por cada arena
        e dos pacotes em <span class="mono">arenas/algs/</span>. Os links abrem
        a documentação renderizada pelo servidor local.
      </p>
      <div id="algorithms-list"><span class="muted">Carregando…</span></div>
    </section>
  </div>

  <div id="tab-analysis" class="tab-panel hidden">
    <section class="card">
      <h2>Análise de resultados</h2>
      <p class="muted" style="margin-top:0">
        Descritivas, IC, teste, tamanho de efeito e p corrigido vêm de
        <span class="mono">arenas/common/report.py</span>; o relatório
        interativo é o mesmo HTML usado pela CLI.
      </p>
      <div class="grid">
        <div class="field span-6">
          <label for="analysis-job">Job / experimento</label>
          <select id="analysis-job"></select>
        </div>
        <div class="field span-6">
          <label for="analysis-source">Arquivo de resultados</label>
          <select id="analysis-source"></select>
        </div>
        <div class="field span-3">
          <label for="analysis-metric">Métrica</label>
          <select id="analysis-metric"></select>
        </div>
        <div class="field span-3">
          <label for="analysis-seed">Unidade independente</label>
          <select id="analysis-seed"></select>
        </div>
        <div class="field span-3">
          <label for="analysis-baseline">Baseline</label>
          <select id="analysis-baseline"></select>
        </div>
        <div class="field span-3">
          <label for="analysis-correction">Correção</label>
          <select id="analysis-correction">
            <option value="holm">Holm-Bonferroni</option>
            <option value="bh">Benjamini-Hochberg</option>
            <option value="none">Nenhuma</option>
          </select>
        </div>
        <div class="field span-3">
          <label for="analysis-confidence">Confiança</label>
          <input id="analysis-confidence" value="0.95">
        </div>
        <div class="field span-3">
          <label for="analysis-bootstrap">Bootstrap</label>
          <input id="analysis-bootstrap" type="number" value="10000">
        </div>
        <div class="field span-3">
          <label for="analysis-alpha">Alpha</label>
          <input id="analysis-alpha" value="0.05">
        </div>
        <div class="field span-3">
          <label for="analysis-filter">Filtro (campo=valor)</label>
          <input id="analysis-filter" placeholder="arena=seek_coin">
        </div>
        <div class="field span-12">
          <label for="analysis-path">Caminho manual (opcional)</label>
          <input id="analysis-path"
                 placeholder="experiments/EXP-AAAAMMDD-slug/results.csv">
        </div>
      </div>
      <div id="analysis-artifacts" class="artifacts" style="margin-top:12px"></div>
      <div class="row" style="margin-top:12px">
        <label class="check">
          <input type="checkbox" id="analysis-paired"> Pareado por seed
        </label>
        <label class="check">
          <input type="checkbox" id="analysis-aggregate">
          Agregar por unidade (--aggregate)
        </label>
        <button id="analysis-run" class="primary">Gerar análise</button>
        <button id="analysis-open">Abrir relatório em nova aba</button>
      </div>
      <div id="analysis-badges" class="row" style="margin-top:12px"></div>
      <div id="analysis-warnings" class="warnings"></div>
      <div id="analysis-frame-wrap" class="hidden">
        <iframe id="analysis-frame" class="report-frame"
                title="Relatório estatístico"></iframe>
      </div>
    </section>
  </div>

  <div id="tab-bo" class="tab-panel hidden">
    <section class="card">
      <h2>Evolução do BO</h2>
      <p class="muted" style="margin-top:0">
        Lê <span class="mono">campaign.json</span>,
        <span class="mono">best.json</span> e
        <span class="mono">trials.jsonl</span> (BO e ASHA) que já existem no
        diretório da campanha. Os gráficos são <b>exploratórios</b>: com
        poucos trials não há causalidade nem “melhor parâmetro” sem
        incerteza. Trials falhos continuam na tabela e são excluídos das
        estatísticas.
      </p>
      <div class="grid">
        <div class="field span-4">
          <label for="bo-job">Job / experimento</label>
          <select id="bo-job"></select>
        </div>
        <div class="field span-5">
          <label for="bo-path">Caminho da campanha (opcional)</label>
          <input id="bo-path" placeholder="experiments/BO-AAAAMMDD-slug">
        </div>
        <div class="field span-3" style="align-self:end">
          <button id="bo-load" class="primary">Carregar BO</button>
        </div>
        <div class="field span-3">
          <label for="bo-alg">Algoritmo</label>
          <select id="bo-alg"></select>
        </div>
        <div class="field span-3">
          <label for="bo-status">Status</label>
          <select id="bo-status">
            <option value="">todos</option>
          </select>
        </div>
        <div class="field span-3">
          <label for="bo-param">Parâmetro</label>
          <select id="bo-param"></select>
        </div>
        <div class="field span-3" style="align-self:end">
          <label>&nbsp;</label>
          <div class="row">
            <button id="bo-export-csv">CSV</button>
            <button id="bo-export-json">JSON</button>
          </div>
        </div>
      </div>
      <div id="bo-warnings" class="warnings"></div>
      <div id="bo-summary" class="bo-summary"></div>
      <div id="bo-meta" class="muted" style="margin-top:8px"></div>
    </section>

    <section class="card">
      <h3>Convergência</h3>
      <p class="muted" style="margin-top:0">
        Score por trial e melhor score acumulado (best-so-far), por algoritmo.
        ★ = trial escolhido como melhor; ◆ = validação final.
      </p>
      <div id="bo-convergence" class="plot"></div>
    </section>

    <section class="grid">
      <div class="card">
        <h3>Efeito do parâmetro</h3>
        <div id="bo-effect" class="plot"></div>
      </div>
      <div class="card">
        <h3>Importância dos parâmetros</h3>
        <div id="bo-importance" class="plot-sm"></div>
      </div>
    </section>

    <section class="card">
      <h3>Melhoria sobre a referência</h3>
      <p class="muted" style="margin-top:0">
        Best-so-far comparado à mediana e ao primeiro trial válido; no BO, a
        validação final aparece separada por orçamento.
      </p>
      <div id="bo-improvement" class="plot-sm"></div>
    </section>

    <section class="grid">
      <div class="card">
        <h3>Interação (2 parâmetros mais importantes)</h3>
        <div id="bo-interaction" class="plot"></div>
      </div>
      <div class="card">
        <h3>Parallel coordinates</h3>
        <div id="bo-parallel" class="plot"></div>
      </div>
    </section>

    <section class="card">
      <h3>Trials</h3>
      <div id="bo-table-meta" class="muted" style="margin-bottom:8px"></div>
      <div class="table-wrap">
        <table id="bo-trials"></table>
      </div>
    </section>
  </div>

  <div id="tab-proposal" class="tab-panel hidden">
    <section class="card">
      <h2>Proposta de experimentos</h2>
      <p class="muted" style="margin-top:0">
        Formulário guiado por
        <span class="mono">docs/experimentos/TEMPLATE.md</span> e pela skill
        <span class="mono">experimenter-helper</span>. Validar e salvar não
        executam nada; enfileirar usa o mesmo caminho da aba Fila.
      </p>
      <form id="proposal-form" autocomplete="off">
        <fieldset>
          <legend>Pergunta e hipótese</legend>
          <div class="grid">
            <div class="field span-4">
              <label>Título</label>
              <input data-proposal="title" required>
            </div>
            <div class="field span-4">
              <label>Slug do ID (opcional)</label>
              <input data-proposal="slug" placeholder="eppo-vs-ppo">
            </div>
            <div class="field span-4">
              <label>Autor</label>
              <input data-proposal="author">
            </div>
            <div class="field span-12">
              <label>Pergunta (uma frase testável)</label>
              <textarea data-proposal="question" required></textarea>
            </div>
            <div class="field span-6">
              <label>Hipótese</label>
              <textarea data-proposal="hypothesis"></textarea>
            </div>
            <div class="field span-6">
              <label>Falsificação</label>
              <textarea data-proposal="falsification"></textarea>
            </div>
            <div class="field span-12">
              <label>Claim permitido</label>
              <textarea data-proposal="claim_allowed"></textarea>
            </div>
            <div class="field span-4">
              <label>Experimentos relacionados (CSV)</label>
              <input data-proposal="prior_experiments"
                     placeholder="EXP-20261115-bo-level3-6h">
            </div>
            <div class="field span-4">
              <label>Setups existentes (CSV)</label>
              <input data-proposal="existing_setups" placeholder="S6">
            </div>
            <div class="field span-4">
              <label>Lacuna</label>
              <input data-proposal="gap">
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Setup e algoritmos</legend>
          <div class="grid">
            <div class="field span-3">
              <label>Arena</label>
              <select data-proposal="arena">
                <option value="seek_coin">seek_coin</option>
                <option value="risk">risk</option>
                <option value="economy">economy</option>
              </select>
            </div>
            <div class="field span-3">
              <label>Nível</label>
              <input data-proposal="level" type="number" min="0">
            </div>
            <div class="field span-3">
              <label>Cenário</label>
              <input data-proposal="scenario">
            </div>
            <div class="field span-3">
              <label>Setup</label>
              <input data-proposal="setup_id" placeholder="S6">
            </div>
            <div class="field span-6">
              <label>Configs YAML (CSV)</label>
              <input data-proposal="config_files"
                     placeholder="configs/seek_coin/ppo-level3.yaml">
            </div>
            <div class="field span-3">
              <label>Candidatos (CSV)</label>
              <input data-proposal="candidates" required
                     placeholder="ppo, eppo">
            </div>
            <div class="field span-3">
              <label>Baseline (CSV)</label>
              <input data-proposal="baseline" required placeholder="ppo">
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Métrica primária</legend>
          <div class="grid">
            <div class="field span-3">
              <label>Nome</label>
              <input data-proposal="metric_name" placeholder="reward">
            </div>
            <div class="field span-3">
              <label>Direção</label>
              <select data-proposal="metric_direction">
                <option value="maximize">maximize</option>
                <option value="minimize">minimize</option>
              </select>
            </div>
            <div class="field span-6">
              <label>Definição operacional</label>
              <input data-proposal="metric_definition"
                     placeholder="reward guloso médio por seed de treino">
            </div>
            <div class="field span-12">
              <label>Métricas secundárias (CSV)</label>
              <input data-proposal="secondary_metrics"
                     placeholder="coins, deaths, env_steps">
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Seeds e orçamento</legend>
          <div class="grid">
            <div class="field span-4">
              <label>Seeds de treino (CSV, 3+)</label>
              <input data-proposal="seeds_train" required
                     placeholder="7, 42, 123">
            </div>
            <div class="field span-2">
              <label>Seeds de avaliação</label>
              <input data-proposal="seeds_eval" placeholder="1000">
            </div>
            <div class="field span-2">
              <label>Episódios de avaliação</label>
              <input data-proposal="eval_episodes" type="number" value="50">
            </div>
            <div class="field span-4">
              <label>Passos de ambiente por run</label>
              <input data-proposal="env_steps_per_run" placeholder="4M">
            </div>
            <div class="field span-3">
              <label>Horas por run</label>
              <input data-proposal="wall_hours_per_run" type="number"
                     step="0.1" value="2">
            </div>
            <div class="field span-3">
              <label>Número de runs</label>
              <input data-proposal="n_runs" type="number">
            </div>
            <div class="field span-6">
              <label>Hardware</label>
              <input data-proposal="hardware"
                     placeholder="1× RTX 3050 6 GB">
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Tuning e avaliação</legend>
          <div class="grid">
            <div class="field span-3">
              <label>Método de tuning</label>
              <select data-proposal="tuning_method">
                <option value="none">none</option>
                <option value="bo">bo</option>
                <option value="asha">asha</option>
                <option value="grid">grid</option>
                <option value="random">random</option>
              </select>
            </div>
            <div class="field span-3">
              <label>Espaço de busca</label>
              <input data-proposal="space_file"
                     placeholder="configs/search_spaces/ppo.yaml">
            </div>
            <div class="field span-3">
              <label>Horas de tuning</label>
              <input data-proposal="tuning_budget_hours" type="number"
                     step="0.1">
            </div>
            <div class="field span-3">
              <label>Checkpoint</label>
              <select data-proposal="checkpoint">
                <option value="latest">latest</option>
                <option value="best_validation">best_validation</option>
                <option value="fixed_step">fixed_step</option>
              </select>
            </div>
            <div class="field span-6">
              <label>Regra de seleção</label>
              <input data-proposal="selection_rule">
            </div>
            <div class="field span-6">
              <label>Truncamento</label>
              <input data-proposal="truncation"
                     placeholder="sem bootstrap; recompensa acumulada">
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Estatística</legend>
          <div class="grid">
            <div class="field span-3">
              <label>Unidade independente</label>
              <input data-proposal="independent_unit" value="train_seed">
            </div>
            <div class="field span-3">
              <label>Teste</label>
              <input data-proposal="test" value="wilcoxon">
            </div>
            <div class="field span-3">
              <label>Tamanho de efeito</label>
              <input data-proposal="effect_size" value="cohens_d">
            </div>
            <div class="field span-3">
              <label>Correção</label>
              <input data-proposal="correction" value="holm">
            </div>
            <div class="field span-3">
              <label>Confiança</label>
              <input data-proposal="confidence" value="0.95">
            </div>
            <div class="field span-9" style="justify-content:flex-end">
              <label>&nbsp;</label>
              <div class="row">
                <label class="check">
                  <input type="checkbox" data-proposal="paired" checked>
                  Pareado
                </label>
                <label class="check">
                  <input type="checkbox" data-proposal="greedy" checked>
                  Política gulosa
                </label>
                <label class="check">
                  <input type="checkbox" data-proposal="same_protocol" checked>
                  Mesmo protocolo
                </label>
              </div>
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Critério de decisão</legend>
          <div class="grid">
            <div class="field span-4">
              <label>Suporta a hipótese</label>
              <textarea data-proposal="support"></textarea>
            </div>
            <div class="field span-4">
              <label>Inconclusivo</label>
              <textarea data-proposal="inconclusive"></textarea>
            </div>
            <div class="field span-4">
              <label>Abandona</label>
              <textarea data-proposal="abandon"></textarea>
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Execução e artefatos</legend>
          <div class="grid">
            <div class="field span-12">
              <label>Comando (argv, sem shell)</label>
              <textarea data-proposal="command"
                placeholder="python train.py --config configs/seek_coin/ppo-level3.yaml"></textarea>
            </div>
            <div class="field span-3">
              <label>cwd</label>
              <input data-proposal="cwd" value=".">
            </div>
            <div class="field span-3">
              <label>Tempo máximo (s)</label>
              <input data-proposal="max_runtime" type="number" value="21600">
            </div>
            <div class="field span-3">
              <label>Prioridade</label>
              <input data-proposal="priority" type="number" value="0">
            </div>
            <div class="field span-3">
              <label>Pasta do experimento</label>
              <input data-proposal="experiment_dir"
                     placeholder="experiments/EXP-...">
            </div>
            <div class="field span-6">
              <label>CSV de resultados</label>
              <input data-proposal="results_csv"
                     placeholder="experiments/EXP-.../results.csv">
            </div>
            <div class="field span-6">
              <label>Riscos (uma linha cada)</label>
              <textarea data-proposal="risks"></textarea>
            </div>
          </div>
        </fieldset>
      </form>
    </section>

    <section class="card">
      <div class="row">
        <button id="proposal-preview" class="primary">Prévia MD + YAML</button>
        <button id="proposal-save">Salvar</button>
        <button id="proposal-enqueue">Salvar e enfileirar</button>
        <label class="check">
          <input type="checkbox" id="proposal-overwrite"> Sobrescrever
        </label>
      </div>
      <div id="proposal-errors" class="warnings"></div>
      <div id="proposal-result" class="muted" style="margin-top:8px"></div>
      <div class="grid" style="margin-top:12px">
        <div class="field span-6">
          <label>YAML estruturado</label>
          <pre id="proposal-yaml" class="code">—</pre>
        </div>
        <div class="field span-6">
          <label>Markdown</label>
          <pre id="proposal-md" class="code">—</pre>
        </div>
      </div>
    </section>
  </div>

  <p class="footer">
    Servidor local. A fila é persistida em JSON; jobs em execução continuam
    mesmo se a página for fechada.
  </p>
</div>

<script>
const $ = (id) => document.getElementById(id);
let state = { jobs: [], selected: null, paused: false, tab: "queue",
              analysisSources: [], analysisSource: null, analysisHtml: "",
              algorithms: null, bo: null, boAlgorithm: null, boParam: null,
              boStatus: "", boSort: {key: "index", dir: 1} };
let loading = false;

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, ch => ({
  "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
}[ch]));

function showError(message) {
  const banner = $("banner");
  banner.textContent = message || "";
  banner.classList.toggle("show", Boolean(message));
}

window.addEventListener("error", (event) => {
  showError(`Erro de interface: ${event.message}`);
});
window.addEventListener("unhandledrejection", (event) => {
  const reason = event.reason;
  showError(`Erro de interface: ${reason && reason.message ? reason.message : reason}`);
});

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function fmtSeconds(value) {
  if (value === null || value === undefined) return "—";
  const s = Math.max(0, Math.round(value));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  if (h) return `${h}h${String(m).padStart(2,"0")}`;
  if (m) return `${m}m${String(s % 60).padStart(2,"0")}`;
  return `${s}s`;
}

function statusBadge(status) {
  const cls = {queued:"s-queued", running:"s-running", done:"s-done",
               failed:"s-failed", canceled:"s-canceled", paused:"s-paused",
               planned:"s-queued", aborted:"s-failed"}[status] || "s-queued";
  return `<span class="status ${cls}">${esc(status)}</span>`;
}

function jobProgress(job) {
  if (job.status === "running" && job.started_at) {
    const elapsed = job.elapsed_seconds ?? 0;
    if (job.max_runtime) return `${fmtSeconds(elapsed)} / ${fmtSeconds(job.max_runtime)}`;
    return fmtSeconds(elapsed);
  }
  if (job.started_at && job.finished_at) return "finalizado";
  return "—";
}

function render() {
  const jobs = state.jobs || [];
  $("toggle").textContent = state.paused ? "Retomar fila" : "Pausar fila";
  $("toggle").classList.toggle("primary", !state.paused);
  const running = jobs.filter(j => j.status === "running").length;
  const queued = jobs.filter(j => j.status === "queued").length;
  $("status-pills").innerHTML = `
    <span class="pill"><b>Fila:</b> ${state.paused ? "pausada" : "ativa"}</span>
    <span class="pill"><b>Rodando:</b> ${running}/${state.max_parallel}</span>
    <span class="pill"><b>Na fila:</b> ${queued}</span>
    <span class="pill"><b>Total:</b> ${jobs.length}</span>`;
  $("counts").textContent = `${jobs.length} jobs · paralelismo ${state.max_parallel}`;
  const rows = $("job-rows");
  if (!jobs.length) {
    rows.innerHTML = '<tr><td colspan="5" class="empty">Nenhum job na fila.</td></tr>';
    return;
  }
  rows.innerHTML = jobs.map(job => `
    <tr>
      <td>
        <b>${esc(job.name)}</b>
        ${job.mode === "local" ? '<span class="tags"> · local</span>' : ''}<br>
        ${job.experiment ? `<span class="tags">EXP: ${esc(job.experiment.id || job.experiment.path)}</span><br>` : ''}
        <span class="mono muted">${esc(job.id)}</span><br>
        <span class="tags">${esc(Object.entries(job.tags || {}).map(([k,v]) => `${k}=${v}`).join(" · "))}</span>
      </td>
      <td>${statusBadge(job.status)}</td>
      <td><div class="cmd mono" title="${esc((job.command || []).join(" "))}">${esc((job.command || []).join(" "))}</div></td>
      <td class="mono">
        ${jobProgress(job)}<br>
        ${job.pid ? `pid ${esc(job.pid)}` : ""}
        ${job.exit_code !== null && job.exit_code !== undefined ? `<br>exit ${esc(job.exit_code)}` : ""}
        ${job.error ? `<br><span class="muted">${esc(job.error)}</span>` : ""}
      </td>
      <td>
        <div class="job-actions">
          <button data-action="up" data-id="${esc(job.id)}" title="Subir prioridade">↑</button>
          <button data-action="down" data-id="${esc(job.id)}" title="Descer prioridade">↓</button>
          <button data-action="cancel" data-id="${esc(job.id)}" class="danger">Cancelar</button>
          <button data-action="retry" data-id="${esc(job.id)}">Retry</button>
          <button data-action="logs" data-id="${esc(job.id)}" class="primary">Logs</button>
          ${job.experiment ? `<button data-action="experiment" data-id="${esc(job.id)}">YAML</button>` : ''}
          <button data-action="artifacts" data-id="${esc(job.id)}">Artefatos</button>
        </div>
      </td>
    </tr>`).join("");
}

function artifactCard(artifact) {
  const label = `${esc(artifact.label || artifact.name)} <span class="kind">${esc(artifact.kind)}</span>`;
  const url = esc(artifact.url);
  if (!artifact.exists) {
    return `<div class="artifact"><span class="muted">${label} · ausente</span><br><span class="mono">${esc(artifact.path)}</span></div>`;
  }
  if (artifact.kind === "video") {
    return `<div class="artifact"><div>${label}</div><video controls preload="metadata" src="${url}"></video><a href="${url}" target="_blank">abrir vídeo</a></div>`;
  }
  if (artifact.kind === "image") {
    return `<div class="artifact"><div>${label}</div><img src="${url}" alt="${esc(artifact.name)}"><a href="${url}" target="_blank">abrir imagem</a></div>`;
  }
  return `<div class="artifact"><div>${label}</div><a href="${url}" target="_blank">${esc(artifact.path)}</a></div>`;
}

async function openExperiment(id) {
  const [exp, artifacts] = await Promise.all([
    api(`/api/jobs/${encodeURIComponent(id)}/experiment`).catch(() => ({yaml: null})),
    api(`/api/jobs/${encodeURIComponent(id)}/artifacts`).catch(() => ({artifacts: []})),
  ]);
  const job = (state.jobs || []).find(item => item.id === id);
  state.experimentId = id;
  $("experiment-title").textContent = job?.experiment?.id
    ? `Experimento · ${job.experiment.id}`
    : `Experimento · ${id}`;
  const experiment = job?.experiment;
  if (experiment) {
    const declared = experiment.declared_status ?? experiment.status ?? "?";
    const effective = experiment.effective_status ?? declared;
    const suffix = experiment.status_mismatch
      ? ' <span class="muted">(divergente)</span>' : '';
    $("experiment-status").innerHTML =
      `YAML: ${statusBadge(declared)} · efetivo: ${statusBadge(effective)}${suffix}`;
    $("experiment-sync").classList.toggle(
      "hidden", !experiment.can_update_status);
  } else {
    $("experiment-status").textContent = "";
    $("experiment-sync").classList.add("hidden");
  }
  $("experiment-yaml").textContent = exp.yaml || "(sem YAML de experimento)";
  const items = artifacts.artifacts || [];
  $("artifact-list").innerHTML = items.length
    ? items.map(artifactCard).join("")
    : '<span class="muted">Nenhum artefato registrado.</span>';
  $("experiment-panel").classList.remove("hidden");
  $("experiment-panel").scrollIntoView({behavior: "smooth", block: "start"});
}

async function refresh() {
  if (loading) return;
  loading = true;
  try {
    const selected = state.selected;
    // Mescla em vez de substituir: chamadas concorrentes (aba de
    // algoritmos, análise) atualizam o mesmo objeto sem perder dados.
    Object.assign(state, await api("/api/state"));
    state.selected = selected;
    render();
    populateAnalysisJobs();
    showError("");
  } catch (error) {
    showError(`Falha ao atualizar: ${error.message}`);
  } finally {
    loading = false;
  }
}

async function refreshLog() {
  if (!state.selected) return;
  try {
    const payload = await api(`/api/jobs/${encodeURIComponent(state.selected)}/log?lines=250`);
    $("log").textContent = payload.log || "(log vazio)";
  } catch (error) {
    $("log").textContent = `Erro ao ler log: ${error.message}`;
  }
}

async function post(path, body = {}) {
  try {
    const payload = await api(path, { method: "POST", body: JSON.stringify(body) });
    showError("");
    await refresh();
    return payload;
  } catch (error) {
    showError(error.message);
    return null;
  }
}

$("refresh").addEventListener("click", refresh);
$("log-refresh").addEventListener("click", refreshLog);
$("toggle").addEventListener("click", () => {
  post(state.paused ? "/api/queue/resume" : "/api/queue/pause");
});

$("add-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const depends = $("depends_on").value.split(",").map(s => s.trim()).filter(Boolean);
  const artifacts = $("artifacts").value.split(",").map(s => s.trim()).filter(Boolean);
  const runtime = $("max_runtime").value.trim();
  const payload = {
    name: $("name").value.trim(),
    command: $("command").value.trim(),
    cwd: $("cwd").value.trim() || ".",
    priority: Number($("priority").value || 0),
    depends_on: depends,
    artifacts,
    max_runtime: runtime ? Number(runtime) : null,
  };
  const result = await post("/api/jobs", payload);
  if (result) {
    event.target.reset();
    $("cwd").value = ".";
    $("priority").value = "0";
  }
});

$("job-rows").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const id = button.dataset.id;
  const action = button.dataset.action;
  if (action === "logs") {
    state.selected = id;
    $("log-title").textContent = `Log · ${id}`;
    await refreshLog();
    return;
  }
  if (action === "cancel") await post(`/api/jobs/${encodeURIComponent(id)}/cancel`);
  if (action === "retry") await post(`/api/jobs/${encodeURIComponent(id)}/retry`);
  if (action === "up") await post(`/api/jobs/${encodeURIComponent(id)}/move`, {delta: 1});
  if (action === "down") await post(`/api/jobs/${encodeURIComponent(id)}/move`, {delta: -1});
  if (action === "experiment" || action === "artifacts") await openExperiment(id);
});

$("experiment-close").addEventListener("click", () => {
  $("experiment-panel").classList.add("hidden");
});
$("experiment-bo").addEventListener("click", () => {
  if (!state.experimentId) return;
  switchTab("bo");
  $("bo-path").value = "";
  $("bo-job").value = state.experimentId;
  loadBo();
});

$("experiment-sync").addEventListener("click", async () => {
  const id = state.experimentId;
  if (!id) return;
  const result = await post(
    `/api/jobs/${encodeURIComponent(id)}/experiment/status`, {});
  if (result) await openExperiment(id);
});

$("experiment-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const path = $("experiment-path").value.trim();
  if (!path) return;
  const result = await post("/api/experiments", {path});
  if (result) {
    $("experiment-path").value = "";
    state.selected = result.id;
    $("log-title").textContent = `Log · ${result.id}`;
    await openExperiment(result.id);
  }
});

// ---------------------------------------------------------------------
// Abas

const TABS = ["queue", "algorithms", "analysis", "bo", "proposal"];

function switchTab(name) {
  if (!TABS.includes(name)) name = "queue";
  document.querySelectorAll(".tab").forEach(button => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  TABS.forEach(tab => {
    const panel = $(`tab-${tab}`);
    if (panel) panel.classList.toggle("hidden", tab !== name);
  });
  state.tab = name;
  try {
    const url = new URL(window.location.href);
    url.searchParams.set("tab", name);
    history.replaceState(null, "", url);
  } catch (error) { /* navegador sem History API */ }
  if (name === "analysis") populateAnalysisJobs();
  if (name === "bo") populateBoJobs();
  if (name === "algorithms") loadAlgorithms();
}

document.querySelectorAll(".tab").forEach(button => {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
});

// ---------------------------------------------------------------------
// Aba: algoritmos

function algorithmKind(row) {
  const parts = [row.on_policy ? "on-policy" : "off-policy"];
  parts.push(row.continuous ? "contínuo (Box)" : "discreto");
  if (row.multi_agent) parts.push("multiagente");
  return parts.join(" · ");
}

function renderAlgorithms() {
  const data = state.algorithms;
  if (!data) return;
  $("algorithms-guide").href = data.choosing_guide || "#";
  $("algorithms-list").innerHTML = (data.arenas || []).map(arena => `
    <div class="algo-arena">
      <div class="row" style="justify-content:space-between">
        <h3>${esc(arena.name)}</h3>
        <div class="row">
          <span class="pill">${esc(arena.action_space === "box" ? "Box" : "Discrete")}</span>
          <a class="muted" href="${esc(arena.readme)}" target="_blank" rel="noopener">README →</a>
        </div>
      </div>
      <p class="muted" style="margin:6px 0 10px">${esc(arena.description)}</p>
      <table>
        <thead><tr>
          <th>Algoritmo</th><th>Tipo</th><th>Documentação</th>
        </tr></thead>
        <tbody>${(arena.algorithms || []).map(row => `
          <tr>
            <td>
              <span class="algo-name">${esc(row.name)}</span>
              ${row.default ? '<span class="badge">padrão</span>' : ''}
              <div class="mono muted">${esc(row.package)}</div>
            </td>
            <td>${esc(algorithmKind(row))}</td>
            <td>
              <div class="doc-links">${(row.docs || []).map(doc =>
                `<a href="${esc(doc.url)}" target="_blank" rel="noopener">${esc(doc.label)}</a>`
              ).join("")}</div>
            </td>
          </tr>`).join("")}
        </tbody>
      </table>
    </div>`).join("");
}

async function loadAlgorithms() {
  if (state.algorithms) {
    renderAlgorithms();
    return;
  }
  try {
    const data = await api("/api/algorithms");
    state.algorithms = data;
    renderAlgorithms();
    showError("");
  } catch (error) {
    $("algorithms-list").textContent =
      `Falha ao carregar algoritmos: ${error.message}`;
  }
}

// ---------------------------------------------------------------------
// Aba: análise de resultados

const QUALITY = {
  primary: ["s-done", "análise primária (seed de treino)"],
  secondary: ["s-paused", "análise secundária (episódios)"],
  unspecified: ["s-unknown", "unidade independente não identificada"],
};

function populateAnalysisJobs() {
  const select = $("analysis-job");
  if (!select) return;
  const jobs = (state.jobs || []).filter(job =>
    job.experiment || job.experiment_yaml ||
    (job.artifacts && job.artifacts.length));
  const signature = jobs.map(job =>
    `${job.id}:${job.status}:${job.name}`).join("|");
  if (select.dataset.signature === signature) return;
  const current = select.value;
  select.dataset.signature = signature;
  const options = ['<option value="">— escolha um job —</option>'];
  jobs.forEach(job => {
    const exp = job.experiment && job.experiment.id
      ? ` · ${job.experiment.id}` : "";
    options.push(
      `<option value="${esc(job.id)}">${esc(job.name + exp)}</option>`);
  });
  select.innerHTML = options.join("");
  if (current && jobs.some(job => job.id === current)) select.value = current;
}

function applyAnalysisSource(source) {
  state.analysisSource = source;
  const metrics = source.metrics || [];
  $("analysis-metric").innerHTML = metrics.length
    ? metrics.map(metric => `<option value="${esc(metric)}"${metric === source.suggested_metric ? " selected" : ""}>${esc(metric)}</option>`).join("")
    : '<option value="">(sem métricas)</option>';
  const seeds = source.seed_fields || [];
  $("analysis-seed").innerHTML = '<option value="">(nenhuma)</option>' +
    seeds.map(field => `<option value="${esc(field)}"${field === source.suggested_seed_field ? " selected" : ""}>${esc(field)}</option>`).join("");
  const algorithms = source.algorithms || [];
  $("analysis-baseline").innerHTML = '<option value="">(comparar todos)</option>' +
    algorithms.map(alg => `<option value="${esc(alg)}">${esc(alg)}</option>`).join("");
}

async function loadAnalysisSources() {
  const jobId = $("analysis-job").value;
  const sourceSelect = $("analysis-source");
  state.analysisSources = [];
  if (!jobId) {
    sourceSelect.innerHTML = '<option value="">—</option>';
    $("analysis-artifacts").innerHTML = "";
    return;
  }
  try {
    const payload = await api(
      `/api/jobs/${encodeURIComponent(jobId)}/results`);
    state.analysisSources = payload.sources || [];
    const artifacts = payload.artifacts || [];
    $("analysis-artifacts").innerHTML = artifacts.length
      ? artifacts.map(artifactCard).join("")
      : '<span class="muted">Nenhum artefato registrado.</span>';
    if (!state.analysisSources.length) {
      sourceSelect.innerHTML =
        '<option value="">(nenhum CSV/JSONL encontrado)</option>';
      return;
    }
    sourceSelect.innerHTML = state.analysisSources.map(source =>
      `<option value="${esc(source.path)}">${esc(source.name)} · ${esc(source.rows)} linhas</option>`).join("");
    applyAnalysisSource(state.analysisSources[0]);
    showError("");
  } catch (error) {
    showError(`Falha ao listar resultados: ${error.message}`);
  }
}

function analysisInput() {
  return $("analysis-path").value.trim() || $("analysis-source").value;
}

function renderAnalysis(result) {
  const [cls, label] = QUALITY[result.analysis_quality] || QUALITY.unspecified;
  $("analysis-badges").innerHTML = `
    <span class="status ${cls}">${esc(label)}</span>
    <span class="pill"><b>Unidade:</b> ${esc(result.analysis_unit || "—")}</span>
    <span class="pill"><b>Métrica:</b> ${esc(result.metric)}</span>
    <span class="pill"><b>Comparação:</b> ${result.paired ? "pareada" : "independente"}</span>
    <span class="pill"><b>Agregação:</b> ${result.aggregate ? "por unidade" : "linha a linha"}</span>`;
  const warnings = result.warnings || [];
  $("analysis-warnings").innerHTML = warnings.length
    ? `<div class="banner show"><b>Avisos (${warnings.length})</b><ul>${warnings.map(warning => `<li>${esc(warning)}</li>`).join("")}</ul></div>`
    : "";
  state.analysisHtml = result.html || "";
  $("analysis-frame").srcdoc = state.analysisHtml;
  $("analysis-frame-wrap").classList.remove("hidden");
}

async function runAnalysis() {
  const input = analysisInput();
  if (!input) {
    showError("Escolha um experimento/arquivo de resultados ou informe um caminho.");
    return;
  }
  const body = {
    input,
    metric: $("analysis-metric").value || null,
    seed_field: $("analysis-seed").value || null,
    baseline: $("analysis-baseline").value || null,
    correction: $("analysis-correction").value,
    alpha: Number($("analysis-alpha").value || 0.05),
    confidence: Number($("analysis-confidence").value || 0.95),
    bootstrap: Number($("analysis-bootstrap").value || 10000),
    filter: $("analysis-filter").value.trim(),
    aggregate: $("analysis-aggregate").checked,
    paired: $("analysis-paired").checked,
  };
  try {
    const result = await api("/api/analysis", {
      method: "POST", body: JSON.stringify(body),
    });
    showError("");
    renderAnalysis(result);
  } catch (error) {
    showError(`Análise falhou: ${error.message}`);
  }
}

$("analysis-job").addEventListener("change", loadAnalysisSources);
$("analysis-source").addEventListener("change", (event) => {
  const source = (state.analysisSources || []).find(
    item => item.path === event.target.value);
  if (source) applyAnalysisSource(source);
});
$("analysis-run").addEventListener("click", runAnalysis);
$("analysis-open").addEventListener("click", () => {
  if (!state.analysisHtml) {
    showError("Gere uma análise antes de abrir o relatório.");
    return;
  }
  const blob = new Blob([state.analysisHtml],
                        {type: "text/html;charset=utf-8"});
  window.open(URL.createObjectURL(blob), "_blank", "noopener");
});

// ---------------------------------------------------------------------
// Aba: BO

const BO_CONFIG = {responsive: true, displaylogo: false,
                   modeBarButtonsToRemove: ["lasso2d", "select2d"]};
const boLayout = (extra = {}) => Object.assign({
  margin: {l: 56, r: 20, t: 40, b: 50},
  paper_bgcolor: "#fff", plot_bgcolor: "#fff",
  font: {size: 12, color: "#0f172a"},
  legend: {orientation: "h", y: -0.25},
}, extra);

const boNum = (value, digits = 2) => {
  if (value === null || value === undefined) return "—";
  const number = Number(value);
  return Number.isFinite(number)
    ? number.toLocaleString("pt-BR", {maximumFractionDigits: digits})
    : "—";
};

function boMessage(id, text) {
  const element = $(id);
  if (!element) return;
  if (window.Plotly) Plotly.purge(element);
  element.innerHTML = `<p class="muted">${esc(text)}</p>`;
}

function populateBoJobs() {
  const select = $("bo-job");
  if (!select) return;
  const jobs = (state.jobs || []).filter(job =>
    job.experiment || job.experiment_yaml ||
    (job.artifacts && job.artifacts.length));
  const signature = jobs.map(job =>
    `${job.id}:${job.status}:${job.name}`).join("|");
  if (select.dataset.signature === signature) return;
  const current = select.value;
  select.dataset.signature = signature;
  select.innerHTML = '<option value="">— escolha um job —</option>' +
    jobs.map(job => {
      const exp = job.experiment && job.experiment.id
        ? ` · ${job.experiment.id}` : "";
      return `<option value="${esc(job.id)}">${esc(job.name + exp)}</option>`;
    }).join("");
  if (current && jobs.some(job => job.id === current)) select.value = current;
}

function boCampaign() {
  const campaigns = (state.bo && state.bo.campaigns) || [];
  if (!campaigns.length) return null;
  return campaigns.find(campaign =>
    campaign.algorithm === state.boAlgorithm) || campaigns[0];
}

function boFilteredTrials(entry) {
  let trials = (entry.trials || []).slice();
  if (state.boStatus) {
    trials = trials.filter(trial => trial.status === state.boStatus);
  }
  const {key, dir} = state.boSort;
  const value = trial => {
    if (key === "params") return JSON.stringify(trial.params);
    return trial[key];
  };
  trials.sort((a, b) => {
    const va = value(a), vb = value(b);
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    if (typeof va === "number" && typeof vb === "number") {
      return (va - vb) * dir;
    }
    return String(va).localeCompare(String(vb), "pt-BR") * dir;
  });
  return trials;
}

function updateBoUrl() {
  if (!state.bo) return;
  try {
    const url = new URL(window.location.href);
    url.searchParams.set("tab", "bo");
    url.searchParams.set("bo", state.bo.root);
    if (state.boAlgorithm) url.searchParams.set("alg", state.boAlgorithm);
    else url.searchParams.delete("alg");
    if (state.boParam) url.searchParams.set("param", state.boParam);
    else url.searchParams.delete("param");
    if (state.boStatus) url.searchParams.set("status", state.boStatus);
    else url.searchParams.delete("status");
    history.replaceState(null, "", url);
  } catch (error) { /* navegador sem History API */ }
}

async function loadBo() {
  const pathValue = $("bo-path").value.trim();
  const jobId = $("bo-job").value;
  let payload;
  try {
    if (pathValue) {
      payload = await api(`/api/bo?path=${encodeURIComponent(pathValue)}`);
    } else if (jobId) {
      payload = await api(`/api/jobs/${encodeURIComponent(jobId)}/bo`);
    } else {
      showError("Escolha um job ou informe o caminho da campanha de BO.");
      return;
    }
  } catch (error) {
    state.bo = null;
    $("bo-summary").innerHTML = "";
    $("bo-meta").textContent = "";
    $("bo-table-meta").textContent = "";
    $("bo-warnings").innerHTML =
      `<div class="banner show"><b>BO:</b> ${esc(error.message)}</div>`;
    showError(`Falha ao carregar BO: ${error.message}`);
    return;
  }
  showError("");
  state.bo = payload;
  state.boAlgorithm = null;
  state.boParam = null;
  state.boStatus = "";
  $("bo-path").value = payload.root;
  const deep = new URLSearchParams(window.location.search);
  const alg = deep.get("alg");
  if (alg && (payload.campaigns || []).some(
          campaign => campaign.algorithm === alg)) {
    state.boAlgorithm = alg;
  }
  state.boParam = deep.get("param");
  state.boStatus = deep.get("status") || "";
  renderBo();
}

function renderBoWarnings(payload, entry) {
  const warnings = [...(payload.warnings || []), ...(entry.warnings || [])];
  $("bo-warnings").innerHTML = warnings.length
    ? `<div class="banner show"><b>Avisos (${warnings.length})</b><ul>${
        warnings.map(warning => `<li>${esc(warning)}</li>`).join("")
      }</ul></div>`
    : "";
}

function renderBoSummary(entry) {
  const improvement = entry.improvement || {};
  const pills = [
    ["trials", entry.n_trials],
    ["ok", entry.n_ok],
    ["falhos", entry.n_failed],
    ["melhor score", boNum(entry.best_score)],
    ["mediana", boNum(improvement.median_score)],
    ["primeiro", boNum(improvement.first_score)],
    ["fonte", entry.source],
  ];
  $("bo-summary").innerHTML = pills.map(([label, value]) =>
    `<span class="pill"><b>${esc(label)}:</b> ${esc(value)}</span>`).join("");
  const bestParams = improvement.best_params || {};
  const final = improvement.final;
  const parts = [
    `Diretório: ${entry.directory}`,
    `Fonte: ${entry.source}`,
  ];
  if (Object.keys(bestParams).length) {
    parts.push(`Melhores parâmetros: ${JSON.stringify(bestParams)}`);
  }
  if (final) {
    parts.push(`Validação final: score ${boNum(final.reward_mean)}${
      final.held_out === true ? " (seed held-out)" : " (seed de seleção)"}`);
  }
  $("bo-meta").textContent = parts.join(" · ");
}

function renderBoConvergence(selected) {
  const traces = [];
  (state.bo.campaigns || []).forEach(campaign => {
    const active = campaign === selected;
    traces.push({
      x: (campaign.best_so_far || []).map(point => point.index),
      y: (campaign.best_so_far || []).map(point => point.best),
      name: `${campaign.algorithm} · best-so-far`,
      type: "scatter", mode: "lines",
      line: {width: active ? 3 : 1.5, dash: active ? "solid" : "dot"},
      opacity: active ? 1 : .5,
      hovertemplate: "trial %{x}<br>melhor %{y:.2f}<extra></extra>",
    });
    traces.push({
      x: campaign.trials.map(trial => trial.index),
      y: campaign.trials.map(trial => trial.score),
      text: campaign.trials.map(trial => trial.status),
      name: `${campaign.algorithm} · score`,
      type: "scatter", mode: "markers",
      marker: {size: active ? 7 : 5, opacity: active ? .8 : .4},
      hovertemplate: "trial %{x}<br>score %{y:.2f}<br>%{text}<extra></extra>",
    });
    const best = campaign.trials.find(trial => trial.is_best);
    if (best) {
      traces.push({
        x: [best.index], y: [best.score],
        name: `${campaign.algorithm} · melhor`,
        type: "scatter", mode: "markers",
        marker: {symbol: "star", size: 15, color: "#f59e0b"},
        hovertemplate: "melhor trial %{x}: %{y:.2f}<extra></extra>",
      });
    }
    const finals = campaign.trials.filter(trial =>
      trial.kind === "final" && trial.score !== null);
    if (finals.length) {
      traces.push({
        x: finals.map(trial => trial.index),
        y: finals.map(trial => trial.score),
        name: `${campaign.algorithm} · final`,
        type: "scatter", mode: "markers",
        marker: {symbol: "diamond", size: 12, color: "#10b981"},
        hovertemplate: "validação final: %{y:.2f}<extra></extra>",
      });
    }
  });
  if (!traces.length) return boMessage("bo-convergence", "Sem trials.");
  Plotly.newPlot("bo-convergence", traces, boLayout({
    xaxis: {title: "trial"}, yaxis: {title: "score"},
  }), BO_CONFIG);
}

function renderBoEffect(entry) {
  const params = entry.params || [];
  const param = params.find(item => item.name === state.boParam) || params[0];
  if (!param) {
    return boMessage("bo-effect",
      "Nenhum parâmetro com trials válidos suficientes.");
  }
  if (param.kind === "numeric") {
    const points = param.points || [];
    const bins = param.bins || [];
    const traces = [{
      x: points.map(point => point.x),
      y: points.map(point => point.score),
      name: "trials",
      type: "scatter", mode: "markers",
      marker: {size: 6, opacity: .5},
      hovertemplate: "%{x}<br>score %{y:.2f}<extra></extra>",
    }, {
      x: bins.map(bin_ => (bin_.lo + bin_.hi) / 2),
      y: bins.map(bin_ => bin_.mean),
      text: bins.map(bin_ => `n=${bin_.n}`),
      name: "média por faixa",
      type: "scatter", mode: "lines+markers",
      line: {width: 2, color: "#2563eb"},
      marker: {size: 8, color: "#2563eb"},
      error_y: {
        type: "data", symmetric: false,
        array: bins.map(bin_ => bin_.ci95 ? bin_.ci95[1] - bin_.mean : 0),
        arrayminus: bins.map(bin_ => bin_.ci95 ? bin_.mean - bin_.ci95[0] : 0),
      },
      hovertemplate: "%{text}<br>média %{y:.2f}<extra></extra>",
    }];
    Plotly.newPlot("bo-effect", traces, boLayout({
      xaxis: {title: param.name}, yaxis: {title: "score"},
      title: {text: `Spearman ρ = ${boNum(param.spearman)} · n = ${param.n}`,
              font: {size: 12}},
    }), BO_CONFIG);
  } else {
    const traces = (param.groups || []).map(group => ({
      type: "box", name: String(group.value), y: group.scores,
      boxpoints: "all", jitter: .35, pointpos: 0,
      marker: {size: 4, opacity: .55},
      line: {width: 1},
      hovertemplate: `${group.value}<br>n=${group.n}<extra></extra>`,
    }));
    Plotly.newPlot("bo-effect", traces, boLayout({
      xaxis: {title: param.name}, yaxis: {title: "score"},
      showlegend: false,
      title: {text: `η² = ${boNum(param.eta_squared)} · n = ${param.n}`,
              font: {size: 12}},
    }), BO_CONFIG);
  }
}

function renderBoImportance(entry) {
  const ranked = (entry.params || [])
    .filter(param => param.importance !== null &&
                     param.importance !== undefined)
    .sort((a, b) => a.importance - b.importance);
  if (!ranked.length) {
    return boMessage("bo-importance", "Sem estimativas de importância.");
  }
  Plotly.newPlot("bo-importance", [{
    type: "bar", orientation: "h",
    x: ranked.map(param => param.importance),
    y: ranked.map(param => param.name),
    text: ranked.map(param =>
      param.kind === "numeric" ? "|ρ|" : "η²"),
    textposition: "outside",
    marker: {color: ranked.map(param =>
      param.kind === "numeric" ? "#2563eb" : "#10b981")},
    hovertemplate: "%{y}<br>%{x:.3f}<extra></extra>",
  }], boLayout({
    margin: {l: 110, r: 30, t: 20, b: 45},
    xaxis: {title: "|Spearman| (numérico) ou η² (categórico)"},
    showlegend: false,
  }), BO_CONFIG);
}

function renderBoImprovement(entry) {
  const series = (entry.best_so_far || [])
    .filter(point => point.best !== null);
  const improvement = entry.improvement || {};
  if (!series.length) {
    return boMessage("bo-improvement", "Sem trials válidos.");
  }
  const traces = [{
    x: series.map(point => point.index),
    y: series.map(point => point.best),
    name: "best-so-far", type: "scatter", mode: "lines",
    line: {width: 2.5, color: "#2563eb"},
    fill: "tozeroy", fillcolor: "rgba(37,99,235,.08)",
    hovertemplate: "trial %{x}<br>melhor %{y:.2f}<extra></extra>",
  }];
  const shapes = [];
  if (improvement.median_score !== null &&
      improvement.median_score !== undefined) {
    shapes.push({
      type: "line", xref: "paper", x0: 0, x1: 1,
      y0: improvement.median_score, y1: improvement.median_score,
      line: {dash: "dash", color: "#94a3b8", width: 1.5},
    });
    traces.push({
      x: [series[0].index, series[series.length - 1].index],
      y: [improvement.median_score, improvement.median_score],
      name: "mediana dos trials", type: "scatter", mode: "lines",
      line: {dash: "dash", color: "#94a3b8"},
      hoverinfo: "skip",
    });
  }
  const finals = entry.trials.filter(trial =>
    trial.kind === "final" && trial.score !== null);
  if (finals.length) {
    traces.push({
      x: finals.map(trial => trial.index),
      y: finals.map(trial => trial.score),
      name: "validação final", type: "scatter", mode: "markers",
      marker: {symbol: "diamond", size: 12, color: "#10b981"},
      hovertemplate: "final: %{y:.2f}<extra></extra>",
    });
  }
  Plotly.newPlot("bo-improvement", traces, boLayout({
    xaxis: {title: "trial"}, yaxis: {title: "score"},
    shapes,
    title: {
      text: `melhor − mediana: ${boNum(improvement.delta_vs_median)} · ` +
            `melhor − primeiro: ${boNum(improvement.delta_vs_first)}`,
      font: {size: 12},
    },
  }), BO_CONFIG);
}

function renderBoInteraction(entry) {
  const interaction = entry.interaction;
  if (!interaction) {
    return boMessage("bo-interaction",
      "Não há dois parâmetros numéricos com trials suficientes.");
  }
  Plotly.newPlot("bo-interaction", [{
    type: "heatmap",
    z: interaction.means,
    x: interaction.x_labels,
    y: interaction.y_labels,
    text: interaction.counts.map(row => row.map(n => `n=${n}`)),
    hovertemplate: `${interaction.y_param} %{y}<br>` +
      `${interaction.x_param} %{x}<br>média %{z:.2f}<br>%{text}<extra></extra>`,
    colorscale: "Viridis",
  }], boLayout({
    xaxis: {title: interaction.x_param},
    yaxis: {title: interaction.y_param},
    margin: {l: 70, r: 20, t: 30, b: 50},
  }), BO_CONFIG);
}

function renderBoParallel(entry) {
  const numeric = (entry.params || [])
    .filter(param => param.kind === "numeric").slice(0, 6);
  const rows = (entry.trials || []).filter(trial =>
    trial.ok && trial.score !== null);
  if (!numeric.length || rows.length < 2) {
    return boMessage("bo-parallel",
      "Não há parâmetros numéricos suficientes para o parallel coordinates.");
  }
  const dimensions = numeric.map(param => ({
    label: param.name,
    values: rows.map(trial => Number(trial.params[param.name])),
  }));
  dimensions.push({label: "score", values: rows.map(trial => trial.score)});
  Plotly.newPlot("bo-parallel", [{
    type: "parcoords",
    line: {
      color: rows.map(trial => trial.score),
      colorscale: "Viridis", showscale: true,
      colorbar: {title: "score"},
    },
    dimensions,
    labelfont: {size: 11},
  }], boLayout({margin: {l: 70, r: 30, t: 30, b: 30}}), BO_CONFIG);
}

function boStatusClass(status) {
  if (status === "ok") return "s-done";
  if (status === "erro" || status === "failed") return "s-failed";
  return "s-unknown";
}

function renderBoTable(entry) {
  const trials = boFilteredTrials(entry);
  const shown = trials.slice(0, 500);
  const arrow = key => state.boSort.key === key
    ? (state.boSort.dir > 0 ? " ▲" : " ▼") : "";
  const columns = [
    ["index", "#"], ["status", "status"], ["score", "score"],
    ["reward", "reward"], ["coins", "moedas"], ["deaths", "mortes"],
    ["duration_seconds", "duração"], ["params", "parâmetros"],
    ["artifact", "artefato"],
  ];
  const head = columns.map(([key, label]) =>
    `<th class="sortable" data-bo-sort="${key}">${esc(label)}${arrow(key)}</th>`
  ).join("") + "<th>erro</th>";
  const rows = shown.map(trial => {
    const params = JSON.stringify(trial.params || {});
    const shortParams = params.length > 90
      ? params.slice(0, 87) + "…" : params;
    const link = trial.log
      ? `<a href="/api/files?path=${encodeURIComponent(trial.log)}" ` +
        `target="_blank" rel="noopener">log</a>`
      : (trial.artifact
          ? `<span class="mono" title="${esc(trial.artifact)}">pasta</span>`
          : "—");
    return `<tr>
      <td class="mono">${esc(trial.index)}${
        trial.is_best ? ' <span class="badge">best</span>' : ''}${
        trial.kind === "final" ? ' <span class="badge">final</span>' : ''}</td>
      <td><span class="status ${boStatusClass(trial.status)}">${
        esc(trial.status)}</span></td>
      <td>${boNum(trial.score)}</td>
      <td>${boNum(trial.reward)}</td>
      <td>${boNum(trial.coins, 1)}</td>
      <td>${boNum(trial.deaths, 1)}</td>
      <td>${fmtSeconds(trial.duration_seconds)}</td>
      <td><span class="mono" title="${esc(params)}">${esc(shortParams)}</span></td>
      <td>${link}</td>
      <td class="muted">${esc(trial.error || "")}</td>
    </tr>`;
  }).join("");
  $("bo-trials").innerHTML = `<thead><tr>${head}</tr></thead>` +
    `<tbody>${rows || '<tr><td colspan="10" class="muted">Nenhum trial no filtro.</td></tr>'}</tbody>`;
  $("bo-table-meta").textContent =
    `${trials.length} trial(s) no filtro · exibindo ${shown.length}` +
    (trials.length > shown.length ? " (limite de 500 linhas)" : "") +
    " · filtros e parâmetro ficam na URL";
  document.querySelectorAll("#bo-trials th[data-bo-sort]").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.boSort;
      state.boSort = {key,
                      dir: state.boSort.key === key ? -state.boSort.dir : 1};
      renderBoTable(entry);
    });
  });
}

function renderBo() {
  const payload = state.bo;
  if (!payload) return;
  const campaigns = payload.campaigns || [];
  if (!campaigns.length) {
    $("bo-summary").innerHTML = "";
    $("bo-meta").textContent = "";
    $("bo-table-meta").textContent = "";
    boMessage("bo-convergence", "Nenhuma campanha encontrada.");
    return;
  }
  if (!campaigns.some(campaign =>
          campaign.algorithm === state.boAlgorithm)) {
    state.boAlgorithm = campaigns[0].algorithm;
  }
  const entry = boCampaign();
  $("bo-alg").innerHTML = campaigns.map(campaign =>
    `<option value="${esc(campaign.algorithm)}">${
      esc(campaign.algorithm)} · ${esc(campaign.source)}</option>`).join("");
  $("bo-alg").value = entry.algorithm;

  const statuses = [...new Set((entry.trials || [])
    .map(trial => trial.status))].sort();
  $("bo-status").innerHTML = '<option value="">todos</option>' +
    statuses.map(status =>
      `<option value="${esc(status)}">${esc(status)}</option>`).join("");
  if (!statuses.includes(state.boStatus)) state.boStatus = "";
  $("bo-status").value = state.boStatus;

  const params = entry.params || [];
  $("bo-param").innerHTML = params.map(param =>
    `<option value="${esc(param.name)}">${esc(param.name)} · ${
      esc(param.kind)}</option>`).join("");
  if (!params.some(param => param.name === state.boParam)) {
    state.boParam = params.length ? params[0].name : "";
  }
  $("bo-param").value = state.boParam;

  renderBoWarnings(payload, entry);
  renderBoSummary(entry);
  renderBoConvergence(entry);
  renderBoEffect(entry);
  renderBoImportance(entry);
  renderBoImprovement(entry);
  renderBoInteraction(entry);
  renderBoParallel(entry);
  renderBoTable(entry);
  updateBoUrl();
}

function boDownload(filename, text, type) {
  const blob = new Blob([text], {type});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

function boExport(kind) {
  const entry = boCampaign();
  if (!entry) {
    showError("Carregue uma campanha antes de exportar.");
    return;
  }
  const trials = boFilteredTrials(entry);
  if (kind === "json") {
    boDownload(`bo-${entry.algorithm}-trials.json`,
      JSON.stringify({campaign: state.bo.root, algorithm: entry.algorithm,
                      source: entry.source, trials}, null, 2),
      "application/json");
    return;
  }
  const quote = value => `"${String(value ?? "").replace(/"/g, '""')}"`;
  const head = ["index", "kind", "status", "score", "reward", "coins",
                "deaths", "duration_seconds", "params"];
  const lines = [head.join(",")];
  trials.forEach(trial => {
    lines.push([trial.index, trial.kind, trial.status, trial.score,
                trial.reward, trial.coins, trial.deaths,
                trial.duration_seconds, JSON.stringify(trial.params)]
      .map(quote).join(","));
  });
  boDownload(`bo-${entry.algorithm}-trials.csv`, lines.join("\n"),
             "text/csv;charset=utf-8");
}

$("bo-load").addEventListener("click", loadBo);
$("bo-job").addEventListener("change", () => {
  $("bo-path").value = "";
  loadBo();
});
$("bo-alg").addEventListener("change", (event) => {
  state.boAlgorithm = event.target.value;
  state.boParam = null;
  renderBo();
});
$("bo-status").addEventListener("change", (event) => {
  state.boStatus = event.target.value;
  renderBoTable(boCampaign());
  updateBoUrl();
});
$("bo-param").addEventListener("change", (event) => {
  state.boParam = event.target.value;
  renderBoEffect(boCampaign());
  updateBoUrl();
});
$("bo-export-csv").addEventListener("click", () => boExport("csv"));
$("bo-export-json").addEventListener("click", () => boExport("json"));

// ---------------------------------------------------------------------
// Aba: proposta de experimentos

function proposalFields() {
  const fields = {};
  document.querySelectorAll("[data-proposal]").forEach(element => {
    fields[element.dataset.proposal] = element.type === "checkbox"
      ? element.checked : element.value;
  });
  return fields;
}

function renderProposalErrors(errors) {
  const list = errors || [];
  $("proposal-errors").innerHTML = list.length
    ? `<div class="banner show"><b>Proposta inválida (${list.length})</b><ul>${list.map(error => `<li>${esc(error)}</li>`).join("")}</ul></div>`
    : "";
}

async function previewProposal() {
  try {
    const data = await api("/api/proposals/preview", {
      method: "POST", body: JSON.stringify({fields: proposalFields()}),
    });
    $("proposal-yaml").textContent = data.yaml;
    $("proposal-md").textContent = data.markdown;
    renderProposalErrors(data.errors);
    $("proposal-result").textContent = data.errors.length
      ? "" : `Prévia pronta · id ${data.proposal.id}`;
    showError("");
    return data;
  } catch (error) {
    showError(error.message);
    return null;
  }
}

async function saveProposal(enqueue) {
  const path = enqueue ? "/api/proposals/enqueue" : "/api/proposals/save";
  try {
    const data = await api(path, {
      method: "POST",
      body: JSON.stringify({
        fields: proposalFields(),
        overwrite: $("proposal-overwrite").checked,
      }),
    });
    showError("");
    renderProposalErrors([]);
    if (data.yaml) $("proposal-yaml").textContent = data.yaml;
    if (data.markdown) $("proposal-md").textContent = data.markdown;
    if (enqueue) {
      $("proposal-result").textContent =
        `Salvo e enfileirado: ${data.job.id}`;
      await refresh();
      switchTab("queue");
      state.selected = data.job.id;
      await openExperiment(data.job.id);
    } else {
      $("proposal-result").textContent =
        `Salvo: ${data.yaml_path} · ${data.markdown_path}`;
    }
  } catch (error) {
    showError(error.message);
    renderProposalErrors([error.message]);
  }
}

$("proposal-preview").addEventListener("click", previewProposal);
$("proposal-save").addEventListener("click", () => saveProposal(false));
$("proposal-enqueue").addEventListener("click", () => saveProposal(true));
$("proposal-form").addEventListener("submit", (event) => {
  event.preventDefault();
  previewProposal();
});

// ---------------------------------------------------------------------
// Inicialização e deep link (?tab=proposal&job=<id>)

refresh();
setInterval(refresh, 2000);
setInterval(refreshLog, 2000);

const params = new URLSearchParams(window.location.search);
const deepTab = params.get("tab");
const deepJob = params.get("job");
const deepBo = params.get("bo");
const deepBoJob = params.get("bojob");
if (deepTab) switchTab(deepTab);
if (deepJob) {
  switchTab("queue");
  setTimeout(() => openExperiment(deepJob), 900);
}
if (deepBo || deepBoJob) {
  setTimeout(() => {
    if (deepBo) $("bo-path").value = deepBo;
    if (deepBoJob) $("bo-job").value = deepBoJob;
    switchTab("bo");
    loadBo();
  }, 900);
}
</script>
</body>
</html>
"""
