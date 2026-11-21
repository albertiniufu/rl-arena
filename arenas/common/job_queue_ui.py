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
</style>
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
              algorithms: null };
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
  const readOnly = Boolean(state.read_only);
  $("toggle").textContent = readOnly
    ? "Somente leitura" : (state.paused ? "Retomar fila" : "Pausar fila");
  $("toggle").classList.toggle("primary", !readOnly && !state.paused);
  $("toggle").disabled = readOnly;
  $("toggle").title = readOnly
    ? "Esta instância só acompanha a fila; use a porta principal para alterar"
    : "";
  document.querySelectorAll(
    '#add-form button[type="submit"], #experiment-form button[type="submit"]'
  ).forEach(button => { button.disabled = readOnly; });
  const running = jobs.filter(j => j.status === "running").length;
  const queued = jobs.filter(j => j.status === "queued").length;
  $("status-pills").innerHTML = `
    <span class="pill"><b>Fila:</b> ${state.paused ? "pausada" : "ativa"}</span>
    <span class="pill"><b>Rodando:</b> ${running}/${state.max_parallel}</span>
    <span class="pill"><b>Na fila:</b> ${queued}</span>
    <span class="pill"><b>Total:</b> ${jobs.length}</span>
    ${readOnly ? '<span class="pill"><b>Modo:</b> somente leitura</span>' : ''}`;
  $("counts").textContent = `${jobs.length} jobs · paralelismo ${state.max_parallel}`;
  const ro = readOnly ? "disabled" : "";
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
          <button data-action="up" data-id="${esc(job.id)}" title="Subir prioridade" ${ro}>↑</button>
          <button data-action="down" data-id="${esc(job.id)}" title="Descer prioridade" ${ro}>↓</button>
          <button data-action="cancel" data-id="${esc(job.id)}" class="danger" ${ro}>Cancelar</button>
          <button data-action="retry" data-id="${esc(job.id)}" ${ro}>Retry</button>
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

const TABS = ["queue", "algorithms", "analysis", "proposal"];

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
if (deepTab) switchTab(deepTab);
if (deepJob) {
  switchTab("queue");
  setTimeout(() => openExperiment(deepJob), 900);
}
</script>
</body>
</html>
"""
