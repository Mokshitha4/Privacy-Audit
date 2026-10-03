"use strict";

/* ------------------------------------------------------------------------
 * Theme
 * ------------------------------------------------------------------------ */
const THEME_KEY = "privaudit-theme";

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch (e) { /* private window etc. */ }
  applyTheme(saved === "dark" ? "dark" : "light");
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
}

/* ------------------------------------------------------------------------
 * Small DOM helpers
 * ------------------------------------------------------------------------ */
function byId(id) { return document.getElementById(id); }
function val(id) { const el = byId(id); return el ? el.value : ""; }
function checked(id) { const el = byId(id); return el ? el.checked : false; }

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* ------------------------------------------------------------------------
 * Job-config assembly -- mirrors privaudit/ui.py's _assemble_config() so the
 * nested JSON this page sends matches exactly what the CLI/validator expect.
 * ------------------------------------------------------------------------ */
function numOrNull(raw) {
  if (raw === null || raw === undefined) return null;
  const text = String(raw).trim();
  if (text === "") return null;
  const n = Number(text);
  if (Number.isNaN(n)) throw new Error(`expected a number, got ${JSON.stringify(raw)}`);
  return n;
}

function parseIntList(text) {
  if (text === null || text === undefined || !String(text).trim()) return null;
  const parts = String(text).replace(/,/g, " ").split(/\s+/).filter(Boolean);
  return parts.map((part) => {
    if (!/^-?\d+$/.test(part)) {
      throw new Error(`ngram_ns must be a comma- or space-separated list of integers, got ${JSON.stringify(text)}`);
    }
    return parseInt(part, 10);
  });
}

function modelBlock(prefix) {
  const source = val(`${prefix}_source`);
  const identifier = val(`${prefix}_identifier`).trim();
  const access = val(`${prefix}_access`);
  const block = { source, identifier, access };
  const baseModel = val(`${prefix}_base_model`).trim();
  if (baseModel) block.base_model = baseModel;
  if (source === "openai") {
    const apiKey = val(`${prefix}_api_key`).trim();
    if (apiKey) block.api_key = apiKey;
    const baseUrl = val(`${prefix}_base_url`).trim();
    if (baseUrl) block.base_url = baseUrl;
  }
  return block;
}

function attackParams(prefix, keys) {
  const params = {};
  for (const key of keys) {
    const n = numOrNull(val(`${prefix}_${key}`));
    if (n !== null) params[key] = n;
  }
  return params;
}

function assembleConfig() {
  const model = modelBlock("model");

  if (checked("ft_enabled")) {
    const finetuning = {};
    const regime = val("ft_regime");
    if (regime) finetuning.regime = regime;
    const loss = val("ft_loss");
    if (loss) finetuning.loss = loss;
    const epsilon = numOrNull(val("ft_epsilon"));
    if (epsilon !== null) finetuning.epsilon = epsilon;
    if (Object.keys(finetuning).length) model.finetuning = finetuning;
  }

  const config = { model };

  if (checked("ezmia_enabled")) {
    config.reference_model = modelBlock("ref");
  }

  const schema = {
    format: val("schema_format"),
    member_file: val("schema_member_file").trim(),
    nonmember_file: val("schema_nonmember_file").trim(),
  };
  const templateRaw = val("schema_text_template");
  const fieldRaw = val("schema_text_field");
  if (templateRaw.trim()) {
    schema.text_template = templateRaw;
  } else if (fieldRaw.trim()) {
    schema.text_field = fieldRaw.trim();
  }
  config.data = { role: val("data_role"), path: val("data_path").trim(), schema };

  const attacks = [];
  if (checked("em_enabled")) {
    const params = attackParams("em", ["prefix_len", "continuation_len", "max_samples"]);
    const ngramNs = parseIntList(val("em_ngram_ns"));
    if (ngramNs && ngramNs.length) params.ngram_ns = ngramNs;
    attacks.push({ family: "EM", variant: "default", params });
  }
  if (checked("mia_enabled")) {
    const params = attackParams("mia", [
      "num_members", "num_nonmembers", "max_length", "k_percent", "n_folds", "batch_size", "seed",
    ]);
    attacks.push({ family: "MIA", variant: "default", params });
  }
  if (checked("ezmia_enabled")) {
    const params = attackParams("ezmia", [
      "num_members", "num_nonmembers", "sequence_length", "batch_size", "seed",
    ]);
    attacks.push({ family: "EZ_MIA", variant: "default", params });
  }
  config.attacks = attacks;
  config.output = { return_raw_generations: checked("return_raw_generations") };
  return config;
}

function buildLlmConfig() {
  return {
    enabled: checked("llm_enabled"),
    provider: val("llm_provider"),
    api_key: val("llm_api_key").trim() || null,
    model: val("llm_model").trim() || null,
    base_url: val("llm_base_url").trim() || null,
  };
}

/* ------------------------------------------------------------------------
 * Conditional field visibility
 * ------------------------------------------------------------------------ */
function evalConditions() {
  document.querySelectorAll(".cond").forEach((el) => {
    const key = el.dataset.show;
    let visible;
    if (el.dataset.showValues) {
      const groupEl = document.querySelector(`[data-group="${key}"]`);
      const values = el.dataset.showValues.split(",");
      visible = groupEl ? values.includes(groupEl.value) : false;
    } else {
      visible = checked(key);
    }
    el.classList.toggle("show", visible);
  });
}

/* ------------------------------------------------------------------------
 * JSON pretty-printing with lightweight syntax highlighting
 * ------------------------------------------------------------------------ */
function highlightJson(value) {
  const json = escapeHtml(JSON.stringify(value, null, 2));
  const pattern = /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false)\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
  return json.replace(pattern, (match) => {
    let cls = "tok-num";
    if (/^"/.test(match)) {
      cls = /:$/.test(match) ? "tok-key" : "tok-str";
    } else if (/^(true|false)$/.test(match)) {
      cls = "tok-bool";
    } else if (match === "null") {
      cls = "tok-null";
    }
    return `<span class="${cls}">${match}</span>`;
  });
}

/* ------------------------------------------------------------------------
 * Form message + technical-details panels
 * ------------------------------------------------------------------------ */
function setFormMessage(text, kind) {
  const el = byId("form-message");
  el.textContent = text || "";
  el.className = "form-message" + (kind ? ` ${kind}` : "");
}

function openTechDetails() {
  const details = document.querySelector("details.card:has(#config-json)");
  if (details && !details.open) details.open = true;
}

function switchTechTab(name) {
  document.querySelectorAll(".tech-tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tech-panel").forEach((p) => p.classList.toggle("active", p.dataset.panel === name));
}

function renderConfigPreview(config) {
  byId("config-json").innerHTML = highlightJson(config);
}

/* ------------------------------------------------------------------------
 * Status log
 * ------------------------------------------------------------------------ */
function clearLog() {
  byId("status-log").innerHTML = "";
}

function logLine(text, cls) {
  const pre = byId("status-log");
  const time = new Date().toLocaleTimeString([], { hour12: false });
  const div = document.createElement("div");
  div.className = "line" + (cls ? ` ${cls}` : "");
  div.innerHTML = `<span class="ts">${time}</span>${escapeHtml(text)}`;
  pre.appendChild(div);
  pre.scrollTop = pre.scrollHeight;
}

function setRunIndicator(state, label) {
  const el = byId("run-indicator");
  el.className = "run-indicator" + (state !== "idle" ? ` ${state}` : "");
  el.innerHTML = `<span class="pulse"></span>${escapeHtml(label)}`;
}

function setRunning(isRunning) {
  ["btn-preview", "btn-validate", "btn-run"].forEach((id) => { byId(id).disabled = isRunning; });
  const label = byId("btn-run").querySelector(".btn-label");
  const icon = byId("btn-run").querySelector(".btn-icon");
  label.textContent = isRunning ? "Running..." : "Run audit";
  icon.innerHTML = isRunning
    ? '<span class="spinner"></span>'
    : '<svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M8 5v14l11-7z"/></svg>';
}

/* ------------------------------------------------------------------------
 * Results rendering
 * ------------------------------------------------------------------------ */
function showPlaceholder(show) {
  byId("results-placeholder").classList.toggle("hidden", !show);
  byId("results-content").classList.toggle("hidden", show);
}

function renderResults(msg) {
  const container = byId("results-content");
  container.innerHTML = "";
  const summary = msg.summary || { families: [], agreement_note: null, disclaimer: "" };

  summary.families.forEach((fam) => {
    const card = document.createElement("div");
    card.className = `verdict-card v-${fam.verdict}`;
    card.innerHTML = `
      <div class="verdict-top">
        <span class="verdict-name">${fam.badge} ${escapeHtml(fam.name)}</span>
        <span class="verdict-pill v-${fam.verdict}">${escapeHtml(fam.verdict_label)}</span>
      </div>
      <p class="verdict-what">${escapeHtml(fam.what_it_checks)}</p>
      <p class="verdict-headline">${escapeHtml(fam.headline)}</p>
      <ul class="verdict-detail">${fam.detail.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}</ul>
    `;
    container.appendChild(card);
  });

  if (summary.agreement_note) {
    const note = document.createElement("div");
    note.className = "agreement-note";
    note.innerHTML = `<strong>Reading these together:</strong> ${escapeHtml(summary.agreement_note)}`;
    container.appendChild(note);
  }

  if (msg.narration) {
    const card = document.createElement("div");
    card.className = "narration-card";
    card.innerHTML = `<div class="narration-title">AI narration</div><div>${escapeHtml(msg.narration).replace(/\n/g, "<br>")}</div>`;
    container.appendChild(card);
  } else if (msg.narration_error) {
    const warn = document.createElement("div");
    warn.className = "narration-warn";
    warn.textContent = `AI narration failed (${msg.narration_error}). Showing the rule-based summary only.`;
    container.appendChild(warn);
  }

  if (summary.disclaimer) {
    const disclaimer = document.createElement("p");
    disclaimer.className = "disclaimer";
    disclaimer.textContent = summary.disclaimer;
    container.appendChild(disclaimer);
  }

  showPlaceholder(false);
  byId("report-json").innerHTML = highlightJson(msg.report);
}

function renderError(msg) {
  const container = byId("results-content");
  container.innerHTML = `
    <div class="error-card">
      <div class="error-title">Run failed</div>
      <div>${escapeHtml(msg.message || "Unknown error.")}</div>
      ${msg.traceback ? `<pre>${escapeHtml(msg.traceback)}</pre>` : ""}
    </div>
  `;
  showPlaceholder(false);
}

/* ------------------------------------------------------------------------
 * Actions: preview / validate / run
 * ------------------------------------------------------------------------ */
function handlePreview() {
  try {
    const config = assembleConfig();
    renderConfigPreview(config);
    switchTechTab("config");
    openTechDetails();
    setFormMessage("Config preview generated below, in Technical details.", "ok");
  } catch (e) {
    setFormMessage(e.message, "err");
  }
}

async function handleValidate() {
  let config;
  try {
    config = assembleConfig();
  } catch (e) {
    setFormMessage(e.message, "err");
    return;
  }
  renderConfigPreview(config);
  setFormMessage("Validating...", "");
  try {
    const res = await fetch("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    });
    const data = await res.json();
    setFormMessage(data.valid ? "Config is valid." : data.error, data.valid ? "ok" : "err");
  } catch (e) {
    setFormMessage(`Could not reach the local server: ${e.message}`, "err");
  }
}

function handleRun() {
  let config;
  try {
    config = assembleConfig();
  } catch (e) {
    setFormMessage(e.message, "err");
    return;
  }
  const llm = buildLlmConfig();

  setFormMessage("", "");
  clearLog();
  logLine("Starting...", "muted");
  renderConfigPreview(config);
  showPlaceholder(true);
  setRunIndicator("running", "running");
  setRunning(true);

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/run`);
  let finished = false;

  ws.onopen = () => ws.send(JSON.stringify({ config, llm }));

  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }

    if (msg.type === "progress") {
      logLine(msg.message);
    } else if (msg.type === "error") {
      finished = true;
      logLine(`Failed: ${msg.message}`, "err");
      setRunIndicator("failed", "failed");
      renderError(msg);
      setRunning(false);
      ws.close();
    } else if (msg.type === "result") {
      finished = true;
      logLine("Done.");
      setRunIndicator("done", "done");
      renderResults(msg);
      setRunning(false);
      ws.close();
    }
  };

  ws.onerror = () => {
    if (finished) return;
    logLine("Lost connection to the local server.", "err");
  };

  ws.onclose = () => {
    if (finished) return;
    finished = true;
    logLine("Connection closed unexpectedly before the run finished.", "err");
    setRunIndicator("failed", "failed");
    setRunning(false);
  };
}

/* ------------------------------------------------------------------------
 * Wire-up
 * ------------------------------------------------------------------------ */
function initTechPanels() {
  document.querySelectorAll(".tech-tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTechTab(tab.dataset.tab));
  });
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const pre = byId(`${btn.dataset.copy}-json`);
      navigator.clipboard.writeText(pre.textContent).then(() => {
        const original = btn.textContent;
        btn.classList.add("copied");
        btn.textContent = "Copied";
        setTimeout(() => { btn.classList.remove("copied"); btn.textContent = original; }, 1400);
      }).catch(() => { /* clipboard unavailable; ignore */ });
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  byId("theme-toggle").addEventListener("click", toggleTheme);

  byId("audit-form").addEventListener("change", evalConditions);
  evalConditions();

  byId("btn-preview").addEventListener("click", handlePreview);
  byId("btn-validate").addEventListener("click", handleValidate);
  byId("btn-run").addEventListener("click", handleRun);

  initTechPanels();
});
