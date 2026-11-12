const state = {
  summaries: {},
  dataset: new URLSearchParams(location.search).get("dataset") || "cic",
  index: Number(new URLSearchParams(location.search).get("sample") || 42),
  timer: null,
  sample: null,
};

const byId = (id) => document.getElementById(id);
const percent = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const fixed = (value, digits = 3) => Number(value).toFixed(digits);
const bytes = (value) => `${(value / (1024 * 1024)).toFixed(2)} MB`;
const compact = (value) => new Intl.NumberFormat("en", { maximumFractionDigits: 0 }).format(value);

function text(id, value) {
  byId(id).textContent = value;
}

function setupCanvas(id) {
  const canvas = byId(id);
  const ratio = Math.max(2, window.devicePixelRatio || 1);
  const width = Math.max(320, canvas.clientWidth);
  const height = Math.max(180, canvas.clientHeight);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.font = "11px Inter, system-ui, sans-serif";
  context.lineWidth = 1;
  return { context, width, height };
}

function drawAxes(context, width, height, bounds, xLabels, yLabels) {
  context.strokeStyle = "#cdd4d5";
  context.fillStyle = "#657176";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(bounds.left, bounds.top);
  context.lineTo(bounds.left, height - bounds.bottom);
  context.lineTo(width - bounds.right, height - bounds.bottom);
  context.stroke();
  xLabels.forEach(({ x, label }) => {
    context.fillText(label, x - context.measureText(label).width / 2, height - 18);
  });
  yLabels.forEach(({ y, label }) => {
    context.fillText(label, 3, y + 4);
    context.strokeStyle = "#edf0ef";
    context.beginPath();
    context.moveTo(bounds.left, y);
    context.lineTo(width - bounds.right, y);
    context.stroke();
  });
}

function drawAttention(rows) {
  const { context, width, height } = setupCanvas("attention-chart");
  const colors = ["#286985", "#157f68", "#b56a17"];
  const left = 82;
  const top = 26;
  const barHeight = 30;
  const gap = 24;
  const maxWidth = width - left - 40;
  rows.forEach((row, index) => {
    const y = top + index * (barHeight + gap);
    context.fillStyle = "#e7eceb";
    context.fillRect(left, y, maxWidth, barHeight);
    context.fillStyle = colors[index % colors.length];
    context.fillRect(left, y, maxWidth * row.weight, barHeight);
    context.fillStyle = "#263033";
    context.fillText(row.branch, 4, y + 20);
    const label = percent(row.weight, 1);
    context.fillStyle = row.weight > 0.18 ? "#ffffff" : "#263033";
    const labelX = Math.max(left + 6, left + maxWidth * row.weight - context.measureText(label).width - 8);
    context.fillText(label, labelX, y + 20);
  });
  context.fillStyle = "#657176";
  context.fillText("Attention weights sum to 100% for the current replay flow", left, height - 12);
}

function drawFaithfulness(rows) {
  const { context, width, height } = setupCanvas("faithfulness-chart");
  const bounds = { left: 42, right: 15, top: 18, bottom: 34 };
  const plotWidth = width - bounds.left - bounds.right;
  const plotHeight = height - bounds.top - bounds.bottom;
  const values = rows.flatMap((row) => [row.ranked_drop, row.random_drop]);
  const minimum = Math.min(0, ...values);
  const maximum = Math.max(0.01, ...values) * 1.12;
  const x = (fraction) => bounds.left + ((fraction - 0.1) / 0.4) * plotWidth;
  const y = (value) => bounds.top + (1 - (value - minimum) / (maximum - minimum)) * plotHeight;
  drawAxes(
    context,
    width,
    height,
    bounds,
    rows.map((row) => ({ x: x(row.fraction), label: `${Math.round(row.fraction * 100)}%` })),
    [0, 0.5, 1].map((step) => ({
      y: bounds.top + plotHeight * step,
      label: (maximum - (maximum - minimum) * step).toFixed(2),
    })),
  );
  const series = [
    { key: "ranked_drop", color: "#157f68", label: "Ranked features" },
    { key: "random_drop", color: "#b56a17", label: "Random control" },
  ];
  series.forEach((item) => {
    context.strokeStyle = item.color;
    context.fillStyle = item.color;
    context.lineWidth = 2.2;
    context.beginPath();
    rows.forEach((row, index) => {
      if (index === 0) context.moveTo(x(row.fraction), y(row[item.key]));
      else context.lineTo(x(row.fraction), y(row[item.key]));
    });
    context.stroke();
    rows.forEach((row) => {
      context.beginPath();
      context.arc(x(row.fraction), y(row[item.key]), 3.5, 0, Math.PI * 2);
      context.fill();
    });
  });
  context.fillStyle = "#157f68";
  context.fillRect(bounds.left, 1, 15, 3);
  context.fillStyle = "#263033";
  context.fillText("Ranked", bounds.left + 20, 6);
  context.fillStyle = "#b56a17";
  context.fillRect(bounds.left + 88, 1, 15, 3);
  context.fillStyle = "#263033";
  context.fillText("Random", bounds.left + 108, 6);
}

function drawCalibration(rows) {
  const { context, width, height } = setupCanvas("calibration-chart");
  const bounds = { left: 42, right: 16, top: 16, bottom: 34 };
  const plotWidth = width - bounds.left - bounds.right;
  const plotHeight = height - bounds.top - bounds.bottom;
  const x = (value) => bounds.left + value * plotWidth;
  const y = (value) => bounds.top + (1 - value) * plotHeight;
  drawAxes(
    context,
    width,
    height,
    bounds,
    [0, .5, 1].map((value) => ({ x: x(value), label: value.toFixed(1) })),
    [0, .5, 1].map((value) => ({ y: y(value), label: value.toFixed(1) })),
  );
  context.setLineDash([5, 4]);
  context.strokeStyle = "#8d989a";
  context.beginPath();
  context.moveTo(x(0), y(0));
  context.lineTo(x(1), y(1));
  context.stroke();
  context.setLineDash([]);
  context.strokeStyle = "#286985";
  context.fillStyle = "#286985";
  context.lineWidth = 2;
  context.beginPath();
  rows.forEach((row, index) => {
    if (index === 0) context.moveTo(x(row.confidence), y(row.accuracy));
    else context.lineTo(x(row.confidence), y(row.accuracy));
  });
  context.stroke();
  rows.forEach((row) => {
    const radius = Math.min(7, 2.5 + Math.log10(Math.max(1, row.count)));
    context.beginPath();
    context.arc(x(row.confidence), y(row.accuracy), radius, 0, Math.PI * 2);
    context.fill();
  });
  context.fillStyle = "#657176";
  context.fillText("Confidence", width / 2 - 26, height - 2);
  context.save();
  context.translate(11, height / 2 + 20);
  context.rotate(-Math.PI / 2);
  context.fillText("Accuracy", 0, 0);
  context.restore();
}

function renderFeatureEvidence(rows) {
  const container = byId("feature-evidence");
  container.replaceChildren();
  const maximum = Math.max(...rows.map((row) => Math.abs(row.impact)), 1e-9);
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "feature-row";
    const name = document.createElement("span");
    name.className = "feature-name";
    name.textContent = row.feature;
    name.title = `${row.feature}; standardized value ${fixed(row.standardized_value, 3)}`;
    const track = document.createElement("div");
    track.className = "impact-track";
    const fill = document.createElement("span");
    fill.className = `impact-fill${row.impact < 0 ? " negative" : ""}`;
    const width = Math.max(1, Math.abs(row.impact) / maximum * 50);
    fill.style.width = `${width}%`;
    fill.style.left = row.impact >= 0 ? "50%" : `${50 - width}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "impact-value";
    value.textContent = row.impact.toExponential(2);
    item.append(name, track, value);
    container.append(item);
  });
}

function renderAgent(agent) {
  text("agent-accepted", agent.accepted);
  text("agent-latency", `${fixed(agent.latency_ms, 1)} ms`);
  const container = byId("agent-findings");
  container.replaceChildren();
  agent.findings.forEach((finding) => {
    const item = document.createElement("div");
    item.className = `finding ${finding.claim_type}`;
    const head = document.createElement("div");
    head.className = "finding-head";
    const title = document.createElement("strong");
    title.textContent = finding.claim_type;
    const verdict = document.createElement("span");
    verdict.textContent = "Evidence verified";
    head.append(title, verdict);
    const paragraph = document.createElement("p");
    paragraph.textContent = finding.statement;
    item.append(head, paragraph);
    container.append(item);
  });
}

function renderSummary(summary) {
  text("run-id", summary.run);
  byId("run-id").title = summary.run;
  text("run-seed", summary.seed);
  text("run-protocol", summary.protocol);
  text("metric-accuracy", percent(summary.metrics.accuracy, 2));
  text("metric-macro-f1", fixed(summary.metrics.macro_f1, 3));
  text("metric-p95", `${fixed(summary.latency.p95_single_ms, 2)} ms`);
  text("metric-throughput", compact(summary.latency.maximum_throughput_flows_s));
  text("metric-size", bytes(summary.resource.model_file_bytes));
  text("metric-parameters", `${compact(summary.resource.parameters)} parameters`);
  text("metric-unsupported", percent(summary.agent.unsupported_claim_rate, 0));
  text("stability-agreement", percent(summary.stability.prediction_agreement, 1));
  text("stability-overlap", percent(summary.stability.feature_overlap, 1));
  text("spec-parameters", compact(summary.resource.parameters));
  text("spec-model-size", bytes(summary.resource.model_file_bytes));
  text("spec-flops", compact(summary.resource.flops_per_flow));
  text("spec-threads", `${summary.resource.threads} CPU`);
  text("spec-rss", bytes(summary.resource.peak_process_rss_bytes));
  text("spec-parity", percent(summary.verification.prediction_agreement, 0));
  text("checkpoint-hash", summary.verification.checkpoint_sha256);
  text("limitation-replay", summary.limitations.replay);
  text("limitation-deployment", summary.limitations.deployment);
  text("limitation-explanation", summary.limitations.explanation);
  renderAgent(summary.agent);
  drawFaithfulness(summary.faithfulness);
  drawCalibration(summary.calibration);
}

function renderSample(sample) {
  state.sample = sample;
  text("event-id", sample.event_id);
  text("flow-position", sample.position + 1);
  text("flow-count", sample.sample_count);
  text("predicted-label", sample.predicted_label);
  text("true-label", sample.true_label);
  text("confidence-value", `${percent(sample.confidence, 2)} confidence`);
  byId("confidence-fill").style.width = `${Math.max(1, sample.confidence * 100)}%`;
  const correctness = byId("correctness");
  correctness.textContent = sample.correct ? "Correct" : "Mismatch";
  correctness.classList.toggle("incorrect", !sample.correct);
  text("local-latency", `${fixed(sample.local_replay_ms, 3)} ms`);
  text("probability-delta", sample.stored_probability_delta.toExponential(2));
  text("sample-scope", sample.scope);

  const probabilities = byId("probability-list");
  probabilities.replaceChildren();
  sample.top_probabilities.forEach((row) => {
    const item = document.createElement("div");
    item.className = "probability-row";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = row.label;
    name.title = row.label;
    const bar = document.createElement("div");
    bar.className = "probability-bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(.4, row.probability * 100)}%`;
    bar.append(fill);
    const value = document.createElement("span");
    value.className = "value";
    value.textContent = percent(row.probability, 1);
    item.append(name, bar, value);
    probabilities.append(item);
  });
  renderFeatureEvidence(sample.feature_evidence);
  drawAttention(sample.attention);
}

async function loadSample(index) {
  const response = await fetch(`/api/sample?dataset=${encodeURIComponent(state.dataset)}&index=${index}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Sample request failed: ${response.status}`);
  const sample = await response.json();
  state.index = sample.position;
  renderSample(sample);
}

async function selectDataset(key) {
  stopReplay();
  state.dataset = key;
  document.querySelectorAll(".dataset-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.dataset === key);
  });
  renderSummary(state.summaries[key]);
  await loadSample(state.index);
}

function stopReplay() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  text("toggle-replay", "Start replay");
}

function toggleReplay() {
  if (state.timer) {
    stopReplay();
    return;
  }
  text("toggle-replay", "Pause replay");
  state.timer = window.setInterval(() => {
    loadSample(state.index + 1).catch((error) => {
      console.error(error);
      stopReplay();
    });
  }, 1400);
}

async function initialize() {
  const response = await fetch("/api/status", { cache: "no-store" });
  if (!response.ok) throw new Error(`Status request failed: ${response.status}`);
  const payload = await response.json();
  state.summaries = payload.datasets;
  if (!state.summaries[state.dataset]) state.dataset = "cic";
  await selectDataset(state.dataset);
  byId("loading-state").classList.add("hidden");
  window.__ATLAS_READY = true;
  const capture = new URLSearchParams(location.search).get("capture");
  if (capture === "explain") byId("explainability").scrollIntoView({ block: "start" });
}

document.querySelectorAll(".dataset-tab").forEach((button) => {
  button.addEventListener("click", () => selectDataset(button.dataset.dataset));
});
byId("previous-flow").addEventListener("click", () => loadSample(state.index - 1));
byId("next-flow").addEventListener("click", () => loadSample(state.index + 1));
byId("toggle-replay").addEventListener("click", toggleReplay);
window.addEventListener("resize", () => {
  if (!state.summaries[state.dataset] || !state.sample) return;
  drawFaithfulness(state.summaries[state.dataset].faithfulness);
  drawCalibration(state.summaries[state.dataset].calibration);
  drawAttention(state.sample.attention);
});

initialize().catch((error) => {
  console.error(error);
  byId("loading-state").textContent = `Deployment bundle failed to load: ${error.message}`;
});
