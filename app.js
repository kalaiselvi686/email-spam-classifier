const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const examples = {
  spam: "Congratulations! You have been selected for a free cash prize. Claim your reward now before this limited-time offer expires. Click here to get started.",
  phishing: "Your bank account is locked due to unusual activity. Verify your password at our secure account link now to prevent suspension.",
  ham: "Hi, could you review the project notes before our planning meeting tomorrow? I have attached the latest slides. Thanks for your help."
};
const viewTitles = { overview: "Overview", analyze: "Analyze email", history: "Scan history", model: "Model & data" };
let toastTimer;

function escapeHtml(value = "") {
  return String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
}

function percent(value) { return `${Math.round((Number(value) || 0) * 100)}%`; }
function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(date);
}
function formatDate(value) {
  if (!value) return "";
  const date = new Date(`${value}T12:00:00`);
  return new Intl.DateTimeFormat(undefined, { weekday: "short" }).format(date);
}
function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3000);
}

function signalsMarkup(keywords = []) {
  if (!keywords.length) return '<span class="muted">No strong terms</span>';
  return `<span class="signal-list">${keywords.slice(0, 3).map((item) => `<span class="signal-tag">${escapeHtml(item.term)}</span>`).join("")}</span>`;
}

function detectionRows(rows = [], history = false) {
  return rows.map((row) => `<tr>
    <td><span class="label-badge badge-${escapeHtml(row.label)}">${escapeHtml(row.label)}</span></td>
    ${history ? `<td class="sender-cell">${escapeHtml(row.sender || "Unknown sender")}</td>` : ""}
    <td class="preview-cell" title="${escapeHtml(row.preview)}">${escapeHtml(row.preview)}</td>
    <td>${signalsMarkup(row.keywords)}</td>
    <td class="confidence-cell">${percent(row.confidence)}</td>
    <td class="time-cell">${formatTime(row.created_at)}</td>
  </tr>`).join("");
}

async function request(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "The request could not be completed.");
  return body;
}

function renderVolume(daily = []) {
  const chart = $("#volume-chart");
  const max = Math.max(1, ...daily.map((day) => day.total));
  chart.innerHTML = daily.map((day) => {
    const spamHeight = day.total ? Math.max(2, (day.spam / max) * 72) : 0;
    const hamHeight = day.total ? Math.max(2, (day.ham / max) * 72) : 0;
    const phishingHeight = day.total ? Math.max(2, (day.phishing / max) * 72) : 0;
    return `<div class="chart-day" title="${escapeHtml(day.date)}: ${day.spam} spam, ${day.ham} ham, ${day.phishing} phishing"><div class="chart-bars"><div class="chart-bar-spam" style="height:${spamHeight}px"></div><div class="chart-bar-ham" style="height:${hamHeight}px"></div><div class="chart-bar-phishing" style="height:${phishingHeight}px"></div></div><span class="chart-day-label">${formatDate(day.date)}</span></div>`;
  }).join("");
}

function renderRecent(rows) {
  $("#recent-rows").innerHTML = detectionRows(rows);
  $("#recent-empty").classList.toggle("visible", rows.length === 0);
}

function renderDashboard(data) {
  $("#total-scans").textContent = data.total_scans.toLocaleString();
  $("#spam-count").textContent = data.spam_count.toLocaleString();
  $("#ham-count").textContent = data.ham_count.toLocaleString();
  $("#phishing-count").textContent = data.phishing_count.toLocaleString();
  $("#today-scans").textContent = `${data.today_scans.toLocaleString()} today`;
  $("#spam-rate").textContent = Math.round(data.spam_share * 100);
  $("#week-scans").textContent = data.daily.reduce((sum, day) => sum + day.total, 0).toLocaleString();
  $("#mix-total").textContent = data.total_scans.toLocaleString();
  $("#mix-spam").textContent = data.spam_count.toLocaleString();
  $("#mix-ham").textContent = data.ham_count.toLocaleString();
  $("#mix-phishing").textContent = data.phishing_count.toLocaleString();
  $("#mix-rate").textContent = `${Math.round(data.spam_share * 100)}%`;
  $("#history-count").textContent = data.total_scans > 99 ? "99+" : data.total_scans;
  const spamDegrees = data.total_scans ? data.spam_count / data.total_scans * 360 : 0;
  const hamDegrees = data.total_scans ? data.ham_count / data.total_scans * 360 : 0;
  const hamStop = spamDegrees + hamDegrees;
  $("#mix-donut").style.background = `conic-gradient(var(--coral) 0deg ${spamDegrees}deg, #70bd9e ${spamDegrees}deg ${hamStop}deg, var(--amber) ${hamStop}deg 360deg)`;
  renderVolume(data.daily);
  renderRecent(data.recent);
  renderModel(data.model);
}

function renderModel(model) {
  if (!model) return;
  const names = { naive_bayes: "Naive Bayes", logistic_regression: "Logistic Regression", linear_svm: "Linear SVM" };
  const modelName = names[model.model] || model.model;
  $("#sidebar-model").textContent = modelName;
  $("#footer-model").textContent = modelName.toUpperCase();
  $("#dataset-name").textContent = model.dataset_name.toUpperCase();
  $("#demo-note").hidden = !model.is_demo;
  $("#model-banner").hidden = !model.is_demo;
  const metrics = [
    ["ACCURACY", model.accuracy, "overall correct"],
    ["PRECISION", model.precision, "spam alerts correct"],
    ["FALSE POSITIVE RATE", model.false_positive_rate, "non-spam flagged as spam"],
    ["RECALL", model.recall, "spam caught"],
    ["F1 SCORE", model.f1, "precision + recall"],
    ["EVALUATION SET", null, `${model.test_size} emails`]
  ];
  $("#model-metrics").innerHTML = metrics.map(([label, value, note]) => `<article class="model-metric"><span>${label}</span><strong>${value === null ? model.test_size : percent(value)}</strong><small>${escapeHtml(note)}</small></article>`).join("");
  const matrix = model.confusion_matrix || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const classes = model.classes || ["ham", "spam", "phishing"];
  const confusion = $("#confusion-matrix");
  confusion.style.gridTemplateColumns = `78px repeat(${classes.length}, minmax(0, 1fr))`;
  confusion.style.gridTemplateRows = `25px repeat(${classes.length}, 64px)`;
  confusion.innerHTML = `<div class="matrix-axis"></div>${classes.map((label) => `<div class="matrix-axis">${escapeHtml(label.toUpperCase())}</div>`).join("")}${classes.map((actual, row) => `<div class="matrix-axis">${escapeHtml(actual.toUpperCase())}</div>${classes.map((predicted, column) => `<div class="matrix-cell ${row === column ? "correct" : row === 0 && predicted === "spam" ? "fp" : "miss"}" title="Actual ${escapeHtml(actual)}, predicted ${escapeHtml(predicted)}"><strong>${matrix[row]?.[column] ?? 0}</strong><span>${row === column ? "CORRECT" : "MISCLASSIFIED"}</span></div>`).join("")}`).join("")}`;
  $("#class-report").innerHTML = classes.map((label) => {
    const scores = model.class_metrics?.[label] || { precision: 0, recall: 0, f1: 0 };
    return `<tr><td><span class="label-badge badge-${escapeHtml(label)}">${escapeHtml(label)}</span></td><td>${percent(scores.precision)}</td><td>${percent(scores.recall)}</td><td>${percent(scores.f1)}</td></tr>`;
  }).join("");
}

async function loadDashboard() {
  const data = await request("/api/dashboard");
  renderDashboard(data);
}

async function loadHistory() {
  const params = new URLSearchParams({ limit: "250", q: $("#history-search").value.trim(), label: $("#history-filter").value });
  const rows = await request(`/api/history?${params}`);
  $("#history-rows").innerHTML = detectionRows(rows, true);
  $("#history-total").textContent = rows.length;
  $("#history-empty").classList.toggle("visible", rows.length === 0);
}

function switchView(name) {
  if (!viewTitles[name]) return;
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  $("#breadcrumb-title").textContent = viewTitles[name];
  if (name === "history") loadHistory().catch((error) => showToast(error.message));
  if (name === "overview") loadDashboard().catch((error) => showToast(error.message));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function classifyEmail(event) {
  event.preventDefault();
  const button = $("#analyze-submit");
  const text = $("#email-input").value.trim();
  if (text.length < 10) return;
  button.disabled = true;
  button.innerHTML = '<span class="loading-dots">Analyzing…</span>';
  try {
    const result = await request("/api/classify", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, sender: $("#sender-input").value.trim() })
    });
    const label = result.label;
    const classTitle = { ham: "Legitimate email", spam: "Spam detected", phishing: "Phishing risk" }[label] || label;
    const classCopy = { ham: "No strong spam or phishing pattern detected.", spam: "This message matches unsolicited bulk-message patterns.", phishing: "This message resembles a credential or identity theft attempt." }[label] || "The model classified this message.";
    $("#result-placeholder").hidden = true;
    $("#result-output").hidden = false;
    const keywords = result.keywords.length ? result.keywords.map((keyword) => `<div class="signal-row"><span>${escapeHtml(keyword.term)}</span><span class="signal-direction ${keyword.direction}">${keyword.direction} signal</span></div>`).join("") : '<div class="signal-row"><span>No known terms in this message</span></div>';
    $("#result-output").innerHTML = `<div class="result-verdict ${escapeHtml(label)}"><svg viewBox="0 0 24 24">${label === "ham" ? '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>' : '<path d="M12 3 22 21H2L12 3Z"/><path d="M12 9v5m0 3h.01"/>'}</svg><div><strong>${escapeHtml(classTitle)}</strong><span>${escapeHtml(classCopy)}</span></div></div><div class="confidence-block"><div class="confidence-heading"><span>CLASSIFICATION CONFIDENCE</span><strong>${percent(result.confidence)}</strong></div><div class="confidence-track"><div class="confidence-fill ${escapeHtml(label)}" style="width:${Math.round(result.confidence * 100)}%"></div></div></div><div class="signal-heading">TOP CONTRIBUTING TERMS</div><div class="signal-rows">${keywords}</div><div class="class-probabilities">${Object.entries(result.class_probabilities).map(([name, value]) => `<div><span>${escapeHtml(name)}</span><strong>${percent(value)}</strong></div>`).join("")}</div><p class="result-disclaimer">Confidence is not a guarantee. Verify suspicious links and sender details independently.</p>`;
    await loadDashboard();
    showToast("Classification saved to local history.");
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = '<svg viewBox="0 0 24 24"><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 5 5M10.8 7v7.6m-3.8-3.8h7.6"/></svg>Classify email';
  }
}

async function trainModel(event) {
  event.preventDefault();
  const file = $("#dataset-file").files[0];
  if (!file) return;
  const button = $("#train-submit");
  const message = $("#train-message");
  button.disabled = true;
  message.classList.remove("error");
  message.textContent = "Training model…";
  const body = new FormData();
  body.append("file", file);
  body.append("model", $("#model-select").value);
  try {
    const result = await request("/api/dataset", { method: "POST", body });
    renderModel(result.metrics);
    await loadDashboard();
    message.textContent = `${result.metrics.training_size} training rows · ${result.metrics.test_size} validation rows`;
    showToast("Model retrained with your dataset.");
  } catch (error) {
    message.textContent = error.message;
    message.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

$("#today-label").textContent = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(new Date()).toUpperCase();
$("#analyze-form").addEventListener("submit", classifyEmail);
$("#dataset-form").addEventListener("submit", trainModel);
$("#email-input").addEventListener("input", (event) => { $("#char-count").textContent = `${event.target.value.length.toLocaleString()} / 50,000`; });
$("#history-search").addEventListener("input", () => loadHistory().catch((error) => showToast(error.message)));
$("#history-filter").addEventListener("change", () => loadHistory().catch((error) => showToast(error.message)));
$("#refresh-button").addEventListener("click", () => loadDashboard().then(() => showToast("Dashboard refreshed.")).catch((error) => showToast(error.message)));
$("#dismiss-note").addEventListener("click", () => { $("#demo-note").hidden = true; });
$$("[data-view]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
$$("[data-open-view]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.openView)));
$$('[data-example]').forEach((button) => button.addEventListener("click", () => {
  $("#email-input").value = examples[button.dataset.example];
  $("#char-count").textContent = `${$("#email-input").value.length} / 50,000`;
  $("#email-input").focus();
}));

loadDashboard().catch((error) => showToast(error.message));