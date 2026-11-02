const state = {
  activeTab: "analyze",
  imageFile: null,
  trainFile: null,
  selectedFakeIds: new Set(),
  charts: {
    pie: null,
    bar: null,
    line: null,
  },
  gmailPollTimer: null,
};

const RESULT_VIEWS = {
  result: {
    cardId: "resultCard",
    placeholderId: "resultPlaceholder",
    badgeId: "resultBadge",
    confId: "resultConf",
    reasonId: "explainReason",
    tipId: "explainTip",
    chipsId: "signalChips",
    realBarId: "realBar",
    suspiciousBarId: "suspiciousBar",
    fakeBarId: "fakeBar",
    realNumId: "realNum",
    suspiciousNumId: "suspiciousNum",
    fakeNumId: "fakeNum",
  },
  ocr: {
    cardId: "ocrResultCard",
    placeholderId: "ocrPlaceholder",
    badgeId: "ocrBadge",
    confId: "ocrConf",
    reasonId: "ocrReason",
    tipId: "ocrTip",
  },
};

document.addEventListener("DOMContentLoaded", () => {
  bindNavigation();
  bindAmbientEffects();
  initVfxCanvas();
  bindAnalyzeActions();
  bindImageActions();
  bindHistoryActions();
  bindDashboardActions();
  bindArchitectureComparison();
  bindGmailActions();
  bindTrainingActions();
  bindSamples();
  bindShortcuts();
  setupTrainDropZone();
  setupImageDropZone();
  loadModelBadge();
  loadModelStats();
  loadHistory();
  loadOcrStatus();
});



function initVfxCanvas() {
  const canvas = $("vfx-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  const pointer = { x: null, y: null, radius: 145 };
  let points = [];
  let width = 0;
  let height = 0;
  let frame = 0;

  function resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const count = Math.min(150, Math.max(55, Math.round((width * height) / 12000)));
    points = Array.from({ length: count }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.72,
      vy: (Math.random() - 0.5) * 0.72,
      radius: 0.7 + Math.random() * 1.8,
      phase: Math.random() * Math.PI * 2,
    }));
    draw(false);
  }

  function draw(move) {
    ctx.clearRect(0, 0, width, height);
    for (const p of points) {
      if (move) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > width) p.vx *= -1;
        if (p.y < 0 || p.y > height) p.vy *= -1;
        if (pointer.x !== null) {
          const dx = p.x - pointer.x;
          const dy = p.y - pointer.y;
          const distance = Math.hypot(dx, dy);
          if (distance > 0 && distance < pointer.radius) {
            const force = (pointer.radius - distance) / pointer.radius * 0.7;
            p.x += dx / distance * force;
            p.y += dy / distance * force;
          }
        }
        p.phase += 0.018;
      }
      const alpha = 0.45 + (Math.sin(p.phase) + 1) * 0.22;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(58, 239, 161, ${alpha})`;
      ctx.fill();
      if (p.radius > 1.7) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius * 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(46, 204, 113, ${alpha * 0.12})`;
        ctx.fill();
      }
    }
    const maxDistance = 156;
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const a = points[i];
        const b = points[j];
        const distance = Math.hypot(a.x - b.x, a.y - b.y);
        if (distance < maxDistance) {
          const opacity = (1 - distance / maxDistance) * 0.48;
          ctx.beginPath();
          ctx.strokeStyle = `rgba(46, 204, 113, ${opacity})`;
          ctx.lineWidth = 1.1;
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
  }

  function animate() {
    if (document.hidden) {
      frame = 0;
      return;
    }
    draw(true);
    frame = window.requestAnimationFrame(animate);
  }

  function start() {
    if (!frame && !document.hidden) frame = window.requestAnimationFrame(animate);
  }

  canvas.style.opacity = "1";
  window.addEventListener("pointermove", (event) => {
    pointer.x = event.clientX;
    pointer.y = event.clientY;
  }, { passive: true });
  window.addEventListener("pointerleave", () => { pointer.x = pointer.y = null; });
  window.addEventListener("resize", resize, { passive: true });
  document.addEventListener("visibilitychange", start);
  resize();
  start();
}

function bindAmbientEffects() {
  if (window.matchMedia("(pointer: fine)").matches) {
    let frame = 0;
    document.addEventListener("pointermove", (event) => {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        document.documentElement.style.setProperty("--pointer-x", `${event.clientX}px`);
        document.documentElement.style.setProperty("--pointer-y", `${event.clientY}px`);
        frame = 0;
      });
    }, { passive: true });
  }
}

function bindNavigation() {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
}

function bindAnalyzeActions() {
  $("analyzeBtn").addEventListener("click", analyzeMessage);
  $("clearAnalyzeBtn").addEventListener("click", clearAnalyze);
}

function bindImageActions() {
  $("dropZone").addEventListener("click", () => $("imgInput").click());
  $("imgInput").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
      setImageFile(file);
    }
  });
  $("ocrBtn").addEventListener("click", analyzeImage);
  $("clearImageBtn").addEventListener("click", clearImage);
}

function bindHistoryActions() {
  $("downloadAllBtn").addEventListener("click", () => downloadFile("/download-all-csv"));
  $("downloadFakeBtn").addEventListener("click", () => downloadFile("/download-fake-csv"));
  $("downloadRealBtn").addEventListener("click", () => downloadFile("/download-real-csv"));
  $("clearHistoryBtn").addEventListener("click", clearHistoryUI);
}

function bindDashboardActions() {
  $("dashboardDownloadAllBtn").addEventListener("click", () => downloadFile("/download-all-csv"));
  $("dashboardDownloadFakeBtn").addEventListener("click", () => downloadFile("/download-fake-csv"));
  $("dashboardDownloadRealBtn").addEventListener("click", () => downloadFile("/download-real-csv"));
}

function bindArchitectureComparison() {
  $("compareModelsBtn").addEventListener("click", compareModelPaths);
  $("architectureInput").addEventListener("input", (event) => {
    const value = event.target.value.trim();
    $("archSampleText").textContent = value
      ? `${value.slice(0, 150)}${value.length > 150 ? "…" : ""}`
      : "Enter a message below to compare both paths.";
  });
}

async function compareModelPaths() {
  const message = $("architectureInput").value.trim();
  const status = $("compareModelStatus");
  if (!message) {
    status.textContent = "Enter a message first.";
    return;
  }
  setButtonBusy($("compareModelsBtn"), true, "Comparing...");
  status.textContent = "Running the same message through both classifiers…";
  try {
    const formData = new FormData();
    formData.append("message", message);
    const data = await apiFetch("/compare-models", { method: "POST", body: formData });
    $("archTfidfReal").textContent = formatPercent(data.tfidf.real_prob);
    $("archTfidfSuspicious").textContent = formatPercent(data.tfidf.suspicious_prob);
    $("archTfidfFake").textContent = formatPercent(data.tfidf.fake_prob);
    $("archTfidfLabel").textContent = data.tfidf.label;
    if (data.transformer) {
      $("archTransformerReal").textContent = formatPercent(data.transformer.real_prob);
      $("archTransformerSuspicious").textContent = formatPercent(data.transformer.suspicious_prob);
      $("archTransformerFake").textContent = formatPercent(data.transformer.fake_prob);
      $("archTransformerLabel").textContent = data.transformer.label;
      status.textContent = "Same input; raw model probabilities (not calibrated confidence). Labels use the shared Real / Suspicious / Fake thresholds.";
    } else {
      $("archTransformerReal").textContent = "--";
      $("archTransformerSuspicious").textContent = "--";
      $("archTransformerFake").textContent = "--";
      $("archTransformerLabel").textContent = "Not retrained";
      status.textContent = `${data.transformer_status} Live result above is from the active three-class TF-IDF model.`;
    }
  } catch (error) {
    status.textContent = `Could not compare models: ${error.message}`;
  } finally {
    setButtonBusy($("compareModelsBtn"), false, "Score message");
  }
}

function bindGmailActions() {
  $("gmailLoginBtn").addEventListener("click", gmailLogin);
  $("gScanBtn").addEventListener("click", scanGmail);
  $("gDelBtn").addEventListener("click", deleteSelected);
  $("gmailDownloadBtn").addEventListener("click", () => downloadFile("/download-gmail-csv"));
  $("gmailDisconnectBtn").addEventListener("click", disconnectGmail);
}

function bindTrainingActions() {
  $("trainDropZone").addEventListener("click", () => $("trainFileInput").click());
  $("trainFileInput").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
      setTrainFile(file);
    }
  });
  $("trainBtn").addEventListener("click", startTraining);
}

function bindSamples() {
  document.querySelectorAll(".sample-card").forEach((button) => {
    button.addEventListener("click", () => {
      $("msgInput").value = button.dataset.sampleMessage || "";
      switchTab("analyze");
      hideAlert("analysisAlert");
    });
  });
}

function bindShortcuts() {
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && state.activeTab === "analyze") {
      analyzeMessage();
    }
  });
}

function switchTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-section").forEach((section) => {
    section.classList.toggle("active", section.id === `tab-${tabName}`);
  });

  if (tabName === "dashboard") {
    refreshDashboard();
  } else if (tabName === "train") {
    loadModelStats();
  }
}

async function analyzeMessage() {
  hideAlert("analysisAlert");
  const message = $("msgInput").value.trim();
  if (!message) {
    showAlert("analysisAlert", "Please paste a message before starting analysis.", "warn");
    return;
  }

  setButtonBusy($("analyzeBtn"), true, "Analyzing...");
  showResultPlaceholder("result");

  try {
    const formData = new FormData();
    formData.append("message", message);
    const data = await apiFetch("/predict", { method: "POST", body: formData });
    renderResultView("result", data);
    await syncAfterPrediction();
  } catch (error) {
    showAlert("analysisAlert", error.message, "error");
  } finally {
    setButtonBusy($("analyzeBtn"), false, "Analyze Message");
  }
}

function clearAnalyze() {
  $("msgInput").value = "";
  showResultPlaceholder("result");
  hideAlert("analysisAlert");
}

function showResultPlaceholder(viewKey) {
  const view = RESULT_VIEWS[viewKey];
  $(view.placeholderId).classList.remove("hidden");
  $(view.cardId).classList.add("hidden");
}

function renderResultView(viewKey, data) {
  const view = RESULT_VIEWS[viewKey];
  $(view.placeholderId).classList.add("hidden");
  $(view.cardId).classList.remove("hidden");

  const badge = $(view.badgeId);
  badge.className = `result-badge ${badgeClassFromLabel(data.label)}`;
  badge.textContent = badgeTextFromLabel(data.label);
  $(view.confId).textContent = `Threat score ${formatPercent(data.confidence)}`;
  $(view.reasonId).textContent = data.explanation?.reason || "No explanation provided.";
  $(view.tipId).textContent = data.explanation?.tip || "Review manually before acting on unfamiliar requests.";

  if (view.chipsId) {
    const signals = data.explanation?.signals || [];
    $(view.chipsId).innerHTML = signals.length
      ? signals.map((signal) => `<span class="chip">${escapeHtml(signal)}</span>`).join("")
      : '<span class="chip">No strong keywords surfaced</span>';
  }

  if (view.realBarId) {
    $(view.realBarId).style.width = `${data.real_prob}%`;
    $(view.fakeBarId).style.width = `${data.fake_prob}%`;
    $(view.realNumId).textContent = formatPercent(data.real_prob);
    $(view.fakeNumId).textContent = formatPercent(data.fake_prob);
    if (view.suspiciousBarId) {
      $(view.suspiciousBarId).style.width = `${data.suspicious_prob || 0}%`;
      $(view.suspiciousNumId).textContent = formatPercent(data.suspicious_prob || 0);
    }
  }
}

async function syncAfterPrediction() {
  await Promise.all([loadHistory(), refreshDashboard()]);
}

async function loadHistory() {
  try {
    const data = await apiFetch("/history");
    const rows = data.history || [];
    $("historyBody").innerHTML = rows.length
      ? rows.slice(0, 15).map(renderHistoryRow).join("")
      : '<tr><td colspan="5" class="empty-row">No checks yet</td></tr>';
  } catch (error) {
    showAlert("analysisAlert", `Could not load history: ${error.message}`, "error");
  }
}

function renderHistoryRow(item) {
  return `
    <tr>
      <td class="mono">${item.id}</td>
      <td title="${escapeHtml(item.message)}">${escapeHtml(item.message)}</td>
      <td class="${labelToneClass(item.label)}">${escapeHtml(item.label)}</td>
      <td class="mono">${formatPercent(item.confidence)}</td>
      <td>${escapeHtml(item.timestamp)}</td>
    </tr>
  `;
}

async function clearHistoryUI() {
  hideAlert("analysisAlert");
  hideAlert("dashboardAlert");
  if (!window.confirm("Clear the entire session history?")) {
    return;
  }

  try {
    await apiFetch("/clear-history", { method: "POST" });
    clearAnalyze();
    clearImage(false);
    resetGmailPanel();
    await Promise.all([loadHistory(), refreshDashboard()]);
  } catch (error) {
    showAlert("analysisAlert", `Could not clear history: ${error.message}`, "error");
  }
}

async function loadOcrStatus() {
  try {
    const data = await apiFetch("/ocr-status");
    if (!data.available) {
      $("ocrBtn").disabled = true;
      showAlert("ocrAlert", "OCR is not available on this machine yet. Install Tesseract to enable screenshot analysis.", "warn");
    }
  } catch (error) {
    showAlert("ocrAlert", `Could not confirm OCR status: ${error.message}`, "warn");
  }
}

function setupImageDropZone() {
  const zone = $("dropZone");
  ["dragenter", "dragover"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    });
  });
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) {
      return;
    }
    if (!file.type.startsWith("image/")) {
      showAlert("ocrAlert", "Please choose an image file such as PNG, JPG, or WEBP.", "error");
      return;
    }
    setImageFile(file);
  });
}

function setImageFile(file) {
  state.imageFile = file;
  hideAlert("ocrAlert");
  const reader = new FileReader();
  reader.onload = (event) => {
    $("imgPreview").src = event.target.result;
    $("imgPreviewWrap").classList.remove("hidden");
  };
  reader.readAsDataURL(file);
  $("ocrBtn").disabled = false;
}

async function analyzeImage() {
  if (!state.imageFile) {
    showAlert("ocrAlert", "Select an image before running OCR analysis.", "warn");
    return;
  }

  setButtonBusy($("ocrBtn"), true, "Processing...");
  hideAlert("ocrAlert");
  $("ocrPlaceholder").classList.remove("hidden");
  $("ocrResultWrap").classList.add("hidden");

  try {
    const formData = new FormData();
    formData.append("image", state.imageFile);
    const data = await apiFetch("/predict-image", { method: "POST", body: formData });
    $("extractedText").textContent = data.extracted_text || "No text extracted.";
    $("ocrResultWrap").classList.remove("hidden");
    renderResultView("ocr", data);
    await syncAfterPrediction();
  } catch (error) {
    showAlert("ocrAlert", error.message, "error");
  } finally {
    setButtonBusy($("ocrBtn"), false, "Extract and Analyze");
  }
}

function clearImage(resetInput = true) {
  state.imageFile = null;
  if (resetInput) {
    $("imgInput").value = "";
  }
  $("imgPreview").src = "";
  $("imgPreviewWrap").classList.add("hidden");
  $("ocrBtn").disabled = true;
  $("ocrPlaceholder").classList.remove("hidden");
  $("ocrResultWrap").classList.add("hidden");
  $("ocrResultCard").classList.add("hidden");
  $("extractedText").textContent = "";
  hideAlert("ocrAlert");
}

async function refreshDashboard() {
  hideAlert("dashboardAlert");
  try {
    const data = await apiFetch("/analytics");
    updateDashboardMetrics(data);
    renderDashboardCharts(data);
    renderDashboardActivity(data.recent || []);
  } catch (error) {
    showAlert("dashboardAlert", `Could not load analytics: ${error.message}`, "error");
  }
}

function updateDashboardMetrics(data) {
  $("dTotal").textContent = data.total;
  $("dFake").textContent = data.fake;
  $("dSusp").textContent = data.suspicious;
  $("dReal").textContent = data.real;
  $("dAcc").textContent = data.model_accuracy ? `${data.model_accuracy}%` : "--";
  $("dSamples").textContent = data.model_samples ?? "--";
  renderModelComparison(data.model_comparison);
  $("activeModelName").textContent = data.active_model || "TF-IDF + Logistic Regression";
  const directCount = data.source_counts?.direct || 0;
  const gmailCount = data.source_counts?.gmail || 0;
  $("dCoverage").textContent = `${data.total} combined checks | ${directCount} direct | ${gmailCount} Gmail`;
  $("dRisk").textContent = dashboardRiskCopy(data);
}

function renderModelComparison(comparison) {
  const tfidf = $("comparisonTfidf");
  if (!comparison) {
    tfidf.textContent = "--";
    $("comparisonTransformer").textContent = "--";
    $("comparisonDelta").textContent = "--";
    $("comparisonStatus").textContent = "Not measured";
    $("comparisonNote").textContent = "Train the three-class model to populate the evaluation.";
    return;
  }

  if (comparison.accuracy_pct !== undefined) {
    tfidf.textContent = formatPercent(comparison.accuracy_pct);
    $("comparisonTransformer").textContent = "Not retrained";
    $("comparisonDelta").textContent = "--";
    $("comparisonStatus").textContent = `${compactNumber(comparison.test_samples)} test messages`;
    const metrics = comparison.class_metrics || {};
    const summary = Object.entries(metrics).map(([label, values]) => `${label} F1 ${values["f1-score"]}%`).join(" · ");
    $("comparisonNote").textContent = `Three-class held-out results: ${compactNumber(comparison.train_samples)} train / ${compactNumber(comparison.test_samples)} test. ${summary}. ${comparison.transformer_status || ""}`;
    return;
  }

  tfidf.textContent = formatPercent(comparison.tfidf_accuracy_pct);
  $("comparisonTransformer").textContent = formatPercent(comparison.transformer_accuracy_pct);
  const delta = Number(comparison.accuracy_difference_pp);
  const deltaNode = $("comparisonDelta");
  deltaNode.textContent = `${delta > 0 ? "+" : ""}${delta.toFixed(2)} pp`;
  deltaNode.classList.toggle("positive", delta > 0);
  deltaNode.classList.toggle("negative", delta < 0);
  $("comparisonStatus").textContent = `${compactNumber(comparison.test_samples)} test messages`;
  $("comparisonNote").textContent = `Same stratified 80/20 split: ${compactNumber(comparison.train_samples)} train and ${compactNumber(comparison.test_samples)} held-out messages. Higher accuracy: ${comparison.winner}.`;
}

function dashboardRiskCopy(data) {
  if (!data.total) {
    return "No data yet";
  }
  if (data.fake_pct >= 60) {
    return "High fake volume";
  }
  if (data.fake_pct >= 30) {
    return "Mixed threat profile";
  }
  return "Mostly lower-risk";
}

function renderDashboardCharts(data) {
  const total = data.total || 0;
  const hasData = total > 0;
  toggleChartState("pieChartState", !hasData);
  toggleChartState("barChartState", !hasData);
  toggleChartState("lineChartState", !hasData);
  renderPieChart(data, hasData);
  renderBarChart(data.conf_buckets || {}, hasData);
  renderLineChart(data.recent || [], hasData);
}

function renderPieChart(data, hasData) {
  destroyChart("pie");
  const context = $("pieChart").getContext("2d");
  const values = hasData ? [data.fake, data.suspicious, data.real] : [1, 1, 1];
  state.charts.pie = new Chart(context, {
    type: "doughnut",
    data: {
      labels: ["Fake", "Suspicious", "Real"],
      datasets: [{
        data: values,
        backgroundColor: hasData
          ? ["rgba(255, 107, 107, 0.75)", "rgba(255, 191, 71, 0.72)", "rgba(79, 240, 166, 0.72)"]
          : ["rgba(142, 168, 191, 0.18)", "rgba(142, 168, 191, 0.18)", "rgba(142, 168, 191, 0.18)"],
        borderColor: hasData
          ? ["#ff6b6b", "#ffbf47", "#4ff0a6"]
          : ["rgba(142, 168, 191, 0.28)", "rgba(142, 168, 191, 0.28)", "rgba(142, 168, 191, 0.28)"],
        borderWidth: 1.5,
      }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: "#eef7ff",
            font: { family: "Space Grotesk" },
          },
        },
      },
      cutout: "68%",
    },
  });
}

function renderBarChart(buckets, hasData) {
  destroyChart("bar");
  const context = $("barChart").getContext("2d");
  const labels = ["0-85", "85-95", "95-100"];
  const values = labels.map((label) => Number(buckets[label] || 0));
  state.charts.bar = new Chart(context, {
    type: "bar",
    data: {
      labels: labels.map((label) => `${label}% threat`),
      datasets: [{
        label: "Messages",
        data: hasData ? values : [0, 0, 0],
        backgroundColor: "rgba(63, 224, 255, 0.42)",
        borderColor: "#3fe0ff",
        borderRadius: 12,
        borderWidth: 1.2,
      }],
    },
    options: chartOptions({ suggestedMax: hasData ? undefined : 1 }),
  });
}

function renderLineChart(recent, hasData) {
  destroyChart("line");
  const context = $("lineChart").getContext("2d");
  state.charts.line = new Chart(context, {
    type: "line",
    data: {
      labels: hasData ? recent.map((item, index) => `Scan ${index + 1}`) : ["Awaiting data"],
      datasets: [{
        label: "Fake probability %",
        data: hasData ? recent.map((item) => item.fake_prob) : [0],
        fill: true,
        tension: 0.34,
        borderColor: "#3fe0ff",
        backgroundColor: "rgba(63, 224, 255, 0.14)",
        pointRadius: hasData ? 4.5 : 0,
        pointBackgroundColor: hasData ? recent.map((item) => labelColor(item.label)) : ["#3fe0ff"],
      }],
    },
    options: chartOptions({ min: 0, max: 100 }),
  });
}

function chartOptions(yOverrides = {}) {
  return {
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: "#eef7ff",
          font: { family: "Space Grotesk" },
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#8ea8bf", font: { family: "Space Grotesk" } },
        grid: { color: "rgba(142, 168, 191, 0.12)" },
      },
      y: {
        min: 0,
        ticks: { color: "#8ea8bf", font: { family: "JetBrains Mono" } },
        grid: { color: "rgba(142, 168, 191, 0.12)" },
        ...yOverrides,
      },
    },
  };
}

function renderDashboardActivity(items) {
  $("dashboardActivity").innerHTML = items.length
    ? items.map(renderActivityItem).join("")
    : '<div class="activity-empty">No prediction activity yet. Analyze text or screenshots to populate the feed.</div>';
}

function renderActivityItem(item) {
  return `
    <div class="activity-item">
      <div class="activity-meta">
        <div class="activity-title">${escapeHtml(item.label)} · ${formatPercent(item.confidence)} threat score</div>
        <div class="activity-copy">${escapeHtml(item.source || "Direct Scan")} | ${escapeHtml(item.message)}</div>
      </div>
      <span class="activity-badge ${badgeClassFromLabel(item.label)}">${badgeTextShort(item.label)}</span>
    </div>
  `;
}

function toggleChartState(id, visible) {
  $(id).classList.toggle("hidden", !visible);
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    state.charts[key] = null;
  }
}

async function loadModelBadge() {
  try {
    const data = await apiFetch("/model-info");
    const summary = `Acc ${data.accuracy}% | ${compactNumber(data.features)} features`;
    $("navModelBadge").textContent = summary;
    $("footerModel").textContent = summary;
  } catch (error) {
    $("navModelBadge").textContent = "Model telemetry unavailable";
    $("footerModel").textContent = "Telemetry unavailable";
  }
}

async function loadModelStats() {
  try {
    const data = await apiFetch("/model-info");
    $("tAcc").textContent = `${data.accuracy}%`;
    $("tFeat").textContent = compactNumber(data.features);
    $("tSamp").textContent = data.training_samples ?? "--";
  } catch (error) {
    showAlert("trainAlert", `Could not load model stats: ${error.message}`, "error");
  }
}

function setupTrainDropZone() {
  const zone = $("trainDropZone");
  ["dragenter", "dragover"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    });
  });
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) {
      return;
    }
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showAlert("trainAlert", "Please choose a CSV file for retraining.", "error");
      return;
    }
    setTrainFile(file);
  });
}

function setTrainFile(file) {
  state.trainFile = file;
  $("trainFileName").textContent = file.name;
  $("trainBtn").disabled = false;
  hideAlert("trainAlert");
}

async function startTraining() {
  if (!state.trainFile) {
    showAlert("trainAlert", "Select a CSV file before starting retraining.", "warn");
    return;
  }

  hideAlert("trainAlert");
  setButtonBusy($("trainBtn"), true, "Training...");
  $("trainProgressBlock").classList.remove("hidden");
  $("trainProgressBar").style.width = "4%";
  $("trainProgressLbl").textContent = "Preparing training job...";

  const pollId = window.setInterval(async () => {
    try {
      const status = await apiFetch("/training-status");
      $("trainProgressBar").style.width = `${status.progress || 0}%`;
      $("trainProgressLbl").textContent = status.message || "Training...";
    } catch (error) {
      // Keep the main request running; the final response will surface any issue.
    }
  }, 800);

  try {
    const formData = new FormData();
    formData.append("file", state.trainFile);
    const data = await apiFetch("/upload-train-data", { method: "POST", body: formData });
    showAlert(
      "trainAlert",
      `Training complete. Accuracy ${data.accuracy}%, samples ${data.samples}, features ${data.features}.`,
      "success",
    );
    $("trainProgressBar").style.width = "100%";
    $("trainProgressLbl").textContent = "Training complete.";
    await Promise.all([loadModelBadge(), loadModelStats(), refreshDashboard()]);
  } catch (error) {
    showAlert("trainAlert", `Training failed: ${error.message}`, "error");
  } finally {
    window.clearInterval(pollId);
    setButtonBusy($("trainBtn"), false, "Start Training");
  }
}

async function gmailLogin() {
  hideAlert("gmailLoginAlert");
  const email = $("gEmail").value.trim();
  const password = $("gPass").value.trim();
  if (!email || !password) {
    showAlert("gmailLoginAlert", "Fill in both Gmail address and App Password.", "warn");
    return;
  }

  setButtonBusy($("gmailLoginBtn"), true, "Connecting...");
  try {
    await apiFetch("/gmail-login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    $("gmailLoginCard").classList.add("hidden");
    $("gmailDash").classList.remove("hidden");
    hideAlert("gmailDashAlert");
  } catch (error) {
    showAlert("gmailLoginAlert", error.message, "error");
  } finally {
    setButtonBusy($("gmailLoginBtn"), false, "Connect Gmail");
  }
}

async function scanGmail() {
  hideAlert("gmailDashAlert");
  state.selectedFakeIds.clear();
  $("gDelBtn").disabled = true;
  $("gmailProgress").classList.remove("hidden");
  setButtonBusy($("gScanBtn"), true, "Scanning...");

  try {
    await apiFetch("/check-gmail", { method: "POST" });
    if (state.gmailPollTimer) {
      window.clearInterval(state.gmailPollTimer);
    }
    state.gmailPollTimer = window.setInterval(pollGmailState, 1200);
    await pollGmailState();
  } catch (error) {
    showAlert("gmailDashAlert", error.message, "error");
    $("gmailProgress").classList.add("hidden");
    setButtonBusy($("gScanBtn"), false, "Scan Inbox");
  }
}

async function pollGmailState() {
  try {
    const data = await apiFetch("/gmail-stats");
    $("gTotal").textContent = data.total;
    $("gFake").textContent = data.fake;
    $("gSusp").textContent = data.suspicious || 0;
    $("gReal").textContent = data.real;
    $("gProgressBar").style.width = `${data.progress || 0}%`;
    $("gProgressLbl").textContent = data.processing ? `Scanning inbox... ${data.progress || 0}%` : "Scan complete.";
    renderGmailMessages(data.messages || []);

    if (!data.processing && state.gmailPollTimer) {
      window.clearInterval(state.gmailPollTimer);
      state.gmailPollTimer = null;
      $("gmailProgress").classList.add("hidden");
      setButtonBusy($("gScanBtn"), false, "Scan Inbox");
      await refreshDashboard();
    }
  } catch (error) {
    if (state.gmailPollTimer) {
      window.clearInterval(state.gmailPollTimer);
      state.gmailPollTimer = null;
    }
    $("gmailProgress").classList.add("hidden");
    setButtonBusy($("gScanBtn"), false, "Scan Inbox");
    showAlert("gmailDashAlert", `Could not read Gmail scan progress: ${error.message}`, "error");
  }
}

function renderGmailMessages(messages) {
  $("gmailMsgs").innerHTML = messages.length
    ? messages.map(renderGmailMessageItem).join("")
    : '<div class="activity-empty">No messages yet. Run Scan Inbox to populate results.</div>';

  document.querySelectorAll(".msg-item").forEach((item) => {
    item.addEventListener("click", () => toggleGmailMsg(item.dataset.id, item));
  });
}

function renderGmailMessageItem(item) {
  const badgeClass = item.badge === "fake" ? "mb-fake" : item.badge === "suspicious" ? "mb-susp" : "mb-real";
  const selectedClass = state.selectedFakeIds.has(item.id) ? " selected-fake" : "";
  return `
    <div class="msg-item${selectedClass}" data-id="${escapeHtml(item.id)}" data-fake="${item.badge === "fake"}">
      <div class="msg-left">
        <div class="msg-subj">${escapeHtml(item.subject)}</div>
        <div class="msg-from">From: ${escapeHtml(item.from)}</div>
        <div class="msg-prev">${escapeHtml(item.preview)}</div>
      </div>
      <span class="msg-badge ${badgeClass}">${escapeHtml(item.label)} ${escapeHtml(item.confidence)}</span>
    </div>
  `;
}

function toggleGmailMsg(id, element) {
  if (element.dataset.fake !== "true") {
    return;
  }
  if (state.selectedFakeIds.has(id)) {
    state.selectedFakeIds.delete(id);
    element.classList.remove("selected-fake");
  } else {
    state.selectedFakeIds.add(id);
    element.classList.add("selected-fake");
  }
  $("gDelBtn").disabled = state.selectedFakeIds.size === 0;
}

async function deleteSelected() {
  hideAlert("gmailDashAlert");
  if (!state.selectedFakeIds.size) {
    showAlert("gmailDashAlert", "Select one or more fake messages first.", "warn");
    return;
  }
  if (!window.confirm(`Delete ${state.selectedFakeIds.size} selected fake message(s)?`)) {
    return;
  }

  setButtonBusy($("gDelBtn"), true, "Deleting...");
  try {
    const data = await apiFetch("/delete-fake-messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message_ids: [...state.selectedFakeIds] }),
    });
    showAlert("gmailDashAlert", `Deleted ${data.deleted} message(s) from Gmail.`, "success");
    state.selectedFakeIds.clear();
    $("gDelBtn").disabled = true;
    await scanGmail();
  } catch (error) {
    showAlert("gmailDashAlert", error.message, "error");
  } finally {
    setButtonBusy($("gDelBtn"), false, "Delete Selected Fake");
  }
}

async function disconnectGmail() {
  hideAlert("gmailDashAlert");
  try {
    await apiFetch("/gmail-logout", { method: "POST" });
  } catch (error) {
    showAlert("gmailDashAlert", `Disconnect failed: ${error.message}`, "error");
    return;
  }

  if (state.gmailPollTimer) {
    window.clearInterval(state.gmailPollTimer);
    state.gmailPollTimer = null;
  }

  $("gmailDash").classList.add("hidden");
  $("gmailLoginCard").classList.remove("hidden");
  $("gEmail").value = "";
  $("gPass").value = "";
  resetGmailPanel();
  hideAlert("gmailLoginAlert");
  hideAlert("gmailDashAlert");
  await refreshDashboard();
}

function resetGmailPanel() {
  state.selectedFakeIds.clear();
  $("gmailProgress").classList.add("hidden");
  $("gTotal").textContent = "0";
  $("gFake").textContent = "0";
  $("gSusp").textContent = "0";
  $("gReal").textContent = "0";
  $("gDelBtn").disabled = true;
  $("gmailMsgs").innerHTML = '<div class="activity-empty">Click Scan Inbox to analyze your recent Gmail messages.</div>';
}

function showAlert(id, message, type = "error") {
  const element = $(id);
  element.innerHTML = escapeHtml(message);
  element.className = `alert ${type}`;
  element.classList.remove("hidden");
}

function hideAlert(id) {
  $(id).classList.add("hidden");
}

function setButtonBusy(button, isBusy, label) {
  button.disabled = isBusy;
  if (button.dataset.defaultLabel === undefined) {
    button.dataset.defaultLabel = button.innerHTML;
  }
  button.innerHTML = isBusy ? label : label || button.dataset.defaultLabel;
  if (!isBusy) {
    button.innerHTML = label || button.dataset.defaultLabel;
  }
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch (error) {
    throw new Error("Server returned an unexpected response.");
  }

  if (!response.ok || data.error) {
    throw new Error(data.error || "Request failed.");
  }

  return data;
}

function downloadFile(path) {
  window.location.href = path;
}

function badgeClassFromLabel(label) {
  if (label === "Fake") return "fake-badge";
  if (label === "Suspicious") return "suspicious-badge";
  return "real-badge";
}

function labelToneClass(label) {
  if (label === "Fake") return "danger-text";
  if (label === "Suspicious") return "warn-text";
  return "success-text";
}

function badgeTextFromLabel(label) {
  if (label === "Fake") return "Fake Threat";
  if (label === "Suspicious") return "Suspicious";
  return "Looks Real";
}

function badgeTextShort(label) {
  if (label === "Fake") return "Fake";
  if (label === "Suspicious") return "Review";
  return "Real";
}

function labelColor(label) {
  if (label === "Fake") return "#ff6b6b";
  if (label === "Suspicious") return "#ffbf47";
  return "#4ff0a6";
}

function compactNumber(value) {
  if (value === null || value === undefined) {
    return "--";
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}K`;
  }
  return String(value);
}

function formatPercent(value) {
  const numeric = Number(value || 0);
  return `${numeric.toFixed(numeric % 1 === 0 ? 0 : 2)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function $(id) {
  return document.getElementById(id);
}

