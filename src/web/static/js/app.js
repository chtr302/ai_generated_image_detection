const IMAGE_LIMIT_BYTES = 10 * 1024 * 1024;
const SUPPORTED_IMAGES = new Set(["jpg", "jpeg", "png", "webp"]);
const THEME_KEY = "ai_image_detection_theme";

const state = {
  selectedFile: null,
  selectedUrl: "",
  modelEnabled: true,
  lastRenderedModelEnabled: null,
  lastResult: null,
  busy: false,
};

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function extensionOf(name) {
  const clean = String(name || "").split("?")[0].split("#")[0];
  return clean.includes(".") ? clean.split(".").pop().toLowerCase() : "";
}

async function apiPostForm(url, formData) {
  const response = await fetch(url, { method: "POST", body: formData });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

async function apiGetJson(url) {
  const response = await fetch(url);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function systemTheme() {
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function savedTheme() {
  return localStorage.getItem(THEME_KEY);
}

function setTheme(theme, shouldSave = false) {
  const resolvedTheme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = resolvedTheme;

  const themeColor = $("themeColor");
  if (themeColor) {
    themeColor.setAttribute("content", resolvedTheme === "light" ? "#f8fafc" : "#171717");
  }

  const toggle = $("themeToggle");
  const toggleText = $("themeToggleText");
  if (toggle && toggleText) {
    const isLight = resolvedTheme === "light";
    toggle.setAttribute("aria-pressed", String(isLight));
    toggle.setAttribute("aria-label", isLight ? "Switch to dark mode" : "Switch to light mode");
    toggleText.textContent = isLight ? "Light" : "Dark";
  }

  if (shouldSave) {
    localStorage.setItem(THEME_KEY, resolvedTheme);
  }
}

function initTheme() {
  setTheme(savedTheme() || document.documentElement.dataset.theme || systemTheme());

  $("themeToggle").addEventListener("click", () => {
    const currentTheme = document.documentElement.dataset.theme === "light" ? "light" : "dark";
    setTheme(currentTheme === "light" ? "dark" : "light", true);
  });

  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
    if (!savedTheme()) {
      setTheme(systemTheme());
    }
  });
}

function setMessage(text, kind = "") {
  const el = $("message");
  el.textContent = text || "";
  el.className = `message ${kind}`.trim();
}

function labelText(label) {
  return label === "AI-generated" ? "Anh AI" : "Anh that";
}

function resultClass(label) {
  return label === "AI-generated" ? "result-ai" : "result-real";
}

function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function setBusy(active, title = "", copy = "") {
  state.busy = active;
  document.body.classList.toggle("is-busy", active);

  const stage = $("previewStage");
  if (!stage) return;

  stage.querySelector(".preview-loading-overlay")?.remove();
  if (!active) return;

  const overlay = document.createElement("div");
  overlay.className = "preview-loading-overlay";
  overlay.setAttribute("role", "status");
  overlay.setAttribute("aria-live", "polite");
  overlay.innerHTML = `
    <span class="loader-ring" aria-hidden="true"></span>
    <strong>${escapeHtml(title || "Dang xu ly")}</strong>
    <small>${escapeHtml(copy || "Vui long cho trong giay lat.")}</small>
  `;
  stage.appendChild(overlay);
}

function setButtonBusy(button, active, text = "") {
  if (!button) return;
  if (active) {
    button.dataset.idleText = button.textContent;
    button.textContent = text || "Dang xu ly...";
  } else {
    button.textContent = button.dataset.idleText || button.textContent;
    delete button.dataset.idleText;
  }
}

function updateSelectedInput(text) {
  const selectedInput = $("selectedInput");
  selectedInput.textContent = text || "Chua chon anh.";
  selectedInput.title = selectedInput.textContent;
}

function setModelControlsDisabled(disabled) {
  ["imageInput", "urlInput", "loadUrlButton", "analyzeButton"].forEach((id) => {
    const el = $(id);
    if (el) el.disabled = disabled;
  });

  const dropZone = $("dropZone");
  if (dropZone) {
    dropZone.classList.toggle("is-disabled", disabled);
  }

  document.body.classList.toggle("model-disabled", disabled);

  const inputPanel = document.querySelector(".input-panel");
  if (inputPanel) inputPanel.setAttribute("aria-disabled", String(disabled));
}

function resetAnalysisUi(disabled = false) {
  state.selectedFile = null;
  state.selectedUrl = "";
  state.lastResult = null;

  const imageInput = $("imageInput");
  const urlInput = $("urlInput");
  if (imageInput) imageInput.value = "";
  if (urlInput) urlInput.value = "";

  updateSelectedInput(disabled ? "Model OFF - input bi khoa." : "Chua chon anh.");

  $("previewStage").className = "preview-stage";
  $("previewStage").innerHTML = `
    <div class="empty-visual">
      <strong>${disabled ? "MODEL OFF" : "Chua co anh"}</strong>
      <span>${disabled ? "Chuc nang load anh va phan tich hien dang tam khoa." : "Vui long chon anh hoac nhap URL."}</span>
    </div>
  `;

  $("summaryCards").innerHTML = `
    <article class="summary-card">
      <span>Nhan</span>
      <strong>-</strong>
      <small>${disabled ? "Model OFF" : "Dang cho anh"}</small>
    </article>
    <article class="summary-card">
      <span>Diem AI</span>
      <strong>-</strong>
      <small>${disabled ? "Da khoa" : "Xac suat anh AI"}</small>
    </article>
    <article class="summary-card">
      <span>Xu ly</span>
      <strong>-</strong>
      <small>${disabled ? "Khong kha dung" : "Thoi gian xu ly"}</small>
    </article>
  `;
}

function renderModelState(modelState) {
  const enabled = Boolean(modelState.enabled);
  const previous = state.lastRenderedModelEnabled;
  state.modelEnabled = enabled;
  state.lastRenderedModelEnabled = enabled;
  setModelControlsDisabled(!enabled);

  if (!enabled && previous !== false) {
    resetAnalysisUi(true);
    setMessage("Model dang tat. Chuc nang load anh va phan tich hien dang tam khoa.", "warning");
  } else if (enabled && previous === false) {
    resetAnalysisUi(false);
    setMessage("");
  }
}

async function refreshModelState() {
  const modelState = await apiGetJson("/api/model-state");
  renderModelState(modelState);
}

function validateImageFile(file) {
  if (!file) return { ok: false, error: "Vui long chon anh truoc." };
  const ext = extensionOf(file.name);
  if (!SUPPORTED_IMAGES.has(ext)) {
    return { ok: false, error: "Dinh dang khong ho tro. Chi nhan jpg, jpeg, png, webp." };
  }
  if (file.size > IMAGE_LIMIT_BYTES) {
    return { ok: false, error: "Anh qua lon. Gioi han la 10 MB." };
  }
  return { ok: true };
}

function validateImageUrl(url) {
  if (!url) return { ok: false, error: "Vui long chon anh hoac nhap URL anh." };
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return { ok: false, error: "URL khong hop le." };
  }
  if (!["http:", "https:"].includes(parsed.protocol)) {
    return { ok: false, error: "URL phai dung http hoac https." };
  }
  return { ok: true };
}

function renderPreview(src, name = "", result = null) {
  const stage = $("previewStage");
  stage.className = `preview-stage ${result ? resultClass(result.final_label) : ""}`.trim();
  stage.innerHTML = `
    <img id="previewImage" src="${escapeHtml(src)}" alt="${escapeHtml(name || "Selected image")}">
    ${
      result
        ? `<div class="preview-result-ribbon ${resultClass(result.final_label)}">
            <span>${escapeHtml(labelText(result.final_label))}</span>
            <strong>AI ${percent(result.final_score)}</strong>
          </div>`
        : ""
    }
  `;
}

function handleFile(file) {
  if (!state.modelEnabled) {
    setMessage("Model dang tat. Khong the load anh.", "warning");
    return;
  }

  const validation = validateImageFile(file);
  if (!validation.ok) {
    setMessage(validation.error, "error");
    return;
  }

  state.selectedFile = file;
  state.selectedUrl = "";
  state.lastResult = null;
  $("urlInput").value = "";

  const reader = new FileReader();
  reader.onload = () => renderPreview(reader.result, file.name);
  reader.readAsDataURL(file);
  updateSelectedInput(`File: ${file.name}`);
  setMessage(`${file.name} da san sang de phan tich.`, "success");
  renderSummaryIdle("San sang", "Bam Phan tich de chay model");
}

async function loadUrlPreview() {
  if (!state.modelEnabled) {
    setMessage("Model dang tat. Khong the load URL.", "warning");
    return;
  }

  const urlValue = $("urlInput").value.trim();
  const validation = validateImageUrl(urlValue);
  if (!validation.ok) {
    setMessage(validation.error, "error");
    updateSelectedInput("URL chua hop le.");
    return;
  }

  const formData = new FormData();
  formData.append("url", urlValue);

  const loadButton = $("loadUrlButton");
  const analyzeButton = $("analyzeButton");
  loadButton.disabled = true;
  analyzeButton.disabled = true;
  setButtonBusy(loadButton, true, "Dang load...");
  setBusy(true, "Dang load anh", "Dang tai preview tu URL.");
  setMessage("Dang load anh tu URL...");
  updateSelectedInput(`URL: ${urlValue}`);

  try {
    const result = await apiPostForm("/api/load-url", formData);
    state.selectedFile = null;
    state.selectedUrl = urlValue;
    state.lastResult = null;
    $("imageInput").value = "";
    renderPreview(result.preview_data_url, result.input_name || "Remote image URL");
    renderSummaryIdle("Da load anh", "Bam Phan tich de chay model");
    setMessage(`Da load anh URL (${Math.round(Number(result.size_bytes || 0) / 1024)} KB).`, "success");
  } catch (error) {
    setMessage(error.message, "error");
    updateSelectedInput("Khong load duoc URL.");
    refreshModelState().catch(() => {});
  } finally {
    setBusy(false);
    setButtonBusy(loadButton, false);
    loadButton.disabled = !state.modelEnabled;
    analyzeButton.disabled = !state.modelEnabled;
  }
}

function decisionText(status) {
  const map = {
    selected_highest_confidence: "Chon model tu tin nhat",
  };
  return map[status] || status;
}

function renderSummary(result) {
  const labelClass = result.final_label === "AI-generated" ? "label-ai" : "label-real";
  const selectedModel = result.selected_model?.name || "-";
  const aiScore = Number(result.final_score || 0);
  const processingMs = Number(result.processing_time_ms || 0);
  $("summaryCards").innerHTML = `
    <article class="summary-card primary-result ${resultClass(result.final_label)}">
      <span>Nhan</span>
      <strong><span class="label-pill ${labelClass}">${escapeHtml(labelText(result.final_label))}</span></strong>
      <small>${escapeHtml(selectedModel)}</small>
    </article>
    <article class="summary-card">
      <span>Diem AI</span>
      <strong>${percent(aiScore)}</strong>
      <small>${aiScore.toFixed(6)}</small>
    </article>
    <article class="summary-card">
      <span>Xu ly</span>
      <strong>${processingMs} ms</strong>
      <small>${escapeHtml(decisionText(result.decision_status))}</small>
    </article>
  `;
}

function renderSummaryIdle(title = "Dang cho anh", copy = "Chua co ket qua") {
  $("summaryCards").innerHTML = `
    <article class="summary-card">
      <span>Nhan</span>
      <strong>-</strong>
      <small>${escapeHtml(title)}</small>
    </article>
    <article class="summary-card">
      <span>Diem AI</span>
      <strong>-</strong>
      <small>${escapeHtml(copy)}</small>
    </article>
    <article class="summary-card">
      <span>Xu ly</span>
      <strong>-</strong>
      <small>Chua chay model</small>
    </article>
  `;
}

function renderSummaryLoading() {
  $("summaryCards").innerHTML = `
    <article class="summary-card loading-card">
      <span>Nhan</span>
      <strong>Dang phan tich</strong>
      <small>Dang cho model tra ket qua</small>
    </article>
    <article class="summary-card loading-card">
      <span>Diem AI</span>
      <strong>...</strong>
      <small>Dang tinh score</small>
    </article>
    <article class="summary-card loading-card">
      <span>Xu ly</span>
      <strong>...</strong>
      <small>Dang xu ly anh</small>
    </article>
  `;
}

function detailValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return value ? "Co" : "Khong";
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

function detailRow(label, value) {
  return `
    <div class="model-detail-row">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(detailValue(value))}</strong>
    </div>
  `;
}

function resultStatusText(result) {
  if (!result) return "Chua co ket qua";
  if (result.low_confidence) return "Can xem xet them";
  return "Da hoan tat";
}

function renderImageResultDetail() {
  const result = state.lastResult;
  if (!result) {
    return `
      <section class="popup-section image-result-empty">
        <div class="popup-section-header">
          <span>Ket qua anh</span>
          <strong>Chua phan tich</strong>
        </div>
        <p>Chon anh va bam Phan tich de hien thi ket qua tai day.</p>
      </section>
    `;
  }

  const score = Number(result.final_score || 0);
  const confidence = Number(result.selected_confidence || 0);
  const percent = Math.round(confidence * 100);
  const selectedModel = result.selected_model?.name || "-";
  const labelClass = result.final_label === "AI-generated" ? "label-ai" : "label-real";
  return `
    <section class="popup-section image-result-detail">
      <div class="popup-section-header">
        <span>Ket qua anh</span>
        <strong>${escapeHtml(resultStatusText(result))}</strong>
      </div>
      <div class="result-highlight">
        <div>
          <span>Nhan du doan</span>
          <strong><span class="label-pill ${labelClass}">${escapeHtml(result.final_label)}</span></strong>
        </div>
        <div>
          <span>Model chon</span>
          <strong>${escapeHtml(selectedModel)}</strong>
        </div>
        <div>
          <span>Confidence</span>
          <strong>${confidence.toFixed(6)}</strong>
        </div>
      </div>
      <div class="score-meter" aria-label="Confidence ${percent}%">
        <span style="width:${Math.max(2, Math.min(100, percent))}%"></span>
      </div>
      <p>Score AI cua model duoc chon: ${score.toFixed(6)}. Thoi gian phan tich: ${Number(result.processing_time_ms || 0)} ms.</p>
      ${result.warning ? `<p class="message warning">${escapeHtml(result.warning)}</p>` : ""}
    </section>
  `;
}

function renderModelScoreDetail() {
  const scores = state.lastResult?.model_scores || [];
  if (!scores.length) return "";

  const rows = scores.map((item) => {
    const voteClass = item.vote === "AI-generated" ? "vote-ai" : "vote-real";
    return `
      <tr>
        <td>${escapeHtml(item.model)}</td>
        <td>${escapeHtml(Array.isArray(item.raw_output) ? item.raw_output.join(", ") : "-")}</td>
        <td>${Number(item.prob_real).toFixed(6)}</td>
        <td>${Number(item.prob_ai).toFixed(6)}</td>
        <td>${Number(item.confidence).toFixed(6)}</td>
        <td><span class="vote-pill ${voteClass}">${escapeHtml(item.vote)}</span></td>
      </tr>
    `;
  }).join("");

  return `
    <section class="popup-section">
      <div class="popup-section-header">
        <span>Output tung model</span>
        <strong>${scores.length} models</strong>
      </div>
      <div class="table-shell model-score-table">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Raw output</th>
              <th>Real</th>
              <th>AI</th>
              <th>Confidence</th>
              <th>Vote</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>
  `;
}

function renderConfiguredModels(models) {
  if (!Array.isArray(models) || !models.length) return "";
  const rows = models.map((model) => {
    const statusClass = model.found ? "vote-real" : "vote-ai";
    const statusText = model.found ? "Da load file" : "Thieu file";
    return `
      <tr>
        <td>${escapeHtml(model.name)}</td>
        <td>${escapeHtml(model.filename)}</td>
        <td>${Number(model.image_size)}px</td>
        <td><span class="vote-pill ${statusClass}">${statusText}</span></td>
      </tr>
    `;
  }).join("");

  return `
    <section class="popup-section">
      <div class="popup-section-header">
        <span>Danh sach model ONNX</span>
        <strong>${models.length} models</strong>
      </div>
      <div class="table-shell model-score-table">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>File</th>
              <th>Input</th>
              <th>Trang thai</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>
  `;
}

function renderModelDetail(detail) {
  const status = detail.model_enabled ? "" : `
    <div class="modal-warning">
      <strong>MODEL OFF</strong>
      <span>Model dang tat, chuc nang load anh va phan tich hien dang tam khoa.</span>
    </div>
  `;

  $("modelDetailContent").innerHTML = `
    ${status}
    ${renderImageResultDetail()}
    ${renderModelScoreDetail()}
    ${renderConfiguredModels(detail.models)}
  `;
}

function closeModelDetail() {
  const modal = $("modelDetailModal");
  modal.hidden = true;
  document.body.classList.remove("modal-open");
}

async function openModelDetail() {
  const modal = $("modelDetailModal");
  $("modelDetailContent").innerHTML = `<p>Dang tai thong so model...</p>`;
  modal.hidden = false;
  document.body.classList.add("modal-open");

  try {
    renderModelDetail(await apiGetJson("/api/model-detail"));
  } catch (error) {
    $("modelDetailContent").innerHTML = `<p class="message error">${escapeHtml(error.message)}</p>`;
  }
}

async function runAnalyze(event) {
  event.preventDefault();
  if (!state.modelEnabled) {
    setMessage("Model dang tat. Khong the phan tich.", "warning");
    return;
  }

  const urlValue = $("urlInput").value.trim();
  const selectedUrl = urlValue || state.selectedUrl;
  const validation = state.selectedFile ? validateImageFile(state.selectedFile) : validateImageUrl(selectedUrl);

  if (!validation.ok) {
    setMessage(validation.error, "error");
    return;
  }

  if (urlValue && !state.selectedFile) {
    if (state.selectedUrl !== urlValue) {
      setMessage("Vui long bam Load URL de hien thi anh truoc khi phan tich.", "warning");
      updateSelectedInput(`URL chua load: ${urlValue}`);
      return;
    }
    state.selectedUrl = urlValue;
    updateSelectedInput(`URL: ${urlValue}`);
  }

  const formData = new FormData();
  if (state.selectedFile) {
    formData.append("image", state.selectedFile);
  } else {
    formData.append("url", selectedUrl);
  }

  $("analyzeButton").disabled = true;
  setButtonBusy($("analyzeButton"), true, "Dang phan tich...");
  setBusy(true, "Dang phan tich anh", "Model dang xu ly, ket qua se hien ngay khi xong.");
  renderSummaryLoading();
  setMessage("Dang phan tich anh...");
  await sleep(150);

  try {
    const result = await apiPostForm("/api/analyze", formData);
    state.lastResult = result;
    if (result.preview_data_url) {
      renderPreview(result.preview_data_url, result.input_name || "Remote image URL", result);
    } else {
      const currentImage = $("previewImage");
      if (currentImage) {
        renderPreview(currentImage.src, result.input_name || "Selected image", result);
      }
    }
    renderSummary(result);
    setMessage(`Hoan tat: ${labelText(result.final_label)} - diem AI ${percent(result.final_score)}.`, result.warning ? "warning" : "success");
  } catch (error) {
    setMessage(error.message, "error");
    renderSummaryIdle("Phan tich loi", "Vui long thu lai");
    refreshModelState().catch(() => {});
  } finally {
    setBusy(false);
    setButtonBusy($("analyzeButton"), false);
    $("analyzeButton").disabled = !state.modelEnabled;
  }
}

function bindEvents() {
  initTheme();
  refreshModelState().catch((error) => setMessage(error.message, "error"));
  window.setInterval(() => {
    refreshModelState().catch(() => {});
  }, 5000);

  $("imageInput").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) handleFile(file);
  });

  $("urlInput").addEventListener("input", (event) => {
    if (!state.modelEnabled) return;
    const value = event.target.value.trim();
    state.selectedFile = null;
    $("imageInput").value = "";
    state.selectedUrl = "";
    state.lastResult = null;
    $("previewStage").className = "preview-stage";
    renderSummaryIdle(value ? "Dang cho Load URL" : "Dang cho anh", value ? "Can load preview truoc" : "Chua co ket qua");
    setMessage(value ? "Bam Load URL de hien thi anh truoc khi phan tich." : "");
    updateSelectedInput(value ? `URL: ${value}` : "Chua chon anh.");
  });

  const dropZone = $("dropZone");
  ["dragenter", "dragover"].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      if (!state.modelEnabled) return;
      dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.remove("is-dragging");
    });
  });
  dropZone.addEventListener("drop", (event) => {
    if (!state.modelEnabled) return;
    const file = event.dataTransfer.files?.[0];
    if (file) handleFile(file);
  });

  $("loadUrlButton").addEventListener("click", loadUrlPreview);
  $("analyzeForm").addEventListener("submit", runAnalyze);
  $("modelDetailButton").addEventListener("click", openModelDetail);
  $("modelDetailClose").addEventListener("click", closeModelDetail);
  $("modelDetailModal").addEventListener("click", (event) => {
    if (event.target?.hasAttribute("data-modal-close")) closeModelDetail();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("modelDetailModal").hidden) closeModelDetail();
  });
}

document.addEventListener("DOMContentLoaded", bindEvents);
