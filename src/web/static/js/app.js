const IMAGE_LIMIT_BYTES = 10 * 1024 * 1024;
const SUPPORTED_IMAGES = new Set(["jpg", "jpeg", "png", "webp"]);
const THEME_KEY = "ai_image_detection_theme";
const STATIC_VERSION = "20260817-mobile-intro-no-repo";

const state = {
  selectedFile: null,
  selectedUrl: "",
  urlCheckTimer: null,
  urlCheckSeq: 0,
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

function ensureFreshStylesheet() {
  if (document.querySelector(`link[data-aigid-version="${STATIC_VERSION}"]`)) return;

  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = `/static/css/styles.css?v=${STATIC_VERSION}`;
  link.dataset.aigidVersion = STATIC_VERSION;
  document.head.appendChild(link);
}

function extensionOf(name) {
  const clean = String(name || "").split("?")[0].split("#")[0];
  return clean.includes(".") ? clean.split(".").pop().toLowerCase() : "";
}

async function apiPostForm(url, formData) {
  const response = await fetch(url, { method: "POST", body: formData });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || "Yêu cầu không thực hiện được. Vui lòng thử lại.");
  }
  return data;
}

async function apiGetJson(url) {
  const response = await fetch(url);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || "Yêu cầu không thực hiện được. Vui lòng thử lại.");
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
    toggle.setAttribute("aria-label", isLight ? "Chuyển sang giao diện tối" : "Chuyển sang giao diện sáng");
    toggleText.textContent = isLight ? "Sáng" : "Tối";
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
  el.title = text || "";
  el.className = `message ${kind}`.trim();
}

function isAiLabel(label) {
  return ["ai-generated", "ai", "fake", "synthetic"].includes(String(label || "").trim().toLowerCase());
}

function labelText(label) {
  return isAiLabel(label) ? "Ảnh AI" : "Ảnh thật";
}

function resultClass(label) {
  return isAiLabel(label) ? "result-ai" : "result-real";
}

function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function setBusy(active, title = "", copy = "") {
  state.busy = active;
  document.body.classList.toggle("is-busy", active);
  syncControlState();

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
    <strong>${escapeHtml(title || "Đang xử lý")}</strong>
    <small>${escapeHtml(copy || "Vui lòng chờ trong giây lát.")}</small>
  `;
  stage.appendChild(overlay);
}

function setButtonBusy(button, active, text = "") {
  if (!button) return;
  if (active) {
    if (!button.dataset.idleText) {
      button.dataset.idleText = button.textContent;
    }
    button.textContent = text || "Đang xử lý...";
  } else {
    button.textContent = button.dataset.idleText || button.textContent;
    delete button.dataset.idleText;
  }
}

function updateSelectedInput(text) {
  const selectedInput = $("selectedInput");
  selectedInput.textContent = text || "Chưa chọn ảnh.";
  selectedInput.title = selectedInput.textContent;
}

function setText(selector, text) {
  const el = document.querySelector(selector);
  if (el) el.textContent = text;
}

function localizeStaticText() {
  setText(".hero h1", "Nhận diện ảnh thật và ảnh AI");
  setText(
    ".hero > p:not(.eyebrow)",
    "Hệ thống hỗ trợ nhận diện dấu hiệu ảnh do AI tạo bằng cách phân tích nội dung ảnh và tổng hợp kết quả từ model."
  );

  const heroItems = document.querySelectorAll(".hero .hero-guide li");
  [
    "Tải ảnh lên hoặc dán URL ảnh cần kiểm tra.",
    "Xác nhận ảnh xem trước, sau đó bấm Phân tích.",
    "Xem kết luận, điểm AI và chi tiết xử lý.",
  ].forEach((text, index) => {
    if (heroItems[index]) heroItems[index].textContent = text;
  });

  setText("#introModalTitle", "Nhận diện ảnh thật và ảnh AI");
  setText(
    ".intro-modal-content > p",
    "Hệ thống hỗ trợ nhận diện dấu hiệu ảnh do AI tạo bằng cách phân tích nội dung ảnh và tổng hợp kết quả từ model."
  );
  const introItems = document.querySelectorAll(".intro-modal .hero-guide li");
  [
    "Tải ảnh lên hoặc dán URL ảnh cần kiểm tra.",
    "Xác nhận ảnh xem trước, sau đó bấm Phân tích.",
    "Xem kết luận, điểm AI và chi tiết xử lý.",
  ].forEach((text, index) => {
    if (introItems[index]) introItems[index].textContent = text;
  });
  setText("#introModalClose", "Đã hiểu");

  setText(".result-panel .panel-header h2", "Kết quả");
  setText(".result-panel .panel-header p", "Dự đoán cuối cùng và thông số xử lý.");
  setText("#modelDetailButton", "Chi tiết kết quả");
  setText(".input-panel .panel-header h2", "Ảnh đầu vào");
  setText(".input-panel .panel-header p", "Hỗ trợ jpg, jpeg, png, webp. Tối đa 10 MB.");
  setText(".drop-title", "Bấm để chọn ảnh");
  setText(".drop-copy", "hoặc kéo ảnh vào khung này.");
  setText(".drop-action", "Chọn ảnh từ máy");
  setText(".field-group label", "Hoặc nhập URL ảnh");
  setText("#analyzeButton", "Phân tích");

  const selectedInput = $("selectedInput");
  if (selectedInput && /Chua|Chưa|chon|chọn|anh|ảnh/i.test(selectedInput.textContent)) {
    updateSelectedInput("Chưa chọn ảnh.");
  }

  const summaryCards = document.querySelectorAll("#summaryCards .summary-card");
  const summaryText = [
    ["Nhãn", "-", "Đang chờ ảnh"],
    ["Điểm AI", "-", "Xác suất ảnh AI"],
    ["Xử lý", "-", "Thời gian xử lý"],
  ];
  summaryCards.forEach((card, index) => {
    const [label, value, copy] = summaryText[index] || [];
    if (!label) return;
    const span = card.querySelector("span");
    const strong = card.querySelector("strong");
    const small = card.querySelector("small");
    if (span) span.textContent = label;
    if (strong && strong.textContent.trim() === "-") strong.textContent = value;
    if (small) small.textContent = copy;
  });

  setText("#previewPanelTitle", "Ảnh đã tải");
  setText("#previewPanelSubtitle", "Chưa có ảnh để xem trước.");
  const clearButton = $("clearImageButton");
  if (clearButton) {
    clearButton.setAttribute("aria-label", "Loại ảnh đang tải");
    clearButton.title = "Loại ảnh đang tải";
  }

  const emptyVisual = document.querySelector("#previewStage .empty-visual");
  if (emptyVisual) {
    setText("#previewStage .empty-visual strong", "Chưa có ảnh");
    setText("#previewStage .empty-visual span", "Vui lòng chọn ảnh hoặc nhập URL.");
  }

  setText("#modelDetailTitle", "Chi tiết kết quả");
  setText("#modelDetailContent p", "Đang tải chi tiết kết quả...");
  const modelClose = $("modelDetailClose");
  if (modelClose) modelClose.setAttribute("aria-label", "Đóng popup");
}

function setPreviewPanelState(title = "Ảnh đã tải", subtitle = "Chưa có ảnh để xem trước.", canClear = false) {
  const titleEl = $("previewPanelTitle");
  const subtitleEl = $("previewPanelSubtitle");
  const clearButton = $("clearImageButton");

  if (titleEl) titleEl.textContent = title;
  if (subtitleEl) {
    subtitleEl.textContent = subtitle;
    subtitleEl.title = subtitle;
  }
  if (clearButton) {
    clearButton.hidden = !canClear;
    clearButton.disabled = state.busy || !canClear;
  }
}

function analyzeUnavailableReason() {
  if (!state.modelEnabled) return "Model đang tắt nên chưa thể phân tích.";
  if (state.busy) return "";

  if (state.selectedFile) {
    const validation = validateImageFile(state.selectedFile);
    return validation.ok ? "" : validation.error;
  }

  const urlValue = $("urlInput")?.value.trim() || "";
  if (!urlValue) return "Chưa có ảnh để phân tích. Hãy chọn ảnh hoặc nhập URL ảnh.";

  const validation = validateImageUrl(urlValue);
  if (!validation.ok) return validation.error;
  if (state.selectedUrl !== urlValue) return "URL chưa được kiểm tra xong. Vui lòng chờ trong giây lát.";
  return "";
}

function syncControlState() {
  const inputLocked = state.busy || !state.modelEnabled;
  const analyzeReason = analyzeUnavailableReason();
  const disabledById = {
    imageInput: inputLocked,
    urlInput: inputLocked,
    analyzeButton: inputLocked,
    modelDetailButton: state.busy,
    modelDetailClose: state.busy,
    themeToggle: state.busy,
    clearImageButton: state.busy || $("clearImageButton")?.hidden,
  };

  Object.entries(disabledById).forEach(([id, disabled]) => {
    const el = $(id);
    if (el) el.disabled = disabled;
  });

  const analyzeButton = $("analyzeButton");
  if (analyzeButton) {
    analyzeButton.classList.toggle("is-soft-locked", Boolean(analyzeReason));
    analyzeButton.setAttribute("aria-disabled", String(Boolean(analyzeReason || inputLocked)));
    if (analyzeReason) {
      analyzeButton.setAttribute("aria-describedby", "analyzeTooltip");
    } else {
      analyzeButton.removeAttribute("aria-describedby");
    }
  }

  const actionRow = document.querySelector(".action-row");
  const analyzeTooltip = $("analyzeTooltip");
  if (actionRow && analyzeTooltip) {
    actionRow.classList.toggle("has-warning", Boolean(analyzeReason));
    analyzeTooltip.hidden = !analyzeReason;
    analyzeTooltip.textContent = analyzeReason;
  }

  const dropZone = $("dropZone");
  if (dropZone) {
    dropZone.classList.toggle("is-disabled", inputLocked);
  }

  document.body.classList.toggle("model-disabled", !state.modelEnabled);

  const inputPanel = document.querySelector(".input-panel");
  if (inputPanel) inputPanel.setAttribute("aria-disabled", String(inputLocked));
}

function clearUrlCheckTimer() {
  if (state.urlCheckTimer) {
    window.clearTimeout(state.urlCheckTimer);
    state.urlCheckTimer = null;
  }
}

function resetAnalysisUi(disabled = false) {
  state.selectedFile = null;
  state.selectedUrl = "";
  state.lastResult = null;
  clearUrlCheckTimer();
  state.urlCheckSeq += 1;

  const imageInput = $("imageInput");
  const urlInput = $("urlInput");
  if (imageInput) imageInput.value = "";
  if (urlInput) urlInput.value = "";

  updateSelectedInput(disabled ? "Model tắt - đầu vào bị khóa." : "Chưa chọn ảnh.");

  renderEmptyPreview(
    disabled ? "MODEL TẮT" : "Chưa có ảnh",
    disabled ? "Chức năng tải ảnh và phân tích hiện đang tạm khóa." : "Vui lòng chọn ảnh hoặc nhập URL."
  );
  setPreviewPanelState(
    disabled ? "Model tắt" : "Ảnh đã tải",
    disabled ? "Chức năng xem trước ảnh hiện đang tạm khóa." : "Chưa có ảnh để xem trước.",
    false
  );

  $("summaryCards").innerHTML = `
    <article class="summary-card">
      <span>Nhãn</span>
      <strong>-</strong>
      <small>${disabled ? "Model tắt" : "Đang chờ ảnh"}</small>
    </article>
    <article class="summary-card">
      <span>Điểm AI</span>
      <strong>-</strong>
      <small>${disabled ? "Đã khóa" : "Xác suất ảnh AI"}</small>
    </article>
    <article class="summary-card">
      <span>Xử lý</span>
      <strong>-</strong>
      <small>${disabled ? "Không khả dụng" : "Thời gian xử lý"}</small>
    </article>
  `;
}

function renderModelState(modelState) {
  const enabled = Boolean(modelState.enabled);
  const previous = state.lastRenderedModelEnabled;
  state.modelEnabled = enabled;
  state.lastRenderedModelEnabled = enabled;
  syncControlState();

  if (!enabled && previous !== false) {
    resetAnalysisUi(true);
    setMessage("Model đang tắt. Chức năng tải ảnh và phân tích hiện đang tạm khóa.", "warning");
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
  if (!file) return { ok: false, error: "Vui lòng chọn ảnh trước." };
  const ext = extensionOf(file.name);
  if (!SUPPORTED_IMAGES.has(ext)) {
    return { ok: false, error: "Định dạng không hỗ trợ. Chỉ nhận jpg, jpeg, png, webp." };
  }
  if (file.size > IMAGE_LIMIT_BYTES) {
    return { ok: false, error: "Ảnh quá lớn. Giới hạn là 10 MB." };
  }
  return { ok: true };
}

function validateImageUrl(url) {
  if (!url) return { ok: false, error: "Vui lòng chọn ảnh hoặc nhập URL ảnh." };
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return { ok: false, error: "URL không hợp lệ. Vui lòng nhập đầy đủ dạng https://ten-mien/anh.jpg." };
  }
  if (!["http:", "https:"].includes(parsed.protocol)) {
    return { ok: false, error: "URL phải dùng http hoặc https." };
  }
  if (!parsed.host) {
    return { ok: false, error: "URL thiếu tên miền hoặc host." };
  }
  return { ok: true };
}

function friendlyUrlError(message) {
  const text = String(message || "").trim();
  if (!text) return "Không kiểm tra được URL. Vui lòng thử lại.";
  if (/request failed/i.test(text) || /\b\d{3}\b/.test(text)) {
    return "Không tải được ảnh từ URL này. Vui lòng kiểm tra lại đường dẫn ảnh.";
  }
  return text;
}

function renderEmptyPreview(title = "Chưa có ảnh", copy = "Vui lòng chọn ảnh hoặc nhập URL.") {
  const stage = $("previewStage");
  stage.className = "preview-stage";
  stage.innerHTML = `
    <div class="empty-visual">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(copy)}</span>
    </div>
  `;
}

function renderPreview(src, name = "", result = null) {
  const stage = $("previewStage");
  stage.className = `preview-stage has-image ${result ? resultClass(result.final_label) : ""}`.trim();
  stage.innerHTML = `
    <img id="previewImage" src="${escapeHtml(src)}" alt="${escapeHtml(name || "Ảnh đã chọn")}">
    ${
      result
        ? `<div class="preview-result-ribbon ${resultClass(result.final_label)}">
            <span>${escapeHtml(labelText(result.final_label))}</span>
            <strong>AI ${percent(result.final_score)}</strong>
          </div>`
        : ""
    }
  `;
  setPreviewPanelState(
    result ? "Ảnh đã phân tích" : "Ảnh xem trước",
    result ? "Đã phân tích ảnh xong." : "Đã tải ảnh và sẵn sàng phân tích.",
    true
  );
}

function clearSelectedImage() {
  if (state.busy) return;

  state.selectedFile = null;
  state.selectedUrl = "";
  state.lastResult = null;
  clearUrlCheckTimer();
  state.urlCheckSeq += 1;

  const imageInput = $("imageInput");
  const urlInput = $("urlInput");
  if (imageInput) imageInput.value = "";
  if (urlInput) urlInput.value = "";

  updateSelectedInput("Chưa chọn ảnh.");
  renderEmptyPreview();
  setPreviewPanelState();
  renderSummaryIdle("Đang chờ ảnh", "Chưa có kết quả");
  setMessage("");
  syncControlState();
}

function handleFile(file) {
  if (state.busy) return;

  if (!state.modelEnabled) {
    setMessage("Model đang tắt. Không thể tải ảnh.", "warning");
    return;
  }

  const validation = validateImageFile(file);
  if (!validation.ok) {
    setMessage(validation.error, "error");
    state.selectedFile = null;
    state.selectedUrl = "";
    state.lastResult = null;
    renderEmptyPreview("Tệp không hợp lệ", "Vui lòng chọn ảnh jpg, jpeg, png hoặc webp.");
    setPreviewPanelState("Tệp không hợp lệ", "Không có ảnh nào được tải.", false);
    renderSummaryIdle("Tệp không hợp lệ", "Vui lòng chọn ảnh khác");
    syncControlState();
    return;
  }

  clearUrlCheckTimer();
  state.urlCheckSeq += 1;
  state.selectedFile = file;
  state.selectedUrl = "";
  state.lastResult = null;
  $("urlInput").value = "";

  const reader = new FileReader();
  reader.onload = () => {
    if (state.selectedFile !== file) return;
    renderPreview(reader.result, file.name);
  };
  reader.readAsDataURL(file);
  updateSelectedInput(`Tệp: ${file.name}`);
  setMessage(`${file.name} đã sẵn sàng để phân tích.`, "success");
  renderSummaryIdle("Sẵn sàng", "Bấm Phân tích để chạy model");
  syncControlState();
}

async function loadUrlPreview() {
  if (state.busy) return;

  if (!state.modelEnabled) {
    setMessage("Model đang tắt. Không thể tải URL.", "warning");
    return;
  }

  const urlValue = $("urlInput").value.trim();
  const validation = validateImageUrl(urlValue);
  if (!validation.ok) {
    setMessage(validation.error, "error");
    updateSelectedInput("URL chưa hợp lệ.");
    return;
  }

  const checkSeq = ++state.urlCheckSeq;
  const formData = new FormData();
  formData.append("url", urlValue);

  const analyzeButton = $("analyzeButton");
  setButtonBusy(analyzeButton, true, "Đang kiểm tra...");
  setBusy(true, "Đang kiểm tra URL", "Đang tải bản xem trước để xác thực ảnh.");
  setPreviewPanelState("Đang kiểm tra URL", "Đang tải bản xem trước để xác thực ảnh.", false);
  setMessage("Đang kiểm tra URL ảnh...");
  updateSelectedInput(`URL: ${urlValue}`);

  try {
    const result = await apiPostForm("/api/load-url", formData);
    if (checkSeq !== state.urlCheckSeq) return;

    state.selectedFile = null;
    state.selectedUrl = urlValue;
    state.lastResult = null;
    $("imageInput").value = "";
    renderPreview(result.preview_data_url, result.input_name || "Ảnh từ URL");
    renderSummaryIdle("Đã tải ảnh", "Bấm Phân tích để chạy model");
    setMessage(`URL hợp lệ (${Math.round(Number(result.size_bytes || 0) / 1024)} KB). Có thể bấm Phân tích.`, "success");
  } catch (error) {
    if (checkSeq !== state.urlCheckSeq) return;

    state.selectedUrl = "";
    setMessage(friendlyUrlError(error.message), "error");
    updateSelectedInput("URL không hợp lệ.");
    renderEmptyPreview("URL không hợp lệ", "Vui lòng nhập URL ảnh hợp lệ khác.");
    setPreviewPanelState("URL không hợp lệ", "Không có ảnh nào được tải từ URL này.", false);
    renderSummaryIdle("URL không hợp lệ", "Vui lòng nhập URL ảnh khác");
    refreshModelState().catch(() => {});
  } finally {
    setBusy(false);
    setButtonBusy(analyzeButton, false);
    syncControlState();
  }
}

function scheduleUrlPreviewCheck() {
  clearUrlCheckTimer();
  state.urlCheckSeq += 1;

  if (state.busy || !state.modelEnabled) return;

  const value = $("urlInput").value.trim();
  state.selectedFile = null;
  state.selectedUrl = "";
  state.lastResult = null;
  $("imageInput").value = "";

  if (!value) {
    renderEmptyPreview();
    setPreviewPanelState();
    renderSummaryIdle("Đang chờ ảnh", "Chưa có kết quả");
    setMessage("");
    updateSelectedInput("Chưa chọn ảnh.");
    syncControlState();
    return;
  }

  const validation = validateImageUrl(value);
  if (!validation.ok) {
    renderEmptyPreview("URL chưa hợp lệ", "Vui lòng nhập URL http hoặc https.");
    setPreviewPanelState("URL chưa hợp lệ", "Vui lòng nhập URL ảnh hợp lệ.", false);
    renderSummaryIdle("URL chưa hợp lệ", "Cần URL http hoặc https");
    setMessage(validation.error, "error");
    updateSelectedInput("URL chưa hợp lệ.");
    syncControlState();
    return;
  }

  renderEmptyPreview("Đang kiểm tra URL", "Hệ thống đang tải bản xem trước của ảnh.");
  setPreviewPanelState("Đang kiểm tra URL", "Hệ thống đang tải bản xem trước của ảnh.", false);
  renderSummaryIdle("Đang kiểm tra URL", "Sẽ mở Phân tích nếu ảnh hợp lệ");
  setMessage("URL đúng định dạng. Đang kiểm tra ảnh...");
  updateSelectedInput(`URL đang kiểm tra: ${value}`);
  syncControlState();

  state.urlCheckTimer = window.setTimeout(() => {
    state.urlCheckTimer = null;
    loadUrlPreview();
  }, 650);
}

function decisionText(status) {
  const map = {
    selected_highest_confidence: "Chọn model tự tin nhất",
  };
  return map[status] || status;
}

function renderSummary(result) {
  const selectedModel = result.selected_model?.name || "-";
  const aiScore = Number(result.final_score || 0);
  const processingMs = Number(result.processing_time_ms || 0);
  $("summaryCards").innerHTML = `
    <article class="summary-card primary-result result-label-card ${resultClass(result.final_label)}">
      <span>Nhãn</span>
      <strong>${escapeHtml(labelText(result.final_label))}</strong>
      <small>${escapeHtml(selectedModel)} - AI ${percent(aiScore)}</small>
    </article>
    <article class="summary-card">
      <span>Điểm AI</span>
      <strong>${percent(aiScore)}</strong>
      <small>${aiScore.toFixed(6)}</small>
    </article>
    <article class="summary-card">
      <span>Xử lý</span>
      <strong>${processingMs} ms</strong>
      <small>${escapeHtml(decisionText(result.decision_status))}</small>
    </article>
  `;
}

function renderSummaryIdle(title = "Đang chờ ảnh", copy = "Chưa có kết quả") {
  $("summaryCards").innerHTML = `
    <article class="summary-card">
      <span>Nhãn</span>
      <strong>-</strong>
      <small>${escapeHtml(title)}</small>
    </article>
    <article class="summary-card">
      <span>Điểm AI</span>
      <strong>-</strong>
      <small>${escapeHtml(copy)}</small>
    </article>
    <article class="summary-card">
      <span>Xử lý</span>
      <strong>-</strong>
      <small>Chưa chạy model</small>
    </article>
  `;
}

function renderSummaryLoading() {
  $("summaryCards").innerHTML = `
    <article class="summary-card loading-card">
      <span>Nhãn</span>
      <strong>Đang phân tích</strong>
      <small>Đang chờ model trả kết quả</small>
    </article>
    <article class="summary-card loading-card">
      <span>Điểm AI</span>
      <strong>...</strong>
      <small>Đang tính score</small>
    </article>
    <article class="summary-card loading-card">
      <span>Xử lý</span>
      <strong>...</strong>
      <small>Đang xử lý ảnh</small>
    </article>
  `;
}

function detailValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return value ? "Có" : "Không";
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
  if (!result) return "Chưa có kết quả";
  if (result.low_confidence) return "Cần xem xét thêm";
  return "Đã hoàn tất";
}

function renderImageResultDetail() {
  const result = state.lastResult;
  if (!result) {
    return `
      <section class="popup-section image-result-empty">
        <div class="popup-section-header">
          <span>Kết quả ảnh</span>
          <strong>Chưa phân tích</strong>
        </div>
        <p>Chọn ảnh và bấm Phân tích để hiển thị kết quả tại đây.</p>
      </section>
    `;
  }

  const score = Number(result.final_score || 0);
  const confidence = Number(result.selected_confidence || 0);
  const percent = Math.round(confidence * 100);
  const selectedModel = result.selected_model?.name || "-";
  const labelClass = isAiLabel(result.final_label) ? "label-ai" : "label-real";
  return `
    <section class="popup-section image-result-detail">
      <div class="popup-section-header">
        <span>Kết quả ảnh</span>
        <strong>${escapeHtml(resultStatusText(result))}</strong>
      </div>
      <div class="result-highlight">
        <div>
          <span>Nhãn dự đoán</span>
          <strong><span class="label-pill ${labelClass}">${escapeHtml(labelText(result.final_label))}</span></strong>
        </div>
        <div>
          <span>Model chọn</span>
          <strong>${escapeHtml(selectedModel)}</strong>
        </div>
        <div>
          <span>Độ tin cậy</span>
          <strong>${confidence.toFixed(6)}</strong>
        </div>
      </div>
      <div class="score-meter" aria-label="Độ tin cậy ${percent}%">
        <span style="width:${Math.max(2, Math.min(100, percent))}%"></span>
      </div>
      <p>Score AI của model được chọn: ${score.toFixed(6)}. Thời gian phân tích: ${Number(result.processing_time_ms || 0)} ms.</p>
      ${result.warning ? `<p class="message warning">${escapeHtml(result.warning)}</p>` : ""}
    </section>
  `;
}

function renderModelScoreDetail() {
  const scores = state.lastResult?.model_scores || [];
  if (!scores.length) return "";

  const rows = scores.map((item) => {
    const voteClass = isAiLabel(item.vote) ? "vote-ai" : "vote-real";
    return `
      <tr>
        <td>${escapeHtml(item.model)}</td>
        <td>${escapeHtml(Array.isArray(item.raw_output) ? item.raw_output.join(", ") : "-")}</td>
        <td>${Number(item.prob_real).toFixed(6)}</td>
        <td>${Number(item.prob_ai).toFixed(6)}</td>
        <td>${Number(item.confidence).toFixed(6)}</td>
        <td><span class="vote-pill ${voteClass}">${escapeHtml(labelText(item.vote))}</span></td>
      </tr>
    `;
  }).join("");

  return `
    <section class="popup-section">
      <div class="popup-section-header">
        <span>Output từng model</span>
        <strong>${scores.length} models</strong>
      </div>
      <div class="table-shell model-score-table">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Output thô</th>
              <th>Ảnh thật</th>
              <th>AI</th>
              <th>Độ tin cậy</th>
              <th>Dự đoán</th>
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
    const statusText = model.found ? "Đã tải tệp" : "Thiếu tệp";
    return `
      <tr>
        <td>${escapeHtml(model.name)}</td>
        <td>${escapeHtml(model.filename)}</td>
        <td>${Number(model.image_size)}px</td>
        <td>${escapeHtml(model.output_mode || "-")}</td>
        <td>${escapeHtml(model.score_transform || "none")}</td>
        <td><span class="vote-pill ${statusClass}">${statusText}</span></td>
      </tr>
    `;
  }).join("");

  return `
    <section class="popup-section">
      <div class="popup-section-header">
        <span>Danh sách model ONNX</span>
        <strong>${models.length} models</strong>
      </div>
      <div class="table-shell model-score-table">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Tệp</th>
              <th>Đầu vào</th>
              <th>Output</th>
              <th>Score</th>
              <th>Trạng thái</th>
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
      <span>Model đang tắt, chức năng tải ảnh và phân tích hiện đang tạm khóa.</span>
    </div>
  `;

  $("modelDetailContent").innerHTML = `
    ${status}
    ${renderImageResultDetail()}
    ${renderModelScoreDetail()}
    ${renderConfiguredModels(detail.models)}
  `;
}

function syncModalOpenState() {
  const hasOpenModal = Boolean(document.querySelector(".modal:not([hidden])"));
  document.body.classList.toggle("modal-open", hasOpenModal);
}

function isMobileViewport() {
  return window.matchMedia("(max-width: 700px)").matches;
}

function openIntroModalOnMobile() {
  if (!isMobileViewport()) return;

  const modal = $("introModal");
  if (!modal) return;

  modal.hidden = false;
  syncModalOpenState();
}

function closeIntroModal() {
  const modal = $("introModal");
  if (!modal) return;

  modal.hidden = true;
  syncModalOpenState();
}

function closeModelDetail() {
  if (state.busy) return;

  const modal = $("modelDetailModal");
  modal.hidden = true;
  syncModalOpenState();
}

async function openModelDetail() {
  if (state.busy) return;

  const modal = $("modelDetailModal");
  $("modelDetailContent").innerHTML = `<p>Đang tải chi tiết kết quả...</p>`;
  modal.hidden = false;
  syncModalOpenState();

  try {
    renderModelDetail(await apiGetJson("/api/model-detail"));
  } catch (error) {
    $("modelDetailContent").innerHTML = `<p class="message error">${escapeHtml(error.message)}</p>`;
  }
}

async function runAnalyze(event) {
  event.preventDefault();
  if (state.busy) return;

  if (!state.modelEnabled) {
    setMessage("Model đang tắt. Không thể phân tích.", "warning");
    return;
  }

  const urlValue = $("urlInput").value.trim();
  const selectedUrl = urlValue || state.selectedUrl;
  const unavailableReason = analyzeUnavailableReason();
  if (unavailableReason) {
    setMessage(unavailableReason, "warning");
    updateSelectedInput("Chưa có ảnh sẵn sàng để phân tích.");
    if (urlValue && !state.selectedFile && state.selectedUrl !== urlValue) {
      scheduleUrlPreviewCheck();
    }
    syncControlState();
    return;
  }

  const validation = state.selectedFile ? validateImageFile(state.selectedFile) : validateImageUrl(selectedUrl);

  if (!validation.ok) {
    setMessage(validation.error, "error");
    updateSelectedInput("Chưa có ảnh sẵn sàng để phân tích.");
    syncControlState();
    return;
  }

  if (urlValue && !state.selectedFile) {
    if (state.selectedUrl !== urlValue) {
      setMessage("URL chưa được kiểm tra hợp lệ. Vui lòng chờ hệ thống kiểm tra xong.", "warning");
      updateSelectedInput(`URL chưa kiểm tra: ${urlValue}`);
      scheduleUrlPreviewCheck();
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

  setButtonBusy($("analyzeButton"), true, "Đang phân tích...");
  setBusy(true, "Đang phân tích ảnh", "Model đang xử lý, kết quả sẽ hiện ngay khi xong.");
  renderSummaryLoading();
  setMessage("Đang phân tích ảnh...");
  await sleep(150);

  try {
    const result = await apiPostForm("/api/analyze", formData);
    state.lastResult = result;
    if (result.preview_data_url) {
      renderPreview(result.preview_data_url, result.input_name || "Ảnh từ URL", result);
    } else {
      const currentImage = $("previewImage");
      if (currentImage) {
        renderPreview(currentImage.src, result.input_name || "Ảnh đã chọn", result);
      }
    }
    renderSummary(result);
    setMessage(`Hoàn tất: ${labelText(result.final_label)} - điểm AI ${percent(result.final_score)}.`, result.warning ? "warning" : "success");
  } catch (error) {
    setMessage(error.message, "error");
    renderSummaryIdle("Phân tích lỗi", "Vui lòng thử lại");
    refreshModelState().catch(() => {});
  } finally {
    setBusy(false);
    setButtonBusy($("analyzeButton"), false);
    syncControlState();
  }
}

function bindEvents() {
  ensureFreshStylesheet();
  localizeStaticText();
  initTheme();
  refreshModelState().catch((error) => setMessage(error.message, "error"));
  window.setInterval(() => {
    refreshModelState().catch(() => {});
  }, 5000);

  $("imageInput").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) handleFile(file);
  });

  $("urlInput").addEventListener("input", () => {
    scheduleUrlPreviewCheck();
  });

  const dropZone = $("dropZone");
  ["dragenter", "dragover"].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      if (state.busy || !state.modelEnabled) return;
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
    if (state.busy || !state.modelEnabled) return;
    const file = event.dataTransfer.files?.[0];
    if (file) handleFile(file);
  });

  $("analyzeForm").addEventListener("submit", runAnalyze);
  $("clearImageButton").addEventListener("click", clearSelectedImage);
  $("modelDetailButton").addEventListener("click", openModelDetail);
  $("modelDetailClose").addEventListener("click", closeModelDetail);
  const introModal = $("introModal");
  $("introModalClose")?.addEventListener("click", closeIntroModal);
  introModal?.addEventListener("click", (event) => {
    if (event.target?.hasAttribute("data-intro-modal-close")) closeIntroModal();
  });
  $("modelDetailModal").addEventListener("click", (event) => {
    if (!state.busy && event.target?.hasAttribute("data-modal-close")) closeModelDetail();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const introModal = $("introModal");
    if (introModal && !introModal.hidden) {
      closeIntroModal();
      return;
    }
    if (!state.busy && !$("modelDetailModal").hidden) closeModelDetail();
  });
  syncControlState();
  openIntroModalOnMobile();
}

document.addEventListener("DOMContentLoaded", bindEvents);
