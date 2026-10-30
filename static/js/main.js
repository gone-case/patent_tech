document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tabs = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const filePreview = document.getElementById("file-preview");
  const selectedFileName = document.getElementById("selected-file-name");
  const selectedFileSize = document.getElementById("selected-file-size");
  const removeFileBtn = document.getElementById("remove-file-btn");

  const patentIdInput = document.getElementById("patent-id-input");
  const textQueryInput = document.getElementById("text-query-input");
  const ipcInput = document.getElementById("ipc-input");
  const topKSelect = document.getElementById("top-k-select");
  const searchBtn = document.getElementById("search-btn");

  const agentStepper = document.getElementById("agent-stepper");
  const stepHarvester = document.getElementById("step-harvester");
  const stepRetrieval = document.getElementById("step-retrieval");
  const stepAnalyst = document.getElementById("step-analyst");

  const resultsContainer = document.getElementById("results-container");
  const targetTitle = document.getElementById("target-title");
  const targetSource = document.getElementById("target-source");
  const targetElapsed = document.getElementById("target-elapsed");
  const targetIpcTags = document.getElementById("target-ipc-tags");
  const targetAbstract = document.getElementById("target-abstract");
  const candidatesGrid = document.getElementById("candidates-grid");
  const downloadReportBtn = document.getElementById("download-report-btn");

  let activeMode = "doc"; // "doc" | "patent" | "text"
  let selectedFile = null;

  // 1. Tab Navigation
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      
      tab.classList.add("active");
      activeMode = tab.dataset.mode;
      const targetContent = document.getElementById(`content-${activeMode}`);
      if (targetContent) targetContent.classList.add("active");
    });
  });

  // 2. Drag & Drop File Handling
  browseBtn.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", (e) => {
    if (e.target !== removeFileBtn && !filePreview.contains(e.target)) {
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = formatBytes(file.size);
    filePreview.classList.remove("hidden");
  }

  removeFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";
    filePreview.classList.add("hidden");
  });

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  // 3. Quick Chips & Presets
  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.getElementById("tab-patent").click();
      patentIdInput.value = chip.dataset.patent;
      patentIdInput.focus();
    });
  });

  document.querySelectorAll(".idea-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.getElementById("tab-text").click();
      textQueryInput.value = chip.dataset.idea;
      ipcInput.value = chip.dataset.ipc || "";
      textQueryInput.focus();
    });
  });

  // 4. Multi-Agent Analysis Execution
  searchBtn.addEventListener("click", async () => {
    let queryPayload = null;
    let formData = null;

    if (activeMode === "doc") {
      if (!selectedFile) {
        alert("Please select or drop a document file (.pdf, .docx, .txt, .md, .json) first.");
        return;
      }
      formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("top_k", topKSelect.value);
      if (ipcInput.value.trim()) {
        formData.append("custom_ipc", ipcInput.value.trim());
      }
    } else if (activeMode === "patent") {
      const pId = patentIdInput.value.trim();
      if (!pId) {
        alert("Please enter a patent publication number (e.g. US10123456B2).");
        patentIdInput.focus();
        return;
      }
      queryPayload = {
        query: pId,
        top_k: parseInt(topKSelect.value, 10),
        custom_ipc: ipcInput.value.trim() || undefined
      };
    } else if (activeMode === "text") {
      const text = textQueryInput.value.trim();
      if (!text) {
        alert("Please describe your invention concept or research summary.");
        textQueryInput.focus();
        return;
      }
      queryPayload = {
        query: text,
        top_k: parseInt(topKSelect.value, 10),
        custom_ipc: ipcInput.value.trim() || undefined
      };
    }

    // UI State: Starting Search
    searchBtn.disabled = true;
    searchBtn.innerHTML = `
      <span class="pulsing-dot"></span>
      Running Agents...
    `;
    agentStepper.classList.remove("hidden");
    resultsContainer.classList.add("hidden");

    // Stepper Animation
    setStepState("step-harvester", "active");
    setStepState("step-retrieval", "idle");
    setStepState("step-analyst", "idle");

    const t1 = setTimeout(() => {
      setStepState("step-harvester", "completed");
      setStepState("step-retrieval", "active");
    }, 1200);

    const t2 = setTimeout(() => {
      setStepState("step-retrieval", "completed");
      setStepState("step-analyst", "active");
    }, 2800);

    try {
      let res;
      if (formData) {
        res = await fetch("/api/search", {
          method: "POST",
          body: formData
        });
      } else {
        res = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(queryPayload)
        });
      }

      clearTimeout(t1);
      clearTimeout(t2);

      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || "Search failed");
      }

      // Mark all steps completed
      setStepState("step-harvester", "completed");
      setStepState("step-retrieval", "completed");
      setStepState("step-analyst", "completed");

      // Render Results
      renderResults(data);

    } catch (err) {
      alert(`Error during analysis: ${err.message}`);
    } finally {
      searchBtn.disabled = false;
      searchBtn.innerHTML = `
        <svg class="btn-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
        </svg>
        Run Multi-Agent Analysis
      `;
    }
  });

  function setStepState(stepId, state) {
    const el = document.getElementById(stepId);
    if (!el) return;
    el.classList.remove("active", "completed");
    if (state === "active") el.classList.add("active");
    if (state === "completed") el.classList.add("completed");
  }

  // 5. Render Results to DOM
  function renderResults(data) {
    const target = data.target || {};
    const candidates = data.candidates || [];

    targetTitle.textContent = target.title || "Invention Query";
    targetSource.textContent = `Source: ${target.source || "Input"}`;
    targetElapsed.textContent = `${data.elapsed_seconds || 0}s execution`;
    targetAbstract.textContent = target.abstract || target.meta || "";

    // Target IPC Tags
    targetIpcTags.innerHTML = "";
    (target.codes || []).forEach(code => {
      const tag = document.createElement("span");
      tag.className = "ipc-tag";
      tag.textContent = code;
      targetIpcTags.appendChild(tag);
    });

    // Render Candidates
    candidatesGrid.innerHTML = "";
    candidates.forEach((cand, idx) => {
      const rank = idx + 1;
      const sdtd = (cand.sdtd_score || 0).toFixed(4);
      const sd = (cand.sd_score || 0).toFixed(4);
      const td = (cand.td_score || 0).toFixed(4);

      let riskClass = "risk-low";
      if (cand.sdtd_score >= 0.65) riskClass = "risk-high";
      else if (cand.sdtd_score >= 0.40) riskClass = "risk-mod";

      const card = document.createElement("article");
      card.className = "candidate-card";

      const sharedIpcText = (cand.shared_ipc && cand.shared_ipc.length > 0)
        ? `Shared IPC: ${cand.shared_ipc.join(", ")}`
        : "No direct IPC overlap";

      card.innerHTML = `
        <div class="cand-top-row">
          <div class="cand-id-box">
            <span class="cand-rank">#${rank}</span>
            <span class="cand-id">${cand.patent_id || `Candidate ${rank}`}</span>
          </div>
          <span class="risk-badge ${riskClass}">${cand.risk_level || "MODERATE OVERLAP"}</span>
        </div>

        <div class="metrics-row">
          <div class="metric-col">
            <span class="metric-label">Hybrid SDTD Score</span>
            <span class="metric-val">${sdtd}</span>
            <div class="metric-bar-bg">
              <div class="metric-bar-fill" style="width: ${Math.min(100, cand.sdtd_score * 100)}%"></div>
            </div>
          </div>
          <div class="metric-col">
            <span class="metric-label">Semantic Sim (SD)</span>
            <span class="metric-val">${sd}</span>
            <div class="metric-bar-bg">
              <div class="metric-bar-fill" style="width: ${Math.min(100, Math.max(0, cand.sd_score) * 100)}%"></div>
            </div>
          </div>
          <div class="metric-col">
            <span class="metric-label">Tech Overlap (TD)</span>
            <span class="metric-val">${td}</span>
            <div class="metric-bar-bg">
              <div class="metric-bar-fill" style="width: ${Math.min(100, cand.td_score * 100)}%"></div>
            </div>
          </div>
        </div>

        <p class="cand-body-text">${escapeHtml(cand.meta || "")}</p>

        <div class="cand-footer">
          <span class="ipc-shared-pill">${sharedIpcText}</span>
          <span>Codes: ${(cand.codes || []).join(", ") || "None"}</span>
        </div>
      `;

      candidatesGrid.appendChild(card);
    });

    resultsContainer.classList.remove("hidden");
    resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // 6. Download Report (.md)
  downloadReportBtn.addEventListener("click", () => {
    window.location.href = "/api/download_report";
  });
});
