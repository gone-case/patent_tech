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

  const agentTrace = document.getElementById("agent-trace");
  const traceList = document.getElementById("trace-list");
  const traceStatus = document.getElementById("trace-status");
  const reviewToggle = document.getElementById("review-toggle");
  const reviewToggleLabel = document.getElementById("review-toggle-label");
  const reviewCard = document.getElementById("review-card");
  const reviewFeatures = document.getElementById("review-features");
  const reviewCodes = document.getElementById("review-codes");
  const reviewCodesSource = document.getElementById("review-codes-source");
  const reviewContinueBtn = document.getElementById("review-continue-btn");
  const agentFindings = document.getElementById("agent-findings");
  const agentNarrative = document.getElementById("agent-narrative");

  let activeMode = "doc"; // "doc" | "patent" | "text"
  let selectedFile = null;
  let pendingThreadId = null;

  const selectedEngine = () => document.querySelector('input[name="engine"]:checked').value;
  document.querySelectorAll('input[name="engine"]').forEach(r => r.addEventListener("change", () => {
    reviewToggleLabel.classList.toggle("disabled", selectedEngine() !== "agents");
  }));

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

    if (selectedEngine() === "agents") {
      await runAgentSearch(formData, queryPayload);
      return;
    }

    // UI State: Starting Search
    agentTrace.classList.add("hidden");
    reviewCard.classList.add("hidden");
    agentFindings.classList.add("hidden");
    searchBtn.disabled = true;
    searchBtn.innerHTML = `
      <span class="pulsing-dot"></span>
      Searching...
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
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
        </svg>
        Search prior art
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
        ? `Shared CPC/IPC: ${cand.shared_ipc.join(", ")}`
        : "No CPC/IPC overlap";

      card.innerHTML = `
        <div class="cand-top-row">
          <div class="cand-id-box">
            <span class="cand-rank">#${rank}</span>
            <span class="cand-id">${cand.patent_id || `Candidate ${rank}`}</span>
          </div>
          <span class="risk-badge ${riskClass}" title="${escapeHtml(cand.risk_level || "")}">${{ "risk-high": "High overlap", "risk-mod": "Moderate overlap", "risk-low": "Low overlap" }[riskClass]}</span>
        </div>

        ${cand.title ? `<h4 class="cand-title">${cand.url && cand.url.startsWith("http")
            ? `<a href="${escapeHtml(cand.url)}" target="_blank" rel="noopener">${escapeHtml(cand.title)}</a>`
            : escapeHtml(cand.title)}</h4>` : ""}

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
          <span>Codes: ${(cand.codes || []).slice(0, 8).join(", ") || "None"}${(cand.codes || []).length > 8 ? ` +${cand.codes.length - 8} more` : ""}</span>
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

  // 6. LangGraph Agents: streamed events
  function setBusy(busy, label) {
    searchBtn.disabled = busy;
    searchBtn.innerHTML = busy
      ? `<span class="pulsing-dot"></span> ${label}`
      : `<svg class="btn-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
         </svg>
         Search prior art`;
  }

  async function runAgentSearch(formData, queryPayload) {
    const review = reviewToggle.checked;
    let req;
    if (formData) {
      formData.append("review", review ? "true" : "false");
      req = fetch("/api/agent/search", { method: "POST", body: formData });
    } else {
      req = fetch("/api/agent/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...queryPayload, review })
      });
    }

    traceList.innerHTML = "";
    agentStepper.classList.add("hidden");
    resultsContainer.classList.add("hidden");
    reviewCard.classList.add("hidden");
    agentTrace.classList.remove("hidden");
    agentTrace.scrollIntoView({ behavior: "smooth", block: "start" });
    await consumeStream(req);
  }

  reviewContinueBtn.addEventListener("click", async () => {
    if (!pendingThreadId) return;
    const features = reviewFeatures.value.split("\n").map(s => s.trim()).filter(Boolean);
    const cpc_codes = reviewCodes.value.split(",").map(s => s.trim()).filter(Boolean);
    reviewCard.classList.add("hidden");
    const req = fetch("/api/agent/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: pendingThreadId, features, cpc_codes })
    });
    pendingThreadId = null;
    await consumeStream(req);
  });

  async function consumeStream(requestPromise) {
    setBusy(true, "Agents working...");
    traceStatus.textContent = "running";
    const waiting = addWaitingItem();
    try {
      const res = await requestPromise;
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (line) handleAgentEvent(JSON.parse(line), waiting);
        }
      }
    } catch (err) {
      addTraceItem({ agent: "Error", message: err.message });
      traceStatus.textContent = "failed";
    } finally {
      waiting.remove();
      setBusy(false);
    }
  }

  function handleAgentEvent(ev, waiting) {
    if (ev.type === "step") {
      addTraceItem(ev);
      traceList.appendChild(waiting);  // keep the "working" row last
    } else if (ev.type === "review") {
      pendingThreadId = ev.thread_id;
      traceStatus.textContent = "waiting for you";
      reviewFeatures.value = ev.invention.features.join("\n");
      reviewCodes.value = ev.invention.cpc_codes.join(", ");
      reviewCodesSource.textContent = `(${ev.codes_source})`;
      reviewCard.classList.remove("hidden");
      reviewCard.scrollIntoView({ behavior: "smooth", block: "start" });
    } else if (ev.type === "result") {
      traceStatus.textContent = `done in ${ev.elapsed_seconds}s, ${ev.rounds} search round(s)`;
      renderAgentResult(ev);
    } else if (ev.type === "error") {
      addTraceItem({ agent: "Error", message: ev.message });
      traceStatus.textContent = "failed";
    }
  }

  function agentClass(agent) {
    return "agent-" + agent.toLowerCase().replace(/#\d+/, "").trim().replace(/\s+/g, "-");
  }

  function addTraceItem(ev) {
    const li = document.createElement("li");
    li.className = `trace-item ${agentClass(ev.agent)}`;
    const secs = ev.seconds !== undefined ? `<span class="trace-secs">${ev.seconds.toFixed(1)}s</span>` : "";
    let details = "";
    const data = ev.data || {};
    // Long lists stay folded, like the claim charts
    if (data.queries) {
      details = `<details class="trace-toggle"><summary>${data.queries.length} queries</summary>
        <ul class="trace-details">${data.queries.map(q =>
          `<li><span class="trace-tag">${q.feature ? "feature " + q.feature : "whole invention"}</span> ${escapeHtml(q.text)}</li>`
        ).join("")}</ul></details>`;
    } else if (data.features) {
      details = `<details class="trace-toggle"><summary>${data.features.length} features</summary>
        <ol class="trace-details">${data.features.map(f => `<li>${escapeHtml(f)}</li>`).join("")}</ol></details>`;
    }
    li.innerHTML = `
      <div class="trace-head"><span class="trace-agent">${escapeHtml(ev.agent)}</span>${secs}</div>
      <div class="trace-msg">${escapeHtml(ev.message)}</div>
      ${details}`;
    traceList.appendChild(li);
  }

  function addWaitingItem() {
    const li = document.createElement("li");
    li.className = "trace-item trace-waiting";
    li.innerHTML = `<div class="trace-head"><span class="pulsing-dot"></span><span class="trace-msg">Next agent working...</span></div>`;
    traceList.appendChild(li);
    return li;
  }

  const STATUS_LABEL = { disclosed: "Disclosed", partial: "Partial", not_found: "Not found" };

  function renderAgentResult(ev) {
    const inv = ev.invention;
    renderResults({
      target: { ...ev.target, title: inv.title, codes: inv.cpc_codes,
                source: `${ev.target.source} · CPC ${ev.codes_source}`,
                abstract: `${inv.problem} ${inv.solution}` },
      candidates: ev.candidates,
      elapsed_seconds: ev.elapsed_seconds
    });

    // Claim chart inside each candidate card
    const cards = candidatesGrid.querySelectorAll(".candidate-card");
    ev.charts.forEach(chart => {
      const card = cards[chart.rank - 1];
      if (!card) return;
      const counts = { disclosed: 0, partial: 0, not_found: 0 };
      chart.rows.forEach(r => counts[r.status]++);
      // Collapsed by default: one summary line, then one row per feature
      const div = document.createElement("details");
      div.className = "claim-chart";
      div.innerHTML = `
        <summary class="claim-similarity">
          <span class="metric-label">Claim similarity</span>
          <span class="metric-val">${Math.round(chart.coverage * 100)}%</span>
          <span class="status-pill status-disclosed">${counts.disclosed} disclosed</span>
          <span class="status-pill status-partial">${counts.partial} partial</span>
          <span class="status-pill status-not_found">${counts.not_found} not found</span>
          <span class="text-sm claim-basis">${chart.claims_compared ? "abstract + claims" : "abstract only"}</span>
        </summary>
        <div class="claim-summary">${escapeHtml(chart.summary)}</div>
        ${chart.rows.map(r => `
          <div class="chart-row">
            <span class="status-pill status-${r.status}">${STATUS_LABEL[r.status]}</span>
            <div>
              <div class="chart-feature">F${r.feature}. ${escapeHtml(shortText(r.text, 12))}</div>
              <div>${escapeHtml(r.explanation)}</div>
              ${r.evidence ? `<div class="chart-evidence">"${escapeHtml(r.evidence)}"${r.verified ? "" : " <em>(quote not found in text)</em>"}</div>` : ""}
            </div>
          </div>`).join("")}`;
      card.appendChild(div);
    });

    agentNarrative.innerHTML = renderMarkdown(ev.narrative || "");
    agentFindings.classList.remove("hidden");
  }

  function shortText(text, maxWords) {
    const words = (text || "").split(/\s+/);
    return words.length <= maxWords ? text : words.slice(0, maxWords).join(" ") + " ...";
  }

  // Minimal markdown for the Reporter's narrative: headings, bullets, bold, code
  function renderMarkdown(md) {
    const inline = s => escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
    let html = "", inList = false;
    md.split("\n").forEach(raw => {
      const line = raw.trim();
      const bullet = line.match(/^[-*]\s+(.*)/);
      if (bullet) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += `<li>${inline(bullet[1])}</li>`;
        return;
      }
      if (inList) { html += "</ul>"; inList = false; }
      const h = line.match(/^(#{2,4})\s+(.*)/);
      if (h) html += `<h4>${inline(h[2])}</h4>`;
      else if (line) html += `<p>${inline(line)}</p>`;
    });
    return html + (inList ? "</ul>" : "");
  }

  // 7. Download Report (.md)
  downloadReportBtn.addEventListener("click", () => {
    window.location.href = "/api/download_report";
  });
});
