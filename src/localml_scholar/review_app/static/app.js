"use strict";

const state = {
  papers: [],
  storage: {},
  selectedPaper: null,
  analysis: null,
  activeTab: "overview",
  currentInteraction: null,
};

const audienceLabels = {
  researcher: "PhD / professor",
  undergraduate: "Undergraduate",
  beginner: "High school / beginner",
};

const elements = {
  fileInput: document.querySelector("#paper-file"),
  dropzone: document.querySelector("#dropzone"),
  uploadProgress: document.querySelector("#upload-progress"),
  paperCount: document.querySelector("#paper-count"),
  paperList: document.querySelector("#paper-list"),
  feedbackCount: document.querySelector("#feedback-count"),
  welcome: document.querySelector("#welcome"),
  workspace: document.querySelector("#paper-workspace"),
  paperSource: document.querySelector("#paper-source"),
  paperTitle: document.querySelector("#paper-title"),
  paperStats: document.querySelector("#paper-stats"),
  question: document.querySelector("#question"),
  askButton: document.querySelector("#ask-button"),
  answerCard: document.querySelector("#answer-card"),
  answerQuestion: document.querySelector("#answer-question"),
  answerText: document.querySelector("#answer-text"),
  answerStatus: document.querySelector("#answer-status"),
  answerAudience: document.querySelector("#answer-audience"),
  evidenceGrid: document.querySelector("#evidence-grid"),
  analysisBody: document.querySelector("#analysis-body"),
  saveFeedback: document.querySelector("#save-feedback"),
  feedbackSaveState: document.querySelector("#feedback-save-state"),
  correctedAnswer: document.querySelector("#corrected-answer"),
  feedbackNotes: document.querySelector("#feedback-notes"),
  storageDialog: document.querySelector("#storage-dialog"),
  storagePaths: document.querySelector("#storage-paths"),
  toast: document.querySelector("#toast"),
};

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status}).`);
  }
  return payload;
}

let toastTimer = null;
function toast(message, isError = false) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("visible");
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 4200);
}

function metric(value, singular, plural = `${singular}s`) {
  return `${value.toLocaleString()} ${value === 1 ? singular : plural}`;
}

function selectedRadio(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value;
}

function renderPaperList() {
  elements.paperCount.textContent = state.papers.length;
  if (!state.papers.length) {
    elements.paperList.innerHTML =
      '<div class="empty-sidebar">Add your first paper to begin.</div>';
    return;
  }
  elements.paperList.innerHTML = state.papers
    .map((paper) => {
      const active =
        state.selectedPaper?.document_id === paper.document_id ? " active" : "";
      const kind = paper.media_type.includes("pdf") ? "PDF" : "TXT";
      const size = paper.page_count
        ? metric(paper.page_count, "page")
        : metric(paper.character_count, "character");
      return `
        <button class="paper-list-item${active}" type="button"
          data-document-id="${escapeHtml(paper.document_id)}">
          <span class="paper-icon">${kind}</span>
          <span>
            <strong>${escapeHtml(paper.title)}</strong>
            <small>${escapeHtml(size)} · ${paper.chunk_count} passages</small>
          </span>
        </button>`;
    })
    .join("");
}

function renderPaperHeader() {
  const paper = state.selectedPaper;
  elements.paperSource.textContent = paper.source_name;
  elements.paperTitle.textContent = paper.title;
  const stats = [
    paper.page_count ? metric(paper.page_count, "page") : null,
    metric(paper.section_count, "section"),
    metric(paper.chunk_count, "indexed passage"),
    metric(paper.character_count, "character"),
  ].filter(Boolean);
  elements.paperStats.innerHTML = stats
    .map((item) => `<span>${escapeHtml(item)}</span>`)
    .join("");
}

async function selectPaper(documentId) {
  const paper = state.papers.find((item) => item.document_id === documentId);
  if (!paper) return;
  state.selectedPaper = paper;
  state.analysis = null;
  state.currentInteraction = null;
  state.activeTab = "overview";
  elements.welcome.classList.add("hidden");
  elements.workspace.classList.remove("hidden");
  elements.answerCard.classList.add("hidden");
  renderPaperList();
  renderPaperHeader();
  setActiveTab("overview");
  elements.analysisBody.innerHTML = `
    <div class="analysis-loading">
      <span class="spinner" aria-hidden="true"></span>
      <span>Building deterministic paper analysis…</span>
    </div>`;
  try {
    state.analysis = await api(
      `/api/papers/${encodeURIComponent(documentId)}/analysis`,
    );
    renderAnalysis();
  } catch (error) {
    elements.analysisBody.innerHTML = `<div class="empty-artifact">${escapeHtml(error.message)}</div>`;
    toast(error.message, true);
  }
}

async function refreshState({ selectNewest = false } = {}) {
  const payload = await api("/api/state");
  state.papers = payload.papers;
  state.storage = payload.storage;
  elements.feedbackCount.textContent = payload.feedback_count;
  renderPaperList();
  renderStoragePaths();
  if (selectNewest && state.papers.length) {
    await selectPaper(state.papers[state.papers.length - 1].document_id);
  }
}

async function uploadPaper(file) {
  if (!file) return;
  elements.uploadProgress.classList.remove("hidden");
  elements.fileInput.disabled = true;
  try {
    const result = await api("/api/papers", {
      method: "POST",
      headers: { "X-Filename": encodeURIComponent(file.name) },
      body: file,
    });
    await refreshState();
    await selectPaper(result.document_id);
    toast(`${result.title} is indexed and ready.`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    elements.uploadProgress.classList.add("hidden");
    elements.fileInput.disabled = false;
    elements.fileInput.value = "";
  }
}

function citationLocation(citation) {
  if (citation?.display) return citation.display;
  if (citation?.page_start) {
    return citation.page_start === citation.page_end
      ? `page ${citation.page_start}`
      : `pages ${citation.page_start}–${citation.page_end}`;
  }
  if (citation?.start_line) {
    return `lines ${citation.start_line}–${citation.end_line}`;
  }
  return "source passage";
}

function renderAnswer(record) {
  const answer = record.answer;
  state.currentInteraction = record;
  elements.answerQuestion.textContent = record.question;
  elements.answerText.innerHTML = escapeHtml(answer.answer_text).replace(
    /\[(C\d+)\]/g,
    '<span class="citation-label">[$1]</span>',
  );
  elements.answerStatus.textContent = answer.abstained
    ? "Insufficient evidence"
    : answer.validation.accepted
      ? "Grounding validated"
      : "Needs review";
  elements.answerAudience.textContent =
    audienceLabels[record.audience_level] || record.audience_level;
  elements.answerStatus.classList.toggle("abstained", answer.abstained);
  elements.evidenceGrid.innerHTML = answer.evidence
    .map(
      (item) => `
        <article class="evidence-card">
          <header>
            <strong>${escapeHtml(item.label)}</strong>
            <span>${escapeHtml(citationLocation(item.citation))}</span>
          </header>
          <p>${escapeHtml(item.text)}</p>
        </article>`,
    )
    .join("");
  if (!answer.evidence.length) {
    elements.evidenceGrid.innerHTML =
      '<div class="empty-artifact">No evidence met the sufficiency threshold.</div>';
  }
  document
    .querySelectorAll('input[name="verdict"], #issue-options input')
    .forEach((input) => {
      input.checked = false;
    });
  document
    .querySelectorAll('input[name="feedback-audience"]')
    .forEach((input) => {
      input.checked = input.value === record.audience_level;
    });
  elements.correctedAnswer.value = "";
  elements.feedbackNotes.value = "";
  elements.feedbackSaveState.textContent = "";
  elements.answerCard.classList.remove("hidden");
  elements.answerCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function askQuestion() {
  if (!state.selectedPaper) return;
  const question = elements.question.value.trim();
  if (!question) {
    toast("Write a question first.", true);
    elements.question.focus();
    return;
  }
  const audienceLevel = selectedRadio("question-audience");
  if (!audienceLevel) {
    toast("Choose the intended audience level.", true);
    return;
  }
  elements.askButton.disabled = true;
  elements.askButton.textContent = "Finding evidence…";
  try {
    const record = await api("/api/questions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        document_id: state.selectedPaper.document_id,
        audience_level: audienceLevel,
      }),
    });
    renderAnswer(record);
  } catch (error) {
    toast(error.message, true);
  } finally {
    elements.askButton.disabled = false;
    elements.askButton.textContent = "Ask with citations";
  }
}

async function saveFeedback() {
  if (!state.currentInteraction) return;
  const verdict = document.querySelector(
    'input[name="verdict"]:checked',
  )?.value;
  if (!verdict) {
    toast("Choose Correct, Partly correct, or Incorrect.", true);
    return;
  }
  const audienceLevel = selectedRadio("feedback-audience");
  if (!audienceLevel) {
    toast("Choose the audience you are reviewing this answer for.", true);
    return;
  }
  const issueCategories = Array.from(
    document.querySelectorAll("#issue-options input:checked"),
  ).map((input) => input.value);
  elements.saveFeedback.disabled = true;
  elements.feedbackSaveState.textContent = "Saving…";
  try {
    await api("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        interaction_id: state.currentInteraction.interaction_id,
        audience_level: audienceLevel,
        verdict,
        issue_categories: issueCategories,
        corrected_answer: elements.correctedAnswer.value,
        notes: elements.feedbackNotes.value,
      }),
    });
    const payload = await api("/api/state");
    elements.feedbackCount.textContent = payload.feedback_count;
    elements.feedbackSaveState.textContent = "Saved locally ✓";
    toast("Feedback saved for later Codex review.");
  } catch (error) {
    elements.feedbackSaveState.textContent = "";
    toast(error.message, true);
  } finally {
    elements.saveFeedback.disabled = false;
  }
}

function artifactValue(item) {
  if (item == null) return "";
  if (typeof item === "string" || typeof item === "number") return String(item);
  if (item.name && item.status && Array.isArray(item.evidence)) {
    const evidence = item.evidence.map((entry) => artifactValue(entry)).join("\n");
    const notes = (item.notes || []).join("\n");
    return [item.status.replaceAll("_", " "), evidence, notes]
      .filter(Boolean)
      .join("\n");
  }
  if (item.item && item.status && Array.isArray(item.values)) {
    const values = item.values.map((entry) => artifactValue(entry)).join("\n");
    return [item.status.replaceAll("_", " "), values, ...(item.notes || [])]
      .filter(Boolean)
      .join("\n");
  }
  if (item.reason) return item.reason;
  if (item.value != null) {
    return typeof item.value === "object"
      ? JSON.stringify(item.value, null, 2)
      : String(item.value);
  }
  if (item.raw_text) return item.raw_text;
  if (item.symbol) return item.symbol;
  if (item.text) return item.text;
  if (item.description) return item.description;
  if (item.title) return item.title;
  return JSON.stringify(item, null, 2);
}

function artifactLabel(item, fallback) {
  return (
    item?.category ||
    item?.symbol ||
    item?.name ||
    item?.item ||
    item?.section ||
    item?.equation_number ||
    item?.validation ||
    item?.confidence ||
    fallback
  );
}

function artifactCards(items, fallbackLabel) {
  if (!items?.length) {
    return '<div class="empty-artifact">No supported items were detected in the extracted text.</div>';
  }
  return `<div class="artifact-grid">${items
    .map((item) => {
      const citation = item.citation || item.selected_definition?.citation;
      return `
        <article class="artifact">
          <div class="artifact-label">${escapeHtml(artifactLabel(item, fallbackLabel))}</div>
          <pre>${escapeHtml(artifactValue(item))}</pre>
          ${citation ? `<span class="source-location">${escapeHtml(citationLocation(citation))}</span>` : ""}
        </article>`;
    })
    .join("")}</div>`;
}

function analysisSection(title, body) {
  return `<section class="analysis-section"><h3>${escapeHtml(title)}</h3>${body}</section>`;
}

function renderOverview(data) {
  const analysis = data.analysis;
  const summary = data.summary;
  const summaryFields = summary.fields || summary.sections || [];
  return [
    analysisSection("Structured summary", artifactCards(summaryFields, "Summary")),
    analysisSection("Claims", artifactCards(analysis.claims, "Claim")),
    analysisSection("Assumptions", artifactCards(analysis.assumptions, "Assumption")),
    analysisSection("Limitations", artifactCards(analysis.limitations, "Limitation")),
    analysis.warnings?.length
      ? analysisSection(
          "Extraction warnings",
          artifactCards(
            analysis.warnings.map((value) => ({ value })),
            "Warning",
          ),
        )
      : "",
  ].join("");
}

function renderEquations(data) {
  const analysis = data.analysis;
  return [
    analysisSection("Detected equations", artifactCards(analysis.equations, "Equation")),
    analysisSection("Notation glossary", artifactCards(analysis.notation, "Symbol")),
    analysisSection(
      "Unresolved symbols",
      artifactCards(
        (analysis.unresolved_symbols || []).map((value) => ({ value })),
        "Unresolved",
      ),
    ),
  ].join("");
}

function renderMethods(data) {
  const analysis = data.analysis;
  return [
    analysisSection("Methodology", artifactCards(analysis.methodology, "Method")),
    analysisSection("Procedures", artifactCards(analysis.procedures, "Procedure")),
    analysisSection("Datasets", artifactCards(analysis.datasets, "Dataset")),
    analysisSection("Metrics", artifactCards(analysis.metrics, "Metric")),
    analysisSection("Baselines", artifactCards(analysis.baselines, "Baseline")),
    analysisSection(
      "Hyperparameters",
      artifactCards(analysis.hyperparameters, "Hyperparameter"),
    ),
    analysisSection("Experiments", artifactCards(analysis.experiments, "Experiment")),
    analysisSection("Results", artifactCards(analysis.results, "Result")),
    analysisSection("Ablations", artifactCards(analysis.ablations, "Ablation")),
  ].join("");
}

function renderChecklist(data) {
  const checklist = data.checklist;
  return [
    analysisSection(
      "Implementation checklist",
      artifactCards(checklist.items, "Checklist item"),
    ),
    analysisSection(
      "Risk and missing-detail flags",
      artifactCards(checklist.risk_flags, "Risk flag"),
    ),
  ].join("");
}

function renderAnalysis() {
  if (!state.analysis) return;
  const renderers = {
    overview: renderOverview,
    equations: renderEquations,
    methods: renderMethods,
    checklist: renderChecklist,
    source: (data) =>
      `<pre class="source-view">${escapeHtml(data.source.text)}</pre>`,
  };
  elements.analysisBody.innerHTML = renderers[state.activeTab](state.analysis);
}

function setActiveTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll("[data-tab]").forEach((button) => {
    const active = button.dataset.tab === tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  renderAnalysis();
}

function renderStoragePaths() {
  elements.storagePaths.innerHTML = Object.entries(state.storage)
    .map(
      ([label, path]) =>
        `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(path)}</dd></div>`,
    )
    .join("");
}

elements.fileInput.addEventListener("change", () =>
  uploadPaper(elements.fileInput.files[0]),
);

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("dragging");
  });
});

elements.dropzone.addEventListener("drop", (event) => {
  uploadPaper(event.dataTransfer.files[0]);
});

elements.paperList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-document-id]");
  if (item) selectPaper(item.dataset.documentId);
});

elements.askButton.addEventListener("click", askQuestion);
elements.question.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") askQuestion();
});

document.querySelector(".prompt-row").addEventListener("click", (event) => {
  const prompt = event.target.dataset.prompt;
  if (prompt) {
    elements.question.value = prompt;
    elements.question.focus();
  }
});

document.querySelector(".tab-list").addEventListener("click", (event) => {
  if (event.target.dataset.tab) setActiveTab(event.target.dataset.tab);
});

elements.saveFeedback.addEventListener("click", saveFeedback);
document.querySelector("#show-storage").addEventListener("click", () => {
  elements.storageDialog.showModal();
});

refreshState().catch((error) => toast(error.message, true));
