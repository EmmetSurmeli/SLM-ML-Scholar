"use strict";

const app = {
  state: null,
  selectedPapers: new Set(),
  sessionId: null,
  currentInteraction: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
})[char]);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({ error: "Invalid server response." }));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status}).`);
  return payload;
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = error ? "show error" : "show";
  window.setTimeout(() => { node.className = ""; }, 3500);
}

function busy(message) { $("#status").textContent = message; }
function lines(value) { return value.split("\n").map((item) => item.trim()).filter(Boolean); }
function paperTitle(id) {
  return app.state.papers.find((paper) => paper.document_id === id)?.title || id;
}
function paperIds(item) {
  if (Array.isArray(item?.paper_ids)) return item.paper_ids;
  if (typeof item?.document_id === "string" && item.document_id) return [item.document_id];
  if (typeof item?.paper_id === "string" && item.paper_id) return [item.paper_id];
  return [];
}

const viewTitles = {
  papers: "Paper corpus", ask: "Grounded conversation", benchmarks: "Benchmark workshop",
  autonomous: "Autonomous corpus curation",
  automation: "Automatic first-pass review", review: "Answer review",
  calibration: "Calibration Lab",
  corrections: "Correction approval", dataset: "Instruction dataset",
  evaluation: "Evaluation dashboard",
};

function showView(name) {
  $$(".nav-link").forEach((node) => node.classList.toggle("active", node.dataset.view === name));
  $$(".view").forEach((node) => node.classList.toggle("active", node.id === `view-${name}`));
  $("#view-title").textContent = viewTitles[name];
}

function renderProgress(container, progress) {
  container.innerHTML = progress.targets.map((target) => `
    <div>
      <div class="progress-label"><span>${target.count} approved</span><span>${Math.round(target.progress * 100)}%</span></div>
      <div class="progress-bar"><i style="width:${Math.round(target.progress * 100)}%"></i></div>
    </div>`).join("");
}

function renderPapers() {
  const papers = app.state.papers;
  const automaticPaper = $("#automation-paper").value;
  const list = $("#paper-list");
  if (!papers.length) {
    list.className = "paper-grid empty";
    list.innerHTML = "<p>No papers indexed yet.</p>";
  } else {
    list.className = "paper-grid";
    list.innerHTML = papers.map((paper) => `
      <article class="paper-card">
        <label class="paper-select"><input type="checkbox" data-select-paper="${escapeHtml(paper.document_id)}" ${app.selectedPapers.has(paper.document_id) ? "checked" : ""}><span><h3>${escapeHtml(paper.title)}</h3><span class="paper-meta">${escapeHtml(paper.source_name)}</span></span></label>
        <p class="paper-meta">${paper.section_count} sections · ${paper.chunk_count} chunks · ${paper.character_count.toLocaleString()} characters<br>${paper.benchmark_question_count} questions · ${paper.reviewed_answer_count} answers · ${paper.correction_count} corrections</p>
        <div class="paper-actions"><button data-inspect-paper="${escapeHtml(paper.document_id)}">Inspect</button><button class="secondary" data-generate-paper="${escapeHtml(paper.document_id)}">Generate 60</button></div>
      </article>`).join("");
  }
  const options = papers.map((paper) => `<option value="${escapeHtml(paper.document_id)}">${escapeHtml(paper.title)}</option>`).join("");
  $("#generate-paper").innerHTML = options || '<option value="">Add a paper first</option>';
  $("#automation-paper").innerHTML = options || '<option value="">Add a paper first</option>';
  if (papers.some((paper) => paper.document_id === automaticPaper)) {
    $("#automation-paper").value = automaticPaper;
  }
  $("#question-paper-filter").innerHTML = '<option value="">All papers</option>' + options;
  if ($("#autonomous-paper-summary")) $("#autonomous-paper-summary").textContent = `${app.selectedPapers.size} papers selected`;
  renderScope();
}

function renderScope() {
  const scope = $("#ask-paper-scope");
  if (!app.state.papers.length) { scope.innerHTML = '<p class="microcopy">Add a paper first.</p>'; return; }
  scope.innerHTML = `
    <div class="form-actions"><button type="button" class="secondary" id="select-all">All</button><button type="button" class="secondary" id="select-none">None</button></div>
    ${app.state.papers.map((paper) => `<label class="scope-item"><input type="checkbox" data-scope-paper="${escapeHtml(paper.document_id)}" ${app.selectedPapers.has(paper.document_id) ? "checked" : ""}>${escapeHtml(paper.title)}</label>`).join("")}`;
  $("#select-all").onclick = () => { app.state.papers.forEach((paper) => app.selectedPapers.add(paper.document_id)); renderPapers(); };
  $("#select-none").onclick = () => { app.selectedPapers.clear(); renderPapers(); };
}

function profileChips(profile) {
  if (!profile) return "";
  const values = [profile.assumed_background, profile.desired_depth, profile.mathematical_depth + " math", profile.output_format];
  if (profile.include_derivation) values.push("derivation");
  if (profile.include_critique) values.push("critique");
  if (profile.include_comparison) values.push("comparison");
  return `<div class="profile-chips">${values.map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("")}</div>`;
}

function evidenceCards(evidence = []) {
  return `<div class="evidence-list">${evidence.map((item) => `
    <div class="evidence-card"><strong>[${escapeHtml(item.label)}] ${escapeHtml(item.title || item.source_name)}</strong><br>${escapeHtml(item.selected_text)}</div>`).join("")}</div>`;
}

function renderConversation() {
  const interactions = app.state.interactions || [];
  const container = $("#conversation");
  if (!interactions.length) { container.className = "conversation empty"; container.innerHTML = "<p>Your local conversation will appear here.</p>"; return; }
  container.className = "conversation";
  container.innerHTML = interactions.slice(0, 12).reverse().map((item) => `
    <article class="turn user"><header><strong>You</strong><span>${escapeHtml(item.created_at)}</span></header><div>${escapeHtml(item.question)}</div></article>
    <article class="turn assistant"><header><strong>LocalML Scholar</strong><span>${item.answer.abstained ? "abstained" : `${item.answer.evidence?.length || 0} evidence items`}</span></header>
      <div class="answer-text">${escapeHtml(item.answer.answer_text)}</div>${profileChips(item.instruction_profile)}
      <div class="profile-chips"><span class="chip">citation coverage ${Math.round((item.diagnostics?.citation_coverage || 0) * 100)}%</span><span class="chip">query coverage ${Math.round((item.diagnostics?.query_term_coverage || 0) * 100)}%</span>${(item.diagnostics?.failure_categories || []).map((failure) => `<span class="chip">${escapeHtml(failure)}</span>`).join("")}</div>
      ${item.comparison?.warning ? `<p class="warning">${escapeHtml(item.comparison.warning)}</p>` : ""}
      ${evidenceCards(item.answer.evidence)}
      <div class="paper-actions"><button data-open-review="${escapeHtml(item.interaction_id)}">Review this answer</button></div>
    </article>`).join("");
}

function renderQuestions() {
  const paperFilter = $("#question-paper-filter").value;
  const statusFilter = $("#question-status-filter").value;
  const items = (app.state.questions || []).filter((item) =>
    (!paperFilter || paperIds(item).includes(paperFilter)) && (!statusFilter || item.review_status === statusFilter));
  $("#question-summary").textContent = `${items.length} shown`;
  $("#benchmark-list").innerHTML = items.length ? items.map((item) => `
    <article class="review-card">
      <div><span class="type">${escapeHtml(item.question_type)}</span><br><span class="status-tag ${escapeHtml(item.review_status)}">${escapeHtml(item.review_status)}</span></div>
      <div><p>${escapeHtml(item.question)}</p><p class="meta">${paperIds(item).map(paperTitle).map(escapeHtml).join(" · ") || "Paper scope unavailable"}${item.parent_question_id ? " · prompt variation" : ""}</p></div>
      <div class="actions"><button data-run-question="${escapeHtml(item.question_id)}">Run</button><button class="secondary" data-edit-question-target="${escapeHtml(item.question_id)}">Edit target</button><button class="secondary" data-question-status="human_approved" data-question-id="${escapeHtml(item.question_id)}">Approve</button><button class="secondary" data-vary-question="${escapeHtml(item.question_id)}">Vary</button><button class="secondary" data-question-status="human_rejected" data-question-id="${escapeHtml(item.question_id)}">Reject</button></div>
    </article>`).join("") : '<div class="empty"><p>No questions match this view.</p></div>';
}

function renderReviewList() {
  const items = app.state.interactions || [];
  $("#review-list").innerHTML = items.length ? items.map((item) => `
    <article class="review-card"><div><span class="type">${escapeHtml(item.question_type || "user question")}</span></div><div><p>${escapeHtml(item.question)}</p><p class="meta">${paperIds(item).map(paperTitle).map(escapeHtml).join(" · ") || "Paper scope unavailable"} · ${item.answer.abstained ? "abstained" : "answered"}</p></div><div class="actions"><button data-open-review="${escapeHtml(item.interaction_id)}">Inspect & label</button></div></article>`).join("") : '<div class="empty"><p>Run a question to create a review item.</p></div>';
}

const reviewLabels = ["correct", "partial", "incorrect", "should_abstain", "benchmark_problem"];

function renderAutomaticReview(review) {
  const pending = review.decision === "pending_user_review";
  const saveable = review.saveable !== false;
  const editable = pending && saveable;
  const evidence = Array.isArray(review.answer?.evidence) ? review.answer.evidence : [];
  const proposedEvidence = new Set(review.final_evidence_ids || review.proposed_evidence_ids || []);
  const label = review.final_label || review.proposed_label;
  const correctedAnswer = review.final_corrected_answer ?? review.proposed_corrected_answer ?? "";
  const requiredFacts = review.final_required_facts || review.proposed_required_facts || [];
  const prohibitedClaims = review.final_prohibited_claims || review.proposed_prohibited_claims || [];
  const secondPass = review.second_pass || {};
  const gateRows = (secondPass.reviewer_results || []).map((result) => {
    const failed = Object.entries(result.gates || {}).filter(([, passed]) => !passed).map(([name]) => name);
    return `<li><strong>${escapeHtml(result.reviewer_profile)}</strong> · ${Math.round((result.confidence || 0) * 100)}% · ${failed.length ? `failed: ${failed.map(escapeHtml).join(", ")}` : "all gates passed"}</li>`;
  }).join("");
  const decisionText = !saveable
    ? "Answer failed"
    : review.decision === "saved_as_user_review"
    ? "Saved as your review"
    : review.decision === "excluded_by_user" ? "Excluded" : "Needs your decision";
  return `
    <details class="auto-review-card" data-auto-review="${escapeHtml(review.review_id)}" ${pending ? "" : "open"}>
      <summary>
        <span class="auto-include-wrap">${pending ? `<input class="auto-include" type="checkbox" ${review.default_selected === false ? "" : "checked"} ${saveable ? "" : "disabled"} aria-label="Include this proposed review">` : ""}</span>
        <span><strong>${escapeHtml(review.question)}</strong><small>${escapeHtml(review.question_type)} · ${paperIds(review).map(paperTitle).map(escapeHtml).join(" · ")}</small></span>
        <span class="status-tag ${pending ? "proposed" : "human_approved"}">${escapeHtml(decisionText)}</span>
      </summary>
      <div class="auto-review-body">
        <div class="auto-diagnostics">
          <span class="chip">confidence ${Math.round((review.proposed_confidence || 0) * 100)}%</span>
          <span class="chip">citation coverage ${Math.round((review.diagnostics?.citation_coverage || 0) * 100)}%</span>
          <span class="chip">query coverage ${Math.round((review.diagnostics?.query_term_coverage || 0) * 100)}%</span>
          ${review.needs_answer_edit ? '<span class="chip attention">answer edit recommended</span>' : ""}
        </div>
        <div class="auto-rationale"><strong>Why this draft?</strong><ul>${(review.rationale || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>
        <details class="second-pass-details"><summary>Second-pass decision · ${escapeHtml(secondPass.review_status || "not run")} · ${Math.round((secondPass.confidence || 0) * 100)}%</summary><p>${(secondPass.rationale || []).map(escapeHtml).join(" ")}</p><ul>${gateRows}</ul>${(secondPass.mandatory_human_categories || []).length ? `<p class="warning">Mandatory human review: ${secondPass.mandatory_human_categories.map(escapeHtml).join(", ")}</p>` : ""}<p class="microcopy">Correlated deterministic profiles; not independent reviewers. Provenance hash: ${escapeHtml(secondPass.provenance?.answer_hash || "unavailable")}</p></details>
        ${pending ? `<button type="button" class="secondary" data-rerun-auto-review="${escapeHtml(review.review_id)}">Send back for re-review</button>` : ""}
        <label>Review label
          <select class="auto-label" ${editable ? "" : "disabled"}>${reviewLabels.map((item) => `<option value="${item}" ${item === label ? "selected" : ""}>${item.replaceAll("_", " ")}</option>`).join("")}</select>
        </label>
        <div class="auto-answer-comparison">
          <div><strong>System answer</strong><div class="answer-box">${escapeHtml(review.answer?.answer_text || "")}</div></div>
          <label>Proposed corrected answer<textarea class="auto-corrected-answer" rows="8" ${editable ? "" : "disabled"}>${escapeHtml(correctedAnswer)}</textarea></label>
        </div>
        <div class="auto-draft-fields">
          <label>Required facts <span>one per line</span><textarea class="auto-required-facts" rows="4" ${editable ? "" : "disabled"}>${escapeHtml(requiredFacts.join("\n"))}</textarea></label>
          <label>Prohibited claims <span>one per line</span><textarea class="auto-prohibited-claims" rows="4" ${editable ? "" : "disabled"}>${escapeHtml(prohibitedClaims.join("\n"))}</textarea></label>
        </div>
        <fieldset class="auto-evidence" ${editable ? "" : "disabled"}><legend>Evidence to retain</legend>
          ${evidence.length ? evidence.map((item) => {
            const evidenceId = item.evidence_id || item.chunk_id || item.label;
            return `<label><input type="checkbox" value="${escapeHtml(evidenceId)}" ${proposedEvidence.has(evidenceId) ? "checked" : ""}><strong>[${escapeHtml(item.label)}]</strong> ${escapeHtml(item.selected_text)}</label>`;
          }).join("") : '<p class="microcopy">No evidence was selected by the answer.</p>'}
        </fieldset>
        ${review.correction_example_id ? `<p class="microcopy">Correction proposal: ${escapeHtml(review.correction_example_id)}</p>` : ""}
      </div>
    </details>`;
}

function renderAutomaticBatches() {
  const batches = app.state.automatic_review_batches || [];
  const pending = batches.reduce((total, batch) => total + (batch.summary?.pending_user_review_count || 0), 0);
  $("#automation-count").textContent = pending;
  const container = $("#automatic-batches");
  if (!batches.length) {
    container.innerHTML = '<div class="empty"><p>No automatic review batch yet. Choose a paper above to run its full question set.</p></div>';
    return;
  }
  container.innerHTML = batches.map((batch, batchIndex) => {
    const summary = batch.summary || {};
    const pendingCount = summary.pending_user_review_count || 0;
    const canFinalize = ["awaiting_user_review", "partially_saved"].includes(batch.status);
    const controls = pendingCount && canFinalize ? `
      <div class="auto-finalize-bar">
        <label>Reviewer name or local identifier<input class="auto-reviewer" placeholder="Example: Emmet" autocomplete="name"></label>
        <div class="form-actions"><button type="button" data-save-auto-batch="${escapeHtml(batch.batch_id)}">Save selected as my reviews</button><button type="button" class="secondary" data-toggle-auto-batch="${escapeHtml(batch.batch_id)}">Toggle all</button></div>
        <p class="microcopy">Unchecked drafts are recorded as excluded. Saved items become correction proposals and still need approval on the Corrections page.</p>
      </div>` : batch.status === "saved" ? `<div class="auto-finalize-bar"><p>This batch is complete. Its saved reviews are correction proposals; inspect and approve them separately.</p><button type="button" data-view="corrections">Open Corrections</button></div>` : ["failed", "stopped"].includes(batch.status) ? `<div class="auto-finalize-bar"><p>This batch stopped early. Its completed drafts are intact; resume from the first unfinished question.</p><button type="button" data-resume-auto-batch="${escapeHtml(batch.batch_id)}">Resume remaining questions</button></div>` : batch.status === "running" ? `<div class="auto-finalize-bar"><p>The batch is processing and saving after each question.</p><button type="button" class="secondary" data-stop-auto-batch="${escapeHtml(batch.batch_id)}">Stop after current question</button></div>` : `<div class="auto-finalize-bar"><p>This batch is unavailable for review.</p></div>`;
    return `<article class="auto-batch ${batchIndex === 0 ? "latest" : ""}" data-auto-batch="${escapeHtml(batch.batch_id)}">
      <header><div><p class="panel-kicker">${batchIndex === 0 ? "LATEST BATCH" : "EARLIER BATCH"}</p><h3>${paperIds(batch).map(paperTitle).map(escapeHtml).join(" · ")}</h3><p class="microcopy">${escapeHtml(batch.created_at)} · ${escapeHtml(batch.status.replaceAll("_", " "))} · deterministic local first pass</p></div>
      <div class="auto-summary"><span><strong>${summary.review_count || 0}</strong> reviewed</span><span><strong>${pendingCount}</strong> awaiting you</span><span><strong>${summary.saved_review_count || 0}</strong> saved</span><span><strong>${summary.execution_error_count || 0}</strong> run errors</span></div></header>
      ${batch.error ? `<p class="warning">${escapeHtml(batch.error)}</p>` : ""}
      ${controls}
      <div class="auto-review-list">${(batch.reviews || []).map(renderAutomaticReview).join("")}</div>
    </article>`;
  }).join("");
}

function renderAutoPolicy() {
  const calibration = app.state.calibration || {};
  const metrics = app.state.second_pass_metrics || {};
  const audit = app.state.audit_queue || {};
  $("#auto-policy-metrics").innerHTML = `<div class="metric-row"><div><strong>${metrics.review_count || 0}</strong><span>second-pass reviews</span></div><div><strong>${calibration.example_count || 0}</strong><span>human outcomes</span></div><div><strong>${audit.selected_count || 0}</strong><span>audit queue</span></div></div><p><span class="status-tag ${escapeHtml(calibration.state || "calibration_required")}">${escapeHtml(calibration.state || "calibration_required")}</span> · agreement ${Math.round((calibration.agreement || 0) * 100)}% · override ${Math.round((calibration.override_rate || 0) * 100)}%</p>`;
  $("#enable-auto-approval").disabled = calibration.state !== "calibration_active";
}

function renderCalibration() {
  const report = app.state.calibration || {};
  const sample = app.state.calibration_sample || {};
  const cards = app.state.calibration_cards || [];
  const reasons = report.reasons || [];
  $("#calibration-count").textContent = report.example_count || 0;
  $("#calibration-metrics").innerHTML = `
    <div class="metric-row"><div><strong>${report.example_count || 0} / ${report.minimum_examples || 50}</strong><span>validated pairs</span></div><div><strong>${Math.round((report.auto_approval_precision || 0) * 100)}%</strong><span>auto precision</span></div><div><strong>${report.false_approval_count || 0}</strong><span>false approvals</span></div></div>
    <p><span class="status-tag ${escapeHtml(report.state || "calibration_required")}">${escapeHtml(report.state || "calibration_required")}</span> · agreement ${Math.round((report.agreement || 0) * 100)}% · false-approval rate ${Math.round((report.false_approval_rate || 0) * 100)}%</p>
    ${reasons.length ? `<ul class="readiness-reasons">${reasons.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : "<p>Every readiness check currently passes.</p>"}`;
  const gaps = sample.coverage_gaps || [];
  $("#calibration-coverage").innerHTML = `<p><strong>${sample.selected_count || 0}</strong> selected from <strong>${sample.population_count || 0}</strong> available reviews.</p>${gaps.length ? `<p class="warning">Coverage gaps: ${gaps.map(escapeHtml).join(", ")}</p>` : "<p>No available stratum is currently missing.</p>"}${(sample.warnings || []).map((item) => `<p class="warning">${escapeHtml(item)}</p>`).join("")}`;
  $("#bulk-auto-review").disabled = report.state !== "auto_approval_enabled";
  $("#enable-calibrated-approval").disabled = report.state !== "calibration_active";
  $("#calibration-cards").innerHTML = cards.length ? cards.map((item, index) => {
    if (item.unavailable) return `<article class="panel warning">${escapeHtml(item.review_id)}: ${escapeHtml(item.error)}</article>`;
    const decision = item.calibration_decision;
    const evidence = Array.isArray(item.answer?.evidence) ? item.answer.evidence : [];
    const second = item.second_pass || {};
    return `<article class="panel calibration-card" data-calibration-card="${escapeHtml(item.review_id)}" data-finished="${decision?.status === "finalized"}">
      <header><div><p class="panel-kicker">${index + 1} · ${escapeHtml(item.question_type || "unknown")}</p><h3>${escapeHtml(item.question)}</h3><p class="microcopy">${paperIds(item).map(paperTitle).map(escapeHtml).join(" · ")} · automatic ${escapeHtml(item.proposed_label)} · ${Math.round((second.confidence ?? item.proposed_confidence ?? 0) * 100)}%</p></div><span class="status-tag">${escapeHtml(decision?.status || "pending")}</span></header>
      <div class="auto-diagnostics"><span class="chip">${escapeHtml(second.review_status || "not rerun")}</span>${(second.mandatory_human_categories || []).map((value) => `<span class="chip attention">${escapeHtml(value)}</span>`).join("")}</div>
      <label>System / corrected answer<textarea class="calibration-answer" rows="6" ${decision ? "disabled" : ""}>${escapeHtml(item.answer?.answer_text || "")}</textarea></label>
      <div class="evidence-checks">${evidence.map((value) => `<label><input class="calibration-evidence" type="checkbox" value="${escapeHtml(value.evidence_id || value.chunk_id || value.label)}" checked ${decision ? "disabled" : ""}><strong>[${escapeHtml(value.label)}]</strong> ${escapeHtml(value.selected_text || "")}</label>`).join("") || "<p>No evidence retained.</p>"}</div>
      <div class="auto-draft-fields"><label>Required facts<textarea class="calibration-facts" rows="3" ${decision ? "disabled" : ""}>${escapeHtml((item.proposed_required_facts || []).join("\n"))}</textarea></label><label>Prohibited claims<textarea class="calibration-prohibited" rows="3" ${decision ? "disabled" : ""}>${escapeHtml((item.proposed_prohibited_claims || []).join("\n"))}</textarea></label><label>Citation IDs <span>one per line</span><textarea class="calibration-citations" rows="3" ${decision ? "disabled" : ""}>${escapeHtml((item.citations || []).join("\n"))}</textarea></label><label>Structured target <span>JSON</span><textarea class="calibration-target" rows="3" ${decision ? "disabled" : ""}>${escapeHtml(JSON.stringify(item.structured_target || {}, null, 2))}</textarea></label></div>
      <details><summary>Gates, failure category, and root cause</summary><pre>${escapeHtml(JSON.stringify({ reviewer_results: second.reviewer_results || [], root_cause: item.root_cause || [], diagnostics: item.diagnostics || {} }, null, 2))}</pre></details>
      ${decision ? `<p><strong>Human decision:</strong> ${escapeHtml(decision.human_label || decision.action || decision.status)}${decision.edited ? " · edited and revalidated" : ""}</p>${decision.edited ? `<details><summary>Before / after correction</summary><div class="auto-answer-comparison"><div><strong>Before</strong><div class="answer-box">${escapeHtml(decision.original_snapshot?.answer?.answer_text || "")}</div></div><div><strong>After</strong><div class="answer-box">${escapeHtml(decision.reviewed_snapshot?.answer?.answer_text || "")}</div></div></div><pre>${escapeHtml(JSON.stringify(decision.revalidation || {}, null, 2))}</pre></details>` : ""}${decision.status === "finalized" && !decision.training_approved ? `<button class="secondary" data-calibration-training="${escapeHtml(decision.pair_id)}">Approve separately for training</button>` : ""}` : `<div class="calibration-actions"><button data-calibration-action="approve_auto">A · Approve auto</button><button class="secondary" data-calibration-action="override_correct">C · Correct</button><button class="secondary" data-calibration-action="override_partial">P · Partial</button><button class="secondary" data-calibration-action="override_incorrect">I · Incorrect</button><button class="secondary" data-calibration-action="override_should_abstain">S · Should abstain</button><button class="secondary" data-calibration-action="benchmark_problem">B · Benchmark problem</button><button class="secondary" data-calibration-action="skip">Skip</button></div>`}
    </article>`;
  }).join("") : '<div class="empty"><p>Create a calibration sample after running automatic reviews.</p></div>';
  const acquisition = app.state.paper_acquisition_queue || [];
  $("#acquisition-list").innerHTML = acquisition.length ? acquisition.map((item) => `<article class="review-card"><div><span class="type">${escapeHtml(item.category)}</span><br><span class="status-tag">${escapeHtml(item.status)}</span></div><div><p><strong>${escapeHtml(item.title)}</strong></p><p class="meta">${escapeHtml(item.reason)}${item.doi ? ` · DOI ${escapeHtml(item.doi)}` : ""}${item.arxiv_id ? ` · arXiv ${escapeHtml(item.arxiv_id)}` : ""}</p></div><div class="actions">${item.status === "suggested" ? `<button data-acquisition-status="obtained" data-acquisition-id="${escapeHtml(item.item_id)}">Mark obtained</button><button class="secondary" data-acquisition-status="declined" data-acquisition-id="${escapeHtml(item.item_id)}">Decline</button>` : ""}</div></article>`).join("") : '<div class="empty"><p>No paper suggestions yet.</p></div>';
}

function renderCorrections() {
  const items = app.state.corrections || [];
  $("#correction-list").innerHTML = items.length ? items.map((item) => `
    <article class="review-card"><div><span class="type">${escapeHtml(item.review_label)}</span><br><span class="status-tag ${escapeHtml(item.effective_trust_status || item.review_status)}">${escapeHtml(item.effective_trust_status || item.review_status)}</span></div><div><p>${escapeHtml(item.turns[item.turns.length - 1].content)}</p><p class="meta">${escapeHtml(item.final_answer)}</p></div><div class="actions">${item.review_status === "proposed" ? `<button data-approve-correction="${escapeHtml(item.example_id)}">Human approve</button><button class="secondary" data-edit-correction="${escapeHtml(item.example_id)}">Edit</button><button class="secondary" data-reject-correction="${escapeHtml(item.example_id)}">Human reject</button>` : item.review_status === "codex_approved" && item.effective_trust_status !== "audited_codex_approved" ? `<button data-audit-correction="${escapeHtml(item.example_id)}">Audit pass</button><button class="secondary" data-reject-correction="${escapeHtml(item.example_id)}">Override / reject</button>` : item.review_status === "codex_approved" ? `<button class="secondary" data-reject-correction="${escapeHtml(item.example_id)}">Override audited decision</button>` : ""}</div></article>`).join("") : '<div class="empty"><p>No correction proposals yet.</p></div>';
}

function renderDataset() {
  const metrics = app.state.dataset_metrics;
  const warnings = app.state.dataset_warnings || [];
  $("#dataset-warnings").innerHTML = warnings.length ? warnings.map((item) => `<p class="warning">${escapeHtml(item)}</p>`).join("") : "<p>No current composition warnings.</p>";
  $("#dataset-metrics").innerHTML = `<strong>${metrics.example_count}</strong> approved examples across <strong>${metrics.paper_count}</strong> papers<br>${metrics.multi_turn_count} multi-turn · ${metrics.multi_paper_count} cross-paper · ${metrics.abstention_count} abstentions · ${metrics.derivation_count} derivations`;
  renderProgress($("#progress-mini"), app.state.progress);
  $("#progress-dashboard").innerHTML = app.state.progress.targets.map((target) => `<article class="target-card"><strong>${target.count}</strong><p>${target.reached ? "Target reached" : `${app.state.progress.approved_examples} approved · ${Math.round(target.progress * 100)}%`}</p><div class="progress-bar"><i style="width:${Math.round(target.progress * 100)}%"></i></div></article>`).join("");
}

function renderAutonomous() {
  const state = app.state.autonomous_curation || {};
  const run = state.latest_run;
  const report = run?.report || {};
  const reliability = report.reliability || {};
  const statuses = state.status_counts || {};
  const diagnostics = run?.claim_diagnostics || [];
  const preflight = run?.preflight_metrics || {};
  const claimDetails = diagnostics.map((item) => {
    const graph = item.supported_claim_graph || {};
    const claims = Object.fromEntries((graph.claims || []).map((claim) => [claim.claim_id, claim]));
    const evidence = Object.fromEntries((item.evidence || []).map((entry) => [entry.label, entry]));
    const sentences = (graph.answer_sentences || []).map((sentence) => {
      const linkedClaims = (sentence.claim_ids || []).map((id) => claims[id]).filter(Boolean);
      const linkedEvidence = (sentence.citation_labels || []).map((label) => evidence[label]).filter(Boolean);
      return `<details><summary>${escapeHtml(sentence.text)}</summary><pre>${escapeHtml(JSON.stringify({ claims: linkedClaims, evidence: linkedEvidence }, null, 2))}</pre></details>`;
    }).join("");
    return `<details><summary>${escapeHtml(item.question || item.question_id)} · ${escapeHtml(item.question_type)}</summary>${sentences}<pre>${escapeHtml(JSON.stringify({ metrics: item.claim_alignment_metrics, repairs: item.repair_history, claim_disagreements: item.claim_level_disagreements }, null, 2))}</pre></details>`;
  }).join("");
  $("#autonomous-count").textContent = statuses.codex_curated || 0;
  $("#autonomous-paper-summary").textContent = `${app.selectedPapers.size} papers selected`;
  $("#autonomous-resume").disabled = run?.status !== "suspended";
  $("#autonomous-metrics").innerHTML = `
    <div class="metric-row"><div><strong>${report.questions_generated || 0}</strong><span>questions</span></div><div><strong>${report.examples_accepted || statuses.codex_curated || 0}</strong><span>Codex-curated</span></div><div><strong>${report.examples_rejected || statuses.rejected || 0}</strong><span>rejected</span></div></div>
    <div class="metric-row"><div><strong>${Math.round((reliability.hard_disagreement_rate || 0) * 100)}%</strong><span>hard disagreement</span></div><div><strong>${Math.round((reliability.soft_disagreement_rate || 0) * 100)}%</strong><span>soft disagreement</span></div><div><strong>${Math.round((reliability.citation_structural_validity || 0) * 100)}%</strong><span>citation structure</span></div></div>
    <div class="metric-row"><div><strong>${Math.round((reliability.citation_support_rate || 0) * 100)}%</strong><span>citation support</span></div><div><strong>${Math.round((reliability.citation_relevance_rate || 0) * 100)}%</strong><span>citation relevance</span></div><div><strong>${reliability.stale_evidence_id_count || 0}</strong><span>stale evidence IDs</span></div></div>
    <div class="metric-row"><div><strong>${reliability.claim_count || 0}</strong><span>claims</span></div><div><strong>${Math.round((reliability.claim_citation_completeness || 0) * 100)}%</strong><span>claim citations</span></div><div><strong>${Math.round((reliability.evidence_to_claim_alignment || 0) * 100)}%</strong><span>evidence alignment</span></div></div>
    <div class="metric-row"><div><strong>${Math.round((reliability.sentence_to_claim_traceability || 0) * 100)}%</strong><span>sentence traceability</span></div><div><strong>${reliability.claim_hard_disagreement_count || 0}</strong><span>claim conflicts</span></div><div><strong>${Math.round((reliability.repair_success_rate || 0) * 100)}%</strong><span>repair success</span></div></div>
    <div class="metric-row"><div><strong>${preflight.healthy_papers || 0}</strong><span>healthy papers</span></div><div><strong>${preflight.unhealthy_papers || 0}</strong><span>unhealthy papers</span></div><div><strong>${preflight.question_templates_suppressed || 0}</strong><span>templates suppressed</span></div></div>
    <div class="metric-row"><div><strong>${preflight.deterministic_preflight_rejections || 0}</strong><span>preflight rejections</span></div><div><strong>${preflight.candidates_sent_to_codex || 0}</strong><span>sent to Codex</span></div><div><strong>${preflight.codex_calls_saved || 0}</strong><span>Codex calls saved</span></div></div>
    <div class="metric-row"><div><strong>${preflight.deterministic_repair_successes || 0}</strong><span>deterministic repairs</span></div><div><strong>${preflight.candidate_construction_failures || 0}</strong><span>candidate failures</span></div><div><strong>${preflight.preflight_cache_hits || 0}</strong><span>cache hits</span></div></div>
    <details><summary>Reviewer reliability and category metrics</summary><pre>${escapeHtml(JSON.stringify({ disagreement_taxonomy: reliability.disagreement_taxonomy || {}, reviewer_pair_matrix: reliability.reviewer_pair_matrix || {}, metrics_by_question_type: reliability.metrics_by_question_type || {}, repair_success_by_type: reliability.repair_success_by_type || {}, representative_failures: reliability.representative_failures || {} }, null, 2))}</pre></details>
    <details><summary>Sentence → claim → evidence traces</summary>${claimDetails || "<p>No 1.2.5 claim traces yet.</p>"}</details>
    <p><span class="status-tag ${escapeHtml(run?.status || "proposed")}">${escapeHtml(run?.status || "not started")}</span> · stage ${escapeHtml(run?.stage || "none")} · Codex ${state.codex_available ? "available" : "unavailable"}</p>
    <p class="microcopy">${run ? `Run ${escapeHtml(run.run_id)} · ${run.paper_ids?.length || 0} papers · updated ${escapeHtml(run.updated_at)}` : "Start with selected papers or select all papers in the corpus."}</p>`;
  $("#autonomous-report").textContent = run ? JSON.stringify({ report, preflight_metrics: preflight, ingestion_health: run.ingestion_health, candidate_failures: run.candidate_failures || [], paper_splits: run.paper_splits, dataset_path: run.dataset_path, manifest_path: run.manifest_path, errors: run.errors }, null, 2) : "No run yet.";
}

function renderState() {
  $("#paper-count").textContent = app.state.papers.length;
  $("#question-count").textContent = app.state.question_count;
  $("#review-count").textContent = app.state.interaction_count;
  $("#correction-count").textContent = app.state.correction_count;
  $("#metric-papers").textContent = app.state.papers.length;
  $("#metric-questions").textContent = app.state.question_count;
  $("#metric-approved").textContent = app.state.approved_example_count;
  renderPapers(); renderConversation(); renderQuestions(); renderAutonomous(); renderAutomaticBatches(); renderAutoPolicy(); renderCalibration(); renderReviewList(); renderCorrections(); renderDataset();
}

async function refresh() {
  busy("Refreshing local state…");
  app.state = await api("/api/state");
  const known = new Set(app.state.papers.map((paper) => paper.document_id));
  app.selectedPapers = new Set([...app.selectedPapers].filter((id) => known.has(id)));
  if (!app.selectedPapers.size && app.state.papers.length) app.selectedPapers.add(app.state.papers[0].document_id);
  renderState(); busy("Local state up to date");
}

async function ensureSession() {
  if (app.sessionId) return app.sessionId;
  const session = await api("/api/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ selected_paper_ids: [...app.selectedPapers], persist_preferences: false }) });
  app.sessionId = session.session_id;
  return app.sessionId;
}

function openReview(interactionId) {
  const item = app.state.interactions.find((candidate) => candidate.interaction_id === interactionId);
  if (!item) return;
  app.currentInteraction = item;
  showView("review");
  $("#review-editor").classList.remove("hidden");
  $("#review-question").textContent = item.question;
  $("#review-answer").textContent = item.answer.answer_text;
  $("#review-interaction-id").value = item.interaction_id;
  $("#corrected-answer").value = item.answer.answer_text;
  $("#required-facts").value = "";
  $("#prohibited-claims").value = "";
  $("#review-notes").value = "";
  $("#review-evidence").innerHTML = (item.answer.evidence || []).map((evidence) => `<label><input type="checkbox" name="retain-evidence" value="${escapeHtml(evidence.evidence_id || evidence.label)}" checked><strong>[${escapeHtml(evidence.label)}]</strong> ${escapeHtml(evidence.selected_text)}</label>`).join("") || '<p class="microcopy">No evidence was selected.</p>';
  $("#evidence-query").value = item.question;
  $("#alternative-evidence").innerHTML = "";
  $("#review-editor").scrollIntoView({ behavior: "smooth" });
}

document.addEventListener("click", async (event) => {
  const nav = event.target.closest("[data-view]"); if (nav) { showView(nav.dataset.view); return; }
  const calibrationAction = event.target.closest("[data-calibration-action]");
  if (calibrationAction) {
    const card = calibrationAction.closest("[data-calibration-card]");
    const reviewer = $("#calibration-reviewer").value.trim();
    if (!reviewer) { toast("Enter your reviewer name before validating a card.", true); return; }
    const original = (app.state.calibration_cards || []).find((item) => item.review_id === card.dataset.calibrationCard);
    const answerText = card.querySelector(".calibration-answer").value;
    const facts = lines(card.querySelector(".calibration-facts").value);
    const prohibited = lines(card.querySelector(".calibration-prohibited").value);
    const evidenceIds = [...card.querySelectorAll(".calibration-evidence:checked")].map((node) => node.value);
    const citations = lines(card.querySelector(".calibration-citations").value);
    let structuredTarget;
    try { structuredTarget = JSON.parse(card.querySelector(".calibration-target").value); } catch (_error) { toast("Structured target must be valid JSON.", true); return; }
    const edits = {};
    if (answerText !== (original.answer?.answer_text || "")) edits.answer_text = answerText;
    if (JSON.stringify(facts) !== JSON.stringify(original.proposed_required_facts || [])) edits.required_facts = facts;
    if (JSON.stringify(prohibited) !== JSON.stringify(original.proposed_prohibited_claims || [])) edits.prohibited_claims = prohibited;
    if (JSON.stringify(evidenceIds) !== JSON.stringify(original.proposed_evidence_ids || [])) edits.evidence_ids = evidenceIds;
    if (JSON.stringify(citations) !== JSON.stringify(original.citations || [])) edits.citations = citations;
    if (JSON.stringify(structuredTarget) !== JSON.stringify(original.structured_target || {})) edits.structured_target = structuredTarget;
    try {
      await api(`/api/calibration/reviews/${encodeURIComponent(card.dataset.calibrationCard)}/decision`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: calibrationAction.dataset.calibrationAction, reviewer, edits: Object.keys(edits).length ? edits : null }) });
      await refresh(); toast("Calibration decision saved. Training approval remains separate.");
    } catch (error) { toast(error.message, true); }
    return;
  }
  const trainingApproval = event.target.closest("[data-calibration-training]");
  if (trainingApproval) {
    const reviewer = $("#calibration-reviewer").value.trim();
    if (!reviewer) { toast("Enter your reviewer name first.", true); return; }
    if (!window.confirm("Approve this validated pair for the training dataset? This is separate from calibration.")) return;
    try { await api(`/api/calibration/pairs/${encodeURIComponent(trainingApproval.dataset.calibrationTraining)}/approve-training`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer }) }); await refresh(); toast("Training approval recorded with provenance."); } catch (error) { toast(error.message, true); }
    return;
  }
  const acquisitionStatus = event.target.closest("[data-acquisition-status]");
  if (acquisitionStatus) {
    try { await api(`/api/acquisition/${encodeURIComponent(acquisitionStatus.dataset.acquisitionId)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: acquisitionStatus.dataset.acquisitionStatus }) }); await refresh(); toast("Acquisition queue updated; no download was performed."); } catch (error) { toast(error.message, true); }
    return;
  }
  const selected = event.target.closest("[data-select-paper], [data-scope-paper]");
  if (selected) {
    const paperId = selected.dataset.selectPaper || selected.dataset.scopePaper;
    selected.checked ? app.selectedPapers.add(paperId) : app.selectedPapers.delete(paperId);
    renderPapers(); return;
  }
  const inspect = event.target.closest("[data-inspect-paper]");
  if (inspect) { try { busy("Extracting scholarly analysis…"); const data = await api(`/api/papers/${encodeURIComponent(inspect.dataset.inspectPaper)}/analysis`); const panel = $("#paper-inspector"); panel.classList.remove("hidden"); panel.innerHTML = `<p class="panel-kicker">SOURCE INSPECTOR</p><h2>${escapeHtml(data.paper.title)}</h2><p>${data.paper.section_count} sections · ${data.paper.chunk_count} chunks</p><details open><summary>Structured analysis</summary><pre>${escapeHtml(JSON.stringify(data.analysis, null, 2))}</pre></details><details><summary>Extracted source</summary><pre>${escapeHtml(data.source.text)}</pre></details>`; busy("Analysis ready"); } catch (error) { toast(error.message, true); } return; }
  const quickGenerate = event.target.closest("[data-generate-paper]");
  if (quickGenerate) { try { busy("Generating diverse candidates…"); await api(`/api/papers/${encodeURIComponent(quickGenerate.dataset.generatePaper)}/questions/generate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ count: 60 }) }); await refresh(); showView("benchmarks"); toast("Generated 60 proposed candidates."); } catch (error) { toast(error.message, true); } return; }
  const run = event.target.closest("[data-run-question]");
  if (run) { try { busy("Running trusted baseline…"); const sessionId = await ensureSession(); await api(`/api/questions/${encodeURIComponent(run.dataset.runQuestion)}/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId }) }); await refresh(); showView("review"); toast("Answer added to the review queue."); } catch (error) { toast(error.message, true); } return; }
  const reviewStatus = event.target.closest("[data-question-status]");
  if (reviewStatus) { try { await api(`/api/questions/${encodeURIComponent(reviewStatus.dataset.questionId)}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ review_status: reviewStatus.dataset.questionStatus }) }); await refresh(); toast(`Question marked ${reviewStatus.dataset.questionStatus}.`); } catch (error) { toast(error.message, true); } return; }
  const editTarget = event.target.closest("[data-edit-question-target]");
  if (editTarget) {
    const item = app.state.questions.find((question) => question.question_id === editTarget.dataset.editQuestionTarget);
    if (!item) return;
    const concepts = window.prompt("Required concepts (comma-separated):", item.required_concepts.join(", "));
    if (concepts === null) return;
    const prohibited = window.prompt("Prohibited claims (comma-separated):", item.prohibited_claims.join(", "));
    if (prohibited === null) return;
    try {
      await api(`/api/questions/${encodeURIComponent(item.question_id)}/review`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review_status: item.review_status, required_concepts: concepts.split(",").map((part) => part.trim()).filter(Boolean), prohibited_claims: prohibited.split(",").map((part) => part.trim()).filter(Boolean) }),
      });
      await refresh(); toast("Candidate target fields updated; approval status unchanged.");
    } catch (error) { toast(error.message, true); }
    return;
  }
  const vary = event.target.closest("[data-vary-question]");
  if (vary) { try { await api(`/api/questions/${encodeURIComponent(vary.dataset.varyQuestion)}/variations`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); await refresh(); toast("Four unapproved prompt variations added."); } catch (error) { toast(error.message, true); } return; }
  const resumeBatch = event.target.closest("[data-resume-auto-batch]");
  if (resumeBatch) {
    try {
      busy("Resuming from the first unfinished question…");
      await api(`/api/automation/batches/${encodeURIComponent(resumeBatch.dataset.resumeAutoBatch)}/resume`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      await refresh();
      showView("automation");
      toast("The remaining questions finished. Review the complete batch below.");
    } catch (error) { busy("Batch resume stopped"); toast(error.message, true); }
    return;
  }
  const stopBatch = event.target.closest("[data-stop-auto-batch]");
  if (stopBatch) {
    try {
      await api(`/api/automation/batches/${encodeURIComponent(stopBatch.dataset.stopAutoBatch)}/stop`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      await refresh(); toast("Stop requested; completed decisions are preserved.");
    } catch (error) { toast(error.message, true); }
    return;
  }
  const rerunReview = event.target.closest("[data-rerun-auto-review]");
  if (rerunReview) {
    try {
      busy("Rerunning answer and second-pass gates…");
      await api(`/api/automation/reviews/${encodeURIComponent(rerunReview.dataset.rerunAutoReview)}/rerun`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      await refresh(); toast("Review rerun from the original question and evidence pipeline.");
    } catch (error) { toast(error.message, true); }
    return;
  }
  const toggleBatch = event.target.closest("[data-toggle-auto-batch]");
  if (toggleBatch) {
    const batch = toggleBatch.closest("[data-auto-batch]");
    const checkboxes = [...batch.querySelectorAll(".auto-include")];
    const shouldCheck = checkboxes.some((checkbox) => !checkbox.checked);
    checkboxes.forEach((checkbox) => { checkbox.checked = shouldCheck; });
    return;
  }
  const saveBatch = event.target.closest("[data-save-auto-batch]");
  if (saveBatch) {
    const batch = saveBatch.closest("[data-auto-batch]");
    const reviewer = batch.querySelector(".auto-reviewer").value.trim();
    if (!reviewer) { toast("Enter your reviewer name or local identifier.", true); return; }
    const cards = [...batch.querySelectorAll("[data-auto-review]")].filter((card) => card.querySelector(".auto-include"));
    const includedCount = cards.filter((card) => card.querySelector(".auto-include").checked).length;
    const message = `Save ${includedCount} draft${includedCount === 1 ? "" : "s"} as your reviews and exclude ${cards.length - includedCount}? Corrections will remain proposed.`;
    if (!window.confirm(message)) return;
    const decisions = cards.map((card) => ({
      review_id: card.dataset.autoReview,
      accepted: card.querySelector(".auto-include").checked,
      review_label: card.querySelector(".auto-label").value,
      corrected_answer: card.querySelector(".auto-corrected-answer").value,
      required_facts: lines(card.querySelector(".auto-required-facts").value),
      prohibited_claims: lines(card.querySelector(".auto-prohibited-claims").value),
      evidence_ids: [...card.querySelectorAll(".auto-evidence input:checked")].map((node) => node.value),
    }));
    try {
      busy("Saving your final batch decisions…");
      const result = await api(`/api/automation/batches/${encodeURIComponent(saveBatch.dataset.saveAutoBatch)}/finalize`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewer, decisions }),
      });
      await refresh();
      showView("automation");
      toast(`${result.saved_review_count} reviews saved; ${result.excluded_count} excluded. Corrections remain proposed.`);
    } catch (error) { toast(error.message, true); }
    return;
  }
  const open = event.target.closest("[data-open-review]"); if (open) { openReview(open.dataset.openReview); return; }
  const approve = event.target.closest("[data-approve-correction]");
  if (approve) { const reviewer = window.prompt("Reviewer name or local identifier:"); if (!reviewer) return; try { await api(`/api/corrections/${encodeURIComponent(approve.dataset.approveCorrection)}/approve`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer }) }); await refresh(); toast("Correction approved for dataset export."); } catch (error) { toast(error.message, true); } }
  const editCorrection = event.target.closest("[data-edit-correction]");
  if (editCorrection) { const item = app.state.corrections.find((candidate) => candidate.example_id === editCorrection.dataset.editCorrection); const finalAnswer = window.prompt("Edit the proposed final answer:", item?.final_answer || ""); if (finalAnswer === null) return; try { await api(`/api/corrections/${encodeURIComponent(editCorrection.dataset.editCorrection)}/edit`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ final_answer: finalAnswer }) }); await refresh(); toast("Correction edited; it remains proposed."); } catch (error) { toast(error.message, true); } return; }
  const auditCorrection = event.target.closest("[data-audit-correction]");
  if (auditCorrection) { const reviewer = window.prompt("Human auditor name or local identifier:"); if (!reviewer) return; try { await api(`/api/corrections/${encodeURIComponent(auditCorrection.dataset.auditCorrection)}/audit`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer, passed: true }) }); await refresh(); toast("Audit pass recorded; provenance remains Codex-approved."); } catch (error) { toast(error.message, true); } return; }
  const rejectCorrection = event.target.closest("[data-reject-correction]");
  if (rejectCorrection) { const reviewer = window.prompt("Reviewer name or local identifier:"); if (!reviewer) return; const reason = window.prompt("Reason for rejecting this suggestion:", "") ?? ""; try { await api(`/api/corrections/${encodeURIComponent(rejectCorrection.dataset.rejectCorrection)}/reject`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer, reason }) }); await refresh(); toast("Correction suggestion rejected."); } catch (error) { toast(error.message, true); } return; }
});

$("#upload-form").addEventListener("submit", async (event) => { event.preventDefault(); const files = [...$("#paper-file").files]; if (!files.length) return; try { busy(`Indexing ${files.length} paper${files.length === 1 ? "" : "s"} locally…`); const title = files.length === 1 ? $("#paper-title").value.trim() : ""; for (const file of files) { await api(`/api/papers${title ? `?title=${encodeURIComponent(title)}` : ""}`, { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) }, body: await file.arrayBuffer() }); } event.target.reset(); await refresh(); toast(`${files.length} paper${files.length === 1 ? "" : "s"} indexed locally.`); } catch (error) { toast(error.message, true); } });

$("#autonomous-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!app.selectedPapers.size) { toast("Select at least one paper first.", true); return; }
  const config = {
    questions_per_paper: Number($("#autonomous-questions").value),
    maximum_examples_per_paper: Number($("#autonomous-maximum").value),
    acceptance_threshold: Number($("#autonomous-threshold").value),
    evidence_threshold: Number($("#autonomous-threshold").value),
    maximum_repair_attempts: Number($("#autonomous-repairs").value),
    validation_fraction: 0.15, test_fraction: 0.15,
    seed: Number($("#autonomous-seed").value),
    include_multi_turn: $("#autonomous-multi-turn").checked,
    include_derivations: $("#autonomous-derivations").checked,
    include_cross_paper: $("#autonomous-cross-paper").checked,
    include_abstentions: $("#autonomous-abstentions").checked,
    per_question_type_cap: 12, maximum_disagreement_rate: 0.10,
  };
  if (!window.confirm(`Build a Codex-curated dataset from ${app.selectedPapers.size} paper(s)?\n\n${config.questions_per_paper} questions per paper\n${config.maximum_repair_attempts} repairs maximum\n${config.acceptance_threshold} acceptance threshold\nPaper-level 70/15/15 splits\nSeed ${config.seed}\n\nNo individual human approval will be requested.`)) return;
  try {
    busy("Starting autonomous curation…");
    const run = await api("/api/autonomous/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ paper_ids: [...app.selectedPapers], config }) });
    await refresh();
    toast(`Run ${run.run_id} started in the background.`);
  } catch (error) { busy("Autonomous curation did not start"); toast(error.message, true); }
});

$("#autonomous-refresh").onclick = () => refresh().catch((error) => toast(error.message, true));
$("#autonomous-process-new").onclick = async () => { try { const run = await api("/api/autonomous/process-new", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ config: { questions_per_paper: 60, maximum_examples_per_paper: 40, acceptance_threshold: 0.97, evidence_threshold: 0.97, maximum_repair_attempts: 2, validation_fraction: 0.15, test_fraction: 0.15, seed: 42, include_multi_turn: true, include_derivations: false, include_cross_paper: false, include_abstentions: true, per_question_type_cap: 12, maximum_disagreement_rate: 0.10 } }) }); await refresh(); toast(`Run ${run.run_id} started for newly uploaded papers.`); } catch (error) { toast(error.message, true); } };
$("#autonomous-resume").onclick = async () => { const run = app.state.autonomous_curation?.latest_run; if (!run) return; try { await api(`/api/autonomous/runs/${encodeURIComponent(run.run_id)}/resume`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); await refresh(); toast("Run resumed from its saved cursor."); } catch (error) { toast(error.message, true); } };

$("#ask-form").addEventListener("submit", async (event) => { event.preventDefault(); if (!app.selectedPapers.size) { toast("Select at least one paper.", true); return; } try { busy("Selecting evidence and interpreting instructions…"); const sessionId = await ensureSession(); await api("/api/questions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: $("#question").value, document_ids: [...app.selectedPapers], session_id: sessionId }) }); $("#question").value = ""; await refresh(); toast("Grounded answer ready for review."); } catch (error) { toast(error.message, true); } });

$("#clear-conversation").onclick = () => { app.sessionId = null; toast("Started a fresh in-memory conversation."); };
$("#refresh").onclick = () => refresh().catch((error) => toast(error.message, true));

$("#create-audit-sample").onclick = async () => { try { const result = await api("/api/automation/audit-sample", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sample_fraction: 0.10, seed: 42 }) }); await refresh(); toast(`Created a deterministic ${result.selected_count}-item audit queue.`); } catch (error) { toast(error.message, true); } };

$("#enable-auto-approval").onclick = async () => { if (!window.confirm("Enable automatic approval for future batches? High-risk categories and failed gates will still require a human.")) return; try { await api("/api/automation/enable", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: true }) }); await refresh(); toast("Automatic approval enabled under the calibrated safety policy."); } catch (error) { toast(error.message, true); } };

$("#enable-calibrated-approval").onclick = async () => { if (!window.confirm("Explicitly enable automatic approval? Mandatory-human routes and audit sampling remain active.")) return; try { await api("/api/automation/enable", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: true }) }); await refresh(); toast("Automatic approval explicitly enabled under the calibrated policy."); } catch (error) { toast(error.message, true); } };

$("#create-calibration-sample").onclick = async () => { try { busy("Selecting representative calibration cases…"); const result = await api("/api/calibration/sample", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target_count: 50, seed: 42 }) }); await refresh(); toast(`Selected ${result.selected_count} calibration cases.`); } catch (error) { toast(error.message, true); } };

$("#rerun-historical").onclick = async () => { const ids = app.state.calibration_sample?.review_ids || []; if (!ids.length) { toast("Create a sample first.", true); return; } try { busy("Appending modern historical reruns…"); const result = await api("/api/calibration/rerun-historical", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ review_ids: ids }) }); await refresh(); toast(`Appended ${result.rerun_count} linked reruns; originals were preserved.`); } catch (error) { toast(error.message, true); } };

$("#bulk-auto-review").onclick = async () => { if (!window.confirm("Run eligible pending questions? Every result remains subject to the audit queue.")) return; try { busy("Running eligible pending reviews…"); const result = await api("/api/calibration/bulk-auto-review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ eligible_only: true }) }); await refresh(); toast(`Reviewed ${result.batch.reviews.length} items; ${result.audit.selected_count} queued for audit.`); } catch (error) { toast(error.message, true); } };

$("#acquisition-form").addEventListener("submit", async (event) => { event.preventDefault(); try { await api("/api/acquisition", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: $("#acquisition-title").value, doi: $("#acquisition-doi").value, arxiv_id: $("#acquisition-arxiv").value, citation: $("#acquisition-citation").value, reason: $("#acquisition-reason").value, category: $("#acquisition-category").value }) }); event.target.reset(); await refresh(); toast("Paper suggestion saved locally; nothing was fetched."); } catch (error) { toast(error.message, true); } });

document.addEventListener("keydown", (event) => {
  if (!$("#view-calibration").classList.contains("active") || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
  const actions = { a: "approve_auto", c: "override_correct", p: "override_partial", i: "override_incorrect", s: "override_should_abstain", b: "benchmark_problem" };
  if (event.key.toLowerCase() === "e") { document.querySelector('.calibration-card[data-finished="false"] .calibration-answer')?.focus(); return; }
  const action = actions[event.key.toLowerCase()];
  if (action) document.querySelector(`.calibration-card[data-finished="false"] [data-calibration-action="${action}"]`)?.click();
});

$("#generate-form").addEventListener("submit", async (event) => { event.preventDefault(); try { busy("Generating proposed questions…"); await api(`/api/papers/${encodeURIComponent($("#generate-paper").value)}/questions/generate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ count: Number($("#generate-count").value) }) }); await refresh(); toast("Candidate pool generated; human review is required."); } catch (error) { toast(error.message, true); } });

$("#automation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const paperId = $("#automation-paper").value;
  if (!paperId) { toast("Add a paper before running automatic review.", true); return; }
  try {
    busy("Running every question and drafting transparent reviews…");
    await api("/api/automation/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper_ids: [paperId], generate_if_empty: true, generated_question_count: Number($("#automation-count-input").value), uncertain_only: $("#automation-uncertain-only").checked }),
    });
    await refresh();
    showView("automation");
    toast("First-pass reviews are ready for your final check.");
  } catch (error) { busy("Automatic review stopped"); toast(error.message, true); }
});

$("#manual-question-form").addEventListener("submit", async (event) => { event.preventDefault(); if (!app.selectedPapers.size) { toast("Select paper scope in Papers or Ask first.", true); return; } try { await api("/api/questions/manual", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: $("#manual-question").value, paper_ids: [...app.selectedPapers], question_type: $("#manual-type").value }) }); event.target.reset(); $("#manual-type").value = "user_authored"; await refresh(); toast("Manual question added as proposed."); } catch (error) { toast(error.message, true); } });

$("#question-paper-filter").onchange = renderQuestions;
$("#question-status-filter").onchange = renderQuestions;

$("#find-evidence").onclick = async () => {
  if (!app.currentInteraction) return;
  try {
    const results = await api("/api/evidence/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: $("#evidence-query").value, paper_ids: paperIds(app.currentInteraction), top_k: 10 }),
    });
    $("#alternative-evidence").innerHTML = results.map((item) => `<label><input type="checkbox" name="retain-evidence" value="${escapeHtml(item.chunk_id)}"><strong>Alternative · ${escapeHtml(item.title || item.source_name)}</strong> ${escapeHtml(item.text)}</label>`).join("") || '<p class="microcopy">No alternative chunks matched.</p>';
  } catch (error) { toast(error.message, true); }
};

$("#review-form").addEventListener("submit", async (event) => { event.preventDefault(); const retained = $$('input[name="retain-evidence"]:checked').map((node) => node.value); try { const correction = await api(`/api/interactions/${encodeURIComponent($("#review-interaction-id").value)}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ review_label: $('input[name="review-label"]:checked').value, replacement_evidence_ids: retained, required_facts: lines($("#required-facts").value), prohibited_claims: lines($("#prohibited-claims").value), corrected_answer: $("#corrected-answer").value, notes: $("#review-notes").value }) }); await refresh(); $("#review-editor").classList.add("hidden"); showView("corrections"); toast(`Correction ${correction.example_id} saved for separate approval.`); } catch (error) { toast(error.message, true); } });

$("#export-dataset").onclick = async () => { try { busy("Validating trust, duplicates, and paper splits…"); const result = await api("/api/dataset/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ seed: 0, trust_tier: $("#dataset-trust-tier").value }) }); $("#dataset-output").textContent = JSON.stringify({ output: result.output, trust_tier: result.trust_tier, report: result.report }, null, 2); toast("Trust-tier dataset exported."); busy("Dataset verified"); } catch (error) { toast(error.message, true); } };

refresh().catch((error) => { busy("Could not load local state"); toast(error.message, true); });
