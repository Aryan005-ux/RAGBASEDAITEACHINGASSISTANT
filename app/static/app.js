const $ = (id) => document.getElementById(id);
const chat = $("chat"), messages = $("messages"), welcome = $("welcome");
const input = $("input"), sendBtn = $("send"), form = $("composer");
const BUILTIN = "sigma";
const SIGMA_SUGGESTIONS = [
  "Where is HTML concluded in this course?",
  "How do I install VS Code?",
  "Explain the CSS box model",
  "What's the difference between inline and block elements?",
  "How do forms and input tags work?",
  "What is SEO and Core Web Vitals?",
];

let busy = false;
let courses = [];
let course = null;          // currently selected course (full record with videos)
let uploads = [];           // in-flight browser uploads: {key, name, progress}
let pollTimer = null;
let serverInfo = { extensions: [], max_upload_mb: 2048 };
let user = null;

// ---------- Helpers ----------
function fmtTime(sec) {
  sec = Math.floor(sec || 0);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
}

// Minimal, safe markdown: input is escaped before any tags are added.
function renderMarkdown(md) {
  const lines = escapeHtml(md).split("\n");
  let html = "", list = null, para = [];
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join("<br>"))}</p>`; para = []; } };
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };

  for (const raw of lines) {
    const line = raw.trim();
    let m;
    if (!line) { flushPara(); closeList(); continue; }
    if ((m = line.match(/^[-*•]\s+(.*)/)) || (m = line.match(/^\d+[.)]\s+(.*)/))) {
      flushPara();
      const type = /^\d/.test(line) ? "ol" : "ul";
      if (list !== type) { closeList(); html += `<${type}>`; list = type; }
      html += `<li>${inline(m[1])}</li>`;
    } else if ((m = line.match(/^&gt;\s?(.*)/))) {
      flushPara(); closeList();
      html += `<blockquote>${inline(m[1])}</blockquote>`;
    } else if ((m = line.match(/^#{1,6}\s+(.*)/))) {
      flushPara(); closeList();
      html += `<p><strong>${inline(m[1])}</strong></p>`;
    } else {
      closeList();
      para.push(line);
    }
  }
  flushPara(); closeList();
  return html;
}

function scrollToBottom() { chat.scrollTop = chat.scrollHeight; }

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function toast(text) {
  const t = el("div", "toast");
  t.textContent = text;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && !path.startsWith("/api/auth/")) showAuth();
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

const jsonPost = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function store(key, value) {
  try { value === undefined ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch {}
}
function recall(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

// ---------- Courses ----------
async function loadCourses(selectId) {
  courses = await api("/api/courses");
  const select = $("courseSelect");
  select.innerHTML = "";
  for (const c of courses) {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.builtin ? `${c.name} (demo)` : c.name;
    select.appendChild(opt);
  }
  const wanted = selectId || recall(`course:${user.id}`) || BUILTIN;
  const id = courses.some((c) => c.id === wanted) ? wanted : BUILTIN;
  select.value = id;
  await selectCourse(id);
}

async function selectCourse(id) {
  store(`course:${user.id}`, id);
  course = await api(`/api/courses/${id}`);
  resetChat();
  renderCourse();
  schedulePoll();
}

async function refreshCourse() {
  if (!course) return;
  try {
    course = await api(`/api/courses/${course.id}`);
    renderVideos();
    renderWelcome();
  } catch { /* course may have been deleted */ }
  schedulePoll();
}

function isProcessing(v) { return !["ready", "failed"].includes(v.status); }

function schedulePoll() {
  clearTimeout(pollTimer);
  // Poll quickly while something is processing, slowly otherwise so uploads from other people show up
  if (course && !course.builtin) {
    pollTimer = setTimeout(refreshCourse, course.videos.some(isProcessing) ? 2000 : 10000);
  }
}

function readyVideos() { return course ? course.videos.filter((v) => v.status === "ready") : []; }

function renderCourse() {
  const custom = !course.builtin;
  $("uploadBtn").hidden = !custom;
  $("deleteCourseBtn").hidden = !custom;
  $("topbarTitle").textContent = course.name;
  $("dropCourse").textContent = course.name;
  renderVideos();
  renderWelcome();
}

function renderWelcome() {
  const custom = !course.builtin;
  const ready = readyVideos();
  const empty = custom && course.videos.length === 0 && uploads.length === 0;

  $("welcomeTitle").textContent = empty ? `Upload videos to “${course.name}”` : `Ask anything about ${course.builtin ? "the course" : "“" + course.name + "”"}`;
  $("welcomeText").textContent = empty
    ? "Your videos are transcribed automatically. Once a video shows Ready, ask questions and I'll point you to the exact moment it's explained."
    : custom && ready.length === 0
      ? "Your videos are being transcribed — you can ask questions as soon as one shows Ready in the sidebar."
      : "I search across every lecture transcript and tell you exactly which video and timestamp covers your question.";
  $("dropzone").hidden = !custom;
  $("translateOpt").hidden = !custom;
  $("dzSub").innerHTML = `or <span class="link">browse files</span> — video or audio, up to ${serverInfo.max_upload_mb} MB each`;

  const sugg = $("suggestions");
  sugg.innerHTML = "";
  const items = course.builtin ? SIGMA_SUGGESTIONS : ready.slice(0, 4).map((v) => `What is explained in “${v.title}”?`);
  for (const q of items) {
    const b = el("button");
    b.type = "button";
    b.textContent = q;
    sugg.appendChild(b);
  }
}

function videoStateHtml(v) {
  if (v.status === "ready") return `<span class="vstate">${fmtTime(v.duration)}${v.language ? " · " + escapeHtml(v.language.toUpperCase()) : ""}</span>`;
  if (v.status === "failed") return `<span class="vstate failed" title="${escapeHtml(v.error || "")}">Failed — ${escapeHtml((v.error || "unknown error").slice(0, 60))}</span>`;
  const pct = Math.round((v.progress || 0) * 100);
  const label = v.status === "transcribing" && pct > 0 ? `Transcribing · ${pct}%` : escapeHtml(v.stage || v.status);
  const indeterminate = !(v.status === "transcribing" && pct > 0) && v.status !== "uploading";
  return `<span class="vstate">${label}</span><span class="bar ${indeterminate ? "indeterminate" : ""}"><i style="width:${pct}%"></i></span>`;
}

function renderVideos() {
  const list = $("videos");
  list.innerHTML = "";
  for (const u of uploads) {
    const li = el("li");
    li.innerHTML = `<button type="button" class="busy" disabled><span class="num">↑</span>
      <span class="vt">${escapeHtml(u.name)}<span class="vstate">Uploading · ${Math.round(u.progress * 100)}%</span>
      <span class="bar"><i style="width:${u.progress * 100}%"></i></span></span></button>`;
    list.appendChild(li);
  }
  for (const v of course.videos) {
    const li = el("li");
    const b = el("button", isProcessing(v) ? "busy" : "");
    b.type = "button";
    b.innerHTML = `<span class="num">${escapeHtml(v.number)}</span><span class="vt">${escapeHtml(v.title)}${videoStateHtml(v)}</span>`;
    b.addEventListener("click", () => {
      if (v.status === "ready" && v.has_media) openPlayer(v, 0);
      else if (v.status === "ready") { closeSidebar(); ask(`What is taught in the video "${v.title}"?`); }
    });
    li.appendChild(b);
    if (!course.builtin) {
      const actions = el("div", "vactions");
      if (v.status === "failed") {
        const retry = el("button", "", "Retry");
        retry.title = "Try processing again";
        retry.addEventListener("click", () => retryVideo(v));
        actions.appendChild(retry);
      }
      const del = el("button", "", '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>');
      del.title = "Delete video";
      del.setAttribute("aria-label", `Delete ${v.title}`);
      del.addEventListener("click", () => deleteVideo(v));
      actions.appendChild(del);
      li.appendChild(actions);
    }
    list.appendChild(li);
  }
  if (!course.videos.length && !uploads.length) {
    list.appendChild(el("li", "empty", course.builtin ? "No videos" : "No videos yet — upload your first lecture."));
  }
}

async function deleteVideo(v) {
  if (!confirm(`Delete “${v.title}”? Its transcript will be removed too.`)) return;
  try {
    await api(`/api/courses/${course.id}/videos/${v.id}`, { method: "DELETE" });
    await refreshCourse();
  } catch (e) { toast(e.message); }
}

async function retryVideo(v) {
  try {
    await api(`/api/courses/${course.id}/videos/${v.id}/retry`, { method: "POST" });
    await refreshCourse();
  } catch (e) { toast(e.message); }
}

// New course dialog
const courseDialog = $("courseDialog");
$("newCourseBtn").addEventListener("click", () => { $("courseName").value = ""; courseDialog.showModal(); });
$("courseCancel").addEventListener("click", () => courseDialog.close());
$("courseForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("courseName").value.trim();
  if (!name) return;
  try {
    const c = await api("/api/courses", jsonPost({ name }));
    courseDialog.close();
    closeSidebar();
    await loadCourses(c.id);
  } catch (err) { toast(err.message); }
});

$("deleteCourseBtn").addEventListener("click", async () => {
  if (!confirm(`Delete the course “${course.name}” and all of its videos? This can't be undone.`)) return;
  try {
    await api(`/api/courses/${course.id}`, { method: "DELETE" });
    await loadCourses(BUILTIN);
  } catch (e) { toast(e.message); }
});

$("courseSelect").addEventListener("change", (e) => { closeSidebar(); selectCourse(e.target.value); });

// ---------- Uploads ----------
const fileInput = $("fileInput");
$("uploadBtn").addEventListener("click", () => fileInput.click());
$("dropzone").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { uploadFiles([...fileInput.files]); fileInput.value = ""; });

function uploadFiles(files) {
  if (!course || course.builtin) {
    toast("Create a course first, then upload your videos to it.");
    return;
  }
  const cid = course.id;
  const translate = $("translateChk").checked;
  for (const file of files) {
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (serverInfo.extensions.length && !serverInfo.extensions.includes(ext)) {
      toast(`${file.name}: unsupported file type`);
      continue;
    }
    if (file.size > serverInfo.max_upload_mb * 1024 * 1024) {
      toast(`${file.name} is larger than ${serverInfo.max_upload_mb} MB`);
      continue;
    }
    const u = { key: Math.random(), name: file.name, progress: 0 };
    uploads.push(u);
    const data = new FormData();
    data.append("file", file);
    data.append("translate", translate ? "true" : "false");
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/courses/${cid}/videos`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) { u.progress = e.loaded / e.total; if (course.id === cid) renderVideos(); }
    };
    xhr.onloadend = () => {
      uploads = uploads.filter((x) => x !== u);
      if (xhr.status === 401) return showAuth();
      if (xhr.status !== 200) {
        let msg = "Upload failed";
        try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
        toast(`${file.name}: ${msg}`);
      }
      if (course.id === cid) refreshCourse();
    };
    xhr.send(data);
  }
  renderVideos();
  renderWelcome();
}

// Drag & drop anywhere on the main area
const main = $("main");
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
main.addEventListener("dragenter", (e) => {
  if (!hasFiles(e) || course?.builtin) return;
  dragDepth++;
  main.classList.add("dragging");
});
main.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; main.classList.remove("dragging"); } });
main.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
main.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  main.classList.remove("dragging");
  if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]);
});

// ---------- Player ----------
const playerDialog = $("playerDialog"), player = $("player");
let playerSrc = null;

function openPlayer(v, start) {
  const src = `/api/courses/${course.id}/videos/${v.id}/media`;
  $("playerTitle").textContent = `Video ${v.number}: ${v.title}`;
  $("playerSub").textContent = start ? `Starting at ${fmtTime(start)}` : "";
  const seek = () => { player.currentTime = Math.max(0, start - 1); player.play().catch(() => {}); };
  if (playerSrc !== src) {
    playerSrc = src;
    player.src = src;
    player.addEventListener("loadedmetadata", seek, { once: true });
  } else {
    seek();
  }
  playerDialog.showModal();
}

function closePlayer() { player.pause(); playerDialog.close(); }
$("playerClose").addEventListener("click", closePlayer);
playerDialog.addEventListener("click", (e) => { if (e.target === playerDialog) closePlayer(); });
playerDialog.addEventListener("close", () => player.pause());

// ---------- Chat rendering ----------
function addUserMessage(text) {
  const msg = el("div", "msg user");
  const bubble = el("div", "bubble");
  bubble.textContent = text;
  msg.appendChild(bubble);
  messages.appendChild(msg);
}

function addBotMessage() {
  const msg = el("div", "msg bot");
  msg.appendChild(el("div", "avatar", "Σ"));
  const body = el("div", "bot-body");
  const answer = el("div", "answer", '<div class="typing"><span></span><span></span><span></span></div>');
  body.appendChild(answer);
  msg.appendChild(body);
  messages.appendChild(msg);
  return { body, answer };
}

function renderSources(body, sources, courseRecord) {
  if (!sources.length) return;
  const top = sources[0].similarity_score || 1;
  body.appendChild(el("div", "sources-label", "Where to watch"));
  const grid = el("div", "sources");
  for (const s of sources) {
    const video = s.video_id && courseRecord.videos.find((v) => v.id === s.video_id);
    const playable = video && video.has_media;
    const pct = Math.max(8, Math.round((s.similarity_score / top) * 100));
    const card = el("button", "source");
    card.type = "button";
    card.title = playable ? `Play from ${fmtTime(s.start)}` : "Show full transcript excerpt";
    card.innerHTML = `
      <div class="source-top">
        <span class="vnum">Video ${escapeHtml(s.number)}</span>
        <span class="time">
          <svg viewBox="0 0 24 24" width="12" height="12" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
          ${fmtTime(s.start)} – ${fmtTime(s.end)}
        </span>
      </div>
      <div class="source-title">${escapeHtml(s.title)}</div>
      <div class="source-text">${escapeHtml(s.text.trim())}</div>
      ${playable ? `<div class="play-hint">▶ Play from ${fmtTime(s.start)}</div>` : ""}
      <div class="match" title="Relevance"><i style="width:${pct}%"></i></div>`;
    card.addEventListener("click", () => {
      if (playable && course.id === courseRecord.id) openPlayer(video, s.start);
      else card.classList.toggle("open");
    });
    grid.appendChild(card);
  }
  body.appendChild(grid);
}

function renderMeta(body, llm) {
  const label = llm === "extractive" ? "Answered from transcripts" : llm === "none" ? "No matching video" : `Answered by ${llm}`;
  body.appendChild(el("div", "meta", `<span class="badge">${escapeHtml(label)}</span>`));
}

function resetChat() {
  messages.innerHTML = "";
  welcome.style.display = "";
  input.placeholder = course ? `Ask a question about ${course.builtin ? "the course" : course.name}…` : "Ask a question…";
}

// ---------- Chat ----------
async function ask(question) {
  question = question.trim();
  if (!question || busy || !course) return;
  busy = true;
  sendBtn.disabled = true;
  welcome.style.display = "none";
  addUserMessage(question);
  const { body, answer } = addBotMessage();
  const courseRecord = course;
  scrollToBottom();

  let text = "", sources = [], llm = null;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, course_id: courseRecord.id }),
    });
    if (res.status === 401) { showAuth(); throw new Error("Your session expired"); }
    if (!res.ok || !res.body) throw new Error(`Server returned ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);
        if (!line.trim()) continue;
        const evt = JSON.parse(line);
        if (evt.type === "sources") {
          sources = evt.sources;
          llm = evt.llm;
        } else if (evt.type === "token") {
          text += evt.text;
          answer.innerHTML = renderMarkdown(text);
          scrollToBottom();
        }
      }
    }
    renderSources(body, sources, courseRecord);
    if (llm) renderMeta(body, llm);
  } catch (err) {
    answer.innerHTML = `<p class="error">Sorry, something went wrong: ${escapeHtml(err.message)}. Is the server running?</p>`;
  } finally {
    busy = false;
    sendBtn.disabled = !input.value.trim();
    scrollToBottom();
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value;
  input.value = "";
  autoGrow();
  ask(q);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    form.requestSubmit();
  }
});

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
  sendBtn.disabled = busy || !input.value.trim();
}
input.addEventListener("input", autoGrow);

$("suggestions").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") ask(e.target.textContent);
});

$("newChat").addEventListener("click", () => {
  if (busy) return;
  resetChat();
  closeSidebar();
  input.focus();
});

// ---------- Sidebar ----------
const sidebar = $("sidebar"), scrim = $("scrim");
function closeSidebar() { sidebar.classList.remove("open"); scrim.classList.remove("open"); }
$("menuBtn").addEventListener("click", () => { sidebar.classList.add("open"); scrim.classList.add("open"); });
scrim.addEventListener("click", closeSidebar);

async function loadStatus() {
  const box = $("status"), text = $("statusText");
  try {
    serverInfo = await api("/api/status");
    const retrieval = serverInfo.retrieval === "ollama" ? "semantic search" : `${serverInfo.retrieval} search`;
    if (serverInfo.llm) {
      box.className = "status ok";
      text.textContent = `${serverInfo.llm} · ${retrieval}`;
    } else {
      box.className = "status partial";
      text.textContent = `Transcript answers · ${retrieval}`;
      box.title = "Start Ollama with llama3.2 to get AI-written answers.";
    }
  } catch {
    box.className = "status";
    text.textContent = "Server offline";
  }
}

// ---------- Accounts ----------
let authMode = "login";

function setAuthMode(mode) {
  authMode = mode;
  const signup = mode === "signup";
  $("tabLogin").classList.toggle("active", !signup);
  $("tabSignup").classList.toggle("active", signup);
  $("nameField").hidden = !signup;
  $("authTitle").textContent = signup ? "Create your account" : "Welcome back";
  $("authSub").textContent = signup ? "Upload your lectures and ask them anything." : "Log in to your teaching assistant.";
  $("authSubmit").textContent = signup ? "Create account" : "Log in";
  $("authPassword").autocomplete = signup ? "new-password" : "current-password";
  $("authError").textContent = "";
  (signup ? $("authName") : $("authEmail")).focus();
}
$("tabLogin").addEventListener("click", () => setAuthMode("login"));
$("tabSignup").addEventListener("click", () => setAuthMode("signup"));

function showAuth() {
  if (!$("auth").hidden) return;
  user = null;
  course = null;
  clearTimeout(pollTimer);
  uploads = [];
  try { playerDialog.close(); courseDialog.close(); } catch {}
  player.removeAttribute("src");
  playerSrc = null;
  $("app").hidden = true;
  $("auth").hidden = false;
  $("authPassword").value = "";
  setAuthMode(authMode);
}

async function enterApp(u) {
  user = u;
  $("auth").hidden = true;
  $("app").hidden = false;
  $("accountAvatar").textContent = (u.name || u.email).trim().charAt(0).toUpperCase();
  $("accountName").textContent = u.name;
  $("accountEmail").textContent = u.email;
  await loadStatus();
  try { await loadCourses(); } catch (e) { toast(e.message); }
  sendBtn.disabled = true;
  input.focus();
}

$("authForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const signup = authMode === "signup";
  const body = { email: $("authEmail").value.trim(), password: $("authPassword").value };
  if (signup) body.name = $("authName").value.trim();
  const err = $("authError");
  if (signup && !body.name) return (err.textContent = "Please enter your name.");
  if (!body.email) return (err.textContent = "Please enter your email.");
  if (signup && body.password.length < 8) return (err.textContent = "Password must be at least 8 characters.");
  if (!body.password) return (err.textContent = "Please enter your password.");

  const btn = $("authSubmit");
  btn.disabled = true;
  err.textContent = "";
  try {
    const u = await api(signup ? "/api/auth/signup" : "/api/auth/login", jsonPost(body));
    $("authPassword").value = "";
    await enterApp(u);
  } catch (ex) {
    err.textContent = ex.message;
  } finally {
    btn.disabled = false;
  }
});

$("logoutBtn").addEventListener("click", async () => {
  try { await api("/api/auth/logout", { method: "POST" }); } catch {}
  messages.innerHTML = "";
  showAuth();
});

(async () => {
  try {
    await enterApp(await api("/api/auth/me"));
  } catch {
    showAuth();
  }
})();
