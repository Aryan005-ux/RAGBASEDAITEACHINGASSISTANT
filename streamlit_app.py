"""Streamlit front end for the RAG AI Teaching Assistant.

Reuses the same backend as the FastAPI site (app/): transcript search, Whisper
transcription worker, course storage and user accounts.

Run locally:  streamlit run streamlit_app.py
"""
import os
import shutil

# Smaller Whisper model by default: Streamlit Community Cloud has limited memory and CPU
os.environ.setdefault("WHISPER_MODEL", "base")

import streamlit as st

st.set_page_config(page_title="AI Teaching Assistant", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading the course index…")
def load_backend():
    """Load the search index and start the single background transcription worker (once per server)."""
    from app.courses import store, worker
    worker.start()
    worker.resume_unfinished()
    return store, worker


store, worker = load_backend()

from app import auth, llm  # noqa: E402  (imported after the backend is initialised)
from app.courses import BUILTIN_ID, VIDEO_EXTS  # noqa: E402
from app.model_loader import model_loader  # noqa: E402

SIGMA_SUGGESTIONS = [
    "Where is HTML concluded in this course?",
    "How do I install VS Code?",
    "Explain the CSS box model",
    "How do forms and input tags work?",
]
LLM_LABELS = {"extractive": "Answered from transcripts", "none": "No matching video"}


def fmt_time(seconds) -> str:
    return llm.fmt_time(seconds or 0)


# ---------- Accounts ----------
def login_page():
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.title("🎓 AI Teaching Assistant")
        st.caption("Upload your lecture videos and ask them anything — I'll point you to the exact moment it's explained.")
        login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
        with login_tab:
            with st.form("login"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", type="primary", width="stretch"):
                    try:
                        ip = getattr(st.context, "ip_address", None) or ""
                        st.session_state.user = auth.authenticate(email, password, ip)
                        st.rerun()
                    except (ValueError, PermissionError) as e:
                        st.error(str(e))
        with signup_tab:
            with st.form("signup"):
                name = st.text_input("Name")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input("Password (at least 8 characters)", type="password", key="signup_password")
                if st.form_submit_button("Create account", type="primary", width="stretch"):
                    try:
                        first_user = auth.user_count() == 0
                        user = auth.create_user(email, name, password)
                        if first_user:
                            store.claim_unowned(user["id"])
                        st.session_state.user = user
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
        st.caption("🔒 Your courses and videos are private to your account.")


# ---------- Sidebar ----------
def sidebar(user: dict) -> str:
    with st.sidebar:
        st.markdown("### 🎓 Teaching Assistant")

        courses = store.list_courses(user["id"])
        labels = {c["id"]: c["name"] + (" (demo)" if c["builtin"] else "") for c in courses}
        if "next_course" in st.session_state:  # set by actions that run after the selectbox exists
            st.session_state.course_id = st.session_state.pop("next_course")
        if st.session_state.get("course_id") not in labels:
            st.session_state.course_id = BUILTIN_ID
        course_id = st.selectbox("Course", list(labels), format_func=labels.get, key="course_id")

        with st.expander("➕ New course"):
            with st.form("new_course", clear_on_submit=True):
                name = st.text_input("Course name", max_chars=120, placeholder="e.g. React for Beginners")
                if st.form_submit_button("Create course") and name.strip():
                    course = store.create_course(name.strip(), user["id"])
                    st.session_state.next_course = course["id"]
                    st.rerun()

        if st.button("🧹 New chat", width="stretch"):
            st.session_state.chats.pop(course_id, None)
            st.session_state.player = None
            st.rerun()

        st.divider()
        mode = "semantic search" if model_loader.backend == "ollama" else f"{model_loader.backend} search"
        answers = llm.OLLAMA_LLM_MODEL if llm.llm_available() else "transcript answers"
        st.caption(f"⚙️ {answers} · {mode}")
        st.markdown(f"**{user['name']}**  \n{user['email']}")
        if st.button("Log out", width="stretch"):
            for key in ("user", "chats", "player", "course_id"):
                st.session_state.pop(key, None)
            st.rerun()
    return course_id


# ---------- Chat ----------
def render_sources(sources: list, course_id: str, key_prefix: str):
    if not sources:
        return
    st.markdown("**Where to watch**")
    cols = st.columns(min(3, len(sources)))
    for i, s in enumerate(sources):
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**Video {s['number']} · {s['title']}**  \n⏱ `{fmt_time(s['start'])} – {fmt_time(s['end'])}`")
                text = s["text"].strip()
                st.caption(text[:220] + ("…" if len(text) > 220 else ""))
                if s.get("video_id") and st.button(f"▶ Play from {fmt_time(s['start'])}", key=f"{key_prefix}-{i}"):
                    st.session_state.player = {"course_id": course_id, "video_id": s["video_id"],
                                               "start": s["start"], "title": f"Video {s['number']}: {s['title']}"}
                    st.rerun()


def player(course_id: str, user: dict):
    p = st.session_state.get("player")
    if not p or p["course_id"] != course_id or not store.owned(course_id, user["id"]):
        return
    video = store.get_video(course_id, p["video_id"])
    if not video:
        st.session_state.player = None
        return
    with st.container(border=True):
        head, close = st.columns([6, 1])
        head.markdown(f"**{p['title']}** — starting at {fmt_time(p['start'])}")
        if close.button("✕ Close", key="close_player"):
            st.session_state.player = None
            st.rerun()
        st.video(store.media_path(course_id, video), start_time=max(0, int(p["start"]) - 1))


def chat_tab(course_id: str, course_name: str, user: dict, ready_titles: list):
    history = st.session_state.chats.setdefault(course_id, [])

    if not history:
        st.markdown(f"#### Ask anything about {course_name}")
        st.caption("I search every lecture transcript and tell you which video and timestamp covers your question.")
        suggestions = SIGMA_SUGGESTIONS if course_id == BUILTIN_ID else [f"What is explained in “{t}”?" for t in ready_titles[:4]]
        if suggestions:
            cols = st.columns(2)
            for i, q in enumerate(suggestions):
                if cols[i % 2].button(q, key=f"sugg-{course_id}-{i}", width="stretch"):
                    st.session_state.pending = q
                    st.rerun()

    for n, msg in enumerate(history):
        with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []), course_id, f"{course_id}-{n}")
                if msg.get("llm"):
                    st.caption(LLM_LABELS.get(msg["llm"], f"Answered by {msg['llm']}"))

    player(course_id, user)  # below the conversation, next to the Play button that opened it

    prompt = st.chat_input(f"Ask a question about {course_name}…") or st.session_state.pop("pending", None)
    if not prompt:
        return
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🎓"):
        index = store.get_index(course_id)
        if not index.metadata:
            answer, source, sources = ("This course has no processed videos yet. Upload a video in the **Videos** tab "
                                       "and wait until it shows Ready, then ask away."), "none", []
            st.markdown(answer)
        else:
            with st.spinner("Searching the transcripts…"):
                chunks = index.search_segments(prompt, top_k=5)
                gen = llm.generate_answer(prompt, chunks, index, course_name)
                source = next(gen)
            answer = st.write_stream(gen)
            sources = chunks if source != "none" else []
    history.append({"role": "assistant", "content": answer, "sources": sources, "llm": source})
    st.rerun()  # redraw from history so the new sources get working Play buttons


# ---------- Videos ----------
@st.fragment(run_every=3)
def video_list(course_id: str, user_id: str):
    """Refreshes every few seconds so transcription progress stays live."""
    course = store.owned(course_id, user_id)
    if not course:
        return
    if not course["videos"]:
        st.info("No videos yet — upload your first lecture above.")
    for v in course["videos"]:
        with st.container(border=True):
            info, actions = st.columns([6, 1])
            info.markdown(f"**{v['number']}. {v['title']}**")
            if v["status"] == "ready":
                info.caption(f"✅ Ready · {fmt_time(v['duration'])}" + (f" · {v['language'].upper()}" if v.get("language") else ""))
            elif v["status"] == "failed":
                info.error(f"Failed: {v.get('error') or 'unknown error'}")
                if actions.button("Retry", key=f"retry-{v['id']}"):
                    store.update_video(course_id, v["id"], error=None)
                    worker.enqueue(course_id, v["id"])
                    st.rerun(scope="fragment")
            else:
                progress = float(v.get("progress") or 0)
                label = f"Transcribing · {int(progress * 100)}%" if v["status"] == "transcribing" and progress else v.get("stage", v["status"])
                info.progress(progress, text=label)
            if actions.button("🗑", key=f"del-{v['id']}", help="Delete this video"):
                store.delete_video(course_id, v["id"])
                st.rerun()


def videos_tab(course_id: str, user: dict):
    if course_id == BUILTIN_ID:
        st.info("This is the built-in demo course (read-only). Create your own course in the sidebar to upload videos.")
        videos = model_loader.index.list_videos()
        st.dataframe(
            [{"#": v["number"], "Title": v["title"], "Length": fmt_time(v["duration"])} for v in videos],
            hide_index=True, width="stretch",
        )
        return

    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader(
            "Upload lecture videos or audio",
            type=sorted(e.lstrip(".") for e in VIDEO_EXTS),
            accept_multiple_files=True,
        )
        translate = st.checkbox("Translate non-English speech to English (recommended for Hindi / Hinglish lectures)", value=True)
        submitted = st.form_submit_button("Upload and transcribe", type="primary")
    if submitted and files:
        for f in files:
            video = store.add_video(course_id, f.name, translate)
            with open(store.media_path(course_id, video), "wb") as out:
                shutil.copyfileobj(f, out)
            worker.enqueue(course_id, video["id"])
        st.success(f"Uploaded {len(files)} file(s). Transcription runs in the background — you can keep using the app.")

    video_list(course_id, user["id"])

    with st.expander("⚠️ Danger zone"):
        sure = st.checkbox("I understand this deletes the course and all of its videos")
        if st.button("Delete this course", disabled=not sure):
            store.delete_course(course_id)
            st.session_state.next_course = BUILTIN_ID
            st.rerun()


# ---------- Page ----------
def main():
    user = st.session_state.get("user")
    if not user:
        login_page()
        return
    st.session_state.setdefault("chats", {})
    st.session_state.setdefault("player", None)

    course_id = sidebar(user)
    if course_id == BUILTIN_ID:
        course_name, ready_titles = store.course_name(BUILTIN_ID), []
    else:
        course = store.owned(course_id, user["id"])
        if not course:
            st.session_state.next_course = BUILTIN_ID
            st.rerun()
        course_name = course["name"]
        ready_titles = [v["title"] for v in course["videos"] if v["status"] == "ready"]

    st.title(course_name)
    ask, videos = st.tabs(["💬 Ask", "🎬 Videos"])
    with ask:
        chat_tab(course_id, course_name, user, ready_titles)
    with videos:
        videos_tab(course_id, user)


main()
