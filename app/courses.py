"""User-created courses: storage, background transcription, and per-course search indexes.

Layout on disk (under DATA_DIR/courses/<course_id>/):
    course.json              course name + list of videos and their processing status
    media/<video_id>.<ext>   the uploaded file
    transcripts/<video_id>.json   Whisper chunks, same shape as the files in jsons/
    embeddings/<video_id>.npy     bge-m3 vectors (only when Ollama was available)
"""
import os
import re
import json
import time
import uuid
import queue
import shutil
import threading
import numpy as np
from app.config import DATA_DIR, WHISPER_MODEL, WHISPER_COMPUTE_TYPE
from app.model_loader import model_loader, TranscriptIndex

BUILTIN_ID = "sigma"
BUILTIN_NAME = "Sigma Web Development Course"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
ID_RE = re.compile(r"^[a-f0-9]{10}$")
COURSES_DIR = os.path.join(DATA_DIR, "courses")


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def valid_id(value: str) -> bool:
    return bool(ID_RE.match(value or ""))


def title_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"^\d+[\s._-]+", "", stem)  # drop "02_" style numbering
    stem = re.sub(r"[_]+", " ", stem).strip()
    return stem or "Untitled video"


class CourseStore:
    def __init__(self):
        os.makedirs(COURSES_DIR, exist_ok=True)
        self.lock = threading.RLock()
        self.index_cache = {}

    # ---------- course.json helpers ----------
    def course_dir(self, cid: str) -> str:
        return os.path.join(COURSES_DIR, cid)

    def load(self, cid: str):
        path = os.path.join(self.course_dir(cid), "course.json")
        if not valid_id(cid) or not os.path.exists(path):
            return None
        with self.lock, open(path) as f:
            return json.load(f)

    def save(self, course: dict):
        path = os.path.join(self.course_dir(course["id"]), "course.json")
        tmp = path + ".tmp"
        with self.lock:
            with open(tmp, "w") as f:
                json.dump(course, f, indent=2)
            os.replace(tmp, path)

    def update_video(self, cid: str, vid: str, **fields):
        """Merge fields into one video's record. Returns False if the video was deleted meanwhile."""
        with self.lock:
            course = self.load(cid)
            if not course:
                return False
            for v in course["videos"]:
                if v["id"] == vid:
                    v.update(fields)
                    self.save(course)
                    return True
            return False

    # ---------- courses ----------
    def all_course_ids(self) -> list:
        return [cid for cid in os.listdir(COURSES_DIR) if self.load(cid)]

    def owned(self, cid: str, user_id: str):
        """The course if it belongs to user_id, else None (callers answer 404 either way)."""
        course = self.load(cid)
        return course if course and course.get("owner_id") == user_id else None

    def claim_unowned(self, user_id: str):
        """Give courses created before accounts existed to the given user."""
        with self.lock:
            for cid in self.all_course_ids():
                course = self.load(cid)
                if not course.get("owner_id"):
                    course["owner_id"] = user_id
                    self.save(course)

    def list_courses(self, user_id: str) -> list:
        builtin = {
            "id": BUILTIN_ID, "name": BUILTIN_NAME, "builtin": True,
            "videos": len(model_loader.index.video_rows), "ready": len(model_loader.index.video_rows),
        }
        courses = []
        for cid in os.listdir(COURSES_DIR):
            c = self.load(cid)
            if c and c.get("owner_id") == user_id:
                courses.append({
                    "id": c["id"], "name": c["name"], "builtin": False, "created": c["created"],
                    "videos": len(c["videos"]), "ready": sum(v["status"] == "ready" for v in c["videos"]),
                })
        courses.sort(key=lambda c: c["created"], reverse=True)
        return [builtin] + courses

    def create_course(self, name: str, owner_id: str) -> dict:
        cid = new_id()
        for sub in ("media", "transcripts", "embeddings"):
            os.makedirs(os.path.join(self.course_dir(cid), sub), exist_ok=True)
        course = {"id": cid, "name": name, "owner_id": owner_id, "created": time.time(), "videos": []}
        self.save(course)
        return course

    def delete_course(self, cid: str) -> bool:
        if not self.load(cid):
            return False
        with self.lock:
            shutil.rmtree(self.course_dir(cid), ignore_errors=True)
            self.index_cache.pop(cid, None)
        return True

    def course_name(self, cid: str) -> str:
        if cid == BUILTIN_ID:
            return BUILTIN_NAME
        c = self.load(cid)
        return c["name"] if c else ""

    # ---------- videos ----------
    def add_video(self, cid: str, filename: str, translate: bool) -> dict:
        with self.lock:
            course = self.load(cid)
            vid = new_id()
            ext = os.path.splitext(filename)[1].lower()
            number = max([int(v["number"]) for v in course["videos"]] + [0]) + 1
            video = {
                "id": vid, "number": str(number), "title": title_from_filename(filename),
                "filename": filename, "ext": ext, "translate": translate,
                "status": "uploading", "stage": "Uploading", "progress": 0.0,
                "duration": None, "language": None, "error": None, "created": time.time(),
            }
            course["videos"].append(video)
            self.save(course)
        return video

    def media_path(self, cid: str, video: dict) -> str:
        return os.path.join(self.course_dir(cid), "media", video["id"] + video["ext"])

    def get_video(self, cid: str, vid: str):
        course = self.load(cid)
        if not course:
            return None
        return next((v for v in course["videos"] if v["id"] == vid), None)

    def delete_video(self, cid: str, vid: str) -> bool:
        with self.lock:
            course = self.load(cid)
            video = next((v for v in course["videos"] if v["id"] == vid), None) if course else None
            if not video:
                return False
            course["videos"].remove(video)
            self.save(course)
            base = self.course_dir(cid)
            for path in (self.media_path(cid, video),
                         os.path.join(base, "transcripts", vid + ".json"),
                         os.path.join(base, "embeddings", vid + ".npy")):
                if os.path.exists(path):
                    os.remove(path)
            self.index_cache.pop(cid, None)
        return True

    # ---------- search ----------
    def get_index(self, cid: str):
        if cid == BUILTIN_ID:
            return model_loader.index
        with self.lock:
            if cid in self.index_cache:
                return self.index_cache[cid]
            course = self.load(cid)
            if not course:
                return None
            metadata, vectors, all_embedded = [], [], True
            base = self.course_dir(cid)
            for v in course["videos"]:
                if v["status"] != "ready":
                    continue
                with open(os.path.join(base, "transcripts", v["id"] + ".json")) as f:
                    chunks = json.load(f)["chunks"]
                for c in chunks:
                    c["number"], c["title"], c["video_id"] = v["number"], v["title"], v["id"]
                    c["chunk_id"] = len(metadata)
                    metadata.append(c)
                emb_path = os.path.join(base, "embeddings", v["id"] + ".npy")
                if os.path.exists(emb_path):
                    vectors.append(np.load(emb_path))
                else:
                    all_embedded = False
            embeddings = np.vstack(vectors) if vectors and all_embedded else None
            index = TranscriptIndex(metadata, embeddings, loader=model_loader)
            self.index_cache[cid] = index
            return index


store = CourseStore()


class TranscriptionWorker(threading.Thread):
    """Processes uploaded videos one at a time: Whisper transcription, then embedding."""

    def __init__(self):
        super().__init__(daemon=True, name="transcription-worker")
        self.jobs = queue.Queue()
        self.whisper = None

    def enqueue(self, cid: str, vid: str):
        store.update_video(cid, vid, status="queued", stage="Waiting in queue", progress=0.0)
        self.jobs.put((cid, vid))

    def resume_unfinished(self):
        """Re-queue videos that were mid-processing when the server stopped."""
        for cid in store.all_course_ids():
            course = store.load(cid)
            for v in course["videos"]:
                if v["status"] in ("queued", "transcribing", "indexing"):
                    self.enqueue(course["id"], v["id"])
                elif v["status"] == "uploading":
                    store.delete_video(course["id"], v["id"])

    def run(self):
        while True:
            cid, vid = self.jobs.get()
            try:
                self.process(cid, vid)
            except Exception as e:
                print(f"Processing {cid}/{vid} failed: {e}")
                message = str(e)
                if "Invalid data" in message or "Errno" in message:
                    # Decoder errors include server file paths; show something useful instead
                    message = "This file couldn't be read as audio or video. Is it a valid, non-corrupted media file?"
                store.update_video(cid, vid, status="failed", stage="Failed", error=message)
            finally:
                store.index_cache.pop(cid, None)

    def load_whisper(self, cid: str, vid: str):
        if self.whisper is None:
            store.update_video(cid, vid, status="transcribing", stage="Loading speech model (first run downloads it)")
            from faster_whisper import WhisperModel
            self.whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
        return self.whisper

    def process(self, cid: str, vid: str):
        video = store.get_video(cid, vid)
        if not video:
            return
        model = self.load_whisper(cid, vid)
        if not store.update_video(cid, vid, status="transcribing", stage="Transcribing", progress=0.0):
            return

        segments, info = model.transcribe(
            store.media_path(cid, video),
            task="translate" if video.get("translate") else "transcribe",
            vad_filter=True,
        )
        duration = info.duration or 0
        chunks, last_update = [], 0.0
        for seg in segments:
            text = seg.text.strip()
            if text:
                chunks.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": " " + text})
            if time.time() - last_update > 1.5 and duration:
                last_update = time.time()
                if not store.update_video(cid, vid, progress=min(seg.end / duration, 0.99)):
                    return  # video was deleted mid-transcription
        if not chunks:
            raise ValueError("No speech was found in this file.")

        base = store.course_dir(cid)
        with open(os.path.join(base, "transcripts", vid + ".json"), "w") as f:
            json.dump({"chunks": chunks, "text": " ".join(c["text"].strip() for c in chunks)}, f)

        if not store.update_video(cid, vid, status="indexing", stage="Building search index", progress=1.0,
                                  duration=duration, language=info.language):
            return
        if model_loader.backend == "ollama":
            texts = [c["text"] for c in chunks]
            vectors = np.vstack([model_loader.embed_text(texts[i:i + 64]) for i in range(0, len(texts), 64)])
            np.save(os.path.join(base, "embeddings", vid + ".npy"), vectors)

        store.update_video(cid, vid, status="ready", stage="Ready")


worker = TranscriptionWorker()
