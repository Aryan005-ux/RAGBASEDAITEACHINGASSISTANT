import json
import requests
from app.config import OLLAMA_URL, OLLAMA_LLM_MODEL, LLM_BACKEND

# Below this retrieval score the question is treated as unrelated to the course
MIN_RELEVANCE = {"keyword": 0.08, "dense": 0.45}


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def llm_available() -> bool:
    if LLM_BACKEND == "none":
        return False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return any(n.split(":")[0] == OLLAMA_LLM_MODEL.split(":")[0] for n in names)
    except Exception:
        return False


def is_relevant(query: str, chunks: list, index) -> bool:
    if not chunks:
        return False
    if chunks[0]["similarity_score"] < MIN_RELEVANCE[index.mode]:
        return False
    if index.mode == "keyword":
        return index.keyword_coverage(query, chunks[0]) >= 0.5
    return True


def build_prompt(query: str, chunks: list, course_name: str) -> str:
    context = [
        {
            "title": c["title"],
            "video_number": c["number"],
            "start": fmt_time(c["start"]),
            "end": fmt_time(c["end"]),
            "text": c["text"],
        }
        for c in chunks
    ]
    return f'''I am teaching the course "{course_name}". Here are video subtitle chunks containing video title, video number, start time, end time, and the text spoken at that time:

{json.dumps(context, ensure_ascii=False)}
---------------------------------
"{query}"
User asked this question related to the video chunks. Answer in a friendly, human way (don't mention the above format, it's just for you). Briefly explain the answer, then say in which video and at what timestamp it is taught and guide the user to go to that particular video. Use short paragraphs or bullet points and **bold** the video names. If the user asks an unrelated question, tell them that you can only answer questions related to the course.
'''


def stream_ollama(prompt: str):
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": True},
        stream=True,
        timeout=(5, 300),
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("response"):
                yield data["response"]
            if data.get("done"):
                break


def extractive_answer(query: str, chunks: list) -> str:
    """Answer built straight from the retrieved transcript segments (no LLM needed)."""
    best = chunks[0]
    lines = [
        f"This is covered in **Video {best['number']}: {best['title']}** — "
        f"jump to **{fmt_time(best['start'])}**.",
        "",
        f"> {best['text'].strip()}",
        "",
    ]
    others = chunks[1:]
    if others:
        lines.append("You'll also find related explanations here:")
        lines.append("")
        for c in others:
            snippet = c["text"].strip()
            if len(snippet) > 140:
                snippet = snippet[:140].rsplit(" ", 1)[0] + "…"
            lines.append(f"- **Video {c['number']}: {c['title']}** at {fmt_time(c['start'])} — “{snippet}”")
    return "\n".join(lines)


def off_topic_answer(course_name: str) -> str:
    return (
        f"I couldn't find that in the videos. I can only answer questions about **{course_name}** — "
        "try asking about a topic that's discussed in one of its videos."
    )


def generate_answer(query: str, chunks: list, index, course_name: str):
    """Yields the answer text piece by piece; the first item names what produced it
    ("none" when off-topic, "extractive" without an LLM, otherwise the Ollama model name)."""
    if not is_relevant(query, chunks, index):
        yield "none"
        yield off_topic_answer(course_name)
        return
    if llm_available():
        try:
            pieces = stream_ollama(build_prompt(query, chunks, course_name))
            first = next(pieces)
            yield OLLAMA_LLM_MODEL
            yield first
            yield from pieces
            return
        except Exception as e:
            print(f"Ollama generation failed, using extractive answer: {e}")
    yield "extractive"
    yield extractive_answer(query, chunks)
