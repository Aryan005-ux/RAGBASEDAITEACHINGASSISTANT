# How to use this RAG AI Teaching assistant on your own data
## Step 1 - Collect your videos
Move all your video files to the videos folder

## Step 2 - Convert to mp3
Convert all the video files to mp3 by ruunning video_to_mp3

## Step 3 - Convert mp3 to json 
Convert all the mp3 files to json by ruunning mp3_to_json

## Step 4 - Convert the json files to Vectors
Use the file preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and feeding to LLM

Read the joblib file and load it into the memory. Then create a relevant prompt as per the user query and feed it to the LLM



## Running the website

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000
```

Then open http://localhost:8000.

- **Without Ollama** it works out of the box: questions are matched against the transcripts with keyword (TF-IDF) search and the answer points you to the right video and timestamp.
- **With Ollama** (`ollama pull bge-m3 && ollama pull llama3.2`, then `ollama serve`) it switches to semantic search using the same bge-m3 embeddings the index was built with, and llama3.2 writes the answers, streamed live. Restart the server after starting Ollama.

The sidebar status shows which mode is active. Settings live in `app/config.py` (`MODEL_BACKEND`, `OLLAMA_URL`, `OLLAMA_LLM_MODEL`, `LLM_BACKEND`).

### Uploading your own videos

After signing up, anyone can create a course (the **+** next to the course picker) and drag-and-drop lecture videos or audio onto it. Each file is processed automatically in the background:

1. **Transcribe** with Whisper ([faster-whisper](https://github.com/SYSTRAN/faster-whisper), runs locally on CPU). Non-English speech can be translated to English, like the original Hindi pipeline.
2. **Chunk and index** the transcript (plus bge-m3 embeddings when Ollama is running).
3. Status shows live in the sidebar (Queued → Transcribing % → Ready). Once a video is ready, ask questions; clicking a source card plays the video from that timestamp.

Uploads are stored under `data/courses/<course_id>/` (the video file, its transcript, and embeddings). The first upload downloads the Whisper model (~460 MB for `small`). Change `WHISPER_MODEL` (`tiny`, `base`, `small`, `medium`, `large-v3`) to trade speed for accuracy, and `MAX_UPLOAD_MB` to change the size limit.

### Accounts

Everyone signs up with a name, email and password. Each course belongs to the account that created it: other users can't see, chat with, play, upload to, or delete it (they get "not found"). The Sigma demo course is available to every logged-in user.

- Accounts and sessions live in SQLite at `data/app.db`. Passwords are hashed with PBKDF2-SHA256 (600k iterations); sessions are HttpOnly, SameSite=Lax cookies valid for 30 days, and the cookie is marked `Secure` automatically when served over HTTPS.
- After 10 failed logins for an email or IP address, logins are paused for 15 minutes.
- Courses created before accounts existed are given to the first account that signs up.
- Not included yet: password reset by email and per-user storage limits. Serve the site over HTTPS when it's public.

With Docker: `docker compose up --build`, then open http://localhost:8080.

## Streamlit version (easiest to deploy)

`streamlit_app.py` is a Streamlit front end with the same features (accounts, courses, uploads with live transcription progress, chat with timestamped sources, playback from the cited moment). It reuses the backend in `app/`.

```bash
.venv/bin/streamlit run streamlit_app.py
```

### Deploy free on Streamlit Community Cloud

1. Push this folder to a GitHub repository (`.gitignore` already leaves out `.venv/` and `data/`).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, pick the repo, branch, and `streamlit_app.py` as the main file.
3. Under **Advanced settings**, choose Python 3.11 or 3.12, then **Deploy**.

Limits to know on Community Cloud:
- **Storage is temporary.** Accounts, uploaded videos and transcripts are kept on the app's local disk, which is wiped whenever the app restarts, redeploys, or wakes up after sleeping. The Sigma demo course is always available because it ships with the repo. For permanent storage, move `data/` to a database and object storage (e.g. Postgres + S3) or host on a server with a disk.
- **Small machine.** Transcription uses Whisper `base` by default here (set `WHISPER_MODEL` in the app's secrets to change it) and runs on a shared CPU, so long videos take a while. Uploads are limited to 200 MB each (`.streamlit/config.toml`).
- **No Ollama.** Answers come straight from the matching transcript passages, using keyword search.
- **Logins last for the browser tab.** Refreshing the page logs you out.
