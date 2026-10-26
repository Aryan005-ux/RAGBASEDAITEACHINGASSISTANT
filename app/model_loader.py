import os
import re
import joblib
import numpy as np
import requests
from app.config import MODEL_NAME, MODEL_BACKEND, INDEX_PATH, OLLAMA_URL, OLLAMA_EMBED_MODEL

# Number of neighbouring subtitle lines merged into each searchable/displayed window.
# Single Whisper segments are only a few words long, so context matters.
CONTEXT_RADIUS = 3


# Question words that say nothing about the topic; ignored so they don't count against a match
QUESTION_WORDS = {
    "does", "did", "doing", "explain", "explained", "explanation", "tell", "mean", "means", "meaning",
    "define", "definition", "difference", "differences", "example", "examples", "work", "works",
    "working", "happen", "happens", "video", "videos", "lecture", "lesson", "course", "teach",
    "taught", "learn", "talk", "talked", "covered", "discussed", "understand", "know", "want", "use", "used", "using",
}


def normalize_text(text: str) -> str:
    # Split camelCase so "useEffect" matches transcripts that say "use effect"
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower()

class TranscriptIndex:
    """Searchable set of transcript chunks for one course.

    Uses dense embeddings when they are available for every chunk and the loader has a
    matching embedding model; otherwise falls back to TF-IDF keyword search."""

    def __init__(self, metadata: list, embeddings: np.ndarray = None, loader=None):
        self.metadata = metadata
        self.loader = loader
        self.embeddings_matrix = None
        if embeddings is not None and len(embeddings) == len(metadata):
            self.embeddings_matrix = embeddings.astype(np.float32)
            self.embeddings_matrix /= np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-9

        # Row positions grouped by video and sorted by time, used to pull neighbouring lines
        self.video_rows = {}
        for idx, item in enumerate(self.metadata):
            self.video_rows.setdefault(str(item['number']), []).append(idx)
        self.row_position = {}
        for rows in self.video_rows.values():
            rows.sort(key=lambda i: self.metadata[i]['start'])
            for pos, idx in enumerate(rows):
                self.row_position[idx] = pos

        if self.metadata:
            self.build_keyword_index()

    @property
    def mode(self) -> str:
        if self.embeddings_matrix is not None and self.loader is not None and self.loader.backend != "keyword":
            return "dense"
        return "keyword"

    def build_keyword_index(self):
        from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
        windows = [f"{self.metadata[i]['title']}. {self.window_text(i)}" for i in range(len(self.metadata))]
        self.vectorizer = TfidfVectorizer(preprocessor=normalize_text, stop_words=list(ENGLISH_STOP_WORDS | QUESTION_WORDS), ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.keyword_matrix = self.vectorizer.fit_transform(windows)
        self.idf = dict(zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_))
        self.max_idf = float(self.vectorizer.idf_.max())

    def keyword_coverage(self, query: str, segment: dict) -> float:
        """IDF-weighted share of the query's words that appear in the matched segment.
        Words never spoken in the course get the highest weight, so off-topic questions score low."""
        analyze = self.vectorizer.build_analyzer()
        q_terms = {t for t in analyze(query) if " " not in t}
        if not q_terms:
            return 0.0
        seg_terms = set(analyze(f"{segment['title']}. {segment['text']}"))
        weights = {t: self.idf.get(t, self.max_idf) for t in q_terms}
        return sum(w for t, w in weights.items() if t in seg_terms) / sum(weights.values())

    def window_rows(self, idx: int, radius: int = CONTEXT_RADIUS) -> list:
        rows = self.video_rows[str(self.metadata[idx]['number'])]
        pos = self.row_position[idx]
        return rows[max(0, pos - radius): pos + radius + 1]

    def window_text(self, idx: int, radius: int = CONTEXT_RADIUS) -> str:
        return " ".join(self.metadata[i]['text'].strip() for i in self.window_rows(idx, radius))

    def score_all(self, query: str) -> np.ndarray:
        if self.mode == "keyword":
            q_vec = self.vectorizer.transform([query])
            return (self.keyword_matrix @ q_vec.T).toarray().ravel()
        # Embed the query and compute dot-product cosine similarity against the pre-loaded matrix
        q_emb = self.loader.embed_text([query])[0]  # Shape: (1024,)
        return np.dot(self.embeddings_matrix, q_emb)  # Shape: (n_chunks,)

    def search_similarity(self, query: str, top_k: int = 5) -> list:
        if not self.metadata:
            return []
        # 1-2. Embed the query and score every chunk
        scores = self.score_all(query)

        # 3. Get top-k indices using argpartition (O(N) search is faster than O(N log N) argsort)
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            partitioned = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = partitioned[np.argsort(scores[partitioned])[::-1]]

        # 4. Extract rows from memory cache (under 0.05ms)
        results = []
        for idx in top_indices:
            item = self.metadata[idx].copy()
            item['similarity_score'] = float(scores[idx])
            results.append(item)

        return results

    def search_segments(self, query: str, top_k: int = 5) -> list:
        """Like search_similarity, but expands each hit with its neighbouring lines and
        merges overlapping hits from the same video into one readable segment."""
        if not self.metadata:
            return []
        scores = self.score_all(query)
        candidates = np.argsort(scores)[::-1][: top_k * 8]

        segments = []
        taken = set()
        for idx in candidates:
            idx = int(idx)
            rows = self.window_rows(idx)
            if taken.intersection(rows):
                continue
            taken.update(rows)
            first, last = self.metadata[rows[0]], self.metadata[rows[-1]]
            segments.append({
                'number': str(first['number']),
                'title': first['title'],
                'start': float(first['start']),
                'end': float(last['end']),
                'text': " ".join(self.metadata[i]['text'].strip() for i in rows),
                'chunk_id': int(self.metadata[idx]['chunk_id']),
                'similarity_score': float(scores[idx]),
                'video_id': first.get('video_id'),
            })
            if len(segments) == top_k:
                break
        return segments

    def list_videos(self) -> list:
        videos = []
        for number, rows in self.video_rows.items():
            videos.append({
                'number': number,
                'title': self.metadata[rows[0]]['title'],
                'duration': float(self.metadata[rows[-1]]['end']),
                'chunks': len(rows),
            })
        return sorted(videos, key=lambda v: int(v['number']) if v['number'].isdigit() else v['number'])


class ModelLoader:
    def __init__(self):
        self.backend = MODEL_BACKEND.lower()
        self.model = None
        self.tokenizer = None

        # Load the model based on the backend
        self.load_model()

        # Load the built-in course index (embeddings dataframe)
        self.load_index()

    def load_index(self):
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"Embeddings database not found at {INDEX_PATH}. Run preprocessing first.")

        print(f"Loading vector index from {INDEX_PATH}...")
        df = joblib.load(INDEX_PATH)
        # Convert metadata columns to standard Python list of dicts for O(1) fast lookup
        # This completely avoids slow pandas DataFrame indexing (.iloc/iterrows) at query time.
        metadata = df[['number', 'title', 'start', 'end', 'text', 'chunk_id']].to_dict(orient="records")
        self.index = TranscriptIndex(metadata, np.vstack(df['embedding'].values), loader=self)
        print(f"Index loaded: {len(metadata)} chunks, search mode: {self.index.mode}")

    # The built-in course index is exposed directly for the original endpoints
    @property
    def metadata(self):
        return self.index.metadata

    def search_similarity(self, query: str, top_k: int = 5) -> list:
        return self.index.search_similarity(query, top_k)

    def search_segments(self, query: str, top_k: int = 5) -> list:
        return self.index.search_segments(query, top_k)

    @staticmethod
    def ollama_available() -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(n.split(":")[0] == OLLAMA_EMBED_MODEL for n in names)
        except Exception:
            return False

    def load_model(self):
        print(f"Loading embedding model using backend: {self.backend}")
        if self.backend == "auto":
            self.backend = "ollama" if self.ollama_available() else "keyword"
            print(f"Auto-selected retrieval backend: {self.backend}")
        if self.backend == "ollama":
            self.model = None
        elif self.backend == "keyword":
            self.model = None
        elif self.backend == "mock":
            print("Model loader configured in MOCK mode. Generating synthetic embeddings.")
            self.model = MockEmbedder()
        elif self.backend == "pytorch":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(MODEL_NAME)
                print("Model loaded successfully via SentenceTransformers (PyTorch).")
            except ImportError:
                print("WARNING: sentence-transformers not installed. Falling back to MOCK mode.")
                self.backend = "mock"
                self.model = MockEmbedder()
        elif self.backend == "onnx":
            # Check for local ONNX files
            onnx_path = os.path.join("model_onnx", "model_quantized.onnx")
            if not os.path.exists(onnx_path):
                print(f"ONNX model not found at {onnx_path}. Attempting to run PyTorch fallback.")
                try:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(MODEL_NAME)
                    self.backend = "pytorch"
                    print("PyTorch model loaded successfully as fallback.")
                except ImportError:
                    print("WARNING: SentenceTransformers and ONNX model not available. Falling back to MOCK.")
                    self.backend = "mock"
                    self.model = MockEmbedder()
            else:
                try:
                    import onnxruntime as ort
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                    self.model = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                    print("ONNX quantized model loaded successfully.")
                except Exception as e:
                    print(f"Error loading ONNX model: {e}. Falling back to MOCK.")
                    self.backend = "mock"
                    self.model = MockEmbedder()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def embed_text(self, texts: list) -> np.ndarray:
        if self.backend == "ollama":
            r = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": OLLAMA_EMBED_MODEL, "input": texts}, timeout=60)
            r.raise_for_status()
            embeddings = np.array(r.json()["embeddings"], dtype=np.float32)
            return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        if self.backend == "keyword":
            raise RuntimeError("No embedding model available in keyword mode")
        if self.backend == "mock":
            return self.model.encode(texts)
        elif self.backend == "pytorch":
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            if len(embeddings.shape) == 1:
                embeddings = np.expand_dims(embeddings, axis=0)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / (norms + 1e-9)
        elif self.backend == "onnx":
            # Standard Hugging Face tokenizer + ONNX execution
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            if "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            
            ort_outs = self.model.run(None, ort_inputs)
            token_embeddings = ort_outs[0]
            cls_embeddings = token_embeddings[:, 0, :]
            norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
            return cls_embeddings / (norms + 1e-9)

class MockEmbedder:
    def __init__(self):
        # Pre-cache a static normalized vector of dimension 1024
        # to avoid dynamic math operations per request
        self.static_vector = np.ones(1024, dtype=np.float32)
        self.static_vector /= np.linalg.norm(self.static_vector)

    def encode(self, texts: list) -> np.ndarray:
        # Replicate static vector for the batch size
        return np.vstack([self.static_vector for _ in texts])

# Global singleton instance
model_loader = ModelLoader()
