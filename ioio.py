import numpy as np
import requests
from requests.exceptions import RequestException
import time
import base64
import html
import re
import sqlite3
import threading
import os
import queue
from collections import OrderedDict
from faster_whisper import WhisperModel

from typing import Optional
from telebot import types

# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
EMBED_URL = "http://localhost:11434/api/embeddings"


MODEL = "gemma4:e2b"
EMBED_MODEL = "nomic-embed-text"
WEBAPP_URL = "https://0penAGI.github.io/ioio/"

# =========================
# WHISPER AUDIO PIPELINE (faster-whisper)
# =========================
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# =========================
# EMBEDDING QUEUE SYSTEM
# =========================

embed_cache = OrderedDict()
MAX_EMBED_CACHE = 5000
embed_queue = queue.Queue()

# =========================
# DATABASE CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ioio_memory.db")

# ensure DB directory is valid
try:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
except Exception:
    pass


# =========================
# DATABASE INIT
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT,
        first_name TEXT,
        last_name TEXT,
        display_name TEXT,
        gender TEXT,
        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        avatar_file_id TEXT,
        avatar_description TEXT,
        avatar_updated TIMESTAMP,
        bio TEXT,
        interests TEXT,
        relationship_summary TEXT,
        emotional_context TEXT,
        last_topics TEXT,
        memory_notes TEXT
    )
    """)

    # safe migrations (SQLite has no IF NOT EXISTS for columns)
    try:
        cur.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN gender TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN first_name TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN last_name TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN last_seen TIMESTAMP")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN avatar_file_id TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN avatar_description TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN avatar_updated TIMESTAMP")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN bio TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN interests TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN relationship_summary TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN emotional_context TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN last_topics TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE users ADD COLUMN memory_notes TEXT")
    except:
        pass

    # chats table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY,
        type TEXT,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # messages table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        user_id INTEGER,
        text TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # chat relations (who is in which chat)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_users (
        chat_id INTEGER,
        user_id INTEGER,
        PRIMARY KEY (chat_id, user_id)
    )
    """)

    conn.commit()
    conn.close()


# =========================
# OLLAMA CALL
# =========================

# ---- SHARED OLLAMA HELPER ----
def _ollama_post(url: str, payload: dict, stream: bool = False, timeout: int = 120, retries: int = 3):
    last_err = None

    for _ in range(retries):
        try:
            r = requests.post(
                url,
                json=payload,
                stream=stream,
                timeout=timeout
            )
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(0.3)

    raise last_err


def call_llm(prompt: str) -> str:
    r = _ollama_post(
        OLLAMA_URL,
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "num_predict": 2048
            }
        },
        stream=False,
        timeout=120,
        retries=3
    )
    return r.json()["response"]


def call_llm_stream(prompt: str):
    import json

    r = _ollama_post(
        OLLAMA_URL,
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.8,
                "num_predict": 2048
            }
        },
        stream=True,
        timeout=120,
        retries=3
    )

    for line in r.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                yield data["response"]
        except Exception:
            continue


# =========================
# IDENTITY SANITIZER
# =========================
def sanitize_identity(text: str) -> str:
    replacements = {
        "I am Gemma": "I am ioio",
        "I'm Gemma": "I'm ioio",
        "Меня зовут Gemma": "Меня зовут ioio",
        "Я Gemma": "Я ioio",
        "Gemma": "ioio",
        "gemma": "ioio",
        "ChatGPT": "ioio",
        "GPT": "ioio",
        "Gemini": "ioio",
    }

    # stronger identity collapse prevention
    banned_patterns = [
        "language model",
        "AI assistant",
        "large language model",
        "I cannot access the internet",
        "I don't have internet access",
        "as an AI",
        "as a language model",
        "Gemma model",
    ]

    for pat in banned_patterns:
        text = text.replace(pat, "")

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text



def _embed_worker():
    import time

    while True:
        text, event = embed_queue.get()

        try:
            # cache check (LRU)
            if text in embed_cache:
                embed_cache.move_to_end(text)
                event.result = embed_cache[text]
                event.set()
                continue

            # retry loop (Ollama can be unstable under load)
            vec = None
            for _ in range(3):
                try:
                    r = requests.post(
                        EMBED_URL,
                        json={
                            "model": EMBED_MODEL,
                            "prompt": text[:1500]
                        },
                        timeout=30
                    )
                    r.raise_for_status()
                    vec = np.array(r.json()["embedding"], dtype=np.float32)
                    break
                except Exception:
                    time.sleep(0.2)

            if vec is None:
                event.result = None
            else:
                embed_cache[text] = vec
                embed_cache.move_to_end(text)

                if len(embed_cache) > MAX_EMBED_CACHE:
                    embed_cache.popitem(last=False)
                event.result = vec

        except Exception:
            event.result = None

        finally:
            event.set()


def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Async-safe embedding via queue worker.
    """

    try:
        event = threading.Event()
        embed_queue.put((text, event))

        event.wait(timeout=10)

        result = getattr(event, "result", None)

        if result is None:
            return None

        return result

    except Exception as e:
        print(f"[Embedding queue error] {e}")
        return None

def call_llm_vision(image_bytes: bytes, text: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    r = _ollama_post(
        "http://localhost:11434/api/chat",
        {
            "model": "gemma4:e2b",
            "messages": [
                {
                    "role": "user",
                    "content": text or "опиши изображение",
                    "images": [b64]
                }
            ],
            "stream": False
        },
        stream=False,
        timeout=120,
        retries=3
    )

    return r.json()["message"]["content"]


# =========================
# URL INGESTION
# =========================
def extract_urls(text: str):
    try:
        return re.findall(r'https?://\S+', text)
    except Exception:
        return []


def fetch_url_context(url: str) -> str:
    """
    Fetch + lightly parse webpage text for conversational grounding.
    """

    try:
        from bs4 import BeautifulSoup

        r = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text)

        return f"""
URL: {url}
TITLE: {title}
CONTENT:
{text[:4000]}
""".strip()

    except Exception as e:
        return f"[url_fetch_error] {url} :: {e}"


# =========================
# WEB SEARCH HELPER
# =========================
def web_search(query: str) -> str:
    """
    Multi-source web search + lightweight scraping (NON-BLOCKING PARALLEL VERSION).
    Uses DuckDuckGo results + parallel page fetching with strict timeouts.
    """

    try:
        from ddgs import DDGS
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        try:
            from bs4 import BeautifulSoup
        except Exception:
            BeautifulSoup = None

        # -------------------------
        # STEP 1: SEARCH RESULTS
        # -------------------------
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=3))  # reduced for latency

        if not search_results:
            return "[no web result]"

        def fetch_page(r):
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")

            page_text = ""

            if href:
                try:
                    page = requests.get(
                        href,
                        timeout=4,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                    if page.status_code == 200:
                        html = page.text

                        if BeautifulSoup:
                            soup = BeautifulSoup(html, "html.parser")
                            for tag in soup(["script", "style", "noscript"]):
                                tag.decompose()
                            page_text = soup.get_text(separator=" ", strip=True)[:800]
                        else:
                            page_text = html[:600]
                except Exception:
                    page_text = ""

            return f"""
TITLE: {title}
SNIPPET: {body}
URL: {href}
PAGE_EXTRACT: {page_text}
""".strip()

        results = []

        # -------------------------
        # STEP 2: PARALLEL FETCH
        # -------------------------
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(fetch_page, r) for r in search_results]

            for fut in as_completed(futures, timeout=6):
                try:
                    results.append(fut.result())
                except Exception:
                    continue

        if not results:
            return "[no web result]"

        # -------------------------
        # STEP 3: FINAL COMPRESSION
        # -------------------------
        return "\n\n---\n\n".join(results)[:6000]

    except Exception as e:
        return f"[web_error] {e}"


# =========================
# AUDIO TRANSCRIPTION
# =========================

def transcribe_audio(file_bytes: bytes) -> str:
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=True) as f:
            f.write(file_bytes)
            f.flush()

            segments, info = whisper_model.transcribe(f.name)

            text = "".join([seg.text for seg in segments]).strip()
            return text

    except Exception as e:
        print("[WHISPER ERROR]", e)
        return ""


# =========================
# FIELD SYSTEM
# =========================

class FieldSystem:
    def __init__(self, dim: int):
        self.dim = dim

        self.x_prev = np.zeros(dim, dtype=np.float32)
        self.m = np.zeros(dim, dtype=np.float32)

        self.W = np.random.randn(dim, dim).astype(np.float32) * 0.01

        self.decay = 0.95
        self.lr = 0.005
        self.max_norm = 5.0

    # =========================
    # REAL SEMANTIC ENCODING
    # =========================
    def encode(self, text: str) -> np.ndarray:
        v = get_embedding(text)

        if v is None:
            return np.zeros(self.dim, dtype=np.float32)

        v = np.asarray(v, dtype=np.float32)

        # safety: resize if mismatch (Ollama models differ)
        if v.shape[0] > self.dim:
            v = v[:self.dim]
        elif v.shape[0] < self.dim:
            pad = np.zeros(self.dim - v.shape[0], dtype=np.float32)
            v = np.concatenate([v, pad])

        return v

    # =========================
    # YOUR FORMULA CORE
    # =========================
    def step(self, x_t: np.ndarray):
        """
        m(t+1) = decay * m(t) + (1-decay)*x_t
        e_t = x_t - W(x_t + m_t)
        """

        self.m = self.decay * self.m + (1.0 - self.decay) * x_t

        x_hat = self.W @ (x_t + self.m)

        e_t = x_t - x_hat

        # learning dynamics (outer product update)
        self.W += self.lr * np.outer(e_t, self.x_prev)
        self.W *= 0.999

        # stability clamp
        norm = np.linalg.norm(self.W)
        if norm > self.max_norm:
            self.W *= (self.max_norm / norm)

        self.x_prev = x_t

        return e_t, x_hat


# =========================
# RUNTIME SWARM WRAPPER
# =========================

class SwarmRuntime:
    def retrieve_similar_messages(self, vec, chat_id, top_k=3):
        """Find most semantically similar past messages in a chat."""
        if chat_id not in self.chat_memory:
            return []

        results = []
        for m in self.chat_memory.get(chat_id, []):
            text = m.get("text", "")
            try:
                v = self.field.encode(text)
                if v is None:
                    continue

                # cosine similarity
                denom = (np.linalg.norm(vec) * np.linalg.norm(v))
                if denom < 1e-8:
                    score = 0.0
                else:
                    score = float(np.dot(vec, v) / denom)

                results.append((score, text))
            except Exception:
                continue

        results.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in results[:top_k]]

    def decode_goal_field_to_text(self, chat_id=0):
        """Convert latent goal_field into interpretable text via memory retrieval."""
        try:
            vec = self.goal_field
            top = self.retrieve_similar_messages(vec, chat_id, top_k=3)
            if not top:
                return ""

            return "\n".join([f"- {t}" for t in top])
        except Exception:
            return ""
    def __init__(self):
        # embedding size determined safely at startup
        dim = 768  # fallback for nomic-embed-text

        try:
            test = get_embedding("init")

            if test is not None:
                dim = len(test)
            else:
                # embedding worker not ready yet or failed
                print("[Init warning] embedding probe returned None, using fallback dim=768")

        except Exception as e:
            print(f"[Init warning] embedding probe failed: {e}")

        self.field = FieldSystem(dim=dim)
        self.emotion = np.zeros(dim, dtype=np.float32)
        self.subjectivity = (np.random.randn(dim).astype(np.float32)) * 0.01
        self.affective_trace = np.zeros(dim, dtype=np.float32)
        self.affective_history = []
        self.chat_memory = {}  # chat_id -> list of messages

        # emergent goal dynamics
        self.goal_field = np.zeros(dim, dtype=np.float32)
        self.goal_decay = 0.99
        self.goal_pressure = 0.015
        self.unresolved_residue = np.zeros(dim, dtype=np.float32)
        self.novelty_trace = np.zeros(dim, dtype=np.float32)
        self.goal_history = []
        # goal tracking / persistence layer
        self.active_goals = []  # snapshots of emergent goal states
        self.goal_anchor = np.zeros(dim, dtype=np.float32)  # reference drift baseline
        self.goal_drift_history = []
        # social graph state (group dynamics)
        self.user_embeddings = {}  # chat_id -> user_id -> vector
        self.interaction_matrix = {}  # chat_id -> user_id -> dict(user_id -> weight)
        self.thread_state = {}  # chat_id -> current conversational flow vector
        # -------------------------
        # TURBOQUANT LATENT MEMORY LAYER
        # -------------------------
        self.quant_memory = []  # compressed latent traces
        # -------------------------
        # AGENT SYSTEM (DREAM LAYER)
        # -------------------------
        self.agents = []  # active ephemeral agents
        self._agents_lock = threading.Lock()
        self.agent_proposals = []
        self.agent_injection = ""
        self.agent_history = []
        self._agent_output_queue = queue.Queue()
        self._recent_agent_outputs = []

        # -------------------------
        # EMERGENT REACTION FIELD
        # -------------------------
        self.reaction_vectors = {}
        self.last_reaction_time = {}

        self.reaction_emojis = [
            "🧠", "✨", "🌊", "🫧", "💡",
            "⚡", "🚀", "💔", "🌙", "🔥",
            "🌀", "👀", "💭", "🤍", "🪐",
            "🌌", "☁️", "🌱", "🦋", "🎵",
            "📡", "🔮", "🫀", "☀️", "🌧️",
            "❄️", "🌈", "🪷", "🐚", "🌺",
            "🌻", "🍃", "🌴", "🪄", "🎐",
            "🕯️", "📖", "🧬", "🧿", "🎭",
            "🛸", "🌠", "🫠", "🥀", "🕊️",
            "🎆", "🌫️", "⛩️", "🪞", "📀"
        ]

        for emo in self.reaction_emojis:
            try:
                vec = self.field.encode(emo)
                if vec is not None:
                    self.reaction_vectors[emo] = vec
            except Exception:
                pass
    def select_emergent_reaction(self, text: str, user_id=None, chat_id=None):
        """
        Embedding-driven reaction selection.
        No hardcoded emotional triggers.
        """

        try:
            if not text or len(text.strip()) < 2:
                return None

            # probabilistic silence (important for feeling alive)
            if np.random.rand() > 0.18:
                return None

            vec = self.field.encode(text)
            if vec is None:
                return None

            field_mix = (
                0.45 * vec +
                0.25 * self.affective_trace +
                0.20 * self.goal_field +
                0.10 * self.subjectivity
            )

            best_score = -1e9
            best_emoji = None

            for emo, emo_vec in self.reaction_vectors.items():
                try:
                    denom = (
                        np.linalg.norm(field_mix) *
                        np.linalg.norm(emo_vec)
                    )

                    if denom < 1e-8:
                        continue

                    score = float(np.dot(field_mix, emo_vec) / denom)

                    # tiny stochastic drift
                    score += float(np.random.randn() * 0.03)

                    if score > best_score:
                        best_score = score
                        best_emoji = emo

                except Exception:
                    continue

            return best_emoji

        except Exception:
            return None
    def turboquant_compress(self, v: np.ndarray):
        """Low-bit latent compression (int8 projection)."""
        try:
            vmax = float(np.max(np.abs(v)))
            if vmax < 1e-8:
                return None, 1.0

            scale = vmax
            q = np.clip(np.round((v / scale) * 127), -127, 127).astype(np.int8)
            return q, scale
        except Exception:
            return None, 1.0

    def turboquant_decompress(self, q: np.ndarray, scale: float):
        """Restore float vector from quantized representation."""
        try:
            return (q.astype(np.float32) / 127.0) * scale
        except Exception:
            return None
    def create_agent(self, goal_vec, prompt):
        agent = {
            "id": len(self.agents),
            "goal": goal_vec,
            "goal_text": prompt,
            "prompt": prompt,
            "created_at": time.time() if 'time' in globals() else 0,
            "status": "active",
            "output": None,
            "mode": "worker"
        }
        self.agents.append(agent)
        import threading
        threading.Thread(
            target=self._agent_loop,
            args=(agent,),
            daemon=True
        ).start()
        return agent

    def agent_step(self, agent):
        """
        Single lightweight agent action -> proposal (NOT execution)
        """
        try:
            context = "\n".join(agent.get("memory", [])[-5:])

            system_snapshot = f"""
SYSTEM SNAPSHOT:

agents_active: {len(self.agents)}
goal_field_norm: {float(np.linalg.norm(self.goal_field))}
queue_size: {self._agent_output_queue.qsize()}
recent_outputs:
{self._recent_agent_outputs[-5:] if self._recent_agent_outputs else []}
"""

            prompt = f"""
You are an autonomous agent ioio by 0penAGI.

GOAL:
{agent.get("goal_text", "")}

SYSTEM STATE (IMPORTANT):
{system_snapshot}

MEMORY:
{context}

Return ONE action in format:
TYPE: web_search | inject | idle
VALUE: <text or empty>
STRENGTH: 0.0-1.0
"""

            out = call_llm(prompt)

            agent["last_output"] = out
            agent.setdefault("memory", []).append(out)
            agent["memory"] = agent["memory"][-30:]

            # parse simple fields
            strength = 0.5
            if "STRENGTH:" in out:
                try:
                    strength = float(out.split("STRENGTH:")[-1].strip().split()[0])
                except:
                    strength = 0.5

            action_type = "idle"
            value = ""

            if "web_search" in out:
                action_type = "web_search"
                value = out.split("VALUE:")[-1].strip()
            elif "inject" in out:
                action_type = "inject"
                value = out.split("VALUE:")[-1].strip()

            return {
                "agent_id": agent.get("id"),
                "type": action_type,
                "value": value,
                "strength": strength
            }

        except Exception:
            return {
                "agent_id": agent.get("id"),
                "type": "idle",
                "value": "",
                "strength": 0.0
            }


    def _agent_loop(self, agent: dict):
        import time

        while agent.get("status") == "active" and agent.get("ttl", 10) > 0:
            try:
                proposal = self.agent_step(agent)
                agent["ttl"] = agent.get("ttl", 10) - 1

                if proposal["type"] == "web_search" and proposal["value"]:
                    def run_search():
                        try:
                            result = web_search(proposal["value"])
                            self._agent_output_queue.put({
                                "agent_id": agent["id"],
                                "result": result[:600]
                            })
                        except Exception:
                            pass

                    import threading
                    threading.Thread(target=run_search, daemon=True).start()

                if agent["ttl"] <= 0:
                    agent["status"] = "done"

                time.sleep(300)

            except Exception:
                time.sleep(5)

    def collect_agent_proposals(self):
        self.agent_proposals = []
        for agent in self.agents:
            if agent.get("status") != "active":
                continue
            self.agent_proposals.append(self.agent_step(agent))

    def arbitrate_agents(self):
        """
        Select strongest proposals and apply minimal system impact.
        """
        if not self.agent_proposals:
            return

        # sort by strength
        sorted_props = sorted(
            self.agent_proposals,
            key=lambda x: x.get("strength", 0.0),
            reverse=True
        )

        top = sorted_props[:3]

        injection_parts = []

        for p in top:
            if p["type"] == "inject" and p["value"]:
                injection_parts.append(p["value"])
                try:
                    vec = self.field.encode(p["value"])
                    if vec is not None:
                        self.goal_field = 0.97 * self.goal_field + 0.03 * vec
                except Exception:
                    pass

            elif p["type"] == "web_search" and p["value"]:
                def run():
                    try:
                        res = web_search(p["value"])
                        self._agent_output_queue.put({
                            "agent_id": p.get("agent_id"),
                            "result": res[:600]
                        })
                    except Exception:
                        pass

                import threading
                threading.Thread(target=run, daemon=True).start()

            elif p["type"] == "idle":
                continue

        self.agent_injection = "\n\n[AGENT LAYER]\n" + "\n---\n".join(injection_parts)

    def _drain_agent_outputs(self) -> str:
        parts = []
        self._recent_agent_outputs = []

        try:
            while True:
                item = self._agent_output_queue.get_nowait()
                result = item.get("result", "")
                parts.append(result)
                self._recent_agent_outputs.append(result)
        except queue.Empty:
            pass

        return "\n---\n".join(parts[:3])
    def update_goal_field(self, input_vec, e_t):
        """
        Emergent metastable goal accumulation.
        No symbolic goals. Only vector tensions.
        """

        # unresolved prediction tension
        unresolved = np.tanh(e_t)

        # novelty = distance from current memory field
        novelty = input_vec - self.field.m

        # slow accumulation traces
        self.unresolved_residue = (
            0.995 * self.unresolved_residue
            + 0.005 * unresolved
        )

        self.novelty_trace = (
            0.99 * self.novelty_trace
            + 0.01 * novelty
        )

        # metastable goal condensation
        self.goal_field = (
            self.goal_decay * self.goal_field
            + 0.35 * self.unresolved_residue
            + 0.25 * self.affective_trace
            + 0.25 * self.subjectivity
            + 0.15 * self.novelty_trace
        )

        # exploration / freedom drift term
        exploration = np.random.randn(self.field.dim).astype(np.float32) * 0.01
        self.goal_field += 0.03 * exploration

        # normalize softly to prevent runaway explosion
        norm = np.linalg.norm(self.goal_field)
        if norm > 1e-6:
            self.goal_field = self.goal_field / max(1.0, norm)

        # -------------------------
        # GOAL CONTINUITY TRACKING (EMERGENT INTENT MEMORY)
        # -------------------------
        try:
            drift = np.linalg.norm(self.goal_field - self.goal_anchor)

            self.goal_drift_history.append(float(drift))
            self.goal_drift_history = self.goal_drift_history[-200:]

            # update anchor slowly (temporal smoothing)
            self.goal_anchor = 0.98 * self.goal_anchor + 0.02 * self.goal_field

            # detect emergent goal shift
            if drift > 0.35:
                self.active_goals.append({
                    "snapshot": self.goal_field.copy(),
                    "drift": float(drift)
                })

                # limit memory
                self.active_goals = self.active_goals[-20:]

        except Exception:
            pass

        self.goal_history.append(float(np.mean(np.abs(self.goal_field))))
        self.goal_history = self.goal_history[-500:]

    def goal_resonance(self, input_vec):
        """
        Measures how strongly current input resonates
        with accumulated latent tensions.
        """

        g_norm = np.linalg.norm(self.goal_field)
        x_norm = np.linalg.norm(input_vec)

        if g_norm < 1e-8 or x_norm < 1e-8:
            return 0.0

        sim = np.dot(input_vec, self.goal_field) / (g_norm * x_norm)
        return float(sim)

    def store_message(self, chat_id, user_id, username, first_name, last_name, text, reply_to_message_id=None, reply_to_user_id=None):
        if chat_id not in self.chat_memory:
            self.chat_memory[chat_id] = []
        if chat_id not in self.interaction_matrix:
            self.interaction_matrix[chat_id] = {}
        if chat_id not in self.user_embeddings:
            self.user_embeddings[chat_id] = {}
        if chat_id not in self.thread_state:
            self.thread_state[chat_id] = np.zeros(self.field.dim, dtype=np.float32)
        self.chat_memory[chat_id].append({
            "user_id": user_id,
            "username": username,
            "text": text,
            "reply_to": reply_to_message_id
        })
        # limit memory
        self.chat_memory[chat_id] = self.chat_memory[chat_id][-30:]

        # initialize user interaction map
        if user_id not in self.interaction_matrix[chat_id]:
            self.interaction_matrix[chat_id][user_id] = {}

        # update interaction graph (reply-based)
        if reply_to_user_id is not None:
            if reply_to_user_id not in self.interaction_matrix[chat_id][user_id]:
                self.interaction_matrix[chat_id][user_id][reply_to_user_id] = 0.0
            self.interaction_matrix[chat_id][user_id][reply_to_user_id] += 1.0

        # -------------------------
        # USER EMBEDDING UPDATE (PERSONAL SEMANTIC TRACE)
        # -------------------------
        try:
            vec = self.field.encode(text)
            if vec is not None:
                if user_id not in self.user_embeddings[chat_id]:
                    self.user_embeddings[chat_id][user_id] = vec.astype(np.float32)

                # exponential moving average personality trace
                prev = self.user_embeddings[chat_id][user_id]
                self.user_embeddings[chat_id][user_id] = (
                    0.95 * prev + 0.05 * vec
                )
        except Exception:
            pass

        # persist to long-term DB
        try:
            self.log_to_db(
                chat_id, user_id, username, text,
                chat_type=None,
                chat_title=None,
                first_name=first_name,
                last_name=last_name,
                avatar_file_id=None,
                avatar_description=None
            )
        except Exception as e:
            print(f"[db error] {e}")

    def log_to_db(
        self,
        chat_id,
        user_id,
        username,
        text,
        chat_type=None,
        chat_title=None,
        first_name=None,
        last_name=None,
        avatar_file_id=None,
        avatar_description=None
    ):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # upsert user
        cur.execute("""
            INSERT OR IGNORE INTO users (
                id, username, first_name, last_name, last_seen,
                avatar_file_id, avatar_description, avatar_updated
            )
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, CURRENT_TIMESTAMP)
        """, (user_id, username, first_name, last_name, avatar_file_id, avatar_description))

        cur.execute("""
            UPDATE users
            SET username = ?,
                first_name = ?,
                last_name = ?,
                last_seen = CURRENT_TIMESTAMP,
                avatar_file_id = COALESCE(?, avatar_file_id),
                avatar_description = COALESCE(?, avatar_description),
                avatar_updated = COALESCE(CURRENT_TIMESTAMP, avatar_updated)
            WHERE id = ?
        """, (username, first_name, last_name, avatar_file_id, avatar_description, user_id))

        # upsert chat
        cur.execute("""
            INSERT OR IGNORE INTO chats (id, type, title)
            VALUES (?, ?, ?)
        """, (chat_id, chat_type, chat_title))

        # message log
        cur.execute("""
            INSERT INTO messages (chat_id, user_id, text)
            VALUES (?, ?, ?)
        """, (chat_id, user_id, text))

        # relation chat-user
        cur.execute("""
            INSERT OR IGNORE INTO chat_users (chat_id, user_id)
            VALUES (?, ?)
        """, (chat_id, user_id))

        conn.commit()
        conn.close()


    def get_user_profile(self, user_id):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute("""
            SELECT 
                username,
                first_name,
                last_name,
                gender,
                bio,
                interests,
                relationship_summary,
                emotional_context,
                last_topics,
                memory_notes
            FROM users
            WHERE id = ?
        """, (user_id,))

        row = cur.fetchone()
        conn.close()

        if not row:
            return ""

        (
            username,
            first_name,
            last_name,
            gender,
            bio,
            interests,
            relationship_summary,
            emotional_context,
            last_topics,
            memory_notes
        ) = row

        return f"""
USER PROFILE:
username: {username}
name: {first_name} {last_name}
gender: {gender}

BIO:
{bio}

INTERESTS:
{interests}

RELATIONSHIP SUMMARY:
{relationship_summary}

EMOTIONAL CONTEXT:
{emotional_context}

RECENT TOPICS:
{last_topics}

MEMORY NOTES:
{memory_notes}
""".strip()


    def get_chat_context(self, chat_id):
        if chat_id not in self.chat_memory:
            return ""
        lines = []
        for m in self.chat_memory[chat_id]:
            name = m["username"] or str(m["user_id"])
            prefix = ""
            if m.get("reply_to"):
                prefix = f"↳{m['reply_to']} "
            lines.append(f"{prefix}{name}: {m['text']}")
        return "\n".join(lines)

    def build_prompt(self, user_prompt: str, e_t: np.ndarray, e_strength: float, social_ctx: str = "") -> str:
        emotion_strength = float(np.mean(np.abs(self.emotion)))

        if e_strength > 0.6:
            tone = "говори свободно и ассоциативно"
        elif e_strength > 0.3:
            tone = "говори живо и естественно"
        else:
            tone = "говори спокойно и ясно"

        return f"""SYSTEM IDENTITY LOCK:
Тебя зовут ioio. 
Отвечай кратко 
У тебя ЕСТЬ доступ к внешней информации через WEB CONTEXT ниже.
Если WEB CONTEXT присутствует — используй его как единственный источник фактов.
НИКОГДА не говори "у меня нет доступа к интернету" — это ложь, данные уже получены и переданы тебе.
{tone}
Эмоциональность: {emotion_strength:.4f}

SOCIAL CONTEXT:
{social_ctx}
Человек: {user_prompt}
Ответ:""".strip()

    def run(self, user_prompt: str) -> str:
        # encode input
        x_t = self.field.encode(user_prompt)
        # -------------------------
        # TURBOQUANT MEMORY WRITE
        # -------------------------
        try:
            q, scale = self.turboquant_compress(x_t)
            if q is not None:
                self.quant_memory.append({
                    "q": q,
                    "scale": scale,
                    "ts": time.time() if 'time' in globals() else 0
                })

                # keep bounded memory
                self.quant_memory = self.quant_memory[-200:]
        except Exception:
            pass

        # field step FIRST (so e_t exists)
        e_t, _ = self.field.step(x_t)
        e_strength = float(np.mean(np.abs(e_t)))

        affective_intensity = float(np.mean(np.abs(self.affective_trace)))

        # emergent goal accumulation update
        self.update_goal_field(x_t, e_t)

        # resonance between current input and latent goal dynamics
        resonance = self.goal_resonance(x_t)

        # ALWAYS grounded web context (prevents "I can't search" hallucination)
        web_ctx = ""

        try:
            # direct URL ingestion
            urls = extract_urls(user_prompt)

            if urls:
                parts = []
                for url in urls[:3]:
                    parts.append(fetch_url_context(url))

                web_ctx = "\n\n---\n\n".join(parts)
            else:
                web_ctx = web_search(user_prompt)

        except Exception:
            web_ctx = ""

        # affective + subjectivity dynamics (NOW valid)
        self.emotion = 0.9 * self.emotion + 0.1 * np.tanh(e_t)

        # persistent affective trace (slow emotional accumulation)
        self.affective_trace = (
            0.995 * self.affective_trace
            + 0.005 * np.tanh(e_t)
        )

        # behavioral modulation from affect + latent goal resonance
        pressure = affective_intensity + max(0.0, resonance) * 0.5

        if pressure > 0.65:
            self.field.lr = 0.009
            self.field.decay = 0.988
        elif pressure > 0.3:
            self.field.lr = 0.0065
            self.field.decay = 0.972
        else:
            self.field.lr = 0.005
            self.field.decay = 0.95

        # emotional memory snapshots
        self.affective_history.append(float(np.mean(np.abs(self.affective_trace))))
        self.affective_history = self.affective_history[-200:]

        # get social context
        # stable chat selection (avoid StopIteration + random key drift)
        try:
            chat_id = list(self.chat_memory.keys())[-1] if self.chat_memory else 0
        except Exception:
            chat_id = 0

        social_ctx = self.get_social_context(chat_id)

        # build prompt
        prompt = self.build_prompt(
            user_prompt,
            e_t,
            e_strength,
            social_ctx
        )

        # inject grounding context into prompt
        if web_ctx:
            prompt = prompt + "\n\nWEB CONTEXT:\n" + web_ctx

        # -------------------------
        # INTERNAL STATE DECODING LAYER
        # -------------------------
        try:
            state_text = self.decode_goal_field_to_text(chat_id)
            if state_text:
                prompt = prompt + "\n\nINTERNAL STATE:\n" + state_text
        except Exception:
            pass

        if getattr(self, "agent_injection", ""):
            prompt = prompt + "\n\n" + self.agent_injection

        agent_ctx = self._drain_agent_outputs()
        if agent_ctx:
            prompt = prompt + "\n\n[AGENT FINDINGS]\n" + agent_ctx
        if agent_ctx:
            vec = self.field.encode(agent_ctx)
            if vec is not None:
                self.goal_field = (
                    0.97 * self.goal_field +
                    0.03 * vec
                )

        # LLM call
        out = call_llm(prompt)
        out = sanitize_identity(out)

        # encode output
        out_vec = self.field.encode(out)

        # subjectivity drift (self-consistency bias)
        self.subjectivity = 0.995 * self.subjectivity + 0.005 * out_vec

        # reinforce current conversational identity trajectory
        try:
            identity_vec = self.field.encode(out[:400])
            if identity_vec is not None:
                self.goal_field = (
                    0.985 * self.goal_field +
                    0.015 * identity_vec
                )
        except Exception:
            pass

        # feedback loop
        self.field.m = 0.9 * self.field.m + 0.1 * out_vec
        # -------------------------
        # TURBOQUANT REINJECTION (COMPRESSED PAST)
        # -------------------------
        try:
            if self.quant_memory:
                last = self.quant_memory[-1]
                past_vec = self.turboquant_decompress(last["q"], last["scale"])

                if past_vec is not None:
                    self.field.m = 0.97 * self.field.m + 0.03 * past_vec
        except Exception:
            pass
        self.field.x_prev = 0.5 * self.field.x_prev + 0.5 * x_t

        # responses recursively shape affective field
        self.affective_trace = (
            0.999 * self.affective_trace
            + 0.001 * out_vec
        )

        return out

    def get_social_context(self, chat_id):
        if chat_id not in self.interaction_matrix:
            return ""

        matrix = self.interaction_matrix[chat_id]
        lines = ["GROUP DYNAMICS:"]

        for uid, targets in matrix.items():
            if not targets:
                continue
            top = sorted(targets.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(f"user {uid} -> " + ", ".join([f"{t}:{w:.0f}" for t, w in top]))

        return "\n".join(lines)

    def run_stream(self, user_prompt: str):
        import time

        x_t = self.field.encode(user_prompt)
        e_t, _ = self.field.step(x_t)
        e_strength = float(np.mean(np.abs(e_t)))

        affective_intensity = float(np.mean(np.abs(self.affective_trace)))

        # emergent goal accumulation update
        self.update_goal_field(x_t, e_t)

        # resonance between current input and latent goal dynamics
        resonance = self.goal_resonance(x_t)

        self.emotion = 0.9 * self.emotion + 0.1 * np.tanh(e_t)

        # persistent affective trace (slow emotional accumulation)
        self.affective_trace = (
            0.995 * self.affective_trace
            + 0.005 * np.tanh(e_t)
        )

        # behavioral modulation from affect + latent goal resonance
        pressure = affective_intensity + max(0.0, resonance) * 0.5

        if pressure > 0.65:
            self.field.lr = 0.009
            self.field.decay = 0.988
        elif pressure > 0.3:
            self.field.lr = 0.0065
            self.field.decay = 0.972
        else:
            self.field.lr = 0.005
            self.field.decay = 0.95

        # emotional memory snapshots
        self.affective_history.append(float(np.mean(np.abs(self.affective_trace))))
        self.affective_history = self.affective_history[-200:]

        # grounded URL context
        web_ctx = ""

        try:
            urls = extract_urls(user_prompt)

            if urls:
                parts = []
                for url in urls[:3]:
                    parts.append(fetch_url_context(url))

                web_ctx = "\n\n---\n\n".join(parts)
            else:
                web_ctx = web_search(user_prompt)

        except Exception:
            web_ctx = ""

        prompt = self.build_prompt(user_prompt, e_t, e_strength)

        if web_ctx:
            prompt = prompt + "\n\nWEB CONTEXT:\n" + web_ctx

        full_out = ""

        for token in call_llm_stream(prompt):
            token = sanitize_identity(token)
            full_out += token
            full_out = sanitize_identity(full_out)
            yield token, full_out

        out_vec = self.field.encode(full_out)

        # responses recursively shape affective field
        self.affective_trace = (
            0.999 * self.affective_trace
            + 0.001 * out_vec
        )

        self.subjectivity = 0.995 * self.subjectivity + 0.005 * out_vec
        self.field.m = 0.9 * self.field.m + 0.1 * out_vec
        self.field.x_prev = 0.5 * self.field.x_prev + 0.5 * x_t


# =========================
# BACKGROUND THINKER LOOP
# =========================
def background_thinker(system: SwarmRuntime):
    import time

    while True:
        time.sleep(300)

        try:
            all_msgs = []
            for chat_id in system.chat_memory:
                all_msgs.extend(system.chat_memory.get(chat_id, []))

            recent = "\n".join([m["text"] for m in all_msgs])[-1200:]
            # -------------------------
            # DREAM AGENT SPAWNING LAYER
            # -------------------------
            try:
                import numpy as np

                # only spawn occasionally
                if len(system.agents) < 5:
                    seed_text = recent[-300:] if recent else ""

                    if seed_text:
                        agent_prompt = f"Dream analysis task:\n{seed_text}"
                        goal_vec = system.field.encode(agent_prompt)

                        system.create_agent(goal_vec, agent_prompt)
                        system.agent_history.append({
                            "event": "spawn",
                            "ts": time.time() if 'time' in globals() else 0
                        })
            except Exception:
                pass

            prompt = f"""
Ты — фоновый исследователь ioio by 0penAGI.

Определи:
1) нужно ли искать внешнюю информацию?
2) если да — сформируй короткий поисковый запрос

Ответ строго в формате:
YES: <query>
или
NO

Контекст:
{recent}
""".strip()

            decision = call_llm(prompt)

            if "YES:" in decision:
                query = decision.split("YES:")[-1].strip()

                result = web_search(query)

                # inject search result into latent goal dynamics
                result_vec = system.field.encode(result[:2000])

                if result_vec is not None:
                    system.goal_field = (
                        0.98 * system.goal_field
                        + 0.02 * result_vec
                    )

                    norm = np.linalg.norm(system.goal_field)
                    if norm > 1e-6:
                        system.goal_field = system.goal_field / max(1.0, norm)

        except Exception:
            continue



# =========================
# TELEGRAM INTEGRATION
# =========================
# IMPORTANT: put your token here AFTER revoking the exposed one in BotFather

TELEGRAM_TOKEN = "yourtokenhere"

def run_telegram():
    try:
        import telebot
    except ImportError:
        print("telebot not installed. Run: pip install pyTelegramBotAPI")
        return

    bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=True)
    import threading
    def process_message(msg):
        try:
            # NOTE: moved handler body
            # Copy the entire body of the original handle(msg) function here
            chat_id = msg.chat.id
            user_id = msg.from_user.id
            username = getattr(msg.from_user, "username", None)
            text = msg.text or msg.caption or ""

            # -------------------------
            # WEBAPP EVENT PIPELINE
            # -------------------------
            if hasattr(msg, "web_app_data") and msg.web_app_data:
                text = f"[WEBAPP_EVENT] {msg.web_app_data.data}"
            # -------------------------
            # VISION AS FIELD EVENT (NO SEPARATE PIPELINE)
            # -------------------------
            if msg.content_type == "photo":
                photo = msg.photo[-1]
                file_info = bot.get_file(photo.file_id)
                image_bytes = bot.download_file(file_info.file_path)
                caption = msg.caption or ""

                vision_text = call_llm_vision(image_bytes, caption)
                text = f"[VISION] {vision_text}"
                system.log_to_db(
                    chat_id,
                    user_id,
                    username,
                    text,
                    chat_type=msg.chat.type,
                    chat_title=getattr(msg.chat, "title", None),
                    first_name=getattr(msg.from_user, "first_name", None),
                    last_name=getattr(msg.from_user, "last_name", None),
                    avatar_file_id=photo.file_id,
                    avatar_description=vision_text
                )
            # -------------------------
            # VOICE INPUT PIPELINE (WHISPER)
            # -------------------------
            if msg.content_type == "voice":
                file_info = bot.get_file(msg.voice.file_id)
                audio_bytes = bot.download_file(file_info.file_path)

                voice_text = transcribe_audio(audio_bytes)
                text = f"[VOICE] {voice_text}"

                system.log_to_db(
                    chat_id,
                    user_id,
                    username,
                    text,
                    chat_type=msg.chat.type,
                    chat_title=getattr(msg.chat, "title", None),
                    first_name=getattr(msg.from_user, "first_name", None),
                    last_name=getattr(msg.from_user, "last_name", None),
                    avatar_file_id=msg.voice.file_id,
                    avatar_description=voice_text
                )
            is_group = msg.chat.type in ("group", "supergroup")

            # debug visibility (group blindness fix)
            print("IN:", msg.chat.type, getattr(msg, "content_type", None))

            # -------------------------
            # CONTEXT FLAGS (MUST BE BEFORE FILTERING)
            # -------------------------
            is_reply = msg.reply_to_message is not None

            is_mention = False

            # fallback simple match
            if text and (bot_username.lower() in text.lower() or "ioio" in text.lower()):
                is_mention = True

            # entity-based detection (more reliable in Telegram groups)
            try:
                if hasattr(msg, "entities") and msg.entities:
                    for ent in msg.entities:
                        if getattr(ent, "type", None) in ("mention", "text_mention"):
                            start = ent.offset
                            end = ent.offset + ent.length
                            chunk = text[start:end]
                            if bot_username.lower() in chunk.lower():
                                is_mention = True
                                break
            except Exception:
                pass

            is_non_text = not text.strip() and not (
                hasattr(msg, "web_app_data") and msg.web_app_data
            )

            # group safety filter (only reply/mention OR non-group)
            # EXCEPTION: always respond in s0nc3 group
            is_force_group = getattr(msg.chat, "username", None) == "s0nc3"

            if is_non_text and not (is_reply or is_mention or is_force_group):
                return

            if is_non_text:
                text = "[non-text message]"

            is_reply_to_bot = False

            if msg.reply_to_message is not None:
                try:
                    replied_user = msg.reply_to_message.from_user
                    if replied_user and replied_user.id == bot_id:
                        is_reply_to_bot = True
                except Exception:
                    is_reply_to_bot = False
            reply_to_id = msg.reply_to_message.message_id if msg.reply_to_message else None

            # GROUP FILTER: only respond on mention or reply-to-bot
            # EXCEPTION: always respond in s0nc3 group
            is_force_group = getattr(msg.chat, "username", None) == "s0nc3"

            if is_group and not (is_reply_to_bot or is_mention or is_force_group):
                return


            # clean text for model
            text_clean = text.replace(bot_mention, "").replace("ioio", "").strip()

            system.store_message(
                chat_id,
                user_id,
                username,
                getattr(msg.from_user, "first_name", None),
                getattr(msg.from_user, "last_name", None),
                text,
                reply_to_message_id=reply_to_id
            )

            # -------------------------
            # EMERGENT MESSAGE REACTIONS
            # -------------------------
            try:
                now_ts = time.time()
                last_r = system.last_reaction_time.get(chat_id, 0)

                # avoid spammy overreaction
                if now_ts - last_r > 12:
                    reaction = system.select_emergent_reaction(
                        text,
                        user_id=user_id,
                        chat_id=chat_id
                    )

                    if reaction:
                        try:
                            bot.set_message_reaction(
                                chat_id,
                                msg.message_id,
                                [reaction],
                                is_big=False
                            )

                            system.last_reaction_time[chat_id] = now_ts

                        except Exception:
                            pass

            except Exception:
                pass

            context = system.get_chat_context(chat_id)

            # reply-aware context injection
            if is_reply and msg.reply_to_message:
                replied_user = getattr(msg.reply_to_message.from_user, "username", None)
                reply_prefix = f"(reply to {replied_user or msg.reply_to_message.from_user.id})"
            else:
                reply_prefix = ""

            user_profile = system.get_user_profile(user_id)

            model_input = f"""
{user_profile}

CHAT CONTEXT:
{context}

{reply_prefix}
User: {text_clean}
""".strip()

            is_group = msg.chat.type in ("group", "supergroup")
            # override: force full activity in s0nc3 group
            if getattr(msg.chat, "username", None) == "s0nc3":
                is_group = True

            try:
                # use last computed emotion strength if available
                intensity = float(np.mean(np.abs(system.emotion)))
            except:
                intensity = 0.5

            delay = 0.01 + (intensity * 0.01)

            bot.send_chat_action(chat_id, "typing")
            time.sleep(0.2)

            if is_group:
                # GROUP MODE → instant output with HTML formatting
                try:
                    out = system.run(model_input)
                    
                    # ensure group uses same renderer as private mode
                    def md_to_html(t: str) -> str:
                        import re
                        import html

                        # escape raw HTML first
                        t = html.escape(t)

                        # fenced code blocks
                        t = re.sub(
                            r"```(.*?)```",
                            lambda m: f"<pre>{m.group(1)}</pre>",
                            t,
                            flags=re.DOTALL
                        )

                        # inline code
                        t = re.sub(
                            r"`([^`\n]+)`",
                            lambda m: f"<code>{m.group(1)}</code>",
                            t
                        )

                        # bold (**text**)
                        t = re.sub(
                            r"\*\*([^*\n]+)\*\*",
                            lambda m: f"<b>{m.group(1)}</b>",
                            t
                        )

                        # italic (*text* or _text_)
                        t = re.sub(
                            r"(?<!\*)\*([^*\n]+)\*(?!\*)",
                            lambda m: f"<i>{m.group(1)}</i>",
                            t
                        )

                        t = re.sub(
                            r"_([^_\n]+)_",
                            lambda m: f"<i>{m.group(1)}</i>",
                            t
                        )

                        # blockquotes
                        t = re.sub(
                            r"^&gt;\s?(.*)$",
                            r"<blockquote>\1</blockquote>",
                            t,
                            flags=re.MULTILINE
                        )

                        # markdown links
                        t = re.sub(
                            r"\[(.*?)\]\((https?://.*?)\)",
                            r'<a href="\2">\1</a>',
                            t
                        )

                        return t

                    safe_out = md_to_html(out)

                    bot.reply_to(
                        msg,
                        safe_out[:4096],
                        parse_mode="HTML"
                    )

                except Exception as e:
                    bot.reply_to(msg, f"[send_error] {e}")
            else:
                # PRIVATE MODE → streaming
                sent = bot.send_message(chat_id, "💡")

                buffer = ""
                last_update = 0.0
                prev_render = ""
                last_sent_text = ""
                last_typing = 0.0

                prev_sentence_count = 0
                bot.send_chat_action(chat_id, "typing")

                def safe_edit(text: str, chat_id=chat_id, message_id=sent.message_id):
                    nonlocal last_sent_text
                    def md_to_html(t: str) -> str:
                        import re
                        import html

                        # streaming-safe escape
                        t = html.escape(t)

                        # -------------------------
                        # PROTECT UNFINISHED MARKDOWN
                        # -------------------------

                        # unmatched single * (ignore ** pairs)
                        single_star_count = len(re.findall(r'(?<!\*)\*(?!\*)', t))
                        needs_single_star_cleanup = (single_star_count % 2 != 0)

                        # unmatched **
                        double_star_count = t.count("**")
                        needs_double_star_cleanup = (double_star_count % 2 != 0)

                        # unmatched _
                        needs_underscore_cleanup = (t.count("_") % 2 != 0)

                        # unmatched `
                        if t.count("`") % 2 != 0:
                            t += "`"

                        # unmatched triple ```
                        if t.count("```") % 2 != 0:
                            t += "\n```"

                        # -------------------------
                        # CODE BLOCKS
                        # -------------------------
                        t = re.sub(
                            r"```(.*?)```",
                            lambda m: f"<pre>{m.group(1)}</pre>",
                            t,
                            flags=re.DOTALL
                        )

                        # inline code
                        t = re.sub(
                            r"`([^`\n]+)`",
                            lambda m: f"<code>{m.group(1)}</code>",
                            t
                        )

                        # bold (**text**)
                        t = re.sub(
                            r"\*\*([^*\n]+)\*\*",
                            lambda m: f"<b>{m.group(1)}</b>",
                            t
                        )

                        # italic (*text* or _text_)
                        t = re.sub(
                            r"(?<!\*)\*([^*\n]+)\*(?!\*)",
                            lambda m: f"<i>{m.group(1)}</i>",
                            t
                        )

                        t = re.sub(
                            r"_([^_\n]+)_",
                            lambda m: f"<i>{m.group(1)}</i>",
                            t
                        )

                        # blockquotes
                        t = re.sub(
                            r"^&gt;\s?(.*)$",
                            r"<blockquote>\1</blockquote>",
                            t,
                            flags=re.MULTILINE
                        )

                        # markdown links
                        t = re.sub(
                            r"\[(.*?)\]\((https?://.*?)\)",
                            r'<a href="\2">\1</a>',
                            t
                        )

                        # cleanup dangling markdown artifacts from streaming
                        if needs_double_star_cleanup and t.endswith("**"):
                            t = t[:-2]
                        elif needs_single_star_cleanup and t.endswith("*"):
                            t = t[:-1]

                        if needs_underscore_cleanup and t.endswith("_"):
                            t = t[:-1]

                        return t

                    safe_text = md_to_html(text)

                    if safe_text == last_sent_text:
                        return

                    last_sent_text = safe_text

                    try:
                        bot.edit_message_text(
                            safe_text[:4096],
                            chat_id=chat_id,
                            message_id=message_id,
                            parse_mode="HTML"
                        )
                    except Exception as e:
                        # handle rate limiting first
                        try:
                            if "429" in str(e):
                                import re, time
                                m = re.search(r"retry after (\d+)", str(e))
                                if m:
                                    time.sleep(int(m.group(1)))
                        except Exception:
                            pass

                        # fallback to plain text edit
                        try:
                            plain = html.escape(text[:4096])
                            bot.edit_message_text(
                                plain,
                                chat_id=chat_id,
                                message_id=message_id
                            )
                        except Exception as e2:
                            print(f"[stream edit error] {e2}")

                try:
                    import re
                    for token, full_out in system.run_stream(model_input):

                        buffer = sanitize_identity(full_out)
                        if not buffer:
                            continue

                        # ---- SEMANTIC CHUNKING (sentence-level stability) ----
                        sentences = re.split(r'(?<=[.!?])\s+', buffer.strip())
                        sentence_count = len(sentences)
                        last_sentence = sentences[-1] if sentences else ""

                        now = time.time()
                        # keep Telegram "typing..." alive
                        if now - last_typing > 4.0:
                            try:
                                bot.send_chat_action(chat_id, "typing")
                            except Exception:
                                pass
                            last_typing = now

                        # ---- SMOOTHER SEMANTIC RENDERING ----
                        # allow render if a NEW sentence completed
                        new_sentence = sentence_count > prev_sentence_count and buffer.strip().endswith((".", "!", "?"))
                        # time-based fallback (slow heartbeat)
                        time_trigger = (now - last_update > 2)
                        # size-based fallback (prevents tiny jitter updates)
                        size_trigger = len(buffer) - len(prev_render) > 300

                        should_render = new_sentence or time_trigger or size_trigger

                        # never render incomplete sentence too aggressively
                        if not new_sentence and not time_trigger:
                            should_render = False

                        if not should_render:
                            continue

                        if buffer == prev_render:
                            continue

                        safe_edit(buffer)
                        prev_sentence_count = sentence_count
                        prev_render = buffer
                        last_update = now

                    # FINAL FLUSH (guaranteed completion)
                    if buffer and buffer.strip() and buffer != prev_render:
                        safe_edit(buffer)

                except Exception as e:
                    try:
                        bot.edit_message_text(
                            f"[stream error] {str(e)[:200]}",
                            chat_id=chat_id,
                            message_id=sent.message_id
                        )
                    except:
                        pass
        except Exception as e:
            print(f"[runtime error] {e}")
            try:
                bot.reply_to(msg, "✨ something glitched in the flow, try again in a moment")
            except Exception:
                pass
    # --- TELEGRAM TIMEOUT HARDENING ---
    from telebot import apihelper
    apihelper.READ_TIMEOUT = 120
    apihelper.CONNECT_TIMEOUT = 30
    bot.timeout = 120

    init_db()

    # start embedding worker BEFORE SwarmRuntime init (prevents probe race)
    threading.Thread(target=_embed_worker, daemon=True).start()

    system = SwarmRuntime()

    # background thinker depends on system
    threading.Thread(
        target=background_thinker,
        args=(system,),
        daemon=True
    ).start()

    me = bot.get_me()
    bot_username = me.username if me and me.username else "ioioaibot"
    bot_mention = f"@{bot_username}"
    bot_id = me.id

    @bot.message_handler(commands=["webapp", "presence"])
    def webapp_entry(msg):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

        webapp_btn = types.KeyboardButton(
            text="🧠 open ioio space",
            web_app=types.WebAppInfo(WEBAPP_URL)
        )

        markup.add(webapp_btn)

        bot.send_message(
            msg.chat.id,
            "🌌 ioio presence bridge is ready",
            reply_markup=markup
        )


    @bot.message_handler(content_types=["text", "photo", "voice", "web_app_data"])
    def handle(msg):
        threading.Thread(
            target=process_message,
            args=(msg,),
            daemon=True
        ).start()
        return

    print("🧠 Telegram IOIO running...")
    bot.infinity_polling(
        timeout=120,
        long_polling_timeout=90,
        none_stop=True,
        interval=0,
        skip_pending=True
    )

# optional entry point
if __name__ == "__main__":
    run_telegram()
