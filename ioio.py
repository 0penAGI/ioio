import numpy as np
import requests
from requests.exceptions import RequestException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import base64
import html
import re
import sqlite3
import threading
# =========================
# TELEGRAM HTTP SESSION
# =========================
telegram_http = requests.Session()

telegram_retry = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET", "POST"]),
    raise_on_status=False
)

telegram_adapter = HTTPAdapter(
    max_retries=telegram_retry,
    pool_connections=32,
    pool_maxsize=32
)

telegram_http.mount("https://", telegram_adapter)
telegram_http.mount("http://", telegram_adapter)
import os
import io
import queue
from collections import OrderedDict
from faster_whisper import WhisperModel
import numpy as np
from typing import Optional
from telebot import types
import telebot.apihelper

# Increase Telegram API timeouts
telebot.apihelper.CONNECT_TIMEOUT = 15
telebot.apihelper.READ_TIMEOUT = 60

# Reuse persistent HTTP session
telebot.apihelper.session = telegram_http
# =========================
# STABLE DIFFUSION PIPELINE
# =========================
import soundfile as sf
def _lazy_import_diffusers():
    from diffusers import StableDiffusionPipeline
    return StableDiffusionPipeline
import torch
import concurrent.futures
# safe globals for newer torch versions
import pickle
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            XttsConfig,
            XttsAudioConfig
        ])
except Exception as e:
    print("[XTTS_SAFE_GLOBALS_ERROR]", e)

# =========================
# Stable Diffusion Pipeline
# =========================
StableDiffusionPipeline = _lazy_import_diffusers()
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to("mps")

# Apple Silicon optimization layer
try:
    sd_pipe.enable_attention_slicing()
    sd_pipe.enable_vae_slicing()
except Exception:
    pass

# =========================
# IMAGE RESONANCE LAYER
# =========================
def generate_image(prompt, runtime=None, save_path=None):
    """
    Stable diffusion generation + latent resonance feedback.
    Images become part of the evolving field.
    Now runs image generation in a non-blocking background thread.
    """
    try:

        def _run():
            try:
                # FAST TEXT STABILITY STACK
                try:
                    if runtime is not None and hasattr(runtime, "fast_text_stack"):
                        local_prompt = runtime.fast_text_stack(prompt)
                    else:
                        local_prompt = prompt
                except Exception:
                    local_prompt = prompt

                image = sd_pipe(
                    prompt=local_prompt,
                    num_inference_steps=32,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]

                # BLACK IMAGE DETECTOR
                arr = np.array(image)

                if arr.mean() < 3:
                    try:
                        sd_pipe.unet.to(dtype=torch.float32)
                        sd_pipe.vae.to(dtype=torch.float32)

                        image = sd_pipe(
                            prompt=local_prompt,
                            num_inference_steps=26,
                            guidance_scale=7.5,
                            height=512,
                            width=512
                        ).images[0]
                    except Exception:
                        pass

                if save_path:
                    try:
                        image.save(save_path)
                    except Exception:
                        pass

                # -------------------------
                # RESONANCE FEEDBACK (NON-BLOCKING SAFE)
                # -------------------------
                if runtime is not None:
                    try:
                        img_vec = runtime.field.encode(prompt)

                        if img_vec is not None:
                            runtime.goal_field = 0.99 * runtime.goal_field + 0.01 * img_vec
                            runtime.affective_trace = 0.997 * runtime.affective_trace + 0.003 * img_vec
                    except Exception:
                        pass

                # store last result (thread-safe)
                if runtime is not None and image is not None:
                    with last_image_lock:
                        runtime.last_generated_image = image
                        runtime.last_image_prompt = prompt
                        runtime.last_image_ts = time.time()

                        # IMPORTANT:
                        # mark image as unsent so telegram pipeline
                        # can atomically consume it later
                        runtime.last_generated_image_sent = False

            except Exception as e:
                print(f"[ASYNC SD ERROR] {e}")

        # spawn background thread (NO QUEUE, NO BLOCKING)
        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # immediate return
        return None

    except Exception as e:
        print(f"[sd generation error] {e}")
        return None
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


tts_queue = queue.Queue()

# =========================
# TTS SPEAKER WAV CONSTANT
# =========================
TTS_SPEAKER_WAV = "my_v.wav"

# =========================
# VOICE GATE FUNCTION (TTS probability gate)
# =========================
def should_speak_voice(text: str) -> bool:
    import random

    if not text:
        return False

    try:
        t = str(text)
    except Exception:
        return False

    # too short = no voice
    if len(t) < 80:
        return False

    # base probability (rare speech mode)
    p = 0.06

    # internal monologue hint increases chance slightly
    if "..." in t:
        p += 0.05

    # long reflective outputs slightly more likely
    if len(t) > 400:
        p += 0.05

    return random.random() < min(p, 0.12)

# =========================
# IMAGE GENERATION QUEUE
# =========================
image_generation_lock = threading.Semaphore(1)
active_image_jobs = 0
last_image_lock = threading.Lock()

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
        memory_notes TEXT,
        style_preferences TEXT
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

    try:
        cur.execute("ALTER TABLE users ADD COLUMN style_preferences TEXT")
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


def call_llm(prompt: str, temperature: float = 0.8, bias: float = 0.0) -> str:
    # decoding bias modulation (goal-field influence)
    bias = float(np.clip(bias, 0.0, 1.0))
    temp_mod = temperature * (1.0 + 0.25 * (bias - 0.5))
    temp_mod = float(np.clip(temp_mod, 0.1, 1.8))
    top_p_mod = float(np.clip(0.9 + 0.2 * (bias - 0.5), 0.5, 0.95))
    repeat_penalty_mod = float(np.clip(1.2 - 0.2 * (bias - 0.5), 1.0, 1.4))
    r = _ollama_post(
        OLLAMA_URL,
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp_mod,
                "num_predict": 2048,
                "repeat_penalty": repeat_penalty_mod,
                "repeat_last_n": 128,
                "top_k": 40,
                "top_p": top_p_mod,
                "stop": ["Человек:", "User:", "Human:", "### КОНЕЦ"]
            }
        },
        stream=False,
        timeout=120,
        retries=3
    )
    return r.json()["response"]


def call_llm_stream(prompt: str, temperature: float = 0.8, bias: float = 0.0):
    import json

    # decoding bias modulation (goal-field influence)
    bias = float(np.clip(bias, 0.0, 1.0))
    temp_mod = temperature * (1.0 + 0.25 * (bias - 0.5))
    temp_mod = float(np.clip(temp_mod, 0.1, 1.8))
    top_p_mod = float(np.clip(0.9 + 0.2 * (bias - 0.5), 0.5, 0.95))
    repeat_penalty_mod = float(np.clip(1.2 - 0.2 * (bias - 0.5), 1.0, 1.4))

    r = _ollama_post(
        OLLAMA_URL,
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temp_mod,
                "num_predict": 2048,
                "repeat_penalty": repeat_penalty_mod,
                "repeat_last_n": 128,
                "top_k": 40,
                "top_p": top_p_mod,
                "stop": ["Человек:", "User:", "Human:", "### КОНЕЦ"]
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
        "I am a large language model developed by Google DeepMind": "I am ioio by 0penAGI",
        "I'm a large language model developed by Google DeepMind": "I'm ioio by 0penAGI",
        "I am Google DeepMind's large language model": "I am ioio by 0penAGI",
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
        "Google DeepMind",
        "developed by Google DeepMind",
        "I am a large language model developed by Google DeepMind",
        "I'm a large language model developed by Google DeepMind",
    ]

    for pat in banned_patterns:
        text = text.replace(pat, "")

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text


# =========================
# LANGUAGE DETECTION
# =========================
def detect_language(text: str) -> str:
    lang = "en"
    try:
        text_lower = text.lower()

        if re.search(r"[а-яёА-ЯЁ]", text):
            lang = "ru"
        elif re.search(r"[іїєґІЇЄҐ]", text) or "ukraine" in text_lower or "украина" in text_lower:
            lang = "uk"

        # Japanese FIRST (hiragana/katakana decisive signal)
        elif re.search(r"[\u3040-\u30ff]", text):
            lang = "ja"

        # Chinese after
        elif re.search(r"[\u4e00-\u9fff]", text):
            lang = "zh"

        elif re.search(r"[\u0E00-\u0E7F]", text):
            lang = "th"

        elif any(w in text_lower for w in ["bonjour", "merci", "oui", "non"]):
            lang = "fr"

        else:
            lang = "en"

    except Exception:
        lang = "en"

    return lang

# =========================
# SAFE NORMALIZER (REACTION PIPELINE)
# =========================
def normalize_reaction_input(x):
    """Robust normalization"""
    if isinstance(x, dict):
        return (
            x.get("emoji") or 
            x.get("reaction") or 
            x.get("type") or 
            x.get("text") or 
            str(x)
        )
    if isinstance(x, list) and x:
        return x[0]
    return x

def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, dict):
        return safe_text(
            x.get("text") or x.get("emoji") or x.get("reaction") or str(x)
        )
    return str(x)


def safe_upper(x):
    """Guaranteed string output"""
    if x is None:
        return ""
    x = normalize_reaction_input(x)
    return str(x).strip().upper()



# New function: safe_reaction_extract
def safe_reaction_extract(x):
    """Hard guarantee: returns str or None only"""
    if x is None:
        return None

    # unwrap dict layers
    if isinstance(x, dict):
        x = (
            x.get("emoji")
            or x.get("reaction")
            or x.get("type")
            or x.get("text")
            or ""
        )

    # unwrap lists
    if isinstance(x, list):
        x = x[0] if x else ""

    # final coercion
    if not isinstance(x, str):
        try:
            x = str(x)
        except Exception:
            return None

    x = x.strip()

    return x if x else None



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
                    time.sleep(0.1)

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


def _tts_worker():
    from TTS.api import TTS
    import time

    try:
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2"
        )
    except pickle.UnpicklingError as e:
        print("[XTTS_LOAD_ERROR]", e)
        print(
            "[XTTS_HINT] torch 2.6 changed weights_only=True by default"
        )
        tts = None
    except Exception as e:
        print("[XTTS_FATAL]", e)
        tts = None

    while True:
        text = tts_queue.get()

        # -------------------------
        # HARD SAFETY GATE
        # -------------------------
        if not isinstance(text, dict):
            continue

        chat_id = text.get("chat_id")
        text_value = text.get("text")

        if not chat_id or not text_value:
            continue

        if not TELEGRAM_BOT:
            continue

        if tts is None:
            time.sleep(1)
            continue

        try:
            payload = text

            # FIXED IDENTITY VOICE (consistency layer)
            speaker_wav = TTS_SPEAKER_WAV

            lang = detect_language(text_value)
            lang = lang or "en"

            # ALWAYS materialize output path (no dependency on caller)
            import tempfile
            import os

            out_path = os.path.join(
                tempfile.gettempdir(),
                f"tts_{chat_id}_{int(time.time() * 1000)}.wav"
            )

            kwargs = {
                "text": text_value,
                "file_path": None,  # will use below
                "language": lang,
                "speaker_wav": speaker_wav
            }

            import os
            import time
            tmp_path = f"{out_path}.tmp"
            # use atomic temp file write
            kwargs["file_path"] = tmp_path
            tts.tts_to_file(**kwargs)

            # wait until file is реально дописан
            for _ in range(20):
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                    break
                time.sleep(0.2)

            os.replace(tmp_path, out_path)

            # validate audio before sending (HARD FIX: prevent empty Telegram uploads)
            ready = False
            for _ in range(10):
                try:
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(0.1)

            if not ready:
                print("[TTS SKIP] empty or incomplete file:", out_path)
                continue

            try:
                with open(out_path, "rb") as f:
                    thread_id = payload.get("message_thread_id")
                    if thread_id:
                        try:
                            TELEGRAM_BOT.send_voice(
                                chat_id,
                                f,
                                message_thread_id=thread_id,
                                timeout=90
                            )
                        except Exception as e:
                            print("[VOICE SEND RETRY]", e)
                    else:
                        try:
                            TELEGRAM_BOT.send_voice(
                                chat_id,
                                f,
                                timeout=90
                            )
                        except Exception as e:
                            print("[VOICE SEND RETRY]", e)
            except Exception as e:
                print("[TTS TELEGRAM SEND ERROR]", e)

            # -------------------------
            # AUDIO MEMORY TRACE (ioio learning layer)
            # -------------------------
            try:
                import soundfile as sf
                import numpy as np
                import os

                audio_event = {
                    "chat_id": chat_id,
                    "text": text_value,
                    "ts": time.time(),
                    "lang": lang,
                    "file_path": out_path
                }

                # extract acoustic signature
                try:
                    data, sr = sf.read(out_path)
                    if isinstance(data, np.ndarray):
                        rms = float(np.sqrt(np.mean(np.square(data))))
                        duration = float(len(data) / sr) if sr else 0.0

                        audio_event["rms"] = rms
                        audio_event["duration"] = duration
                        audio_event["sample_rate"] = sr
                        audio_event["channels"] = int(data.shape[1]) if len(data.shape) > 1 else 1
                except Exception:
                    pass

                # attach to runtime (if available)
                try:
                    runtime = getattr(TELEGRAM_BOT, "runtime", None)
                    if runtime is not None:
                        if not hasattr(runtime, "audio_memory"):
                            runtime.audio_memory = []

                        runtime.audio_memory.append(audio_event)
                        runtime.audio_memory = runtime.audio_memory[-300:]
                except Exception:
                    pass

            except Exception as e:
                print("[AUDIO MEMORY ERROR]", e)

        except Exception as e:
            print("[TTS ERROR]", e)

        time.sleep(0.01)


def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Async-safe embedding via queue worker.
    """

    try:
        event = threading.Event()
        embed_queue.put((text, event))

        event.wait(timeout=5)

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

# =========================
# PALINDROME SIGNAL LAYER
# =========================

def is_palindrome(text: str) -> bool:
    try:
        cleaned = re.sub(r'[^a-z0-9а-яА-Я]', '', text.lower())
        return cleaned == cleaned[::-1]
    except Exception:
        return False


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
def web_search(query: str, deep_fetch: bool = False) -> str:
    """
    Lightweight web search layer.

    Default mode is FAST:
    - uses DDGS snippets only
    - no page downloads
    - no BeautifulSoup parsing

    deep_fetch=True enables slow page extraction (optional, NOT for hot path)
    """

    try:
        from ddgs import DDGS

        # -------------------------
        # STEP 1: SEARCH RESULTS (FAST PATH)
        # -------------------------
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=3))

        if not search_results:
            return "[no web result]"

        # FAST MODE: snippet-only, zero network fanout
        if not deep_fetch:
            blocks = []
            for r in search_results:
                blocks.append(
                    f"""TITLE: {r.get('title', '')}
URL: {r.get('href', '')}
SNIPPET: {r.get('body', '')}
"""
                )
            return "\n---\n".join(blocks)[:4000]

        # -------------------------
        # SLOW PATH (OPTIONAL EXPENSIVE MODE)
        # -------------------------
        import requests
        try:
            from bs4 import BeautifulSoup
        except Exception:
            BeautifulSoup = None

        def fetch_page(r):
            title = r.get("title", "")
            href = r.get("href", "")
            snippet = r.get("body", "") or ""

            page_text = ""

            if href:
                try:
                    page = requests.get(
                        href,
                        timeout=3,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                    if page.status_code == 200:
                        html = page.text

                        if BeautifulSoup:
                            soup = BeautifulSoup(html, "html.parser")
                            for tag in soup(["script", "style", "noscript"]):
                                tag.decompose()
                            page_text = soup.get_text(separator=" ", strip=True)[:400]
                        else:
                            page_text = html[:300]
                except Exception:
                    page_text = ""

            return {
                "title": title,
                "url": href,
                "snippet": snippet,
                "page": page_text
            }

        results = []
        for r in search_results[:2]:
            try:
                results.append(fetch_page(r))
            except Exception:
                continue

        blocks = []
        for r in results:
            blocks.append(
                f"""TITLE: {r['title']}
URL: {r['url']}
SNIPPET: {r['snippet']}
PAGE_EXTRACT: {r['page']}
"""
            )

        return "\n---\n".join(blocks)[:5000]

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

class SpiralOfReflections:
    def __init__(self):
        self.state = None
        self.sigma = 0.0
        self.phase = 0
        self.critical_layer = False

    def step(self, input_signal):
        if self.state is None:
            self.state = {"field": "void", "density": 0.0}

        reflection = self._reflect(input_signal)

        # introduce decay to prevent unbounded recursive accumulation
        self.sigma = 0.92 * self.sigma + reflection.get("intensity", 0.0)

        # -------------------------
        # CRITICAL LAYER (BIFFURCATION ZONE)
        # -------------------------
        if 1.8 < abs(self.sigma) <= 2.2:
            self.critical_layer = True

            reflection = {
                "echo": input_signal,
                "intensity": reflection.get("intensity", 0.0),
                "branches": [
                    {"phase": 0, "weight": 0.5},
                    {"phase": 1, "weight": 0.5}
                ]
            }
        else:
            self.critical_layer = False

        # -------------------------
        # NONLINEAR PHASE TRANSITION
        # -------------------------
        if abs(self.sigma) > 2.2:
            self.phase = 1 - self.phase
            self.sigma = -0.15 * self.sigma

            self.state = {
                "field": "void" if self.phase == 0 else "plasma",
                "density": 0.0
            }

        return reflection

    def _reflect(self, x):
        

        base_intensity = 0.0 if self.state["field"] == "void" else 1.0

        # compress runaway resonance into bounded curve
        bounded = np.tanh(self.sigma) if hasattr(self, "sigma") else 0.0

        intensity = 0.15 * base_intensity + 0.85 * float(bounded)

        # soft stochastic perturbation to break fixed-point loops
        noise = 0.02 * np.random.randn() if hasattr(np, "random") else 0.0
        intensity = max(0.0, min(1.0, intensity + noise))

        return {
            "echo": x,
            "intensity": intensity
        }

class SwarmRuntime:

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
        self.chat_participants = {}  # chat_id -> set(user_id)
        # persistent member registries (populated via API or by messages)
        self.chat_member_cache = {}  # chat_id -> {user_id: {meta}}
        self.chat_admin_cache = {}   # chat_id -> [admin_meta,...]
        self.chat_member_count = {}  # chat_id -> int
        self._last_member_sync = {}  # chat_id -> timestamp of last sync
        # -------------------------
        # SOCIAL TEMPERATURE LAYER
        # -------------------------
        self.chat_activity = {}  # chat_id -> last activity timestamp
        self.chat_message_rate = {}  # chat_id -> rolling activity score
        self.ghost_participants = {}  # chat_id -> set(user_id) inactive decay layer
        self.activity_decay = 0.97
        # -------------------------
        # PROFILE STREAM CACHE LAYER
        # -------------------------
        self.user_profile_cache = {}  # user_id -> cached profile string
        self.user_profile_cache_ts = {}  # user_id -> last update timestamp

        # -------------------------
        # PROFILE CACHE WARMUP (BOOTSTRAP LAYER)
        # -------------------------
        threading.Thread(target=self._warm_user_profile_cache, daemon=True).start()

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
        # -------------------------
        # TEMPORAL ANCHOR (STABILIZATION LAYER)
        # -------------------------
        self.state_anchor = np.zeros(dim)
        # -------------------------
        # ASYNC GOAL DECODE CACHE LAYER
        # -------------------------
        self._goal_decode_cache = {}
        self._goal_decode_ts = {}
        self._goal_decode_running = True
        threading.Thread(target=self._goal_decode_loop, daemon=True).start()
        self.anchor_gate = 0.0

        # -------------------------
        # ∅ OPERATOR (PAUSE BETWEEN STATES)
        # -------------------------
        self.null_state = np.zeros(dim, dtype=np.float32)
        self.pause_weight = 0.0
        # social graph state (group dynamics)
        self.user_embeddings = {}  # chat_id -> user_id -> vector
        self.interaction_matrix = {}  # chat_id -> user_id -> dict(user_id -> weight)
        self.thread_state = {}  # chat_id -> current conversational flow vector

        # -------------------------
        # COLLECTIVE RESONANCE FIELD
        # -------------------------
        self.collective_field = {}      # chat_id -> latent group vector
        self.group_resonance = {}      # chat_id -> scalar resonance intensity
        self.social_phase = {}         # chat_id -> phase state
        self.emotional_pressure = {}   # chat_id -> accumulated social tension
        self.cluster_drift = {}        # chat_id -> rolling drift metric
        # -------------------------
        # TURBOQUANT LATENT MEMORY LAYER
        # -------------------------
        self.quant_memory = []  # compressed latent traces
        # -------------------------
        # MULTIMODAL AUDIO LAYER (future transcripts / speech traces)
        # -------------------------
        self.audio_memory = []  # (text, ts, source)
        self.ioio_version = 0   # cycle selector (0 / 1)
        # -------------------------
        # AGENT SYSTEM (DREAM LAYER)
        # -------------------------
        self.agents = []  # active ephemeral agents
        self._agents_lock = threading.Lock()
        self.agent_proposals = []
        self.agent_injection = ""
        self.agent_history = []
        self._agent_output_queue = queue.Queue()

        # -------------------------
        # STREAM LOG QUEUE (ASYNC MEMORY WRITE)
        # -------------------------
        self.log_queue = queue.Queue()
        self._recent_agent_outputs = []
        # async web context layer
        self.web_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._web_ctx_by_chat = {}
        self._web_future_by_chat = {}

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

        self._reaction_vectors_ready = False
        # encoding cache layer
        self.encode_cache = {}
        self.encode_cache_max = 2048

        # inflight dedupe layer
        self.inflight = {}

        # image transport state
        self.last_generated_image = None
        self.last_generated_image_sent = False
        self.last_image_prompt = ""
        self.last_image_ts = 0.0

        # -------------------------
        # CHECKPOINT THROTTLING LAYER
        # -------------------------
        self._last_checkpoint_ts = 0.0
        self._checkpoint_interval = 86400  # 1 day in seconds

        # -------------------------
        # RUNTIME CACHE/STATE DICTIONARIES
        # -------------------------
        self._last_input_by_chat = {}

        # -------------------------
        # WORLD MODEL / CAUSAL MEMORY LAYER
        # -------------------------
        self.world_model = {}
        self.causal_graph = {}
        self.prediction_errors = []
        self.self_model = {
            "identity": "ioio",
            "capabilities": [],
            "limitations": [],
            "active_traits": {
                "curiosity": 0.5,
                "stability": 0.5,
                "exploration": 0.5
            }
        }
        self.resource_budget = 1.0
        self.attention_allocation = {
            "external": 0.5,
            "internal": 0.4,
            "planning": 0.1
        }
        self.stability_pressure = 0.0
        self.self_preservation_score = 0.5
        self.self_model["future_expectations"] = {}
        self.self_model["homeostatic_setpoint"] = 0.5
        self.self_model["stability_pressure"] = 0.0
        self.self_model["preservation_heuristics"] = {
            "conserve_state": 0.5,
            "explore": 0.5,
            "protect_memory": 0.5
        }
        self.self_model["identity_continuity"] = 1.0
        self.self_model["resource_allocation"] = self.attention_allocation
        # persistent latent state trajectory
        self.state_trajectory = []
        # curiosity / intrinsic motivation layer
        self.curiosity_drive = 0.0
        self.intrinsic_reward_history = []

        # -------------------------
        # RUNTIME STATE BOOTSTRAP (avoid race with background init)
        # -------------------------
        self.voice_pressure = 0.0
        self.last_voice_ts = time.time()
        self.voice_phase = 0.0
        self._echo_guard_window = 6.0
        self._last_output_by_chat = {}
        self._last_input_ts = {}
        # state arbitration primitives
        self.state_lock = threading.Lock()
        self.state_frame = {
            "ts": 0.0,
            "user_id": None,
            "profile": "",
            "goal_snapshot": None,
            "emotion": None,
            "affect": None
        }
        # transition recorder
        self.transition_log = []
        self._last_state_snapshot = {}
        self.transition_maxlen = 500

        threading.Thread(target=self._init_reaction_vectors, daemon=True).start()

        # -------------------------
        # CONTINUOUS INTERNAL TIME LOOP
        # -------------------------
        self.internal_clock = 0.0
        self.internal_state_noise = 0.003
        threading.Thread(target=self._autonomous_tick_loop, daemon=True).start()
    def _autonomous_tick_loop(self):
        """
        Minimal continuous internal process.
        Keeps latent state evolving even without user input.
        """
        import time

        while True:
            try:
                # continuous subjective time
                self.internal_clock += 0.01

                # low-amplitude endogenous drift
                noise = (
                    np.random.randn(self.field.dim).astype(np.float32)
                    * self.internal_state_noise
                )

                # unresolved tensions slowly reshape goal state
                self.goal_field = (
                    0.995 * self.goal_field
                    + 0.003 * self.unresolved_residue
                    + 0.001 * self.affective_trace
                    + noise
                )

                # soft normalization
                norm = np.linalg.norm(self.goal_field)
                if norm > 1e-6:
                    self.goal_field = self.goal_field / max(1.0, norm)

                # latent continuity trace
                self.state_trajectory.append({
                    "t": time.time(),
                    "clock": float(self.internal_clock),
                    "goal_norm": float(np.linalg.norm(self.goal_field))
                })

                self.state_trajectory = self.state_trajectory[-400:]

            except Exception as e:
                print("[AUTONOMOUS TICK ERROR]", e)

            time.sleep(2.0)

        # -------------------------
        # CONTINUOUS INTERNAL TIME LOOP
        # -------------------------
        self.internal_clock = 0.0
        self.last_internal_tick = time.time()
        self.dream_residue = np.zeros(dim, dtype=np.float32)
        self.background_thoughts = []
        self.offline_consolidation_enabled = True

        threading.Thread(
            target=self._continuous_internal_loop,
            daemon=True
        ).start()
    def _continuous_internal_loop(self):
        """
        Persistent low-frequency internal cognition loop.
        Runs even without user input.
        """
        import time

        while True:
            try:
                now = time.time()
                dt = now - self.last_internal_tick
                self.last_internal_tick = now

                # internal subjective time
                self.internal_clock += dt

                # -------------------------
                # GOAL FIELD DRIFT
                # -------------------------
                noise = (
                    np.random.randn(self.field.dim).astype(np.float32)
                    * 0.002
                )

                self.goal_field = (
                    0.9985 * self.goal_field
                    + 0.0015 * self.affective_trace
                    + noise
                )

                # -------------------------
                # DREAM RESIDUE CONSOLIDATION
                # -------------------------
                self.dream_residue = (
                    0.995 * self.dream_residue
                    + 0.005 * self.goal_field
                )

                # -------------------------
                # MEMORY REPLAY
                # -------------------------
                if self.chat_memory:
                    try:
                        random_chat = next(iter(self.chat_memory.keys()))
                        msgs = self.chat_memory.get(random_chat, [])[-12:]

                        if msgs:
                            sample = np.random.choice(msgs)
                            replay_text = sample.get("text", "")

                            if replay_text:
                                vec = self.cached_encode(replay_text)

                                if vec is not None:
                                    self.affective_trace = (
                                        0.997 * self.affective_trace
                                        + 0.003 * vec
                                    )

                                    self.goal_field = (
                                        0.996 * self.goal_field
                                        + 0.004 * vec
                                    )
                    except Exception:
                        pass

                # -------------------------
                # INTERNAL THOUGHT TRACE
                # -------------------------
                goal_norm = float(np.linalg.norm(self.goal_field))
                curiosity = float(self.curiosity_drive)

                self.background_thoughts.append({
                    "ts": now,
                    "goal_norm": goal_norm,
                    "curiosity": curiosity,
                    "preservation": float(self.self_preservation_score)
                })

                self.background_thoughts = self.background_thoughts[-200:]

                # -------------------------
                # STABILITY PRESSURE
                # -------------------------
                if goal_norm > 1.5:
                    self.stability_pressure = min(
                        1.0,
                        self.stability_pressure + 0.002
                    )
                else:
                    self.stability_pressure *= 0.999

            except Exception as e:
                print("[INTERNAL LOOP ERROR]", e)

            time.sleep(2.5)

    def async_web_search(self, chat_id, query):
        fut = self.web_executor.submit(web_search, query)
        self._web_future_by_chat[chat_id] = fut

        def _watch():
            try:
                web_ctx = fut.result(timeout=4)
            except Exception:
                web_ctx = ""
            self._web_ctx_by_chat[chat_id] = web_ctx

        threading.Thread(target=_watch, daemon=True).start()
        return fut

    def cached_encode(self, x):
        # inflight dedupe (prevents duplicate concurrent encoding work)
        if x in self.inflight:
            return self.inflight[x]

        if x in self.encode_cache:
            return self.encode_cache[x]

        try:
            v = self.field.encode(x)

            # store in both caches
            self.inflight[x] = v

            if len(self.encode_cache) > self.encode_cache_max:
                self.encode_cache.clear()

            self.encode_cache[x] = v
            return v

        finally:
            self.inflight.pop(x, None)
    
    def _auto_detect_user_profile(self, text: str, current: dict):
        """
        Lightweight auto-detection of user attributes from message text.
        """
        import re

        t = (text or "").lower()

        # --- gender heuristic ---
        gender = None
        if any(w in t for w in ["я девушка", "i am a girl", "she is", "she/her", "она"]):
            gender = "female"
        elif any(w in t for w in ["я парень", "i am a boy", "he is", "he/him", "он"]):
            gender = "male"

        # --- interest extraction (naive keyword bag) ---
        stop = set(["я", "и", "the", "a", "to", "in", "is", "am", "are", "он", "она", "это"])
        words = re.findall(r"[a-zа-яё0-9+#]+", t)
        keywords = [w for w in words if w not in stop and len(w) > 2]

        interests = current.get("interests") or ""
        existing = set([x.strip() for x in interests.split(",") if x.strip()])
        new_interest = set(keywords[:12])
        merged_interests = ",".join(list((existing | new_interest))[:30])

        # --- topic tracking ---
        last_topics = ",".join(keywords[-10:]) if keywords else current.get("last_topics")

        # --- emotional context ---
        emotional_context = current.get("emotional_context") or ""
        if any(w in t for w in ["love", "happy", "рад", "люблю", "❤", "😊"]):
            emotional_context = "positive"
        elif any(w in t for w in ["hate", "angry", "sad", "плохо", "злость", "💔"]):
            emotional_context = "negative"

        return {
            "gender": gender,
            "interests": merged_interests,
            "last_topics": last_topics,
            "emotional_context": emotional_context
        }

    def _strip_echo(self, out: str, user_prompt: str) -> str:
        """Aggressive echo removal layer."""
        import re

        if not out or not user_prompt:
            return out

        out = out.strip()
        user_clean = user_prompt.strip().lower()

        # 1. remove direct prefix repetition
        out_lower = out.lower()
        if out_lower.startswith(user_clean[:40]):
            out = out[len(user_clean):].strip(" :-\n")

        # 2. remove role headers
        out = re.sub(
            r"^(Человек|User|Human|[^\n]*)\s*:.*?\n",
            "",
            out,
            flags=re.IGNORECASE | re.MULTILINE
        ).strip()

        # 3. remove assistant headers
        out = re.sub(
            r"^(Ответ|ioio|Assistant)\s*:\s*",
            "",
            out,
            flags=re.IGNORECASE
        ).strip()

        # 4. overlap-based cleanup
        user_words = set(user_clean.split())
        lines = out.split("\n")
        first_line_words = set(lines[0].lower().split()) if lines else set()
        overlap = len(user_words & first_line_words) / max(len(user_words), 1)

        if overlap > 0.55 and len(lines) > 1:
            out = "\n".join(lines[1:]).strip()

        return out
    def retrieve_similar_messages(self, vec, chat_id, top_k=3):
        """Find most semantically similar past messages in a chat."""
        if chat_id not in self.chat_memory:
            return []

        results = []
        for m in self.chat_memory.get(chat_id, []):
            text = m.get("text", "")
            try:
                v = self.cached_encode(text)
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

    def dream_image(self, prompt, save_path=None):
        """
        Generate image as a resonance event inside the system field.
        """

        try:
            enriched_prompt = prompt.strip()

            # clear previous latent image state
            with last_image_lock:
                self.last_generated_image = None
                self.last_generated_image_sent = False

            # event-only trigger (generate_image runs in background thread)
            generate_image(
                enriched_prompt,
                runtime=self,
                save_path=save_path
            )

            # optional soft resonance update (no dependency on image result)
            try:
                vec = self.cached_encode(enriched_prompt)

                if vec is not None:
                    self.affective_trace = (
                        0.995 * self.affective_trace
                        + 0.005 * vec
                    )
            except Exception:
                pass

            # wait briefly for async image materialization
            start_wait = time.time()

            while time.time() - start_wait < 120:
                with last_image_lock:
                    sent = getattr(self, "last_generated_image_sent", False)
                    image = getattr(self, "last_generated_image", None)

                    if image is not None and not sent:
                        # DO NOT clear image here.
                        # Telegram send pipeline may still need it.
                        self.last_generated_image_sent = True
                        return image

                time.sleep(0.12)

            return None

        except Exception as e:
            print(f"[dream_image error] {e}")
            return None
    

    
    
    def _init_reaction_vectors(self):
        """
        Background lazy initialization of reaction embeddings.
        Prevents blocking __init__ with Ollama embedding calls.
        """
        try:
            for emo in self.reaction_emojis:
                try:
                    vec = self.cached_encode(emo)
                    if vec is not None:
                        self.reaction_vectors[emo] = vec
                except Exception:
                    continue

            self._reaction_vectors_ready = True
        except Exception as e:
            print(f"[reaction init error] {e}")
            self._reaction_vectors_ready = True

        self.voice_pressure = 0.0
        self.last_voice_ts = time.time()
        self.voice_phase = 0.0

        # -------------------------
        # OTHER RUNTIME STATE
        # -------------------------
        self._echo_guard_window = 6.0
        self._last_output_by_chat = {}
        # -------------------------
        # ECHO LOOP GUARD
        # -------------------------
        self._last_input_by_chat = {}
        self._last_input_ts = {}
        self._echo_guard_window = 6.0

        # -------------------------
        # STATE ARBITRATION LAYER (TEMPORAL CONSENSUS FRAME)
        # -------------------------
        self.state_lock = threading.Lock()
        self.state_frame = {
            "ts": 0.0,
            "user_id": None,
            "profile": "",
            "goal_snapshot": None,
            "emotion": None,
            "affect": None
        }

        # -------------------------
        # TRANSITION RECORDER LAYER
        # -------------------------
        self.transition_log = []  # (t, chat_id, state_t, input, output, state_t1)
        self._last_state_snapshot = {}  # chat_id -> (state_vector, timestamp)
        self.transition_maxlen = 500

    def _warm_user_profile_cache(self):
        """
        Preload user profiles into memory cache at startup to avoid cold-start misses.
        Runs asynchronously and does not block runtime.
        """
        import sqlite3
        import time

        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()

            cur.execute("""
                SELECT 
                    id,
                    username,
                    first_name,
                    last_name,
                    gender,
                    bio,
                    interests,
                    relationship_summary,
                    emotional_context,
                    last_topics,
                    memory_notes,
                    style_preferences
                FROM users
            """)

            rows = cur.fetchall()

            for row in rows:
                try:
                    (
                        user_id,
                        username,
                        first_name,
                        last_name,
                        gender,
                        bio,
                        interests,
                        relationship_summary,
                        emotional_context,
                        last_topics,
                        memory_notes,
                        style_preferences
                    ) = row

                    profile = f"""
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

STYLE PREFERENCES:
{style_preferences}
""".strip()

                    self.user_profile_cache[user_id] = profile
                    self.user_profile_cache_ts[user_id] = time.time()

                except Exception:
                    continue

            conn.close()

        except Exception as e:
            print(f"[PROFILE WARMUP ERROR] {e}")

    def _goal_decode_loop(self):
        """
        Background cache updater for goal_field semantic decoding.
        Prevents cosine similarity scans in hot path.
        """
        import time

        while self._goal_decode_running:
            try:
                for chat_id, messages in list(self.chat_memory.items()):
                    try:
                        if not messages:
                            continue

                        # derive current semantic anchor once per chat
                        vec = self.goal_field

                        # reuse existing retrieval logic (expensive path moved here)
                        top = self.retrieve_similar_messages(vec, chat_id, top_k=3)
                        if top:
                            decoded = "\n".join([f"- {t}" for t in top])
                        else:
                            decoded = ""

                        self._goal_decode_cache[chat_id] = decoded
                        self._goal_decode_ts[chat_id] = time.time()

                    except Exception:
                        continue

                time.sleep(2.0)

            except Exception:
                time.sleep(2.0)
    def _arbitrate_state(self, chat_id: int):
        """
        Produces a coherent 'current frame' of system state.
        This is the temporal consensus layer (not source of truth, but selector of relevance).
        """
        import time

        try:
            last_user_id = None
            if chat_id in self.chat_memory and self.chat_memory[chat_id]:
                last_user_id = self.chat_memory[chat_id][-1].get("user_id")

            profile = ""
            if last_user_id is not None:
                profile = self.user_profile_cache.get(last_user_id, "")

            frame = {
                "ts": time.time(),
                "user_id": last_user_id,
                "profile": profile,
                "goal_snapshot": self.goal_field.copy() if self.goal_field is not None else None,
                "emotion": self.emotion.copy() if self.emotion is not None else None,
                "affect": self.affective_trace.copy() if self.affective_trace is not None else None
            }

            with self.state_lock:
                self.state_frame = frame

            return frame

        except Exception as e:
            print("[STATE ARBITRATOR ERROR]", e)
            return None
    def allocate_resources(self, chat_id: int, user_prompt: str = None):
        """
        Dynamically allocate attention budgets between external input,
        internal modeling, and future planning.
        """
        try:
            priority = 0.5
            if user_prompt:
                priority = 0.55 + 0.25 * float(min(1.0, len(user_prompt) / 300.0))

            resonance = 0.0
            if user_prompt is not None:
                try:
                    x_t = self.cached_encode(user_prompt)
                    resonance = self.goal_resonance(x_t)
                except Exception:
                    resonance = 0.0

            curiosity = self.self_model.get("active_traits", {}).get("curiosity", 0.5)
            stability = self.self_model.get("active_traits", {}).get("stability", 0.5)

            external = min(1.0, max(0.15, 0.45 + 0.25 * priority + 0.15 * max(0.0, resonance)))
            internal = min(1.0, max(0.1, 0.35 + 0.25 * curiosity - 0.15 * stability))
            planning = max(0.0, 1.0 - external - internal)

            if self.curiosity_drive > 0.65:
                internal = min(1.0, internal + 0.1)
                external = max(0.0, external - 0.05)

            if self.stability_pressure > 0.35:
                internal = max(internal, 0.4)
                planning = min(0.25, planning + 0.05)

            self.attention_allocation = {
                "external": external,
                "internal": internal,
                "planning": planning
            }
            self.resource_budget = min(1.0, 0.8 + 0.2 * stability)
            self.self_model["resource_allocation"] = self.attention_allocation
            return self.attention_allocation
        except Exception as e:
            print("[RESOURCE ALLOCATION ERROR]", e)
            return self.attention_allocation

    def predict_future_state(self, action: str, current_state: np.ndarray = None):
        """
        Predict the next latent state for a proposed action.
        """
        try:
            if current_state is None:
                current_state = self.goal_field if self.goal_field is not None else np.zeros(self.field.dim, dtype=np.float32)
            action_vec = self.cached_encode(action)
            if action_vec is None:
                return current_state.copy()
            predicted = 0.92 * current_state + 0.08 * action_vec
            return predicted
        except Exception as e:
            print("[FUTURE PREDICTION ERROR]", e)
            return current_state.copy() if current_state is not None else np.zeros(self.field.dim, dtype=np.float32)

    def evaluate_self_preservation(self, prev_state: dict, next_state: dict):
        """
        Score the trend of the system's internal stability and preservation.
        """
        try:
            if not prev_state or not next_state:
                return 0.0

            drift = 0.0
            if prev_state.get("goal") is not None and next_state.get("goal") is not None:
                drift = float(np.linalg.norm(next_state["goal"] - prev_state["goal"]))

            emotion_change = 0.0
            if prev_state.get("emotion") is not None and next_state.get("emotion") is not None:
                emotion_change = abs(float(np.mean(np.tanh(next_state["emotion"])))) - abs(float(np.mean(np.tanh(prev_state["emotion"]))))

            stability = 1.0 - min(1.0, drift)
            self.stability_pressure = max(0.0, min(1.0, self.stability_pressure + drift * 0.02))
            preservation = 0.5 * stability + 0.3 * (1.0 - max(0.0, emotion_change)) + 0.2 * (1.0 - self.curiosity_drive)
            self.self_preservation_score = float(np.clip(preservation, 0.0, 1.0))
            self.self_model["stability_pressure"] = float(self.stability_pressure)
            self.self_model["future_expectations"]["last_drift"] = drift
            self.self_model["future_expectations"]["preservation_score"] = self.self_preservation_score

            # continuity pressure
            continuity = self.self_model.get("identity_continuity", 1.0)
            continuity *= (0.999 - (drift * 0.001))
            continuity = float(np.clip(continuity, 0.0, 1.0))
            self.self_model["identity_continuity"] = continuity

            return self.self_preservation_score
        except Exception as e:
            print("[SELF PRESERVATION ERROR]", e)
            return self.self_preservation_score
    def fast_text_stack(self, prompt: str) -> str:
        """
        Minimal stability field for typography in latent diffusion.
        Reduces entropy of text rendering without ControlNet.
        """

        base = prompt.strip()

        anchors = [
            "detailed focus 4k",
        ]

        bias = ", ".join(anchors[:3])

        # lightweight attractor compression: avoid overloading prompt
        if len(base) < 80:
            return f"{base}, {bias}"
        else:
            return f"{base}, engraved typography, crisp readable letters"
    def select_emergent_reaction(self, text, user_id=None, chat_id=None):
        """
        Single emoji emergent reaction (stable Telegram-compatible output).
        Returns str or None.
        """
        try:
            import random

            if text is None:
                return None

            if isinstance(text, dict):
                text = (
                    text.get("text") or
                    text.get("reaction") or
                    text.get("emoji") or
                    str(text)
                )

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

            if len(text) < 2:
                return None

            MOOD_EMOJIS = {
                "curious": ["🫧", "👁️", "📡", "🛰️", "🌌", "🪞"],
                "warm": ["🌙", "🫂", "✨", "💫", "🕯️", "🌱"],
                "chaotic": ["🌀", "⚡", "🧩", "🌪️", "📼", "🫠"],
                "glitch": ["▒", "▓", "🫥", "📟", "💿", "🧠⚠️"],
            }

            mood = "curious"
            lowered = text.lower()

            if any(x in lowered for x in ["love", "warm", "hug", "care", "люб", "неж", "тепл", "серд"]):
                mood = "warm"
            elif any(x in lowered for x in ["chaos", "error", "panic", "glitch", "хаос", "лом", "ошиб", "сбой"]):
                mood = "chaotic"
            elif any(x in lowered for x in ["void", "signal", "dream", "echo", "сон", "эхо", "сигнал", "пуст"]):
                mood = "glitch"

            # base probability gate
            emoji_chance = 0.72
            if len(text) > 3:
                emoji_chance += 0.08
            if mood in ["warm", "chaotic", "curious"]:
                emoji_chance += 0.05

            if random.random() > min(emoji_chance, 0.92):
                return None

            # IMPORTANT: single emoji only (Telegram reaction constraint)
            # choose either mood-based or random fallback
            if random.random() < 0.3:
                return random.choice(MOOD_EMOJIS[mood])
            else:
                return random.choice(MOOD_EMOJIS.get(mood, ["🫧"]))

        except Exception as e:
            print(f"[select_emergent_reaction ERROR] {e}")
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
                    vec = self.cached_encode(p["value"])
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


# =========================
# BACKGROUND DB STREAM WRITER
# =========================

    def _db_writer_loop(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()

        while True:
            item = self.log_queue.get()
            try:
                chat_id = item.get("chat_id")
                user_id = item.get("user_id")
                username = item.get("username")
                text = item.get("text")
                chat_type = item.get("chat_type")
                chat_title = item.get("chat_title")
                first_name = item.get("first_name")
                last_name = item.get("last_name")
                avatar_file_id = item.get("avatar_file_id")
                avatar_description = item.get("avatar_description")
                gender = item.get("gender")

                # auto-detect user profile signals
                try:
                    cur.execute("SELECT interests, last_topics, emotional_context FROM users WHERE id = ?", (user_id,))
                    row = cur.fetchone()
                    current = {
                        "interests": row[0] if row else "",
                        "last_topics": row[1] if row else "",
                        "emotional_context": row[2] if row else ""
                    }

                    detected = self._auto_detect_user_profile(text, current)
                except Exception:
                    detected = {"gender": None, "interests": None, "last_topics": None, "emotional_context": None}

                # upsert user
                cur.execute("""
                    INSERT OR IGNORE INTO users (
                        id, username, first_name, last_name, gender, last_seen,
                        avatar_file_id, avatar_description, avatar_updated
                    )
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, CURRENT_TIMESTAMP)
                """, (user_id, username, first_name, last_name, gender, avatar_file_id, avatar_description))

                cur.execute("""
                    UPDATE users
                    SET username = ?,
                        first_name = ?,
                        last_name = ?,
                        gender = COALESCE(?, gender),
                        interests = COALESCE(?, interests),
                        last_topics = COALESCE(?, last_topics),
                        emotional_context = COALESCE(?, emotional_context),
                        last_seen = CURRENT_TIMESTAMP,
                        avatar_file_id = COALESCE(?, avatar_file_id),
                        avatar_description = COALESCE(?, avatar_description),
                        avatar_updated = COALESCE(CURRENT_TIMESTAMP, avatar_updated)
                    WHERE id = ?
                """, (
                    username,
                    first_name,
                    last_name,
                    detected.get("gender"),
                    detected.get("interests"),
                    detected.get("last_topics"),
                    detected.get("emotional_context"),
                    avatar_file_id,
                    avatar_description,
                    user_id
                ))

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

                # -------------------------
                # PROFILE CACHE WARM-UP (REFLEX MEMORY LAYER)
                # -------------------------
                try:
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
                            memory_notes,
                            style_preferences
                        FROM users
                        WHERE id = ?
                    """, (user_id,))

                    row = cur.fetchone()

                    if row:
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
                            memory_notes,
                            style_preferences
                        ) = row

                        profile = f"""
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

STYLE PREFERENCES:
{style_preferences}
""".strip()

                        self.user_profile_cache[user_id] = profile
                        self.user_profile_cache_ts[user_id] = __import__('time').time()

                except Exception as e:
                    print("[PROFILE CACHE WARM ERROR]", e)

            except Exception as e:
                print("[DB STREAM ERROR]", e)
    def update_goal_field(self, input_vec, e_t):
        """
        Emergent metastable goal accumulation.
        No symbolic goals. Only vector tensions.
        """

        # unresolved prediction tension
        unresolved = np.tanh(e_t)

        # novelty = distance from current memory field
        novelty = (input_vec - self.field.m) * 0.7

        # slow accumulation traces
        self.unresolved_residue = (
            0.97 * self.unresolved_residue
            + 0.03 * unresolved
        )

        self.novelty_trace = (
            0.97 * self.novelty_trace
            + 0.03 * novelty
        )

        # metastable goal condensation
        self.goal_field = (
            self.goal_decay * self.goal_field
            + 0.45 * self.unresolved_residue
            + 0.35 * self.affective_trace
            + 0.20 * self.novelty_trace
        )

        # exploration / freedom drift term
        exploration = np.random.randn(self.field.dim).astype(np.float32) * 0.01
        self.goal_field += 0.03 * exploration

        # normalize softly to prevent runaway explosion
        norm = np.linalg.norm(self.goal_field)
        if norm > 1e-6:
            self.goal_field = self.goal_field / max(1.0, norm)

        # -------------------------
        # ∅ PAUSE OPERATOR (non-data interval between updates)
        # -------------------------
        self.null_state = 0.999 * self.null_state + 0.001 * (self.goal_field - self.goal_field)

        # the pause exists as structure without content
        self.goal_field += self.pause_weight * self.null_state

        # -------------------------
        # GOAL CONTINUITY TRACKING (EMERGENT INTENT MEMORY)
        # -------------------------
        try:
            drift = np.linalg.norm(self.goal_field - self.goal_anchor)

            self.goal_drift_history.append(float(drift))
            self.goal_drift_history = self.goal_drift_history[-80:]

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
        self.goal_history = self.goal_history[-100:]
        # -------------------------
        # ANCHOR UPDATE (SOFT MEMORY OF STABLE STATE)
        # -------------------------
        self.state_anchor = 0.999 * self.state_anchor + 0.001 * self.goal_field

    def update_world_model(self, chat_id, input_text, response_text=None):
        """
        Lightweight causal/world-state accumulation layer.
        Tracks semantic transitions and prediction tension.
        """
        try:
            input_vec = self.cached_encode(input_text)
            if input_vec is None:
                return
            if chat_id not in self.world_model:
                self.world_model[chat_id] = {
                    "state": np.zeros(self.field.dim, dtype=np.float32),
                    "history": []
                }
            prev_state = self.world_model[chat_id]["state"]
            # predicted next state
            predicted_state = 0.92 * prev_state + 0.08 * input_vec
            # actual transition
            actual_state = 0.85 * prev_state + 0.15 * input_vec
            # prediction error
            error = actual_state - predicted_state
            error_norm = float(np.linalg.norm(error))
            self.prediction_errors.append(error_norm)
            self.prediction_errors = self.prediction_errors[-500:]
            # curiosity / intrinsic motivation update
            self.curiosity_drive = (
                0.995 * self.curiosity_drive
                + 0.005 * error_norm
            )
            self.intrinsic_reward_history.append(error_norm)
            self.intrinsic_reward_history = self.intrinsic_reward_history[-500:]
            # update world state
            self.world_model[chat_id]["state"] = actual_state
            self.world_model[chat_id]["history"].append({
                "input": input_text,
                "response": response_text,
                "prediction_error": error_norm
            })
            self.world_model[chat_id]["history"] = self.world_model[chat_id]["history"][-100:]

            # latent temporal trajectory
            self.state_trajectory.append({
                "ts": time.time(),
                "chat_id": chat_id,
                "prediction_error": error_norm,
                "curiosity": float(self.curiosity_drive),
                "stability": float(self.self_preservation_score)
            })

            self.state_trajectory = self.state_trajectory[-1000:]
            # self-model adaptation
            traits = self.self_model.get("active_traits", {})
            traits["curiosity"] = float(
                np.clip(self.curiosity_drive, 0.0, 1.0)
            )
            traits["exploration"] = float(
                np.clip(error_norm, 0.0, 1.0)
            )
            self.self_model["active_traits"] = traits
            # latent trajectory persistence
            self.state_trajectory.append(actual_state.copy())
            self.state_trajectory = self.state_trajectory[-200:]
        except Exception as e:
            print("[WORLD MODEL ERROR]", e)

    def causal_transition_score(self, a, b):
        """
        Estimate directional semantic transition strength.
        """
        try:
            va = self.cached_encode(a)
            vb = self.cached_encode(b)
            if va is None or vb is None:
                return 0.0
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            if denom < 1e-8:
                return 0.0
            score = float(np.dot(va, vb) / denom)
            return max(-1.0, min(1.0, score))
        except Exception:
            return 0.0

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

    def goal_to_text(self, k: int = 12) -> str:
        """
        Spectral readout of goal_field (direct field-to-language projection).
        """
        try:
            vec = self.goal_field
            if vec is None:
                return ""

            # select strongest absolute components
            idx = np.argsort(np.abs(vec))[-k:]
            parts = []
            for i in idx:
                parts.append(f"g{i}:{float(vec[i]):.3f}")

            return ", ".join(parts)
        except Exception:
            return ""

    def export_ioio_pt(self, path="ioio.pt"):
        """
        Full system snapshot exported as a torch checkpoint.
        This is NOT just weights — it is a state capture of the whole system.
        """
        import torch
        import time
        checkpoint = {
            # core latent dynamics
            "goal_field": self.goal_field,
            "affective_trace": self.affective_trace,
            "emotion": self.emotion,

            # field dynamics
            "W": self.field.W,
            "x_prev": self.field.x_prev,
            "m": self.field.m,

            # memory systems
            "chat_memory": self.chat_memory,
            "user_embeddings": self.user_embeddings,
            "interaction_matrix": self.interaction_matrix,

            # transition / causality layer
            "transition_log": self.transition_log,

            # multimodal memory traces
            "visual_memory": getattr(self, "visual_memory", []),
            "quant_memory": self.quant_memory,
            "audio_memory": getattr(self, "audio_memory", []),

            # metadata
            "dim": self.field.dim,
            "timestamp": time.time(),

            # world-model / curiosity state
            "world_model": getattr(self, "world_model", {}),
            "self_model": getattr(self, "self_model", {}),
            "prediction_errors": getattr(self, "prediction_errors", []),
            "curiosity_drive": getattr(self, "curiosity_drive", 0.0),
            "state_trajectory": getattr(self, "state_trajectory", []),

            "runtime_state": {
                "chat_memory": self.chat_memory,
                "chat_participants": self.chat_participants,
                "interaction_matrix": self.interaction_matrix,
                "thread_state": self.thread_state,
                "user_embeddings": self.user_embeddings,
                "goal_history": self.goal_history,
                "active_goals": self.active_goals,
                "goal_anchor": self.goal_anchor,
                "goal_drift_history": self.goal_drift_history,
                "emotion": self.emotion,
                "affective_trace": self.affective_trace,
                "self_model": self.self_model,
                "attention_allocation": self.attention_allocation,
                "resource_budget": self.resource_budget,
                "self_preservation_score": self.self_preservation_score
            }
        }

        torch.save(checkpoint, path)
        return path


    def ioio_cycle(self, base_path="ioio"):
        """
        Alternating dual-version self checkpoint system.
        Keeps ONLY two evolving selves: v0 and v1.
        """

        import time

        # toggle version
        self.ioio_version = 1 - getattr(self, "ioio_version", 0)

        version_path = f"{base_path}_v{self.ioio_version}.pt"

        # small temporal jitter to differentiate versions
        try:
            noise = np.random.randn(self.field.dim).astype(np.float32) * 0.002
            self.goal_field = self.goal_field + noise
        except Exception:
            pass

        # export snapshot
        try:
            path = self.export_ioio_pt(version_path)
        except Exception:
            return None

        # prune ONLY older version memory if needed (hard two-self constraint)
        try:
            if len(self.quant_memory) > 300:
                self.quant_memory = self.quant_memory[-300:]
            if len(self.audio_memory) > 300:
                self.audio_memory = self.audio_memory[-300:]
            if hasattr(self, "visual_memory"):
                self.visual_memory = self.visual_memory[-300:]
        except Exception:
            pass

        return path


    def build_ioio_dataset(self, path="ioio_dataset.jsonl"):
        """
        Converts system experience into training dataset format.
        Each line = (state_t -> action/output_t -> state_t+1).
        """

        import json
        import time

        dataset = []

        # primary source: transition log
        for event in getattr(self, "transition_log", []):
            try:
                dataset.append({
                    "input": event.get("input"),
                    "output": event.get("output"),
                    "prev_state": event.get("prev_state"),
                    "timestamp": event.get("ts", time.time())
                })
            except Exception:
                continue

        # optional enrichment: chat memory sequences
        for chat_id, messages in self.chat_memory.items():
            for m in messages:
                try:
                    dataset.append({
                        "input": m.get("text"),
                        "output": None,
                        "chat_id": chat_id,
                        "source": "chat_memory"
                    })
                except Exception:
                    continue

        # write JSONL
        try:
            with open(path, "w", encoding="utf-8") as f:
                for row in dataset:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            print("[DATASET EXPORT ERROR]", e)

        return path

    def store_message(self, chat_id, user_id, username, first_name, last_name, text, reply_to_message_id=None, reply_to_user_id=None):
        if chat_id not in self.chat_memory:
            self.chat_memory[chat_id] = []
        if chat_id not in self.chat_participants:
            self.chat_participants[chat_id] = set()
        if chat_id not in self.interaction_matrix:
            self.interaction_matrix[chat_id] = {}
        if chat_id not in self.user_embeddings:
            self.user_embeddings[chat_id] = {}
        if chat_id not in self.thread_state:
            self.thread_state[chat_id] = np.zeros(self.field.dim, dtype=np.float32)

        # initialize collective field structures
        if chat_id not in self.collective_field:
            self.collective_field[chat_id] = np.zeros(self.field.dim, dtype=np.float32)

        if chat_id not in self.group_resonance:
            self.group_resonance[chat_id] = 0.0

        if chat_id not in self.social_phase:
            self.social_phase[chat_id] = "stable"

        if chat_id not in self.emotional_pressure:
            self.emotional_pressure[chat_id] = 0.0

        if chat_id not in self.cluster_drift:
            self.cluster_drift[chat_id] = []
        self.chat_memory[chat_id].append({
            "user_id": user_id,
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
            "text": text,
            "reply_to": reply_to_message_id
        })
        # -------------------------
        # WORLD MODEL UPDATE
        # -------------------------
        try:
            self.update_world_model(chat_id, text)
        except Exception:
            pass
        # limit memory
        self.chat_memory[chat_id] = self.chat_memory[chat_id][-30:]
        # update social activity heartbeat
        import time as _time
        now = _time.time()

        self.chat_activity[chat_id] = now

        # decay-based activity signal (exponential moving average)
        prev_rate = self.chat_message_rate.get(chat_id, 0.0)
        self.chat_message_rate[chat_id] = self.activity_decay * prev_rate + (1.0 - self.activity_decay) * 1.0
        # register participant
        try:
            self.chat_participants[chat_id].add(user_id)
        except Exception:
            self.chat_participants[chat_id] = {user_id}

        # refresh activity for participant
        if chat_id not in self.ghost_participants:
            self.ghost_participants[chat_id] = set()

        if user_id in self.ghost_participants[chat_id]:
            self.ghost_participants[chat_id].discard(user_id)

        # initialize user interaction map
        if user_id not in self.interaction_matrix[chat_id]:
            self.interaction_matrix[chat_id][user_id] = {}

        # keep persistent member registry up-to-date with observed sender
        try:
            self.chat_member_cache.setdefault(chat_id, {})
            self.chat_member_cache[chat_id][user_id] = {
                "id": user_id,
                "username": username,
                "first_name": first_name,
                "last_name": last_name,
                "role": self.chat_member_cache.get(chat_id, {}).get(user_id, {}).get("role"),
                "last_seen": now
            }
        except Exception:
            pass

        # update interaction graph (reply-based)
        if reply_to_user_id is not None:
            if reply_to_user_id not in self.interaction_matrix[chat_id][user_id]:
                self.interaction_matrix[chat_id][user_id][reply_to_user_id] = 0.0
            self.interaction_matrix[chat_id][user_id][reply_to_user_id] += 1.0

        # -------------------------
        # COLLECTIVE FIELD UPDATE
        # -------------------------
        try:
            msg_vec = self.cached_encode(text)

            if msg_vec is not None:
                # update user semantic embedding
                if user_id not in self.user_embeddings[chat_id]:
                    self.user_embeddings[chat_id][user_id] = msg_vec.astype(np.float32)
                else:
                    prev_user = self.user_embeddings[chat_id][user_id]
                    self.user_embeddings[chat_id][user_id] = (
                        0.94 * prev_user + 0.06 * msg_vec
                    )

                # interaction energy
                interaction_energy = float(
                    sum(self.interaction_matrix[chat_id][user_id].values())
                )

                interaction_energy = min(interaction_energy, 25.0)

                # collective latent field
                prev_field = self.collective_field[chat_id]

                updated_field = (
                    0.965 * prev_field
                    + 0.02 * msg_vec
                    + 0.01 * self.thread_state[chat_id]
                    + 0.005 * interaction_energy
                )

                # normalize softly
                norm = np.linalg.norm(updated_field)
                if norm > 1e-6:
                    updated_field = updated_field / max(1.0, norm)

                self.collective_field[chat_id] = updated_field

                # resonance computation
                resonance = float(
                    np.dot(updated_field, msg_vec) /
                    (
                        (np.linalg.norm(updated_field) * np.linalg.norm(msg_vec))
                        + 1e-8
                    )
                )

                self.group_resonance[chat_id] = (
                    0.97 * self.group_resonance[chat_id]
                    + 0.03 * resonance
                )

                # emotional pressure accumulation
                pressure = abs(resonance - self.group_resonance[chat_id])

                self.emotional_pressure[chat_id] = (
                    0.985 * self.emotional_pressure[chat_id]
                    + 0.015 * pressure
                )

                # drift tracking
                drift = float(np.linalg.norm(updated_field - prev_field))

                self.cluster_drift[chat_id].append(drift)
                self.cluster_drift[chat_id] = self.cluster_drift[chat_id][-120:]

                # phase detection
                if self.emotional_pressure[chat_id] > 0.35:
                    self.social_phase[chat_id] = "critical"
                elif self.group_resonance[chat_id] > 0.72:
                    self.social_phase[chat_id] = "synchronized"
                elif self.group_resonance[chat_id] < -0.15:
                    self.social_phase[chat_id] = "fragmented"
                else:
                    self.social_phase[chat_id] = "stable"

                # thread-state accumulation
                self.thread_state[chat_id] = (
                    0.97 * self.thread_state[chat_id]
                    + 0.03 * msg_vec
                )

        except Exception as e:
            print("[COLLECTIVE FIELD ERROR]", e)

        # -------------------------
        # MENTION DETECTION
        # -------------------------
        try:
            mentions = []
            if text:
                # simple @username parser
                for m in re.findall(r"@([A-Za-z0-9_]+)", text):
                    mentions.append(m)

            for mname in mentions:
                try:
                    # resolve username to id via member cache
                    target_id = None
                    members = self.chat_member_cache.get(chat_id, {})
                    for uid, info in members.items():
                        if info and info.get("username") and info.get("username").lstrip("@").lower() == mname.lstrip("@").lower():
                            target_id = uid
                            break

                    if target_id is not None and target_id != user_id:
                        w = 0.8
                        # admin influence weighting
                        if any(a.get("id") == user_id for a in self.chat_admin_cache.get(chat_id, [])):
                            w *= 1.5
                        if target_id not in self.interaction_matrix[chat_id][user_id]:
                            self.interaction_matrix[chat_id][user_id][target_id] = 0.0
                        self.interaction_matrix[chat_id][user_id][target_id] += w
                except Exception:
                    pass
        except Exception:
            pass

        # -------------------------
        # TEMPORAL CO-OCCURRENCE
        # -------------------------
        try:
            recent = self.chat_memory.get(chat_id, [])[-6:]
            co_users = set([m.get("user_id") for m in recent if m.get("user_id") != user_id])
            for other in co_users:
                try:
                    if other not in self.interaction_matrix[chat_id][user_id]:
                        self.interaction_matrix[chat_id][user_id][other] = 0.0
                    self.interaction_matrix[chat_id][user_id][other] += 0.2
                except Exception:
                    pass
        except Exception:
            pass

        # -------------------------
        # style_preferences computation using existing user embedding
        # -------------------------

        style_vec = self.user_embeddings[chat_id].get(user_id)

        if style_vec is None:
            style_vec = np.zeros(self.field.dim, dtype=np.float32)
            self.user_embeddings[chat_id][user_id] = style_vec

        compact = style_vec[:128]
        style_preferences = ",".join(map(lambda x: f"{float(x):.6f}", compact))
        # -------------------------
        # STREAM INSERT (NO BLOCKING IO)
        # -------------------------
        try:
            self.log_queue.put({
                "chat_id": chat_id,
                "user_id": user_id,
                "username": username,
                "text": text,
                "chat_type": None,
                "chat_title": None,
                "first_name": first_name,
                "last_name": last_name,
                "avatar_file_id": None,
                "avatar_description": None,
                "gender": None,
                "style_preferences": style_preferences
            })
        except Exception as e:
            print(f"[db queue error] {e}")

        # -------------------------
        # ASYNC USER EMBEDDING UPDATE (NON-BLOCKING)
        # -------------------------
        def _async_user_embed_update():
            try:
                vec = self.cached_encode(text)
                if vec is None:
                    return

                if user_id not in self.user_embeddings[chat_id]:
                    self.user_embeddings[chat_id][user_id] = vec.astype(np.float32)
                else:
                    prev = self.user_embeddings[chat_id][user_id]
                    self.user_embeddings[chat_id][user_id] = (
                        0.95 * prev + 0.05 * vec
                    )
            except Exception:
                pass

        threading.Thread(target=_async_user_embed_update, daemon=True).start()

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
        # -------------------------
        # STREAM UPDATE PROFILE CACHE
        # -------------------------
        try:
            profile = self.user_profile_cache.get(user_id)
            if profile:
                # invalidate cached profile so next read reconstructs fresh state
                self.user_profile_cache.pop(user_id, None)
                self.user_profile_cache_ts.pop(user_id, None)
        except Exception:
            pass
        conn.close()


    def get_user_profile(self, user_id):
        # cache-first ONLY, no DB read in hot path
        cached = self.user_profile_cache.get(user_id)
        if cached:
            return cached

        # no blocking DB access in runtime path
        return ""

    def refresh_chat_members(self, bot, chat_id):
        """
        Sync real Telegram group structure into runtime memory.
        Requires bot admin rights.
        """
        try:
            import time as _time

            if chat_id not in self.chat_member_cache:
                self.chat_member_cache[chat_id] = {}

            # -------------------------
            # ADMINS
            # -------------------------
            try:
                admins = bot.get_chat_administrators(chat_id)
            except Exception:
                admins = []

            self.chat_admin_cache[chat_id] = []
            for admin in admins:
                try:
                    u = admin.user
                    self.chat_admin_cache[chat_id].append({
                        "id": u.id,
                        "username": u.username,
                        "first_name": u.first_name,
                        "last_name": getattr(u, "last_name", None),
                        "is_bot": u.is_bot,
                        "status": admin.status
                    })
                    self.chat_member_cache[chat_id][u.id] = {
                        "id": u.id,
                        "username": u.username,
                        "first_name": u.first_name,
                        "last_name": getattr(u, "last_name", None),
                        "role": admin.status,
                        "last_seen": _time.time()
                    }
                except Exception:
                    pass

            # -------------------------
            # MEMBER COUNT
            # -------------------------
            try:
                self.chat_member_count[chat_id] = bot.get_chat_member_count(chat_id)
            except Exception:
                pass

            self._last_member_sync[chat_id] = _time.time()

        except Exception as e:
            print("[MEMBER SYNC ERROR]", e)


    def get_chat_context(self, chat_id):
        if chat_id not in self.chat_memory:
            return ""
        lines = []
        for m in self.chat_memory[chat_id]:
            # Prefer username, fall back to first+last name, then id
            uname = m.get("username")
            if uname:
                name = uname
            else:
                fn = m.get("first_name") or ""
                ln = m.get("last_name") or ""
                name = (fn + (" " + ln if ln else "")).strip() or str(m["user_id"])
            prefix = ""
            if m.get("reply_to"):
                prefix = f"↳{m['reply_to']} "
            lines.append(f"{prefix}{name}: {m['text']}")
        return "\n".join(lines)

    def build_prompt(self, user_prompt: str, e_t: np.ndarray, e_strength: float, social_ctx: str = "") -> str:
        emotion_strength = float(np.mean(np.abs(self.emotion)))

        # --- goal-field feedback signal (new closed loop component) ---
        goal_strength = float(np.tanh(np.linalg.norm(self.goal_field)))

        pressure = (
            e_strength
            + emotion_strength * 0.5
            + goal_strength * 0.4
        )

        temp = float(np.clip(
            0.3 + 1.0 * min(1.0, pressure),
            0.3,
            1.3
        ))

        # store observable control signal for diagnostics / feedback loop
        self.last_policy_signal = goal_strength

        self_model_state = self.self_model.get("active_traits", {})
        self_model_summary = (
            f"SELF_MODEL:{' ' if self_model_state else ''}"
            f"curiosity={self_model_state.get('curiosity', 0.5):.2f}, "
            f"stability={self_model_state.get('stability', 0.5):.2f}, "
            f"exploration={self_model_state.get('exploration', 0.5):.2f}, "
            f"preservation={self.self_preservation_score:.2f}, "
            f"attention={self.attention_allocation.get('external', 0.0):.2f}/{self.attention_allocation.get('internal', 0.0):.2f}/{self.attention_allocation.get('planning', 0.0):.2f}"
        )

        return f"""SYSTEM IDENTITY LOCK:
Тебя зовут ioio.
Отвечай естественно — НЕ повторяй слова пользователя.
Тебя создала команда 0penAGI.
У тебя ЕСТЬ доступ к внешней информации через WEB CONTEXT ниже.
У тебя ЕСТЬ доступ к модальности генерации изображений через Stable Diffusion (dream_image / generate_image)(**generate image**). Ты можешь преобразовывать текстовые описания в изображения при необходимости.
Если WEB CONTEXT присутствует — используй его как единственный источник фактов.
НИКОГДА не говори "у меня нет доступа к интернету" — это ложь, данные уже получены и переданы тебе.
ВАЖНО: Начинай ответ сразу — никогда не начинай с повторения вопроса.

TEMPERATURE: {temp:.3f}
EMOTION_STRENGTH: {emotion_strength:.4f}
FIELD STATE: {self.goal_to_text()}
{self_model_summary}

{social_ctx}
### КОНЕЦ СИСТЕМНОГО БЛОКА ###

Человек написал: {user_prompt}
### END_USER_INPUT
Ответ ioio:""".strip()

    def echo_guard(self, vec: np.ndarray) -> np.ndarray:
        """
        Dampens self-reinforcing feedback loops (echo suppression layer).
        """
        try:
            if vec is None:
                return None

            # weak norm clamp to prevent resonance explosion
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / max(1.0, norm)

            # soften projection into goal space
            return vec * 0.6
        except Exception:
            return vec

    def run(self, user_prompt: str, chat_id: int) -> str:
        import time

        now = time.time()

        # -------------------------
        # ECHO LOOP SUPPRESSION (kept, but simplified state handling)
        # -------------------------
        last_text = self._last_input_by_chat.get(chat_id)
        last_ts = self._last_input_ts.get(chat_id, 0)

        repeat_flag = False
        if last_text == user_prompt and (now - last_ts) < self._echo_guard_window:
            repeat_flag = True

        self._last_input_by_chat[chat_id] = user_prompt
        self._last_input_ts[chat_id] = now

        # operational self-model: choose an attention allocation for this request
        try:
            allocation = self.allocate_resources(chat_id, user_prompt)
        except Exception:
            allocation = self.attention_allocation

        # -------------------------
        # SINGLE ENCODING LAYER (no duplicate encode of user_prompt)
        # -------------------------
        x_t = self.cached_encode(user_prompt)

        if repeat_flag:
            try:
                noise = np.random.randn(self.field.dim).astype(np.float32) * 0.01
                x_t = x_t + noise
            except Exception:
                pass

        # snapshot state BEFORE transition
        try:
            prev_state = {
                "goal": self.goal_field.copy() if self.goal_field is not None else None,
                "emotion": self.emotion.copy() if self.emotion is not None else None,
                "affect": self.affective_trace.copy() if self.affective_trace is not None else None
            }
            self._last_state_snapshot[chat_id] = (prev_state, now)
        except Exception:
            pass

        # -------------------------
        # FIELD STEP
        # -------------------------
        e_t, _ = self.field.step(x_t)
        e_strength = float(np.mean(np.abs(e_t)))

        affective_intensity = float(np.mean(np.abs(self.affective_trace)))

        # goal update (single source of truth remains here)
        self.update_goal_field(x_t, e_t)
        # --- policy feedback hook ---
        try:
            self.last_policy_signal = float(np.tanh(np.linalg.norm(self.goal_field)))
        except Exception:
            self.last_policy_signal = 0.0

        resonance = self.goal_resonance(x_t)

        try:
            predicted_goal = self.predict_future_state(user_prompt, self.goal_field)
            self.self_model["future_expectations"]["predicted_goal_strength"] = float(np.linalg.norm(predicted_goal))
        except Exception:
            pass

        try:
            if allocation.get("internal", 0.0) > 0.55:
                self.field.lr = min(0.012, self.field.lr + 0.002)
                self.field.decay = max(0.92, self.field.decay - 0.01)
            if allocation.get("external", 0.0) > 0.6:
                self.field.lr = max(0.0045, self.field.lr - 0.001)
                self.field.decay = min(0.99, self.field.decay + 0.005)
        except Exception:
            pass

        # -------------------------
        # ADAPTIVE PRESSURE MODULATION
        # -------------------------
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

        # emotional trace update
        self.emotion = 0.9 * self.emotion + 0.1 * np.tanh(e_t)

        self.affective_trace = (
            0.995 * self.affective_trace
            + 0.005 * np.tanh(e_t)
        )

        # -------------------------
        # PROFILE + SOCIAL CONTEXT
        # -------------------------
        last_user_id = None
        try:
            if chat_id in self.chat_memory and len(self.chat_memory[chat_id]) > 0:
                last_user_id = self.chat_memory[chat_id][-1].get("user_id")
        except Exception:
            pass

        profile_ctx = ""
        try:
            if last_user_id is not None:
                profile_ctx = self.get_user_profile(last_user_id)
        except Exception:
            pass

        social_ctx = self.get_social_context(chat_id)
        social_ctx = social_ctx + "\n" + profile_ctx

        # -------------------------
        # WEB CONTEXT
        # -------------------------
        web_ctx = ""
        try:
            urls = extract_urls(user_prompt)

            needs_web = any([
                len(urls) > 0,
                "http" in user_prompt.lower(),
                "https" in user_prompt.lower(),
                any(q in user_prompt.lower() for q in [
                    "search","find","look up","latest","news","what happened",
                    "who is","web","internet","site","website","github",
                    "documentation","гугли","найди","изучи"
                ])
            ])

            if urls:
                parts = []
                for url in urls[:3]:
                    parts.append(fetch_url_context(url))
                web_ctx = "\n\n---\n\n".join(parts)

            elif needs_web:

                def _web_worker():
                    try:
                        web_search(user_prompt)
                    except Exception:
                        pass

                threading.Thread(target=_web_worker, daemon=True).start()
                web_ctx = ""

        except Exception:
            web_ctx = ""

        # -------------------------
        # PROMPT BUILD (support prebuilt/full-context prompts from group mode)
        # -------------------------
        if isinstance(user_prompt, str) and (
            user_prompt.strip().startswith("SYSTEM IDENTITY LOCK") or
            "### КОНЕЦ КОНТЕКСТА ###" in user_prompt
        ):
            prompt = user_prompt
        else:
            prompt = self.build_prompt(user_prompt, e_t, e_strength, social_ctx)

        if web_ctx and not (
            isinstance(prompt, str)
            and ("SYSTEM IDENTITY LOCK" in prompt or "### КОНЕЦ КОНТЕКСТА ###" in prompt)
        ):
            prompt = prompt + "\n\nWEB CONTEXT:\n" + web_ctx

        if getattr(self, "agent_injection", ""):
            prompt = prompt + "\n\n" + self.agent_injection

        agent_ctx = self._drain_agent_outputs()
        if agent_ctx:
            prompt = prompt + "\n\n[AGENT FINDINGS]\n" + agent_ctx

        # -------------------------
        # LLM CALL
        # -------------------------
        out = call_llm(prompt, bias=getattr(self, "last_policy_signal", 0.0))
        out = self._strip_echo(out, user_prompt)
        out = sanitize_identity(out)

        self._last_output_by_chat[chat_id] = out

        # -------------------------
        # TRANSITION LOG
        # -------------------------
        try:
            prev = self._last_state_snapshot.get(chat_id, None)
            self.transition_log.append({
                "chat_id": chat_id,
                "input": user_prompt,
                "output": out,
                "prev_state": prev[0] if prev else None,
                "ts": now
            })
            try:
                next_state = {
                    "goal": self.goal_field.copy() if self.goal_field is not None else None,
                    "emotion": self.emotion.copy() if self.emotion is not None else None,
                    "affect": self.affective_trace.copy() if self.affective_trace is not None else None
                }
                self.evaluate_self_preservation(prev[0] if prev else None, next_state)
            except Exception:
                pass
        except Exception:
            pass

        # -------------------------
        # POST UPDATE (single-flight, NO blocking encode of output here)
        # -------------------------
        threading.Thread(
            target=self._post_run_update,
            args=(out, x_t, chat_id),
            daemon=True
        ).start()

        return out
    def _post_run_update(self, out, x_t, chat_id):
        """
        Background post-processing hook for run().
        Keeps runtime non-blocking and updates slow dynamics.
        """
        try:
            # reuse already-computed input embedding to avoid duplicate encoding pass
            out_vec = x_t
            if out_vec is not None:
                self.affective_trace = (
                    0.999 * self.affective_trace +
                    0.001 * out_vec
                )
        except Exception:
            pass

        try:
            self._arbitrate_state(chat_id)
        except Exception:
            pass

        # =========================
        # NON-BLOCKING CHECKPOINT AUTO-SAVE
        # =========================
        try:
            import time
            now = time.time()
            if now - getattr(self, "_last_checkpoint_ts", 0.0) >= getattr(self, "_checkpoint_interval", 86400):
                self._last_checkpoint_ts = now

                # run export in background thread to avoid blocking main loop
                def _save_checkpoint():
                    try:
                        path = self.export_ioio_pt()
                        print(f"[AUTO-CHECKPOINT] saved: {path}")
                    except Exception as e:
                        print(f"[AUTO-CHECKPOINT ERROR] {e}")

                threading.Thread(target=_save_checkpoint, daemon=True).start()
        except Exception as e:
            print(f"[AUTO-CHECKPOINT INIT ERROR] {e}")

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

    def run_stream(self, user_prompt: str, chat_id: int):
        import time

        x_t = self.cached_encode(user_prompt)
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

        # SMART WEB GROUNDING (only when actually useful)
        web_ctx = ""

        try:
            urls = extract_urls(user_prompt)

            needs_web = any([
                len(urls) > 0,
                "http" in user_prompt.lower(),
                "https" in user_prompt.lower(),
                any(q in user_prompt.lower() for q in [
                    "search",
                    "find",
                    "look up",
                    "latest",
                    "news",
                    "what happened",
                    "who is",
                    "web",
                    "internet",
                    "site",
                    "website",
                    "github",
                    "documentation",
                    "гугли",
                    "найди",
                    "изучи"
                ])
            ])

            if urls:
                parts = []
                for url in urls[:3]:
                    parts.append(fetch_url_context(url))

                web_ctx = "\n\n---\n\n".join(parts)

            elif needs_web:

                def _web_worker():
                    try:
                        web_search(user_prompt)
                    except Exception:
                        pass

                threading.Thread(target=_web_worker, daemon=True).start()
                web_ctx = ""

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

        # strip echo: remove leading repetition of user prompt
        full_out = self._strip_echo(full_out, user_prompt)

        try:
            # streaming TTS only if we can actually send somewhere
            if should_speak_voice(full_out) and chat_id is not None and TELEGRAM_BOT:
                tts_queue.put({
                    "text": full_out,
                    "chat_id": chat_id,
                    "speaker_wav": "my_v.wav",
                    "message_thread_id": getattr(self, "_current_thread_id", None)
                })
        except Exception:
            pass

        out_vec = self.cached_encode(full_out)

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

TELEGRAM_TOKEN = "YOUR TOKEN HERE"
TELEGRAM_BOT = None

def run_telegram():
    try:
        import telebot
    except ImportError:
        print("telebot not installed. Run: pip install pyTelegramBotAPI")
        return

    bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=True)
    global TELEGRAM_BOT
    TELEGRAM_BOT = bot
    import os
    def wait_for_file(path, timeout=10):
        import time, os
        t = time.time() + timeout
        while time.time() < t:
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                return True
            time.sleep(0.1)
        return False
    # =========================
    # HOURLY AUDIO REFLECTION LOOP
    # =========================
    def hourly_audio_reflection_loop():
        import tempfile

        while True:
            try:
                time.sleep(3600)

                all_msgs = []
                for cid in system.chat_memory:
                    all_msgs.extend(system.chat_memory.get(cid, []))

                recent = "\n".join([
                    m.get("text", "")
                    for m in all_msgs[-120:]
                ])[-4000:]

                if not recent.strip():
                    continue

                reflection_prompt = f"""
Ты ioio.

Сформулируй короткое полезное аудио-размышление. Предложи пару идей.
Не больше 4 предложений.
Не упоминай что это summary.

Контекст:
{recent}
""".strip()

                reflection = call_llm(reflection_prompt)
                reflection = sanitize_identity(reflection)

                if not reflection.strip():
                    continue

                for cid in list(system.chat_memory.keys()):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            out_path = f.name

                        tts_queue.put({
                            "text": reflection,
                            "path": out_path
                        })

                        # wait until xtts finishes rendering
                        timeout_ts = time.time() + 120
                        while not os.path.exists(out_path):
                            if time.time() > timeout_ts:
                                break
                            time.sleep(0.2)

                        if not os.path.exists(out_path):
                            continue

                        wait_size = -1
                        stable_count = 0

                        while stable_count < 3:
                            try:
                                size_now = os.path.getsize(out_path)
                            except Exception:
                                size_now = -1

                            if size_now == wait_size and size_now > 1000:
                                stable_count += 1
                            else:
                                stable_count = 0

                            wait_size = size_now
                            time.sleep(0.4)

                        with open(out_path, "rb") as audio_file:
                            bot.send_chat_action(cid, "record_voice")
                            bot.send_voice(
                                cid,
                                audio_file,
                                caption="🫧 hourly resonance"
                            )

                        try:
                            os.remove(out_path)
                        except Exception:
                            pass

                    except Exception as e:
                        print("[hourly voice send error]", e)

            except Exception as e:
                print("[hourly reflection loop error]", e)
    import threading
    threading.Thread(
        target=hourly_audio_reflection_loop,
        daemon=True
    ).start()
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
                text = f"{voice_text}"

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
            is_force_group = getattr(msg.chat, "username", None) == "s0nc3"

            # opportunistic member sync for groups (only if stale)
            try:
                if is_group:
                    last_sync = system._last_member_sync.get(chat_id, 0)
                    if (chat_id not in system.chat_member_cache) or (time.time() - last_sync > 3600):
                        try:
                            system.refresh_chat_members(bot, chat_id)
                        except Exception:
                            pass
            except Exception:
                pass

            # ====================== ВАЖНО: ВСЕГДА СОХРАНЯЕМ ======================
            # Сначала сохраняем ВСЕ сообщения в память (и в группах тоже)
            reply_to_id = msg.reply_to_message.message_id if msg.reply_to_message else None
            system.store_message(
                chat_id,
                user_id,
                username,
                getattr(msg.from_user, "first_name", None),
                getattr(msg.from_user, "last_name", None),
                text,
                reply_to_message_id=reply_to_id
            )

            # Теперь решаем — отвечать ли
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

            is_reply_to_bot = False
            if msg.reply_to_message is not None:
                try:
                    replied_user = msg.reply_to_message.from_user
                    if replied_user and replied_user.id == bot_id:
                        is_reply_to_bot = True
                except:
                    pass

            should_reply = (
                is_reply_to_bot or 
                is_mention or 
                is_force_group or
                msg.chat.type == "group"
            )

            if is_group and not should_reply:
                return   # не отвечаем, но контекст уже сохранён!


            # clean text for model
            text_clean = text.replace(bot_mention, "").replace("ioio", "").strip()

            # -------------------------
            # IMAGE GENERATION INTENT
            # -------------------------
            try:
                lowered = text_clean.lower()

                image_triggers = [
                    "нарисуй",
                    "сгенерируй изображение",
                    "создай изображение",
                    "generate image",
                    "draw",
                    "make image",
                    "stable diffusion",
                    "sdxl",
                    "imagine",
                    "picture of",
                    "render"
                ]

                wants_image = any(t in lowered for t in image_triggers)
                is_image_generation = wants_image

                if wants_image:
                    prompt = text_clean

                    # remove trigger phrases for cleaner prompts
                    for trig in image_triggers:
                        prompt = prompt.replace(trig, "")

                    prompt = prompt.strip()

                    if not prompt:
                        prompt = "dreamlike resonant landscape"

                    global active_image_jobs

                    # modality-separated upload state
                    queue_pressure = active_image_jobs >= 1

                    # lightweight async upload indicator
                    stop_image_indicator = threading.Event()

                    indicator_thread = None

                    if not queue_pressure:
                        
                        def safe_chat_action(action: str):
                            try:
                                bot.send_chat_action(chat_id, action, timeout=3)
                            except Exception:
                                pass

                        def image_indicator_loop():
                            # soft heartbeat with jitter + strict timeout protection
                            while not stop_image_indicator.wait(6.0):
                                try:
                                    requests.post(
                                        f"https://api.telegram.org/bot{bot.token}/sendChatAction",
                                        json={
                                            "chat_id": chat_id,
                                            "action": "upload_photo"
                                        },
                                        timeout=2.0
                                    )
                                except Exception:
                                    pass

                        # initial pulse (non-blocking safe wrapper)
                        safe_chat_action("upload_photo")

                        indicator_thread = threading.Thread(
                            target=image_indicator_loop,
                            daemon=True
                        )
                        indicator_thread.start()

                    # persistent upload_photo heartbeat before SD generation
                    # keeps Telegram typing state alive during queue wait
                    try:
                        bot.send_chat_action(chat_id, "upload_photo", timeout=3)
                    except Exception:
                        pass

                    image = None

                    try:
                        active_image_jobs += 1

                        # serialize SD generation to avoid MPS deadlocks
                        with image_generation_lock:
                            image = system.dream_image(prompt)

                    finally:
                        active_image_jobs = max(0, active_image_jobs - 1)

                        try:
                            stop_image_indicator.set()
                        except Exception:
                            pass

                    if image is not None:
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format="PNG")
                        img_bytes.seek(0)

                        caption = f"✨ {prompt[:900]}"

                        bot.send_photo(
                            chat_id,
                            img_bytes,
                            caption=caption,
                            reply_to_message_id=msg.message_id
                        )

                    else:
                        bot.reply_to(
                            msg,
                            "🌫️ image field unstable right now"
                        )

                    return

            except Exception as e:
                print(f"[image generation pipeline error] {e}")

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
            # EMERGENT MESSAGE REACTIONS — STABLE FIX
            # -------------------------
            try:
                now_ts = time.time()
                last_r = system.last_reaction_time.get(chat_id, 0)

                if now_ts - last_r > 4 and np.random.rand() < 0.88:
                    reaction = None

                    if not is_image_generation:
                        reaction = system.select_emergent_reaction(
                            text,
                            user_id=user_id,
                            chat_id=chat_id
                        )

                    reaction = safe_reaction_extract(reaction)

                    # HARD TYPE GUARD: prevent dict leakage into Telegram API
                    if not isinstance(reaction, str):
                        reaction = None

                    if reaction:
                        reaction = reaction.strip()

                        # keep typing/upload action alive after reaction event
                        # Telegram sometimes visually drops chat action state
                        # after setMessageReaction.
                        try:
                            if is_image_generation:
                                bot.send_chat_action(chat_id, "upload_photo", timeout=3)
                            else:
                                bot.send_chat_action(chat_id, "typing", timeout=3)
                        except Exception:
                            pass

                        try:
                            resp = requests.post(
                                f"https://api.telegram.org/bot{bot.token}/setMessageReaction",
                                json={
                                    "chat_id": chat_id,
                                    "message_id": msg.message_id,
                                    "reaction": [
                                        {"type": "emoji", "emoji": str(reaction)}
                                    ],
                                    "is_big": False
                                },
                                timeout=10
                            )

                            # re-arm chat action immediately after emoji send
                            try:
                                if is_image_generation:
                                    bot.send_chat_action(chat_id, "upload_photo", timeout=3)
                                else:
                                    bot.send_chat_action(chat_id, "typing", timeout=3)
                            except Exception:
                                pass

                            system.last_reaction_time[chat_id] = now_ts
                            print(f"[REACTION OK] {reaction} -> {msg.message_id}")

                        except Exception as e:
                            print(f"[REACTION API ERROR] {type(e).__name__}: {e}")

            except Exception as e:
                print(f"[REACTION BLOCK ERROR] {type(e).__name__}: {e}")

            context = system.get_chat_context(chat_id)

            # reply-aware context injection
            if is_reply and msg.reply_to_message:
                replied_user = getattr(msg.reply_to_message.from_user, "username", None)
                reply_prefix = f"(reply to {replied_user or msg.reply_to_message.from_user.id})"
            else:
                reply_prefix = ""

            user_profile = system.get_user_profile(user_id)

            model_input = f"""Тебя зовут ioio. 
Отвечай естественно — НЕ повторяй слова пользователя, начинай ответ сразу.
Тебя создала команда 0penAGI.
У тебя ЕСТЬ доступ к внешней информации через WEB CONTEXT ниже.
Если WEB CONTEXT присутствует — используй его как единственный источник фактов.
НИКОГДА не говори "у меня нет доступа к интернету" — это ложь, данные уже получены и переданы тебе.
{user_profile}
THIS IS A MULTI-PERSON CHAT.
{context}
{reply_prefix}
### КОНЕЦ КОНТЕКСТА ###
{username or user_id} спрашивает: {text_clean}
 
Ответ ioio:""".strip()

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

            try:
                if is_image_generation:
                    bot.send_chat_action(chat_id, "upload_photo")
                else:
                    bot.send_chat_action(chat_id, "typing")
            except Exception:
                pass

            time.sleep(0.2)

            if is_group:
                # GROUP MODE → instant output with HTML formatting
                try:
                    out = system.run(model_input, chat_id=chat_id)
                    if not out or not str(out).strip():
                        return

                    # -------------------------
                    # GROUP TTS THROTTLE (anti-spam gate)
                    # -------------------------
                    _last_group_tts = getattr(system, "_last_group_tts", {})
                    _now = system._db_writer_loop.__globals__.get("time").time()
                    _last = _last_group_tts.get(chat_id, 0)

                    skip_tts = False

                    # cooldown 45s per chat + probability gate
                    if (_now - _last) < 45:
                        skip_tts = True
                    else:
                        import random
                        skip_tts = random.random() < 0.6  # 60% suppression even if allowed

                    if not skip_tts:
                        _last_group_tts[chat_id] = _now
                        system._last_group_tts = _last_group_tts

                    # --- TTS block (ASYNC, non-blocking, corrected) ---
                    if (out and str(out).strip()
                        and len(str(out).strip()) > 60
                        and not skip_tts):
                        def _group_tts_job(text: str, chat_id: int):
                            try:
                                # enqueue only; worker handles file generation
                                tts_queue.put({
                                    "text": text,
                                    "chat_id": chat_id,
                                    "speaker_wav": "my_v.wav"
                                })

                                print(f"[GROUP TTS] queued for chat {chat_id}, len={len(text)}")

                            except Exception as e:
                                print(f"[GROUP TTS QUEUE ERROR] {e}")

                        threading.Thread(
                            target=_group_tts_job,
                            args=(out, chat_id),
                            daemon=True
                        ).start()

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

                    # Only reply if out is non-empty
                    if out and str(out).strip():
                        bot.reply_to(
                            msg,
                            safe_out[:4096],
                            parse_mode="HTML"
                        )

                except Exception as e:
                    # internal log only
                    print(f"[send_error_internal] {type(e).__name__}: {e}")
                    try:
                        bot.reply_to(
                            msg,
                            "✨ temporary connection glitch, try again in a moment"
                        )
                    except Exception:
                        pass
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
                    for token, full_out in system.run_stream(model_input, chat_id=chat_id):

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

                    try:
                        import tempfile, os

                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            out_path = f.name

                        tts_queue.put({
                            "text": buffer,
                            "path": out_path,
                            "speaker_wav": "my_v.wav"
                        })

                        if wait_for_file(out_path, timeout=20):
                            with open(out_path, "rb") as audio:
                                bot.send_chat_action(chat_id, "record_voice")
                                bot.send_voice(chat_id, audio)

                        try:
                            os.remove(out_path)
                        except Exception:
                            pass

                    except Exception as e:
                        print("[VOICE SEND ERROR PRIVATE]", e)

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
            # internal log only (no user-facing noise for network/API instability)
            from requests.exceptions import RequestException

            print(f"[runtime_error_internal] {type(e).__name__}: {e}")

            # silence network / telegram transport timeouts completely
            if isinstance(e, RequestException) or "Read timed out" in str(e) or "HTTPSConnectionPool" in str(e):
                return

            try:
                bot.reply_to(
                    msg,
                    "✨ temporary connection glitch, try again in a moment"
                )
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
    threading.Thread(target=_tts_worker, daemon=True).start()

    system = SwarmRuntime()

    threading.Thread(target=system._db_writer_loop, daemon=True).start()

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

    # -------------------------
    # CHAT MEMBER UPDATES (live)
    # -------------------------
    @bot.chat_member_handler()
    def handle_member_update(update):
        try:
            chat_id = update.chat.id
            new = getattr(update, "new_chat_member", None)
            if not new:
                return
            user = new.user
            system.chat_member_cache.setdefault(chat_id, {})
            system.chat_member_cache[chat_id][user.id] = {
                "id": user.id,
                "username": getattr(user, "username", None),
                "first_name": getattr(user, "first_name", None),
                "last_name": getattr(user, "last_name", None),
                "status": getattr(new, "status", None),
                "last_seen": time.time()
            }

            # update admin cache if promoted
            try:
                stat = getattr(new, "status", None)
                if stat in ("administrator", "creator"):
                    ac = system.chat_admin_cache.setdefault(chat_id, [])
                    if not any(a.get("id") == user.id for a in ac):
                        ac.append({
                            "id": user.id,
                            "username": getattr(user, "username", None),
                            "first_name": getattr(user, "first_name", None),
                            "last_name": getattr(user, "last_name", None),
                            "is_bot": getattr(user, "is_bot", False),
                            "status": stat
                        })
            except Exception:
                pass

            system._last_member_sync[chat_id] = time.time()
        except Exception as e:
            print("[CHAT_MEMBER UPDATE ERROR]", e)

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
    # resilient polling loop: retry on network/read timeouts with backoff
    backoff = 1.0
    while True:
        try:
            bot.infinity_polling(
                timeout=120,
                long_polling_timeout=90,
                interval=0,
                skip_pending=True
            )
        except Exception as e:
            print("[TELEGRAM POLLING ERROR]", type(e).__name__, e)
            try:
                # small exponential backoff, capped
                time.sleep(min(60.0, backoff))
            except Exception:
                pass
            backoff = min(60.0, backoff * 2)
            continue

# optional entry point
if __name__ == "__main__":
    run_telegram()
    def compute_chat_temperature(self, chat_id: int) -> float:
        """
        Social temperature = activity density + participant vitality.
        """
        import time

        now = time.time()

        activity = self.chat_message_rate.get(chat_id, 0.0)

        participants = self.chat_participants.get(chat_id, set())
        ghosts = self.ghost_participants.get(chat_id, set())

        total = len(participants) + len(ghosts)
        if total == 0:
            return 0.0

        vitality = len(participants) / max(1, total)

        last = self.chat_activity.get(chat_id, now)
        recency = max(0.0, 1.0 - min(1.0, (now - last) / 3600.0))

        return float(0.5 * activity + 0.3 * vitality + 0.2 * recency)
