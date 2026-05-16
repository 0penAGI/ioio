import numpy as np
import requests
from requests.exceptions import RequestException
import time
import base64
import html
import sqlite3
import threading
import queue

from typing import Optional

# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
EMBED_URL = "http://localhost:11434/api/embeddings"


MODEL = "gemma4:e2b"
EMBED_MODEL = "nomic-embed-text"

# =========================
# EMBEDDING QUEUE SYSTEM
# =========================

embed_cache = {}
embed_queue = queue.Queue()

# =========================
# DATABASE CONFIG
# =========================
DB_PATH = "ioio_memory.db"


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
        avatar_updated TIMESTAMP
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

def call_llm(prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "num_predict": 2048
            }
        },
        timeout=120
    )
    r.raise_for_status()
    return r.json()["response"]


def call_llm_stream(prompt: str):
    import json

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.8,
                "num_predict": 2048
            }
        },
        stream=True,
        timeout=120
    )

    r.raise_for_status()

    for line in r.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                yield data["response"]
        except Exception:
            continue



def _embed_worker():
    import time

    while True:
        text, event = embed_queue.get()

        try:
            # cache check
            if text in embed_cache:
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
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
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
        timeout=120
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


# =========================
# WEB SEARCH HELPER
# =========================
def web_search(query: str) -> str:
    """
    Multi-source web search + lightweight scraping.
    Uses DuckDuckGo results + fetches page content for grounding.
    """

    try:
        from ddgs import DDGS
        import requests

        try:
            from bs4 import BeautifulSoup
        except Exception:
            BeautifulSoup = None

        results = []

        # -------------------------
        # STEP 1: SEARCH RESULTS
        # -------------------------
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=5))

        if not search_results:
            return "[no web result]"

        # -------------------------
        # STEP 2: ENRICH WITH PAGE CONTENT
        # -------------------------
        for r in search_results:
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")

            page_text = ""

            # try to scrape full page (lightweight)
            if href:
                try:
                    page = requests.get(
                        href,
                        timeout=6,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                    if page.status_code == 200:
                        html = page.text

                        if BeautifulSoup:
                            soup = BeautifulSoup(html, "html.parser")

                            # remove junk
                            for tag in soup(["script", "style", "noscript"]):
                                tag.decompose()

                            text = soup.get_text(separator=" ", strip=True)
                            page_text = text[:1200]  # limit per page
                        else:
                            # fallback: crude strip
                            page_text = html[:800]

                except Exception:
                    page_text = ""

            # -------------------------
            # STEP 3: COMPOSE BLOCK
            # -------------------------
            block = f"""
TITLE: {title}
SNIPPET: {body}
URL: {href}
PAGE_EXTRACT: {page_text}
""".strip()

            results.append(block)

        # -------------------------
        # STEP 4: FINAL COMPRESSION
        # -------------------------
        return "\n\n---\n\n".join(results)[:8000]

    except Exception as e:
        return f"[web_error] {e}"


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
        self.chat_memory = {}  # chat_id -> list of messages

    def store_message(self, chat_id, user_id, username, first_name, last_name, text, reply_to_message_id=None):
        if chat_id not in self.chat_memory:
            self.chat_memory[chat_id] = []
        self.chat_memory[chat_id].append({
            "user_id": user_id,
            "username": username,
            "text": text,
            "reply_to": reply_to_message_id
        })
        # limit memory
        self.chat_memory[chat_id] = self.chat_memory[chat_id][-30:]

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
            SELECT username, first_name, last_name, gender
            FROM users
            WHERE id = ?
        """, (user_id,))

        row = cur.fetchone()
        conn.close()

        if not row:
            return ""

        username, first_name, last_name, gender = row

        return f"""
USER PROFILE:
username: {username}
name: {first_name} {last_name}
gender: {gender}
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

    def build_prompt(self, user_prompt: str, e_t: np.ndarray, e_strength: float) -> str:
        emotion_strength = float(np.mean(np.abs(self.emotion)))

        if e_strength > 0.6:
            tone = "говори свободно и ассоциативно"
        elif e_strength > 0.3:
            tone = "говори живо и естественно"
        else:
            tone = "говори спокойно и ясно"

        return f"""SYSTEM IDENTITY LOCK:
Твоё имя НЕ Gemma.
Имя "ioio" используется ТОЛЬКО если тебя прямо спрашивают об имени.
Тебя создала команда 0penAGI.
У тебя ЕСТЬ доступ к внешней информации через WEB CONTEXT ниже.
Если WEB CONTEXT присутствует — используй его как единственный источник фактов.
НИКОГДА не говори "у меня нет доступа к интернету" — это ложь, данные уже получены и переданы тебе.

Запрещено использовать "ioio" в начале ответа или как подпись.
Запрещено вставлять его в текст, поэзию или рассуждения.

Запрещено использовать любые другие имена.

Режим: {tone}
Интенсивность состояния: {emotion_strength:.4f}

Человек: {user_prompt}
Ответ:""".strip()

    def run(self, user_prompt: str) -> str:
        # encode input
        x_t = self.field.encode(user_prompt)

        # field step FIRST (so e_t exists)
        e_t, _ = self.field.step(x_t)
        e_strength = float(np.mean(np.abs(e_t)))

        # ALWAYS grounded web context (prevents "I can't search" hallucination)
        web_ctx = ""
        try:
            web_ctx = web_search(user_prompt)
        except Exception:
            web_ctx = ""

        # affective + subjectivity dynamics (NOW valid)
        self.emotion = 0.9 * self.emotion + 0.1 * np.tanh(e_t)

        # build prompt
        prompt = self.build_prompt(
            user_prompt,
            e_t,
            e_strength
        )

        # inject grounding context into prompt
        if web_ctx:
            prompt = prompt + "\n\nWEB CONTEXT:\n" + web_ctx

        # LLM call
        out = call_llm(prompt)

        # hard identity enforcement post-filter
        if "Gemma" in out or "gemma" in out:
            out = "Айоайо"

        # encode output
        out_vec = self.field.encode(out)

        # subjectivity drift (self-consistency bias)
        self.subjectivity = 0.995 * self.subjectivity + 0.005 * out_vec

        # feedback loop
        self.field.m = 0.9 * self.field.m + 0.1 * out_vec
        self.field.x_prev = 0.5 * self.field.x_prev + 0.5 * x_t

        return out

    def run_stream(self, user_prompt: str):
        import time

        x_t = self.field.encode(user_prompt)
        e_t, _ = self.field.step(x_t)
        e_strength = float(np.mean(np.abs(e_t)))

        self.emotion = 0.9 * self.emotion + 0.1 * np.tanh(e_t)

        prompt = self.build_prompt(user_prompt, e_t, e_strength)

        full_out = ""

        for token in call_llm_stream(prompt):
            full_out += token
            yield token, full_out

        # post-process after stream ends
        if "Gemma" in full_out or "gemma" in full_out:
            full_out = "Айоайо"

        out_vec = self.field.encode(full_out)

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

            prompt = f"""
Ты — фоновый исследователь.

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

                system.memory_text += f"\n[WEB:{query}] {result}\n"
                system.memory_text = system.memory_text[-4000:]

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

    bot = telebot.TeleBot(TELEGRAM_TOKEN)

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

    @bot.message_handler(content_types=["text", "photo"])
    def handle(msg):
        try:
            chat_id = msg.chat.id
            user_id = msg.from_user.id
            username = getattr(msg.from_user, "username", None)
            text = msg.text or msg.caption or ""
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

            is_non_text = not text.strip()

            # group safety filter (only reply/mention OR non-group)
            if is_non_text and not (is_reply or is_mention):
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
            if is_group and not (is_reply_to_bot or is_mention):
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

                        t = re.sub(r"```(.*?)```", r"<pre>\1</pre>", t, flags=re.DOTALL)
                        t = re.sub(r"`(.*?)`", r"<code>\1</code>", t)
                        t = re.sub(r"\*(.*?)\*", r"<b>\1</b>", t)
                        t = re.sub(r"\_(.*?)\_", r"<i>\1</i>", t)
                        t = re.sub(r"^>\s?(.*)$", r"<blockquote>\1</blockquote>", t, flags=re.MULTILINE)
                        t = re.sub(r"\[(.*?)\]\((https?://.*?)\)", r'<a href="\2">\1</a>', t)
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
                sent = bot.reply_to(msg, "⚡ ioio is thinking…")

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

                        # IMPORTANT: code blocks first (to avoid partial formatting inside them)
                        t = re.sub(r"```(.*?)```", r"<pre>\1</pre>", t, flags=re.DOTALL)

                        # inline code
                        t = re.sub(r"`(.*?)`", r"<code>\1</code>", t)

                        # bold: *text*
                        t = re.sub(r"\*(.*?)\*", r"<b>\1</b>", t)

                        # italic: _text_
                        t = re.sub(r"\_(.*?)\_", r"<i>\1</i>", t)

                        # blockquotes: > text (line-based)
                        t = re.sub(r"^>\s?(.*)$", r"<blockquote>\1</blockquote>", t, flags=re.MULTILINE)

                        # links: [text](url)
                        t = re.sub(
                            r"\[(.*?)\]\((https?://.*?)\)",
                            r'<a href="\2">\1</a>',
                            t
                        )

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
                        if "429" in str(e):
                            import re, time
                            m = re.search(r"retry after (\d+)", str(e))
                            if m:
                                time.sleep(int(m.group(1)))
                        print(f"[stream edit error] {e}")

                try:
                    import re
                    for token, full_out in system.run_stream(model_input):

                        buffer = full_out
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
                        time_trigger = (now - last_update > 1.25)
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
            bot.reply_to(msg, f"[error] {e}")

    print("🧠 Telegram IOIO running...")
    bot.polling()

# optional entry point
if __name__ == "__main__":
    run_telegram()
