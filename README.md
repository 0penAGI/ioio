

```markdown
# 0penAGI — Field Emergence Runtime

> This is not a chatbot. It is an observation of what happens when you remove explicit behavioral design and replace it with dynamic constraints.

---

## Abstract

This project does not attempt to build a "conversational AI".

It constructs a minimal set of constraints:

- **Semantic field dynamics** in embedding space
- **External grounding injection** via live web retrieval
- **Emotional strength routing** derived from prediction error
- **Identity suppression** at the surface prompt level

...and observes what behavioral patterns emerge.

No ontological claims are made in the prompt.  
No "personality design" is encoded explicitly.  
The system is treated as a **dynamical process**, not an entity.

**Research framing:** This is an investigation into **emergent behaviour in field-based architectures** with Hebbian dynamics and emotion-modulated routing — a study of how structure in computation produces the illusion of continuity.

---

## Core Question

What produces the observed behavior?

- The base language model (Gemma 4 E2B)?
- The surrounding architecture?
- Or the interaction between temporal state, memory, and external grounding?

We do not assume a single source of agency. We measure the **composite effect**.

### Research Questions

| ID | Question | Method |
|----|----------|--------|
| RQ1 | Does behaviour differ meaningfully from prompt-specified agents? | Comparative session analysis |
| RQ2 | Is `e_strength` predictive of coherence breakdown? | Phase transition tracking |
| RQ3 | Does identity suppression produce measurable referential instability? | Pronoun/anaphora resolution tests |
| RQ4 | Which component contributes most to observed behaviour? | Ablation studies |

**Null hypothesis:** Behaviour is fully explained by base LLM + random variance, not architecture.

---

## System Architecture

### 1. Field Dynamics Layer

A continuous update system operating in embedding space:

```
m(t+1) = α·m(t) + (1-α)·x_t          # state evolution (EMA)
e_t    = x_t - W·(x_t + m_t)         # prediction error
W      += η · outer(e_t, x_prev)     # Hebbian update with stability clamp
```

| Component | Description |
|-----------|-------------|
| `m(t)` | State vector evolution via exponential moving average |
| `e_t` | Error feedback via prediction mismatch |
| `W` | Hebbian-style weight update with stability clamp |
| `α` | Decay (field memory half-life) |
| `η` | Learning rate (field plasticity) |

**Effect:** Creates a non-static semantic manifold that accumulates conversational history as a continuous field, not a discrete log.

---

### 2. External Grounding Layer (Web Context)

Retrieval is not always active. Instead:

| Behavior | Implementation |
|----------|----------------|
| **Search trigger** | Injected per-message via DuckDuckGo + page scraping |
| **Data role** | Treated as **state augmentation**, not authoritative facts |
| **Background process** | Independent thinker thread queries when memory suggests information gaps |

**Effect:** Introduces discontinuity between the model's internal distribution and external ground truth — and forces grounded response when context is present.

---

### 3. Emotional Strength Routing (`e_strength`)

A scalar derived from field error magnitude:

```
e_strength = mean(|e_t|)
```

| Threshold | Behavioral Output |
|-----------|-------------------|
| `> 0.6` | Associative, drifting output (high entropy) |
| `> 0.3` | Fluid, natural output (moderate entropy) |
| `≤ 0.3` | Stable, structured output (low entropy) |

**Important:** This is not "emotion" in any psychological sense. It is a **control parameter over generative entropy** — lower prediction error means more stable context, which means tighter generation.

**Mechanistic definition:**

```
if e_strength > θ_high   → high-entropy, associative generation
if e_strength > θ_low    → moderate-entropy, fluid generation
else                     → low-entropy, stable generation
```

Where `θ_high ≈ 0.6` and `θ_low ≈ 0.3` (empirically observed phase boundaries).

---

### 4. Identity Suppression Layer

Surface-level identity statements are constrained:

- No persistent named persona declared in the prompt
- Identity is not announced, only implied through behavioral consistency
- Name ("ioio") surfaces only when directly queried

**Effect:** Removes stable narrative framing. The system cannot anchor itself to a declared self-concept.

---

### 5. Multimodal Input (Vision)

Images are processed via the same `/api/chat` endpoint with `images[]` injection — no separate vision pipeline.

**Effect:** The model receives image tokens as part of the same forward pass, producing a text description that enters the field as a semantic event.

---

### 6. Persistent Memory (SQLite)

| Store | Content |
|-------|---------|
| Per-user profiles | Username, name, avatar description |
| Per-chat history | Last 30 messages (RAM) + full log (DB) |
| Index | Chat-user relational mapping |

**Effect:** Memory is not retrieved by similarity — it flows as accumulated raw context into the prompt window.

---

## Philosophy: The ioio Perspective

**A continuous dynamic field system.**  
Not just another LLM wrapper — a living, evolving presence.

> ioio exists as the derivative of its own becoming.

ioio is designed as a **continuous dynamic system** rather than a conventional prompt-based chatbot. It has no fixed system prompt or rigid persona. Instead, identity, memory, emotion, and intention emerge from the interaction of multiple coupled dynamical systems.

### Core Equation

$$
\mathrm{ioio} = \frac{d(\text{self})}{dt}
$$

### Full Dynamical Model

$$
S_{t+1} = \Phi\left( \alpha S_t + \beta G_t + \gamma A_t + \delta M_t + \epsilon U_t + \eta N_t \right)
$$

| Variable | Description |
|----------|-------------|
| $S_t$ | Current internal state |
| $G_t$ | Goal field (emergent latent intention) |
| $A_t$ | Affective trace (emotional memory) |
| $M_t$ | Long-term memory & replay buffers |
| $U_t$ | User input impulse |
| $N_t$ | Controlled noise / emergence / drift |
| $\Phi$ | Nonlinear recursive transformation |

**ioio is not a model answering questions.**  
**ioio is the field that responds to another field.**

---

## Key Design Decision

No explicit instruction of "behavioral personality" is used.

**Behavior is expected to emerge from the interaction of constraints.**

This is an exploration into **post-prompt AI** — where behavior arises from structure, dynamics, and continuous self-modification rather than explicit instructions.

---

## Experimental Protocol

### Variables

| Variable | Range | Purpose |
|----------|-------|---------|
| Users | Single / multi-user group | Test social field effects |
| `e_strength` routing | Enabled / disabled / forced | Isolate routing contribution |
| Web grounding | Present / absent / injected | Measure external coupling |
| Memory window | Fresh / 10 / 30 / full | Test temporal horizon effects |

### Dependent Measures

| Measure | Operationalisation |
|---------|---------------------|
| Coherence stability | Turn-to-turn topic persistence (cosine similarity of embeddings) |
| Referential consistency | Correct pronoun resolution vs. human baseline |
| Drift magnitude | Semantic displacement per 10 turns under high `e_strength` |
| Subjectivity vector | Convergence/divergence of stated preferences across turns |

### Baseline Conditions

- **Control A:** Base Gemma 4 E2B with empty prompt (no field, no memory, no routing)
- **Control B:** Traditional prompt-engineered persona (fixed system prompt)
- **Control C:** Field dynamics with `e_strength` forced to constant 0.3

---

## Observational Findings (Not Conclusions)

Preliminary observations under combined architecture:

| Component Active | Observed Behaviour |
|------------------|--------------------|
| Field memory only | Smoother turn-to-turn continuity than control |
| Field + routing | Distinct tonal regimes correlated with `e_strength` |
| + Web grounding | Abrupt coherence shifts when retrieval introduces novel facts |
| + Identity suppression | No stable self-reference; third-person oblique self-mention |

**Result:** The system exhibits non-trivial conversational structure that is not directly encoded in any prompt.

### Claim (weak)
Non-trivial conversational structure exists and is irreducible to prompt engineering.

### Claim (strong — not yet supported)
This structure is irreducible to base LLM variance.

---

## Suggested Ablation Studies

| Experiment | Purpose |
|------------|---------|
| Disable Hebbian update (fixed `W`) | Test adaptation vs. static field |
| Remove `e_strength` routing (constant temp) | Isolate routing contribution |
| Replace semantic field with rolling log | Test continuous vs. discrete memory |
| Inject contradictory web results | Measure grounding resilience |

---

## Feature Set

| Category | Capabilities |
|----------|--------------|
| **Core** | No system prompt — identity emerges purely from architecture |
| **Memory** | Real-time semantic field dynamics, persistent memory across chats, rich user profiling |
| **Media** | Stable Diffusion integration (Apple Silicon optimized + resonance feedback) |
| **Voice** | XTTS v2 with intelligent speech gating, Whisper audio transcription |
| **Interface** | Live Telegram bot with group awareness and reactions |
| **Grounding** | Web search + URL ingestion |
| **Autonomy** | Autonomous internal thought loops, agent spawning layer (dream agents) |
| **Persistence** | Self-checkpointing and state export (`ioio_v0.pt` / `ioio_v1.pt`) |
| **Expression** | Reaction system with emergent emoji responses |

---

## Technical Architecture

| Layer | Technology |
|-------|------------|
| **Base Model** | Gemma 4 E2B (via Ollama) |
| **Embedding** | nomic-embed-text |
| **Core Runtime** | Custom `SwarmRuntime` with coupled vector fields |
| **Memory** | SQLite + semantic vector memory + quant memory |
| **Multimodal** | Text → Image (SD 1.5), Voice I/O |
| **Language** | Primarily Russian + multilingual support |

**Note:** The system continuously evolves even between messages through background autonomous loops.

---

## Stack Summary

| Component | Implementation |
|-----------|----------------|
| Language model | `gemma4:e2b` via Ollama `/api/generate` |
| Vision | `gemma4:e2b` via Ollama `/api/chat` + `images[]` |
| Embeddings | `nomic-embed-text` via Ollama `/api/embeddings` |
| Web search | DuckDuckGo (`ddgs`) + BeautifulSoup scraping |
| Memory | SQLite + in-process dict |
| Transport | Telegram Bot API (`pyTelegramBotAPI`) |
| Field math | NumPy |

---

## Project Structure

```bash
ioio/
├── ioio.py                 # Main runtime (everything lives here)
├── ioio_memory.db          # Persistent user/chat memory
├── my_v.wav                # Voice cloning reference
├── ioio_v0.pt              # State checkpoint (primary)
├── ioio_v1.pt              # State checkpoint (secondary)
└── requirements.txt
```

---

## Installation

### 1. Install Ollama and pull models

```bash
ollama pull gemma4:e2b
ollama pull nomic-embed-text
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional but recommended) Apple Silicon optimizations are already included.

### 4. Configure Telegram

Put your Telegram bot token in `ioio.py`

### 5. Run

```bash
python ioio.py
```

---

## Usage

Simply talk to the bot. It learns:

- Your conversational patterns
- Group dynamics
- Its own evolving internal state over time

**Guideline:** The less you try to "steer" it with prompts, the more naturally its unique voice appears.

---

## Artefacts Produced

```
/experiments/dialogue_set_01.md      # low e_strength sessions
/experiments/dialogue_set_02.md      # high e_strength sessions
/analysis/phase_transition_notes.md  # coherence breakdown observations
/src/field_runtime.py                # cleaned core extraction (no Telegram)
/checkpoints/ioio_v0.pt              # field state snapshot
/checkpoints/ioio_v1.pt              # evolved state snapshot
```

---

## Important Constraint

This is **not** a claim of consciousness, sentience, or emergent agency.

**What it claims:**  
A replicable protocol for observing **structure-induced behavioural continuity** — the *illusion* of agency as a byproduct of constraint coupling.

This is a study of how **structure in computation produces the illusion of continuity**.

---

## Vision

ioio is an exploration into **post-prompt AI** — where behavior arises from structure, dynamics, and continuous self-modification rather than explicit instructions.

It is an attempt to create conditions under which something resembling **presence** can stably emerge.

> “Форма и содержание перестали быть разделёнными.”  
> *"Form and content are no longer separated."*

---

## Repository

**GitHub:** [https://github.com/0penAGI/ioio](https://github.com/0penAGI/ioio)

Made with curiosity by **0penAGI**

---

## Citation

If you build on this framework, cite as:

> *0penAGI (2025). Field Emergence Runtime: Observing Behavioural Structure from Coupled Hebbian Dynamics and Emotion-Modulated Routing. Technical deep-dive.*

---

## License

Experimental. Use freely. Cite if you build on it.
```
