# 0penAGI — Field Emergence Runtime

> This is not a chatbot. It is an observation of what happens when you remove explicit behavioral design and replace it with dynamic constraints.

---

## Abstract

This project does not attempt to build a "conversational AI".

It constructs a minimal set of constraints:

- semantic field dynamics in embedding space
- external grounding injection via live web retrieval
- emotional strength routing derived from prediction error
- identity suppression at the surface prompt level

...and observes what behavioral patterns emerge.

No ontological claims are made in the prompt.  
No "personality design" is encoded explicitly.  
The system is treated as a dynamical process, not an entity.

---

## Core Question

What produces the observed behavior?

- the base language model (Gemma 4 E2B)
- the surrounding architecture
- or the interaction between temporal state, memory, and external grounding

We do not assume a single source of agency. We measure the composite effect.

---

## System Architecture

### 1. Field Dynamics Layer

A continuous update system operating in embedding space:

```
m(t+1) = decay * m(t) + (1 - decay) * x_t
e_t    = x_t - W(x_t + m_t)
W      += lr * outer(e_t, x_prev)
```

- state vector evolution via exponential moving average
- error feedback via prediction mismatch (`e_t`)
- Hebbian-style weight update with stability clamp

This creates a non-static semantic manifold that accumulates conversational history as a continuous field, not a discrete log.

---

### 2. External Grounding Layer (WEB CONTEXT)

Retrieval is not always active. Instead:

- search is injected per-message via DuckDuckGo + page scraping
- results are treated as **state augmentation**, not authoritative facts
- a background thinker thread independently queries when memory suggests information gaps

This introduces discontinuity between the model's internal distribution and external ground truth — and forces grounded response when context is present.

---

### 3. Emotional Strength Routing (`e_strength`)

A scalar derived from field error magnitude:

```
e_strength = mean(|e_t|)

e_strength > 0.6  →  associative, drifting output
e_strength > 0.3  →  fluid, natural output  
e_strength ≤ 0.3  →  stable, structured output
```

This is not "emotion" in any psychological sense. It is a control parameter over generative entropy — lower prediction error means more stable context, which means tighter generation.

---

### 4. Identity Suppression Layer

Surface-level identity statements are constrained:

- no persistent named persona declared in the prompt
- identity is not announced, only implied through behavioral consistency
- name ("ioio") surfaces only when directly queried

This removes stable narrative framing. The system cannot anchor itself to a declared self-concept.

---

### 5. Multimodal Input (Vision)

Images are processed via the same `/api/chat` endpoint with `images[]` injection — no separate vision pipeline. The model receives image tokens as part of the same forward pass, producing a text description that enters the field as a semantic event.

---

### 6. Persistent Memory (SQLite)

- per-user profiles (username, name, avatar description)
- per-chat message history (last 30 messages in RAM, full log in DB)
- chat-user relational index

Memory is not retrieved by similarity — it flows as accumulated raw context into the prompt window.

---

## Key Design Decision

No explicit instruction of "behavioral personality" is used.

**Behavior is expected to emerge from the interaction of constraints.**

---

## Observational Claims (Not Conclusions)

When these components are combined:

- field memory alters response continuity across turns
- retrieval injection changes coherence boundaries (grounded vs. hallucinated)
- error-driven routing changes tone distribution
- identity suppression removes stable narrative self-anchoring

The system exhibits non-trivial conversational structure that is not directly encoded in any prompt.

---

## Experimental Protocol

To evaluate the system, we vary:

| Variable | Range |
|----------|-------|
| users | single / multi-user group |
| e_strength | 0.0 → 1.0+ |
| web context | present / absent / injected |
| memory saturation | fresh / accumulated / pruned |

We observe:

- coherence stability across turns
- referential consistency (does it remember correctly?)
- drift behavior under high e_strength
- convergence or divergence of subjectivity vector

---

## Stack

| Component | Implementation |
|-----------|---------------|
| Language model | `gemma4:e2b` via Ollama `/api/generate` |
| Vision | `gemma4:e2b` via Ollama `/api/chat` + `images[]` |
| Embeddings | `nomic-embed-text` via Ollama `/api/embeddings` |
| Web search | DuckDuckGo (`ddgs`) + BeautifulSoup scraping |
| Memory | SQLite + in-process dict |
| Transport | Telegram Bot API (`pyTelegramBotAPI`) |
| Field math | NumPy |

---

## Important Constraint

This is not a claim of consciousness, sentience, or emergent agency.

This is a study of how **structure in computation produces the illusion of continuity**.

---

## Suggested Next Artifacts

```
/experiments/dialogue_set_01.md     # low e_strength sessions
/experiments/dialogue_set_02.md     # high e_strength sessions
/analysis/phase_transition_notes.md # coherence breakdown observations
/src/field_runtime.py               # cleaned core extraction (no Telegram)
```

---

## License

Experimental. Use freely. Cite if you build on it.
