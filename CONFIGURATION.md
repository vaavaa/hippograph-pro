# HippoGraph Pro — Configuration Guide

HippoGraph ships with defaults tuned for **personal AI memory** — an agent that knows you, remembers your history, and builds relational context over time. But the same system can be tuned for very different use cases by adjusting a handful of parameters in your `.env` file.

This guide explains what each tuning dial does, what you gain, what you give up, and which profile fits your use case.

---

## The Three Profiles

### Profile 1 — Personal AI Memory (default)

The agent knows *you*. It remembers what you've discussed, how you work, what matters to you. Emotional weighting means a note about a critical security incident stays prominent; a note about a minor detail fades. Spreading activation surfaces connections you didn't explicitly ask for — the way human memory does.

**Best for:** Personal assistant, AI identity continuity, long-term AI-user relationship.

```env
# .env — Personal Memory (default, no changes needed)
DECAY_ENABLED=true
DECAY_HALF_LIFE_DAYS=30
RERANK_WEIGHT=0.3
ANN_TOP_K=15
BLEND_ALPHA=0.6
BLEND_BETA=0.25
BLEND_GAMMA=0.15
EMOTIONAL_WEIGHT=true
```

**What you get:**
- Memory fades naturally — important stays prominent, trivial fades
- Associative retrieval — related memories surface without explicit query
- Emotional context shapes what's retrieved and how
- Identity continuity across sessions and model versions

**What you give up:**
- Slightly lower precision on specific factual queries
- Older notes may surface less often even if still relevant

---

### Profile 2 — Project Memory (pure task focus)

The agent knows your *project*, not you. No emotional weighting, no personal history, no relational context. Retrieval is optimized for precision: you ask a specific question, you get the most relevant documents. Closer to RAG than to memory.

**Best for:** Project knowledge base, technical documentation assistant, team onboarding, research context — any scenario where the agent needs to know the work, not the person.

```env
# .env — Project Memory
DECAY_ENABLED=false
RERANK_WEIGHT=0.8
ANN_TOP_K=5
BLEND_ALPHA=0.5
BLEND_BETA=0.35
BLEND_GAMMA=0.15
EMOTIONAL_WEIGHT=false
```

**What you get:**
- Higher retrieval precision — exact answers to specific questions
- Nothing fades — all project knowledge stays equally accessible
- No personal or emotional noise in results
- Benchmark-validated: 78.7% Recall@5 on LOCOMO with this config

**What you give up:**
- No associative retrieval — the agent won't surface unexpected connections
- No relational context — the agent doesn't build a model of you over time
- No memory of *why* decisions were made, only *what* was decided

---

### Profile 3 — Hybrid (work context + minimal personal layer)

The agent knows the project and has a thin model of who you are — enough to adapt communication style and remember key decisions you've made, but without building a deep personal relationship. Decay is slow rather than off; emotional weighting is minimal.

**Best for:** Work assistant where some personal context is useful but the primary focus is task execution. The agent remembers you made a specific architectural decision and why, but doesn't build emotional weight around it.

```env
# .env — Hybrid
DECAY_ENABLED=true
DECAY_HALF_LIFE_DAYS=90
RERANK_WEIGHT=0.6
ANN_TOP_K=10
BLEND_ALPHA=0.55
BLEND_BETA=0.30
BLEND_GAMMA=0.15
EMOTIONAL_WEIGHT=false
```

**What you get:**
- Good precision on task-specific queries
- Slow decay — old decisions stay accessible for months
- Minimal personal noise
- Associative retrieval still active but less dominant

**What you give up:**
- Less associative richness than Personal profile
- Less precision than pure Project profile
- Middle ground means neither strength is fully realized

---

## Parameter Reference

### Decay

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DECAY_ENABLED` | `true` | Whether memory edges weaken over time. `false` = all notes equally accessible forever. |
| `DECAY_HALF_LIFE_DAYS` | `30` | How fast edges decay. 30 days = a note from a month ago has half the edge weight of a fresh note. Increase for slower forgetting. |
| `DISABLE_CATEGORY_DECAY` | `false` | Override: disable decay for protected anchor categories regardless of `DECAY_ENABLED`. Always `true` in benchmark runs. |

**Cost/Profit:**
- Decay ON → more natural memory behavior, important stays prominent, but older valid knowledge may surface less
- Decay OFF → everything equally accessible, higher precision on direct queries, no forgetting
- For project memory: **decay OFF**. For personal memory: **decay ON**.

---

### Retrieval Precision

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RERANK_WEIGHT` | `0.3` | How much the cross-encoder reranker influences final ranking. Range: 0.0–1.0. Higher = reranker dominates over blend score. |
| `ANN_TOP_K` | `15` | Candidate pool size before reranking. Lower = reranker works on a tighter, more precise set. Higher = broader recall before reranking. |

**Cost/Profit:**
- High `RERANK_WEIGHT` (0.7–0.9) + low `ANN_TOP_K` (5–10) → maximum precision, best for specific factual queries. Benchmark-optimal: 78.7% Recall@5.
- Low `RERANK_WEIGHT` (0.2–0.4) + high `ANN_TOP_K` (15–20) → broader associative recall, better for open-ended context retrieval.
- Reranking adds ~50–150ms latency. At `RERANK_WEIGHT=0.0` the reranker is effectively disabled.

---

### Blend Weights

The retrieval pipeline combines three signals. The blend weights control their relative contribution.

| Parameter | Default | Signal | Description |
|-----------|---------|--------|-------------|
| `BLEND_ALPHA` | `0.6` | Semantic | Pure embedding similarity. How close the query is to the note in vector space. |
| `BLEND_BETA` | `0.25` | Spreading activation | Graph-based associative score. How connected the note is to other activated nodes. |
| `BLEND_GAMMA` | `0.15` | BM25 | Keyword overlap. Exact term matching. |

> α + β + γ should sum to 1.0.

**Cost/Profit:**
- Higher `BLEND_ALPHA` → more direct semantic matching, less associative
- Higher `BLEND_BETA` → more associative retrieval, related memories surface even without explicit query terms — core of HippoGraph's value for personal memory
- Higher `BLEND_GAMMA` → better exact-term recall, useful when queries use specific technical terms
- For project memory: increase α, decrease β. For personal memory: keep β high.

---

### Emotional Weighting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMOTIONAL_WEIGHT` | `true` | Whether emotional intensity scores influence retrieval ranking. High-intensity notes get a small boost. |

**Cost/Profit:**
- ON → emotionally significant events stay more accessible. Useful for personal memory where importance correlates with emotional weight.
- OFF → purely content-based retrieval. Recommended for project memory where emotional metadata is noise.

---

### Entity Extraction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENTITY_EXTRACTOR` | `gliner` | Entity extraction model. `gliner` = GLiNER (higher quality, ~250ms, 600MB RAM). `spacy` = spaCy (faster, ~10ms, lower RAM). |

**Cost/Profit:**
- `gliner` → richer entity graph, better spreading activation, higher RAM
- `spacy` → minimal hardware, faster ingestion, less rich graph connections
- For minimal hardware (4GB RAM): use `spacy`

---

## Quick Decision Guide

```
I want the agent to know ME over time
    → Profile 1 (Personal Memory, default)

I want the agent to know my PROJECT / codebase / docs
    → Profile 2 (Project Memory)
      DECAY_ENABLED=false, RERANK_WEIGHT=0.8, ANN_TOP_K=5

I want work context + light personal layer
    → Profile 3 (Hybrid)
      DECAY_HALF_LIFE_DAYS=90, RERANK_WEIGHT=0.6

I'm running on minimal hardware (4GB RAM)
    → Any profile + ENTITY_EXTRACTOR=spacy

I want maximum retrieval precision (benchmark mode)
    → RERANK_WEIGHT=0.8, ANN_TOP_K=5, DECAY_ENABLED=false
      (This is the benchmark-validated config: 78.7% Recall@5)
```

---

## Applying Changes

All parameters live in `.env`. After editing:

```bash
# Full restart required — docker restart does NOT reload .env
docker-compose down && docker-compose up -d
```

> ⚠️ `docker-compose restart` does **not** reload environment variables. Always use `down/up`.

---

## What Doesn't Change Between Profiles

These behaviors are architectural — they don't change regardless of configuration:

- **No automatic deletion** — notes are never silently removed. Decay weakens edges, it doesn't delete nodes.
- **Anchor protection** — `self-reflection`, `protocol`, `security`, `milestone`, and other protected categories never decay, regardless of `DECAY_ENABLED`.
- **Zero LLM cost** — all retrieval runs locally. No API calls required regardless of profile.
- **Single user** — HippoGraph is not multi-tenant. All profiles assume one user, one knowledge base.

---

*For full benchmark results and methodology, see [BENCHMARK.md](BENCHMARK.md).*
*For setup and MCP connection, see [ONBOARDING.md](ONBOARDING.md).*