# AI-SEO-Blog-Generator
# 🧠 AI Technical Blog Generator — LangGraph Multi-Agent Pipeline

A multi-agent content pipeline built with **LangGraph** that turns a single topic string into a **publish-ready, fact-checked, SEO-tagged technical blog post** — complete with auto-generated diagrams and a self-correcting citation revision loop.

It ships with a **Streamlit frontend** (`app.py`) so you can run it as a real app instead of a notebook.

---

## ✨ What makes this different from a basic "LLM writes a blog post" script

Most LLM blog generators are a single prompt. This project is a real **agentic graph** with routing, parallel work, and a feedback loop:

```
START
  └─▶ router                (decides: do we even need web research?)
        ├─▶ research         (only if needed — Tavily search + evidence extraction)
        └─▶ orchestrator     (plans the post: title, tasks, audience, tone)
              └─▶ worker × N (parallel — one LLM call per section)
                    └─▶ fact_check              (citation firewall)
                          ├─▶ worker (revise flagged sections)  ⟲ loop
                          └─▶ reducer subgraph:
                                merge_content
                                  └─▶ seo_analysis        (SEO title/description/keywords + readability)
                                        └─▶ decide_images  (plans up to 3 diagrams)
                                              └─▶ generate_and_place_images (Gemini image gen)
                                                    └─▶ attach_frontmatter (writes final .md)
                                                          └─▶ END
```

---

## 🧩 Pipeline stages, explained

### 1. `router` — decides if research is even needed
Before writing anything, an LLM classifies the topic into one of three modes:

| Mode | Meaning | Example |
|---|---|---|
| `closed_book` | Evergreen concept, no research needed | "How does backpropagation work?" |
| `hybrid` | Evergreen but benefits from current examples/tools | "Best vector databases for RAG" |
| `open_book` | Time-sensitive / news / "latest" / pricing / policy | "This week in AI: model releases" |

Based on the mode, it sets a `recency_days` window (7 days for news roundups, 45 for hybrid, ~10 years for closed-book) that later gets used to filter out stale search results.

### 2. `research` — evidence gathering (Tavily + LLM extraction)
Only runs if the router says research is needed. It:
- Fires each generated search query at **Tavily Search**.
- Feeds the raw results to an LLM with a structured output schema (`EvidencePack`) to produce clean `EvidenceItem` objects: `title`, `url`, `published_at`, `snippet`, `source`.
- Deduplicates by URL.
- For `open_book` mode, hard-filters out anything older than the recency window — the model is not allowed to guess a date, only use what it can reliably infer.

This evidence list becomes the **only source of truth** that later stages are allowed to cite from.

### 3. `orchestrator` — plans the post
Produces a structured `Plan`:
- `blog_title`, `audience`, `tone`, `blog_kind` (`explainer` / `tutorial` / `news_roundup` / `comparison` / `system_design`)
- 5–9 `Task` objects, each with a `goal`, 3–6 `bullets`, a `target_words` count, and flags (`requires_research`, `requires_citations`, `requires_code`)

If the router chose `open_book`, the plan is force-set to `blog_kind="news_roundup"` and instructed not to drift into tutorial content.

### 4. `worker` — parallel section writers
Each `Task` from the plan is fanned out to its own `worker` invocation via LangGraph's `Send` API — so all sections are drafted **in parallel**, not sequentially. Each worker:
- Writes exactly one Markdown section covering all of its bullets.
- If `mode == "open_book"`, may only cite URLs that appear verbatim in the evidence list, or must write *"Not found in provided sources"* instead of inventing a claim.
- If `requires_code` is set, includes at least one code snippet.

### 5. `fact_check` — the citation firewall + self-revision loop ⭐
This is the core safety net of the whole pipeline. After all sections are drafted:
- It scans every section's Markdown links and flags any URL that **isn't in the evidence list** — i.e. a hallucinated citation.
- In `open_book` mode, it also flags any section that makes claims with **zero citations** and no explicit "not found" disclaimer.
- If issues are found and we're still under the revision budget (`max_revisions`, configurable, default 1), it routes **only the flagged sections** back to `worker` with a pointed correction note, then re-runs the check. This creates a real loop in the graph:
  `worker → fact_check → worker (revision) → fact_check → reducer`
- If a section is revised, `merge_content` automatically keeps the newest version and discards the earlier draft.
- Produces a `citation_report` (score out of 100 + list of issues) that's surfaced in the frontend.

This means the pipeline **catches and fixes its own hallucinated sources** before publishing, instead of just hoping the model behaved.

### 6. `reducer` (subgraph) — merge → SEO → images → publish
A nested subgraph that runs once all sections are finalized:

- **`merge_content`** — stitches sections into one document in task order.
- **`seo_analysis`** *(feature)* — generates `seo_title` (≤60 chars), `meta_description` (≤155 chars), `slug`, `primary_keyword`, `secondary_keywords`, and a social share blurb via structured LLM output. Also computes readability stats with a **hand-rolled Flesch Reading Ease calculator** (syllable counting via regex, zero extra dependencies): word count, sentence count, reading level, and estimated reading time.
- **`decide_images`** — an LLM decides whether the post needs up to 3 diagrams, inserting `[[IMAGE_1]]`-style placeholders and an image prompt for each.
- **`generate_and_place_images`** — calls **Gemini's image generation model** (`gemini-2.5-flash-image`) for each planned image, saves it to `images/`, and swaps the placeholder for a real Markdown image tag. If generation fails (quota/safety/SDK issue), it gracefully falls back to a visible "image generation failed" callout instead of breaking the pipeline.
- **`attach_frontmatter`** *(feature)* — prepends YAML frontmatter (SEO fields + readability metrics) to the final Markdown and writes it to disk as `<slug>.md` — ready to drop into Jekyll, Hugo, or a Next.js MDX blog.

---

## 🗂️ Project structure

```
.
├── backend.py        # LangGraph pipeline: state, nodes, edges, compiled `app`
├── app.py             # Streamlit frontend
├── images/            # Auto-generated diagrams land here (created at runtime)
└── <slug>.md          # Final generated blog post (created at runtime)
```

---

## 🧱 Core data models (Pydantic)

| Model | Purpose |
|---|---|
| `RouterDecision` | Output of the router: mode, whether research is needed, search queries |
| `EvidenceItem` / `EvidencePack` | A single cited source / the deduplicated collection of them |
| `Task` / `Plan` | The outline: one section spec / the full post plan |
| `CitationReport` | Score + issues + flagged section IDs from the fact-check node |
| `SEOMeta` | SEO title, description, slug, keywords, social summary |
| `ImageSpec` / `GlobalImagePlan` | One diagram's placeholder/prompt/caption / the full image plan for the post |

## 🔀 LangGraph `State` fields

| Field | Type | Set by |
|---|---|---|
| `topic`, `as_of` | `str` | you (initial input) |
| `mode`, `needs_research`, `queries`, `recency_days` | — | `router` |
| `evidence` | `List[EvidenceItem]` | `research` |
| `plan` | `Plan` | `orchestrator` |
| `sections` | `List[(task_id, markdown)]`, accumulates via `operator.add` | `worker` (one write per parallel branch) |
| `citation_report`, `flagged_task_ids` | — | `fact_check` |
| `revision_count` | `int`, reduced with `max()` to avoid parallel-write conflicts | `fact_check` loop |
| `max_revisions` | `int` | you (initial input, default 1) |
| `merged_md` | `str` | `merge_content` |
| `seo`, `readability` | `dict` | `seo_analysis` |
| `md_with_placeholders`, `image_specs` | — | `decide_images` |
| `final` | `str` | `generate_and_place_images` → `attach_frontmatter` |

> **Why `revision_count` uses a custom reducer:** when the fact-check loop fans out `Send`s to multiple flagged sections in the same step, each one writes the same `revision_count`. Without an explicit reducer (`max`), LangGraph throws an "multiple updates for channel" error because it doesn't know how to merge concurrent writes to the same key.

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install langgraph langchain-groq langchain-community python-dotenv pydantic streamlit google-genai
```

### 2. Environment variables (`.env`)
```env
GROQ_API_KEY=your_groq_key          # required — powers all LLM calls (llama-3.3-70b-versatile)
TAVILY_API_KEY=your_tavily_key      # optional — enables web research (hybrid/open_book modes)
GOOGLE_API_KEY=your_gemini_key      # optional — enables diagram/image generation
```

If `TAVILY_API_KEY` is missing, research silently returns no results — the pipeline still runs, just without external evidence (closed-book-style output). If `GOOGLE_API_KEY` is missing, image generation fails gracefully and is replaced with a visible placeholder callout instead of breaking the run.

### 3. Run the frontend
```bash
streamlit run app.py
```

### 4. …or call it directly in Python
```python
from backend import app

result = app.invoke({
    "topic": "How vector databases power retrieval-augmented generation",
    "as_of": "2026-07-04",
    "max_revisions": 1,
    "revision_count": 0,
})

print(result["final"])
```

---

## 🖥️ Streamlit frontend (`app.py`)

- **Sidebar** — topic, as-of date, and a slider for how many citation-revision passes to allow before publishing as-is.
- **Live run status** — streams the graph node-by-node (`app.stream(..., stream_mode="updates", subgraphs=True)`) so you see the router's decision, evidence being gathered, the plan being generated, each section being drafted, fact-check pass/fail with flagged sections, SEO generation, and image placement — as they happen, not just at the end.
- **Result tab** — rendered or raw Markdown of the final post, with a one-click download.
- **Plan tab** — the outline: title, audience, tone, constraints, and each section's goal/bullets/word target.
- **Evidence & citations tab** — every source the research node pulled in, plus the fact-check report (score, flagged section IDs, and the exact issues found).
- **SEO / readability tab** — the generated SEO title/description/slug/keywords/social copy, plus word count, Flesch reading ease score, reading level, and estimated reading time.

---

## 🛡️ Design decisions worth knowing about

- **Grounding is enforced at the prompt *and* checked deterministically.** Workers are instructed to only cite evidence URLs — but instructions alone aren't reliable, so `fact_check` re-verifies every link against the evidence list after the fact and forces a rewrite if it's wrong, rather than trusting the model's word.
- **Revisions replace, they don't duplicate.** Because `sections` accumulates via `operator.add`, a revised section is *appended*, not overwritten in place. `merge_content` resolves this by keeping only the latest entry per `task_id` when assembling the final document.
- **The revision loop is bounded.** `max_revisions` caps how many times a section can be sent back for a rewrite, so a persistently uncooperative model can't loop forever — it'll eventually publish with the citation issue still flagged in the report rather than hang.
- **Image generation failure doesn't fail the whole run.** Any error calling Gemini (quota, safety filter, SDK changes) is caught per-image and swapped for a visible "image generation failed" block with the original prompt/alt/caption, so the post is still usable.
- **No forced image usage.** `decide_images` can legitimately decide a post needs zero diagrams — decorative images are explicitly discouraged in its system prompt.

---

## 🚧 Possible extensions

- Swap `ChatGroq` for any other LangChain chat model — nothing in the graph is Groq-specific beyond the `llm` instantiation.
- Raise `max_revisions` or extend `fact_check` with a semantic (LLM-judged) check in addition to the current deterministic URL check.
- Add a human-in-the-loop approval node before `attach_frontmatter` using LangGraph's `interrupt` support.
- Persist runs with a LangGraph checkpointer (e.g. SQLite/Postgres) so long-running generations can be resumed instead of re-run from scratch.
