from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)



class CitationReport(BaseModel):
    score: int
    issues: List[str] = Field(default_factory=list)
    flagged_task_ids: List[int] = Field(default_factory=list)


class SEOMeta(BaseModel):
    seo_title: str = Field(..., description="<=60 chars, includes the primary keyword")
    meta_description: str = Field(..., description="<=155 chars, compelling summary")
    slug: str = Field(..., description="lowercase-hyphenated, no special characters")
    primary_keyword: str
    secondary_keywords: List[str] = Field(default_factory=list)
    social_summary: str = Field(..., description="1-2 sentence summary for Twitter/LinkedIn share")


def _max_reducer(a: int, b: int) -> int:
    return max(a, b)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # NEW: citation fact-check / revision loop state
    citation_report: Optional[dict]
    flagged_task_ids: List[int]
    revision_count: Annotated[int, _max_reducer]
    max_revisions: int

    # NEW: SEO + readability state
    seo: Optional[dict]
    readability: Optional[dict]

    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""


def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


def _tavily_search(query: str, max_results: int = 2) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}


ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don't invent events).

Output must match Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)) using a URL copied EXACTLY from Evidence.
  Never invent a URL or reuse a URL that is not listed in Evidence.
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims, using the exact URLs given.

Code:
- If requires_code==true, include at least one minimal snippet.
"""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    # NEW: if this is a revision pass triggered by the fact-check loop, add a
    # sharp, targeted correction note so the model doesn't repeat the mistake.
    revision_note = payload.get("revision_note")
    revision_block = ""
    if revision_note:
        revision_block = (
            "\n\nREVISION REQUIRED (previous draft failed the citation check):\n"
            f"{revision_note}\n"
            "Rewrite this section from scratch. Every external claim must link to a URL "
            "copied verbatim from the Evidence list below, or explicitly state "
            "\"Not found in provided sources.\" Do not invent or reuse any other URL.\n"
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n"
                    f"{revision_block}\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# ---------------------------------------------------------------------------
# NEW FEATURE #1: Citation fact-check + self-revision loop
#
# After every worker finishes, we deterministically scan the generated
# sections for any Markdown link whose URL isn't in the Evidence list
# (a hallucinated citation), and in open_book mode we also flag sections
# that assert claims with zero citations at all. If problems are found and
# we're under the revision budget, we route back to `worker` with a
# pointed correction note for ONLY the flagged sections. Otherwise we
# proceed to the reducer as before.
# ---------------------------------------------------------------------------

_MD_LINK_URL_RE = re.compile(r"\]\((https?://[^\s)]+)\)")


def _extract_markdown_link_urls(md: str) -> List[str]:
    return _MD_LINK_URL_RE.findall(md)


def fact_check_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")

    if mode == "closed_book":
        # Evergreen content isn't evidence-bound; nothing to fact-check.
        return {"citation_report": None, "flagged_task_ids": []}

    evidence_urls = {e.url for e in state.get("evidence", []) if e.url}

    # Keep only the most recent draft per task id (in case this is a
    # second pass through the loop after a revision).
    latest_by_id: dict[int, str] = {}
    for tid, md in state.get("sections", []):
        latest_by_id[tid] = md

    issues: List[str] = []
    flagged_ids: List[int] = []

    for tid, md in latest_by_id.items():
        used_urls = _extract_markdown_link_urls(md)
        bad_urls = [u for u in used_urls if u not in evidence_urls]

        if bad_urls:
            flagged_ids.append(tid)
            issues.append(f"Task {tid}: cited URL(s) not present in Evidence: {', '.join(bad_urls)}")
        elif mode == "open_book" and not used_urls and "not found in provided sources" not in md.lower():
            flagged_ids.append(tid)
            issues.append(
                f"Task {tid}: section makes claims with no citations and no "
                "'Not found in provided sources' disclaimer."
            )

    flagged_ids = sorted(set(flagged_ids))
    report = {
        "score": max(0, 100 - 20 * len(flagged_ids)),
        "issues": issues,
        "flagged_task_ids": flagged_ids,
    }
    return {"citation_report": report, "flagged_task_ids": flagged_ids}


def route_after_fact_check(state: State):
    flagged = state.get("flagged_task_ids") or []
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 1)

    if not flagged or revision_count >= max_revisions:
        return "reducer"

    plan = state["plan"]
    assert plan is not None
    tasks_by_id = {t.id: t for t in plan.tasks}

    issues = (state.get("citation_report") or {}).get("issues", [])
    issue_by_task: dict[int, str] = {}
    for issue in issues:
        m = re.match(r"Task (\d+): (.*)", issue)
        if m:
            issue_by_task[int(m.group(1))] = m.group(2)

    sends: List[Send] = []
    for tid in flagged:
        task = tasks_by_id.get(tid)
        if not task:
            continue
        sends.append(
            Send(
                "worker",
                {
                    "task": task.model_dump(),
                    "topic": state["topic"],
                    "mode": state["mode"],
                    "as_of": state["as_of"],
                    "recency_days": state["recency_days"],
                    "plan": plan.model_dump(),
                    "evidence": [e.model_dump() for e in state.get("evidence", [])],
                    "revision_count": revision_count + 1,
                    "revision_note": issue_by_task.get(tid, "Citation check failed."),
                },
            )
        )

    return sends if sends else "reducer"


def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")

    # Keep only the latest revision per task id (revisions are appended
    # after the original, so a plain overwrite-by-id keeps the newest one).
    latest_by_id: dict[int, str] = {}
    for tid, md in state["sections"]:
        latest_by_id[tid] = md

    ordered_sections = [latest_by_id[tid] for tid in sorted(latest_by_id.keys())]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


# ---------------------------------------------------------------------------
# NEW FEATURE #2: SEO metadata + readability scoring -> YAML frontmatter
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _readability_stats(markdown_text: str) -> dict:
    plain = re.sub(r"`[^`]*`", " ", markdown_text)
    plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
    plain = re.sub(r"[#*_>`-]", " ", plain)

    sentences = [s for s in re.split(r"[.!?]+", plain) if s.strip()]
    words = re.findall(r"[A-Za-z']+", plain)

    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)
    num_syllables = sum(_count_syllables(w) for w in words)

    flesch = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    flesch = round(flesch, 1)

    if flesch >= 70:
        level = "Easy (general audience)"
    elif flesch >= 50:
        level = "Medium (some technical background helpful)"
    else:
        level = "Advanced (technical/expert audience)"

    return {
        "word_count": num_words,
        "sentence_count": num_sentences,
        "flesch_reading_ease": flesch,
        "reading_level": level,
        "estimated_reading_minutes": max(1, round(num_words / 220)),
    }


SEO_SYSTEM = """You are an SEO editor for a technical engineering blog.
Given a full blog draft, produce publish-ready metadata.

Rules:
- seo_title: <=60 characters, includes the primary keyword, no clickbait.
- meta_description: <=155 characters, accurate and compelling.
- slug: lowercase, hyphenated, no special characters.
- social_summary: 1-2 sentences suitable for a Twitter/LinkedIn share.
"""


def seo_analysis_node(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None
    merged_md = state["merged_md"]

    readability = _readability_stats(merged_md)

    seo_generator = llm.with_structured_output(SEOMeta)
    seo = seo_generator.invoke(
        [
            SystemMessage(content=SEO_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n\n"
                    f"Draft:\n{merged_md[:6000]}"
                )
            ),
        ]
    )

    return {"seo": seo.model_dump(), "readability": readability}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""


def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    if not image_specs:
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
                    f"> **Alt:** {spec.get('alt', '')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    return {"final": md}


def attach_frontmatter_node(state: State) -> dict:
    """NEW: prepend YAML frontmatter (SEO + readability metadata) and write the file."""
    plan = state["plan"]
    assert plan is not None
    seo = state.get("seo") or {}
    readability = state.get("readability") or {}
    final_md = state["final"]

    def esc(s: str) -> str:
        return (s or "").replace('"', "'")

    keywords_str = ", ".join(f'"{esc(k)}"' for k in seo.get("secondary_keywords", []))

    frontmatter_lines = [
        "---",
        f'title: "{esc(seo.get("seo_title") or plan.blog_title)}"',
        f'description: "{esc(seo.get("meta_description", ""))}"',
        f'slug: "{esc(seo.get("slug") or _safe_slug(plan.blog_title))}"',
        f'primary_keyword: "{esc(seo.get("primary_keyword", ""))}"',
        f"secondary_keywords: [{keywords_str}]",
        f'social_summary: "{esc(seo.get("social_summary", ""))}"',
        f'reading_time_minutes: {readability.get("estimated_reading_minutes", "")}',
        f'reading_level: "{esc(readability.get("reading_level", ""))}"',
        f'flesch_reading_ease: {readability.get("flesch_reading_ease", "")}',
        f'word_count: {readability.get("word_count", "")}',
        "---",
        "",
    ]

    final_with_frontmatter = "\n".join(frontmatter_lines) + final_md

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(final_with_frontmatter, encoding="utf-8")

    return {"final": final_with_frontmatter}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("seo_analysis", seo_analysis_node)  # NEW
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_node("attach_frontmatter", attach_frontmatter_node)  # NEW

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "seo_analysis")
reducer_graph.add_edge("seo_analysis", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", "attach_frontmatter")
reducer_graph.add_edge("attach_frontmatter", END)
reducer_subgraph = reducer_graph.compile()

g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("fact_check", fact_check_node)  # NEW
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "fact_check")  # was: worker -> reducer
g.add_conditional_edges("fact_check", route_after_fact_check, ["reducer", "worker"])  # NEW
g.add_edge("reducer", END)

app = g.compile()
app


