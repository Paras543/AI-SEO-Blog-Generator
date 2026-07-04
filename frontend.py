"""
Streamlit frontend for the LangGraph technical-blog-generation pipeline
defined in backend.py.

Run with:
    streamlit run app.py

Requires (in addition to backend.py's own deps):
    pip install streamlit
"""

from __future__ import annotations

import traceback
from datetime import date
from pathlib import Path

import streamlit as st

from backened import app  # your compiled LangGraph, from backend.py

st.set_page_config(page_title="AI Blog Generator", page_icon="📝", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in {
    "running": False,
    "final_md": None,
    "seo": None,
    "readability": None,
    "citation_report": None,
    "plan": None,
    "evidence": None,
    "mode": None,
    "log": [],
    "error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _reset_run_state() -> None:
    st.session_state.final_md = None
    st.session_state.seo = None
    st.session_state.readability = None
    st.session_state.citation_report = None
    st.session_state.plan = None
    st.session_state.evidence = None
    st.session_state.mode = None
    st.session_state.log = []
    st.session_state.error = None


def _unpack_event(event):
    """Normalize a stream event into (namespace_tuple, {node_name: update})."""
    if isinstance(event, tuple) and len(event) == 2 and isinstance(event[1], dict):
        return event
    return (), event


# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------

st.sidebar.title("📝 Blog Generator")
st.sidebar.caption("Powered by a LangGraph multi-agent pipeline (router → research → plan → write → fact-check → publish).")

topic = st.sidebar.text_area(
    "Blog topic",
    placeholder="e.g. How vector databases power retrieval-augmented generation",
    height=90,
)
as_of = st.sidebar.date_input("As-of date", value=date.today())
max_revisions = st.sidebar.slider(
    "Max citation-revision passes",
    min_value=0,
    max_value=3,
    value=1,
    help="If the fact-checker flags a hallucinated or missing citation, how many times should the "
    "affected section be rewritten before we give up and publish as-is?",
)

run_clicked = st.sidebar.button("🚀 Generate blog post", type="primary", disabled=st.session_state.running or not topic.strip())

st.sidebar.divider()
st.sidebar.caption(
    "The pipeline decides on its own whether it needs live web research "
    "(closed_book / hybrid / open_book) based on your topic."
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("AI Technical Blog Generator")
st.caption("LangGraph pipeline: router → (optional research) → orchestrator → parallel section writers → citation fact-check loop → SEO/readability → images.")

status_placeholder = st.container()
result_tab, plan_tab, evidence_tab, meta_tab = st.tabs(["📄 Result", "🗂️ Plan", "🔎 Evidence & citations", "📈 SEO / readability"])

if run_clicked:
    st.session_state.running = True
    _reset_run_state()

    initial_state = {
        "topic": topic.strip(),
        "as_of": as_of.isoformat(),
        "max_revisions": max_revisions,
        "revision_count": 0,
    }

    total_tasks = None
    completed_tasks = 0

    with status_placeholder:
        with st.status("Running pipeline...", expanded=True) as status:
            try:
                for raw_event in app.stream(initial_state, stream_mode="updates", subgraphs=True):
                    _, updates = _unpack_event(raw_event)

                    for node_name, update in updates.items():
                        if update is None:
                            continue

                        if node_name == "router":
                            mode = update.get("mode")
                            needs_research = update.get("needs_research")
                            st.session_state.mode = mode
                            status.write(f"**Router:** mode=`{mode}`, needs_research=`{needs_research}`")

                        elif node_name == "research":
                            evidence = update.get("evidence", [])
                            st.session_state.evidence = evidence
                            status.write(f"**Research:** gathered {len(evidence)} evidence item(s).")

                        elif node_name == "orchestrator":
                            plan = update.get("plan")
                            if plan is not None:
                                plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
                                st.session_state.plan = plan_dict
                                total_tasks = len(plan_dict.get("tasks", []))
                                status.write(
                                    f"**Plan ready:** \"{plan_dict.get('blog_title')}\" "
                                    f"— {total_tasks} section(s), kind=`{plan_dict.get('blog_kind')}`"
                                )

                        elif node_name == "worker":
                            completed_tasks += 1
                            label = f"{completed_tasks}/{total_tasks}" if total_tasks else str(completed_tasks)
                            status.write(f"**Writer:** drafted section {label}")

                        elif node_name == "fact_check":
                            report = update.get("citation_report")
                            st.session_state.citation_report = report
                            if report:
                                flagged = report.get("flagged_task_ids", [])
                                if flagged:
                                    status.write(f"**Fact-check:** ⚠️ flagged section(s) {flagged} — triggering revision.")
                                else:
                                    status.write(f"**Fact-check:** ✅ passed, score={report.get('score')}.")

                        elif node_name == "seo_analysis":
                            st.session_state.seo = update.get("seo")
                            st.session_state.readability = update.get("readability")
                            status.write("**SEO / readability:** metadata generated.")

                        elif node_name == "decide_images":
                            specs = update.get("image_specs", [])
                            status.write(f"**Images:** planned {len(specs)} image(s).")

                        elif node_name == "generate_and_place_images":
                            status.write("**Images:** generated and placed.")

                        elif node_name == "attach_frontmatter":
                            final_md = update.get("final")
                            if final_md:
                                st.session_state.final_md = final_md
                                status.write("**Publish:** frontmatter attached, file written.")

                status.update(label="Pipeline complete ✅", state="complete", expanded=False)

            except Exception as e:  # noqa: BLE001
                st.session_state.error = f"{e}\n\n{traceback.format_exc()}"
                status.update(label="Pipeline failed ❌", state="error", expanded=True)

    st.session_state.running = False

if st.session_state.error:
    st.error("Something went wrong while running the pipeline.")
    with st.expander("Error details"):
        st.code(st.session_state.error)

# ---------------------------------------------------------------------------
# Result tab
# ---------------------------------------------------------------------------

with result_tab:
    if st.session_state.final_md:
        plan = st.session_state.plan or {}
        title = plan.get("blog_title", "blog_post")
        filename = f"{title.strip().lower().replace(' ', '_') or 'blog_post'}.md"

        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                "⬇️ Download Markdown",
                data=st.session_state.final_md,
                file_name=filename,
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            images_dir = Path("images")
            if images_dir.exists() and any(images_dir.iterdir()):
                st.caption(f"Generated images saved to `{images_dir.resolve()}`")

        view_mode = st.radio("View as", ["Rendered", "Raw markdown"], horizontal=True)
        st.divider()
        if view_mode == "Rendered":
            st.markdown(st.session_state.final_md, unsafe_allow_html=False)
        else:
            st.code(st.session_state.final_md, language="markdown")
    else:
        st.info("Enter a topic in the sidebar and click **Generate blog post** to get started.")

# ---------------------------------------------------------------------------
# Plan tab
# ---------------------------------------------------------------------------

with plan_tab:
    plan = st.session_state.plan
    if plan:
        st.subheader(plan.get("blog_title", ""))
        c1, c2, c3 = st.columns(3)
        c1.metric("Blog kind", plan.get("blog_kind", "—"))
        c2.metric("Mode", st.session_state.mode or "—")
        c3.metric("Sections", len(plan.get("tasks", [])))

        st.write(f"**Audience:** {plan.get('audience', '—')}")
        st.write(f"**Tone:** {plan.get('tone', '—')}")
        if plan.get("constraints"):
            st.write("**Constraints:**")
            for c in plan["constraints"]:
                st.write(f"- {c}")

        st.divider()
        for task in plan.get("tasks", []):
            with st.expander(f"#{task['id']} — {task['title']} (~{task['target_words']} words)"):
                st.write(f"**Goal:** {task['goal']}")
                st.write("**Bullets:**")
                for b in task.get("bullets", []):
                    st.write(f"- {b}")
                tags = ", ".join(task.get("tags", []))
                st.caption(
                    f"tags: {tags or '—'} · requires_research={task.get('requires_research')} · "
                    f"requires_citations={task.get('requires_citations')} · requires_code={task.get('requires_code')}"
                )
    else:
        st.info("The plan will appear here once the orchestrator has run.")

# ---------------------------------------------------------------------------
# Evidence & citations tab
# ---------------------------------------------------------------------------

with evidence_tab:
    evidence = st.session_state.evidence
    if evidence:
        st.subheader(f"Evidence gathered ({len(evidence)})")
        for e in evidence:
            e = e.model_dump() if hasattr(e, "model_dump") else e
            with st.container(border=True):
                st.write(f"**[{e.get('title')}]({e.get('url')})**")
                st.caption(f"{e.get('source') or ''} · {e.get('published_at') or 'date unknown'}")
                if e.get("snippet"):
                    st.write(e["snippet"])
    else:
        st.info("No external evidence was gathered (topic was likely handled closed-book, or research hasn't run yet).")

    report = st.session_state.citation_report
    if report:
        st.divider()
        st.subheader("Citation fact-check report")
        st.metric("Citation score", report.get("score"))
        flagged = report.get("flagged_task_ids", [])
        if flagged:
            st.warning(f"Flagged section IDs: {flagged}")
            for issue in report.get("issues", []):
                st.write(f"- {issue}")
        else:
            st.success("No citation issues detected.")

# ---------------------------------------------------------------------------
# SEO / readability tab
# ---------------------------------------------------------------------------

with meta_tab:
    seo = st.session_state.seo
    readability = st.session_state.readability

    if readability:
        st.subheader("Readability")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Words", readability.get("word_count"))
        c2.metric("Flesch score", readability.get("flesch_reading_ease"))
        c3.metric("Reading time", f"{readability.get('estimated_reading_minutes')} min")
        c4.metric("Level", readability.get("reading_level"))

    if seo:
        st.divider()
        st.subheader("SEO metadata")
        st.write(f"**SEO title:** {seo.get('seo_title')}")
        st.write(f"**Meta description:** {seo.get('meta_description')}")
        st.write(f"**Slug:** `{seo.get('slug')}`")
        st.write(f"**Primary keyword:** {seo.get('primary_keyword')}")
        if seo.get("secondary_keywords"):
            st.write("**Secondary keywords:** " + ", ".join(seo["secondary_keywords"]))
        st.write(f"**Social summary:** {seo.get('social_summary')}")

    if not seo and not readability:
        st.info("SEO and readability metadata will appear here after the pipeline finishes.")


