import re

import streamlit as st

from pipeline import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Multi-Agent AI Research System",
    page_icon="🔎",
    layout="wide",
)


# ------------------------------------------------------------------- helpers
@st.cache_resource
def get_agents():
    """Build the agents once and reuse them across Streamlit reruns."""
    return build_search_agent(), build_reader_agent()


def last_message_text(result: dict) -> str:
    """Extract the final answer text from a create_agent result."""
    content = result["messages"][-1].content
    if isinstance(content, list):  # some models return a list of content blocks
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


def extract_score(critique: str):
    """Pull 'Score: X/10' out of the critic's response."""
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", critique, re.IGNORECASE)
    return match.group(1) if match else None


def run_pipeline(topic: str) -> None:
    """Run all four stages and store every output in session_state."""
    search_agent, reader_agent = get_agents()

    # Stage 1: search
    with st.status("🔍 Search agent is finding sources...", expanded=False) as status:
        search_result = search_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Find the most relevant, recent and reliable sources about: {topic}. "
                        "Use the web search tool. Return the top 5 results as a list, "
                        "each with its URL and a one-line summary.",
                    )
                ]
            }
        )
        st.session_state.search_results = last_message_text(search_result)
        status.update(label="✅ Search complete", state="complete")

    # Stage 2: read / scrape
    with st.status("📖 Reader agent is scraping pages...", expanded=False) as status:
        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Topic: {topic}\n\nSearch results:\n{st.session_state.search_results}\n\n"
                        "Pick the 2-3 most relevant URLs and scrape each one using the tool. "
                        "Then summarise the key facts from each page and keep the source URL "
                        "next to every summary.",
                    )
                ]
            }
        )
        st.session_state.scraped_content = last_message_text(reader_result)
        status.update(label="✅ Reading complete", state="complete")

    research = (
        f"SEARCH RESULTS:\n{st.session_state.search_results}\n\n"
        f"SCRAPED CONTENT:\n{st.session_state.scraped_content}"
    )

    # Stage 3: write
    with st.status("✍️ Writer is drafting the report...", expanded=False) as status:
        st.session_state.report = writer_chain.invoke(
            {"topic": topic, "research": research}
        )
        status.update(label="✅ Report written", state="complete")

    # Stage 4: critique
    with st.status("🧐 Critic is reviewing the report...", expanded=False) as status:
        st.session_state.critique = critic_chain.invoke(
            {"report": st.session_state.report}
        )
        status.update(label="✅ Review complete", state="complete")

    st.session_state.topic = topic


# ------------------------------------------------------------- session state
for key in ("topic", "search_results", "scraped_content", "report", "critique"):
    st.session_state.setdefault(key, None)


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Research Settings")
    topic_input = st.text_input(
        "Research topic",
        placeholder="e.g. Impact of quantum computing on cryptography",
    )
    run_clicked = st.button("🚀 Run Research", type="primary", use_container_width=True)

    st.divider()
    st.markdown(
        """
**How it works**
1. 🔍 Search agent finds sources
2. 📖 Reader agent scrapes pages
3. ✍️ Writer drafts the report
4. 🧐 Critic scores it
"""
    )


# --------------------------------------------------------------- main header
st.title("🔎 Multi-Agent AI Research System")
st.caption("Search → Read → Write → Critique, powered by LangChain agents.")


# ------------------------------------------------------------------ run logic
if run_clicked:
    if not topic_input.strip():
        st.warning("Please enter a research topic first.")
    else:
        # clear old results so stale output isn't shown if a stage fails
        for key in ("search_results", "scraped_content", "report", "critique"):
            st.session_state[key] = None
        try:
            run_pipeline(topic_input.strip())
        except Exception as e:
            st.error(f"Something went wrong while running the pipeline: {e}")


# ------------------------------------------------------------------- results
if st.session_state.report:
    st.subheader(f"Results for: {st.session_state.topic}")

    tab_report, tab_critique, tab_research = st.tabs(
        ["📄 Report", "🧐 Critique", "🗂️ Raw Research"]
    )

    with tab_report:
        st.markdown(st.session_state.report)
        st.download_button(
            "⬇️ Download report (.md)",
            data=st.session_state.report,
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_critique:
        score = extract_score(st.session_state.critique or "")
        if score:
            st.metric("Critic score", f"{score}/10")
        st.markdown(st.session_state.critique)

    with tab_research:
        with st.expander("Search results", expanded=False):
            st.markdown(st.session_state.search_results)
        with st.expander("Scraped content summary", expanded=True):
            st.markdown(st.session_state.scraped_content)
else:
    st.info("Enter a topic in the sidebar and click **Run Research** to begin.")