# src/ui/streamlit_app.py

import streamlit as st
import requests

BACKEND_DEFAULT = "http://localhost:8000"
DEFAULT_MAX_SOURCES = 8


def build_use_case(backend_url: str, brief: str, max_sources: int | None):
    resp = requests.post(
        f"{backend_url}/use_case/build",
        json={"brief": brief, "max_sources": max_sources},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def ask_use_case(backend_url: str, session_id: str, question: str):
    resp = requests.post(
        f"{backend_url}/use_case/qa",
        json={"session_id": session_id, "question": question},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def build_global_index(backend_url: str):
    resp = requests.post(f"{backend_url}/build_index", timeout=600)
    resp.raise_for_status()
    return resp.json()


def ask_global(backend_url: str, question: str):
    resp = requests.post(
        f"{backend_url}/qa",
        json={"question": question},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    st.set_page_config(page_title="GraphRAG Agentic Pipeline", layout="wide")
    st.title("GraphRAG Agentic Pipeline")
    st.caption(
        "Brief-aware ingestion → knowledge graph + vector index → GraphRAG QA with citations and graph paths."
    )

    backend_url = st.text_input("Backend URL", value=BACKEND_DEFAULT)

    if "use_case_session" not in st.session_state:
        st.session_state.use_case_session = None
        st.session_state.selected_sources = []
        st.session_state.brief = ""

    st.header("1) Provide your use-case brief")
    brief = st.text_area(
        "Short brief",
        value=st.session_state.brief
        or "Ex: evolution of the AI courses in ESILV",
        height=90,
    )
    max_sources = st.number_input(
        "Max sources to keep",
        min_value=1,
        max_value=50,
        value=DEFAULT_MAX_SOURCES,
        step=1,
    )

    if st.button("Build use-case graph + vector index"):
        with st.spinner("Selecting sources and building the pipeline..."):
            try:
                result = build_use_case(backend_url, brief, int(max_sources))
                st.session_state.use_case_session = result["session_id"]
                st.session_state.selected_sources = result.get(
                    "selected_sources", []
                )
                st.session_state.brief = result.get("brief", brief)
                st.success(
                    f"Use-case index built (session {result['session_id']}). "
                    f"{result['num_chunks']} chunks / {result['num_sources']} sources."
                )
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.selected_sources:
        st.subheader("Selected sources for this brief")
        source_rows = [
            {
                "path": s["path"],
                "score": round(s.get("score", 0), 3),
                "overlap": round(s.get("overlap", 0), 3),
                "embedding": round(s.get("embedding_score", 0), 3),
            }
            for s in st.session_state.selected_sources
        ]
        st.dataframe(source_rows, use_container_width=True, hide_index=True)

    st.header("2) Ask questions on this use-case graph")
    disabled = st.session_state.use_case_session is None
    question = st.text_area(
        "Question",
        value="Pose une question sur le use-case choisi.",
        height=90,
        disabled=disabled,
    )
    if st.button("Ask (GraphRAG)", disabled=disabled):
        with st.spinner("Running GraphRAG retrieval and LLM reasoning..."):
            try:
                result = ask_use_case(
                    backend_url, st.session_state.use_case_session, question
                )
                st.subheader("Answer")
                st.markdown(result["answer"])

                st.subheader("Citations")
                for c in result.get("citations", []):
                    st.markdown(f"- `{c.get('label')}`")

                st.subheader("Graph paths")
                paths = result.get("graph_paths") or []
                if not paths:
                    st.write("No graph paths found for the mentioned entities.")
                else:
                    for idx, p in enumerate(paths, start=1):
                        names = " → ".join(
                            [n.get("name") or n.get("id") for n in p["nodes"]]
                        )
                        st.markdown(f"{idx}. {names}")

                with st.expander("Graph context (nodes & edges)"):
                    st.json(result.get("graph_context", {}))
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    with st.expander("Legacy global index (ingest everything)"):
        st.write("Optional: build and query the global corpus without a brief.")
        if st.button("Build / rebuild global index"):
            with st.spinner("Building global index..."):
                try:
                    info = build_global_index(backend_url)
                    st.success(
                        f"Global index built: {info['num_chunks']} chunks from {info['num_sources']} sources."
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

        global_q = st.text_input("Global question", value="", key="global_q")
        if st.button("Ask global QA"):
            with st.spinner("Querying global index..."):
                try:
                    result = ask_global(backend_url, global_q)
                    st.markdown(result["answer"])
                    st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
