
"""CareWell Medical Store chatbot and RAG evaluation dashboard."""

import pandas as pd
import streamlit as st

from support_bot import SupportBot
from evaluation.evaluate import run_evaluation


st.set_page_config(
    page_title="CareWell Support",
    page_icon="💊",
    layout="wide",
)


@st.cache_resource
def get_bot(reranker_name: str) -> SupportBot:
    return SupportBot(reranker=reranker_name)


@st.cache_data
def get_evaluation_report():
    return run_evaluation()


def show_chatbot():
    st.title("💊 CareWell Medical Store")
    st.caption(
        "Demo support bot for a fictional store. "
        "It answers store questions only, never medical advice."
    )

    with st.sidebar:
        reranker = st.selectbox(
            "Chatbot reranker",
            ["lexical", "none"],
            help="'none' keeps the plain cosine order.",
        )

        st.markdown(
            "Try: *Is delivery free?*, "
            "*Can I return medicines?*, "
            "*Which medicine is best for a headache?*"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! Ask me about opening hours, delivery, "
                    "prescriptions, payment or returns."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Your question"):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        reply = get_bot(reranker).ask(question)

        with st.chat_message("assistant"):
            st.markdown(reply.text)

            if reply.sources:
                st.caption("Sources: " + ", ".join(reply.sources))

            with st.expander("How the bot decided"):
                st.write(f"Route: `{reply.route}`")

                for hit in reply.ranked:
                    st.write(
                        f"- {hit.chunk.source} / {hit.chunk.heading}: "
                        f"cosine {hit.cosine:.2f}, "
                        f"rerank {hit.rerank:.2f}"
                    )

        st.session_state.messages.append(
            {"role": "assistant", "content": reply.text}
        )


def show_evaluation_dashboard():
    st.title("📊 RAG Evaluation Dashboard")
    st.caption(
        "Evaluation of document retrieval using a fixed set of "
        "CareWell support questions."
    )

    try:
        report = get_evaluation_report()
    except Exception as error:
        st.error(f"Could not run the evaluation: {error}")
        st.info(
            "Check that evaluation/dataset.json and the knowledge "
            "documents are available."
        )
        return

    results = report["results"]

    st.subheader("Evaluation overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Evaluation questions", report["question_count"])
    col2.metric("Knowledge chunks", report["chunk_count"])
    col3.metric("Rerankers compared", len(results))

    st.divider()
    st.subheader("Reranker performance")

    summary_rows = [
        {
            "Reranker": result["reranker"].title(),
            "Recall@3": result["recall_at_3"],
            "MRR": result["mrr"],
        }
        for result in results
    ]

    summary_df = pd.DataFrame(summary_rows).set_index("Reranker")

    st.bar_chart(summary_df, y=["Recall@3", "MRR"])

    st.dataframe(
        summary_df.style.format("{:.3f}"),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Question-level results")

    selected_reranker = st.selectbox(
        "Select reranker",
        ["Both", "None", "Lexical"],
        key="evaluation_reranker_filter",
    )

    show_failures_only = st.checkbox("Show only retrieval failures")

    detail_rows = []

    for result in results:
        for item in result["question_results"]:
            detail_rows.append(
                {
                    "Reranker": result["reranker"].title(),
                    "Question": item["question"],
                    "Expected source": ", ".join(item["expected_sources"]),
                    "Top 3 retrieved": ", ".join(item["retrieved_sources"]),
                    "Correct document rank": (
                        item["relevant_rank"]
                        if item["relevant_rank"] is not None
                        else "Not found"
                    ),
                    "Recall@3": item["recall_at_3"],
                    "MRR": item["reciprocal_rank"],
                    "Result": "Pass" if item["passed"] else "Fail",
                }
            )

    details_df = pd.DataFrame(detail_rows)

    if selected_reranker != "Both":
        details_df = details_df[
            details_df["Reranker"] == selected_reranker
        ]

    if show_failures_only:
        details_df = details_df[details_df["Result"] == "Fail"]

    st.dataframe(
        details_df,
        use_container_width=True,
        hide_index=True,
    )

    failures = sum(
        1
        for row in detail_rows
        if row["Result"] == "Fail"
    )

    if failures == 0:
        st.success("No retrieval failures in this evaluation dataset.")
    else:
        st.warning(
            f"{failures} failed question/reranker result(s) found. "
            "Review the table above."
        )

    st.caption(
        "Recall@3 measures whether an expected source appears in the "
        "top three retrieved results. MRR rewards higher ranking of "
        "the first relevant source. Results depend on this evaluation "
        "dataset and are not a guarantee of performance on all questions."
    )

    if st.button("Refresh evaluation"):
        get_evaluation_report.clear()
        st.rerun()


with st.sidebar:
    st.title("CareWell")
    page = st.radio(
        "Navigation",
        ["Chatbot", "RAG Evaluation"],
    )

if page == "Chatbot":
    show_chatbot()
else:
    show_evaluation_dashboard()
