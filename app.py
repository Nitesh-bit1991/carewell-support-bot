"""Streamlit chat UI. Run with: streamlit run app.py"""
import streamlit as st

from support_bot import SupportBot

st.set_page_config(page_title="CareWell Support", page_icon="💊")
st.title("💊 CareWell Medical Store")
st.caption("Demo support bot for a fictional store. It answers store questions only, never medical advice.")

with st.sidebar:
    reranker = st.selectbox("Reranker", ["lexical", "none"], help="'none' keeps the plain cosine order")
    st.markdown("Try: *Is delivery free?*, *Can I return medicines?*, *Which medicine is best for a headache?*")


@st.cache_resource
def get_bot(reranker_name: str) -> SupportBot:
    return SupportBot(reranker=reranker_name)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me about opening hours, delivery, prescriptions, payment or returns."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    reply = get_bot(reranker).ask(question)
    with st.chat_message("assistant"):
        st.markdown(reply.text)
        if reply.sources:
            st.caption("Sources: " + ", ".join(reply.sources))
        with st.expander("How the bot decided"):
            st.write(f"Route: `{reply.route}`")
            for h in reply.ranked:
                st.write(f"- {h.chunk.source} / {h.chunk.heading}: cosine {h.cosine:.2f}, rerank {h.rerank:.2f}")
    st.session_state.messages.append({"role": "assistant", "content": reply.text})
