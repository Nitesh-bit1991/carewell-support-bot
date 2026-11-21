"""Write the final answer from the retrieved passages."""
import os

from .retrieval import Hit

SYSTEM_PROMPT = (
    "You are a customer support assistant for CareWell Medical Store. "
    "Answer ONLY from the provided context about store services. "
    "Never give medical advice. Keep answers short and friendly. "
    "Mention the source file in square brackets, e.g. [delivery.md]. "
    "If the context does not contain the answer, reply with exactly: NO_ANSWER"
)


def generate_answer(question: str, hits: list[Hit], use_llm: bool = True) -> str:
    """Use Claude if ANTHROPIC_API_KEY is set, otherwise quote the best passage."""
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            context = "\n\n".join(f"[{h.chunk.source}]\n{h.chunk.text}" for h in hits)
            response = anthropic.Anthropic().messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
            )
            return response.content[0].text.strip()
        except ImportError:
            pass  # anthropic not installed: fall back to extractive mode
    top = hits[0].chunk
    return f"{top.text} [{top.source}]"
