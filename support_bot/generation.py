"""Write the final answer from the retrieved passages."""
import os
import re

from .retrieval import Hit

_STOCK_QUESTION = re.compile(
    r"\b(available|availability|in stock|stock|have you got|do you have)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are a customer support assistant for CareWell Medical Store. "
    "Answer ONLY using relevant provided context about store services. "
    "Never give medical advice, diagnosis, dosage, or product recommendations. "
    "Do not infer stock availability for a named medicine or product. "
    "If the context says live stock cannot be checked, explain that limitation and tell the customer to contact the store. "
    "Ignore context passages that do not directly help answer the question. "
    "Mention the source file in square brackets, e.g. [delivery.md]. "
    "If the relevant context does not contain the answer, reply with exactly: NO_ANSWER"
)


def _fallback_answer(question: str, hits: list[Hit]) -> str:
    """Deterministic fallback that preserves stock and contact policies."""
    if _STOCK_QUESTION.search(question):
        stock = next(
            (h.chunk for h in hits
             if h.chunk.source == "products.md" and h.chunk.heading.lower() == "stock availability"),
            None,
        )
        contact = next(
            (h.chunk for h in hits
             if h.chunk.source == "contact.md" and h.chunk.heading.lower() == "phone, whatsapp and email"),
            None,
        )
        if stock:
            answer = (
                "We cannot show live stock levels online, so I cannot confirm whether "
                "this item is currently available. Please call or message us to check. "
                "[products.md]"
            )
            if contact:
                answer += f"\n\n{contact.text} [contact.md]"
            return answer

    top = hits[0].chunk
    return f"{top.text} [{top.source}]"


def generate_answer(question: str, hits: list[Hit], use_llm: bool = True) -> str:
    """Use Claude when configured; otherwise return a safe deterministic answer."""
    if not hits:
        return "NO_ANSWER"

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
            answer = response.content[0].text.strip()
            return answer if answer else "NO_ANSWER"
        except ImportError:
            pass
    return _fallback_answer(question, hits)
