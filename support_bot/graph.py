"""The LangGraph workflow.

    START -> safety --(greeting)--> greet -> END
                |--(medical advice)--> refuse -> END
                +--> retrieve -> rerank --(relevant)--> generate -> END
                                     |
                                     +--(not relevant)--> handoff -> END
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .generation import generate_answer
from .knowledge import DOCS_DIR, load_chunks
from .retrieval import CosineRetriever, Hit, has_lexical_overlap, make_reranker
from .safety import asks_for_medical_advice

REFUSAL = (
    "I can't give medical advice, such as doses, side effects or which medicine to take. "
    "Please ask our pharmacist (call or visit the store) or your doctor. "
    "If this is an emergency, call your local emergency number right away."
)
HANDOFF = (
    "I couldn't find that in our store information, so I don't want to guess. "
    "Please call us on +49 681 000111 (Mon-Sat 8:30-21:00) or write to care@carewell.example."
)
GREETING = (
    "Hello! Welcome to CareWell Medical Store. I can help with store information "
    "such as opening hours, delivery, prescriptions, payment and returns. "
    "What would you like to know?"
)
_GREETING = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening)"
    r"[.!?,\s]*$",
    re.IGNORECASE,
)
_STOCK_QUESTION = re.compile(
    r"\b(available|availability|in stock|stock|have you got|do you have)\b",
    re.IGNORECASE,
)


class BotState(TypedDict, total=False):
    question: str
    candidates: list[Hit]
    ranked: list[Hit]
    answer: str
    route: str


@dataclass
class Reply:
    text: str
    route: str
    sources: list[str] = field(default_factory=list)
    ranked: list[Hit] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.route == "answered"


class SupportBot:
    def __init__(
        self,
        docs_dir: Path | str = DOCS_DIR,
        reranker: str = "lexical",
        min_cosine: float = 0.20,
        candidates_k: int = 6,
        final_k: int = 3,
        use_llm: bool = True,
    ):
        self.retriever = CosineRetriever(load_chunks(docs_dir))
        self.reranker = make_reranker(reranker)
        self.min_cosine = min_cosine
        self.candidates_k = candidates_k
        self.final_k = final_k
        self.use_llm = use_llm
        self.graph = self._build_graph()

    def _safety(self, state: BotState) -> BotState:
        return {}

    def _greet(self, state: BotState) -> BotState:
        return {"answer": GREETING, "route": "greeting", "ranked": []}

    def _refuse(self, state: BotState) -> BotState:
        return {"answer": REFUSAL, "route": "medical_refusal", "ranked": []}

    def _retrieve(self, state: BotState) -> BotState:
        return {"candidates": self.retriever.search(state["question"], self.candidates_k)}

    def _rerank(self, state: BotState) -> BotState:
        question = state["question"]
        relevant = [
            hit for hit in state["candidates"]
            if has_lexical_overlap(question, hit.chunk)
        ]

        # Availability answers need both the stock policy and a way to contact
        # the store. Add the dedicated contact section even when its wording
        # does not overlap with the product name in the question.
        if _STOCK_QUESTION.search(question):
            contact_hits = self.retriever.search("phone WhatsApp email contact", len(self.retriever.chunks))
            phone_hit = next(
                (hit for hit in contact_hits
                 if hit.chunk.source == "contact.md"
                 and hit.chunk.heading.lower() == "phone, whatsapp and email"),
                None,
            )
            if phone_hit and all(h.chunk != phone_hit.chunk for h in relevant):
                relevant.append(phone_hit)

        ranked = self.reranker.rerank(question, relevant)
        return {"ranked": ranked[: self.final_k]}

    def _generate(self, state: BotState) -> BotState:
        text = generate_answer(state["question"], state["ranked"], self.use_llm)
        if text == "NO_ANSWER":
            return {"answer": HANDOFF, "route": "handoff"}
        return {"answer": text, "route": "answered"}

    def _handoff(self, state: BotState) -> BotState:
        return {"answer": HANDOFF, "route": "handoff", "ranked": []}

    def _after_safety(self, state: BotState) -> str:
        question = state["question"].strip()
        if _GREETING.fullmatch(question):
            return "greet"
        return "refuse" if asks_for_medical_advice(question) else "retrieve"

    def _after_rerank(self, state: BotState) -> str:
        best_cosine = max((h.cosine for h in state.get("ranked", [])), default=0.0)
        return "generate" if state.get("ranked") and best_cosine >= self.min_cosine else "handoff"

    def _build_graph(self):
        g = StateGraph(BotState)
        g.add_node("safety", self._safety)
        g.add_node("greet", self._greet)
        g.add_node("refuse", self._refuse)
        g.add_node("retrieve", self._retrieve)
        g.add_node("rerank", self._rerank)
        g.add_node("generate", self._generate)
        g.add_node("handoff", self._handoff)

        g.add_edge(START, "safety")
        g.add_conditional_edges(
            "safety", self._after_safety,
            {"greet": "greet", "refuse": "refuse", "retrieve": "retrieve"},
        )
        g.add_edge("retrieve", "rerank")
        g.add_conditional_edges("rerank", self._after_rerank, {"generate": "generate", "handoff": "handoff"})
        for node in ("greet", "refuse", "generate", "handoff"):
            g.add_edge(node, END)
        return g.compile()

    def ask(self, question: str) -> Reply:
        state = self.graph.invoke({"question": question})
        ranked = state.get("ranked", [])
        sources = sorted({h.chunk.source for h in ranked}) if state["route"] == "answered" else []
        return Reply(state["answer"], state["route"], sources, ranked)
