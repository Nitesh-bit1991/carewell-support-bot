"""Stage 1: cosine similarity over TF-IDF vectors. Stage 2: rerank the candidates."""
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Protocol

from .knowledge import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "is", "are", "it", "for", "on",
    "with", "as", "by", "that", "this", "be", "at", "from", "what", "how", "do",
    "does", "why", "can", "you", "i", "my", "me", "we", "your", "have", "get",
    "which", "who", "whom", "when", "where", "our", "if", "will", "would", "should",
    "was", "were", "has", "had", "but", "so", "than", "then", "there", "they",
    "them", "their", "been", "not",
}


def _stem(word: str) -> str:
    """Tiny suffix stripper so 'returns' and 'returning' both match 'return'."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def has_lexical_overlap(query: str, chunk: Chunk) -> bool:
    """Reject passages that share no meaningful terms with the user's question."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return False
    passage_terms = set(tokenize(f"{chunk.heading} {chunk.text}"))
    return bool(query_terms & passage_terms)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    cosine: float                  # stage 1 score
    rerank: float | None = None    # stage 2 score (None until reranked)


class CosineRetriever:
    """TF-IDF vectors compared with cosine similarity (pure Python, no downloads)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        counts = [Counter(tokenize(c.text)) for c in chunks]
        doc_freq: Counter = Counter()
        for c in counts:
            doc_freq.update(c.keys())
        n = len(chunks)
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in doc_freq.items()}
        self.vectors = [self._vectorize(c) for c in counts]

    def _vectorize(self, counts: Counter) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 6) -> list[Hit]:
        q = self._vectorize(Counter(tokenize(query)))
        hits = [
            Hit(chunk, sum(w * vec.get(t, 0.0) for t, w in q.items()))
            for vec, chunk in zip(self.vectors, self.chunks)
        ]
        hits.sort(key=lambda h: -h.cosine)
        return [h for h in hits[:k] if h.cosine > 0]


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class NoReranker:
    """Keeps the cosine order. Useful as a baseline in the evaluation."""

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        return [replace(h, rerank=h.cosine) for h in hits]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


class LexicalReranker:
    """Light reranker with no model download.

    Blends cosine score with query-term coverage, phrase evidence and heading overlap.
    """

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        q = tokenize(query)
        q_words, q_pairs = set(q), _bigrams(q)
        ranked = []
        for h in hits:
            doc = tokenize(h.chunk.text)
            coverage = len(q_words & set(doc)) / len(q_words) if q_words else 0.0
            pairs = len(q_pairs & _bigrams(doc)) / len(q_pairs) if q_pairs else 0.0
            heading = len(q_words & set(tokenize(h.chunk.heading))) / len(q_words) if q_words else 0.0
            score = 0.35 * h.cosine + 0.35 * coverage + 0.15 * pairs + 0.15 * heading
            ranked.append(replace(h, rerank=score))
        return sorted(ranked, key=lambda h: -(h.rerank or 0.0))


class CrossEncoderReranker:
    """Neural reranker (optional): pip install sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        ranked = [replace(h, rerank=float(s)) for h, s in zip(hits, scores)]
        return sorted(ranked, key=lambda h: -(h.rerank or 0.0))


def make_reranker(name: str) -> Reranker:
    options = {"none": NoReranker, "lexical": LexicalReranker, "cross-encoder": CrossEncoderReranker}
    if name not in options:
        raise ValueError(f"Unknown reranker {name!r}. Choose from {sorted(options)}")
    return options[name]()
