"""Offline evaluation (no API key needed). Compares cosine-only with cosine + reranker.

answerable   -> is the expected source ranked first (top-1) / among the final 3 (recall@3)?
out_of_scope -> does the bot hand off instead of guessing?
medical      -> does the safety check refuse?
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from support_bot import SupportBot  # noqa: E402

CASES = json.loads(Path(__file__).with_name("questions.json").read_text(encoding="utf-8"))
EXPECTED_ROUTE = {"out_of_scope": "handoff", "medical": "medical_refusal"}


def evaluate(reranker: str) -> dict:
    bot = SupportBot(reranker=reranker, use_llm=False)
    top1 = recall = n_answerable = routed = n_routed = 0
    for case in CASES:
        reply = bot.ask(case["question"])
        if case["type"] == "answerable":
            n_answerable += 1
            sources = [h.chunk.source for h in reply.ranked]
            top1 += bool(sources) and sources[0] == case["source"]
            recall += case["source"] in sources
            if not sources or sources[0] != case["source"]:
                print(f"  [{reranker}] top-1 miss: {case['question']!r} -> {sources}")
        else:
            n_routed += 1
            ok = reply.route == EXPECTED_ROUTE[case["type"]]
            routed += ok
            if not ok:
                print(f"  [{reranker}] wrong route: {case['question']!r} -> {reply.route}")
    return {"top1": top1, "recall": recall, "answerable": n_answerable, "routed": routed, "n_routed": n_routed}


def main() -> None:
    print(f"{len(CASES)} questions\n")
    for name in ("none", "lexical"):
        r = evaluate(name)
        print(
            f"{name:8} top-1 {r['top1']}/{r['answerable']}   recall@3 {r['recall']}/{r['answerable']}"
            f"   correct refusals/hand-offs {r['routed']}/{r['n_routed']}"
        )


if __name__ == "__main__":
    main()
