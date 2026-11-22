
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from support_bot.knowledge import load_chunks
from support_bot.retrieval import CosineRetriever, make_reranker


DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset.json"
TOP_K = 3
CANDIDATE_K = 6
RERANKERS = ["none", "lexical"]


def calculate_metrics(results, expected_sources, k=TOP_K):
    """Calculate source-level Recall@K and reciprocal rank."""
    expected = set(expected_sources)

    ranked_sources = [
        Path(hit.chunk.source).name
        for hit in results[:k]
    ]

    found_sources = set(ranked_sources) & expected
    recall = len(found_sources) / len(expected) if expected else 0.0

    reciprocal_rank = 0.0

    for rank, source in enumerate(ranked_sources, start=1):
        if source in expected:
            reciprocal_rank = 1.0 / rank
            break

    return recall, reciprocal_rank


def evaluate(reranker_name, dataset, retriever, verbose=True):
    """Evaluate one reranker and return summary plus per-question results."""
    reranker = make_reranker(reranker_name)

    total_recall = 0.0
    total_mrr = 0.0
    question_results = []

    if verbose:
        print(f"\nResults for: {reranker_name}")
        print("-" * 65)

    for item in dataset:
        question = item["question"]
        expected_sources = item["expected_sources"]

        candidates = retriever.search(question, k=CANDIDATE_K)
        ranked_hits = reranker.rerank(question, candidates)

        recall, reciprocal_rank = calculate_metrics(
            ranked_hits,
            expected_sources,
        )

        top_hits = ranked_hits[:TOP_K]
        top_sources = [
            Path(hit.chunk.source).name
            for hit in top_hits
        ]

        relevant_rank = next(
            (
                rank
                for rank, hit in enumerate(ranked_hits, start=1)
                if Path(hit.chunk.source).name in set(expected_sources)
            ),
            None,
        )

        question_result = {
            "question": question,
            "expected_sources": expected_sources,
            "retrieved_sources": top_sources,
            "recall_at_3": recall,
            "reciprocal_rank": reciprocal_rank,
            "relevant_rank": relevant_rank,
            "passed": recall > 0.0,
        }

        question_results.append(question_result)

        total_recall += recall
        total_mrr += reciprocal_rank

        if verbose:
            print(f"Question: {question}")
            print(f"Expected: {', '.join(expected_sources)}")
            print(
                f"Recall@{TOP_K}: {recall:.2f} | "
                f"Reciprocal rank: {reciprocal_rank:.2f}"
            )
            print()

    count = len(dataset)

    return {
        "reranker": reranker_name,
        "recall_at_3": total_recall / count if count else 0.0,
        "mrr": total_mrr / count if count else 0.0,
        "question_results": question_results,
    }


def run_evaluation():
    """Load the dataset and evaluate all configured rerankers."""
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    chunks = load_chunks()
    retriever = CosineRetriever(chunks)

    results = []

    for reranker_name in RERANKERS:
        results.append(
            evaluate(
                reranker_name,
                dataset,
                retriever,
                verbose=False,
            )
        )

    return {
        "question_count": len(dataset),
        "chunk_count": len(chunks),
        "results": results,
    }


def main():
    report = run_evaluation()

    print(f"Evaluation questions: {report['question_count']}")
    print(f"Knowledge chunks: {report['chunk_count']}")

    for result in report["results"]:
        print(f"\nResults for: {result['reranker']}")
        print("-" * 65)

        for item in result["question_results"]:
            print(f"Question: {item['question']}")
            print(f"Expected: {', '.join(item['expected_sources'])}")
            print(
                f"Recall@{TOP_K}: {item['recall_at_3']:.2f} | "
                f"Reciprocal rank: {item['reciprocal_rank']:.2f}"
            )
            print()

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"{'Reranker':<15} {'Recall@3':>12} {'MRR':>12}")

    for result in report["results"]:
        print(
            f"{result['reranker']:<15}"
            f"{result['recall_at_3']:>12.3f}"
            f"{result['mrr']:>12.3f}"
        )


if __name__ == "__main__":
    main()
