
from types import SimpleNamespace

from evaluation.evaluate import calculate_metrics


def make_hit(source):
    return SimpleNamespace(
        chunk=SimpleNamespace(source=source)
    )


def test_recall_and_mrr_when_expected_source_is_first():
    results = [
        make_hit("contact.md"),
        make_hit("delivery.md"),
        make_hit("returns.md"),
    ]

    recall, mrr = calculate_metrics(
        results,
        ["contact.md"],
        k=3,
    )

    assert recall == 1.0
    assert mrr == 1.0


def test_recall_and_mrr_when_expected_source_is_second():
    results = [
        make_hit("delivery.md"),
        make_hit("contact.md"),
        make_hit("returns.md"),
    ]

    recall, mrr = calculate_metrics(
        results,
        ["contact.md"],
        k=3,
    )

    assert recall == 1.0
    assert mrr == 0.5


def test_recall_and_mrr_when_expected_source_is_missing():
    results = [
        make_hit("delivery.md"),
        make_hit("returns.md"),
    ]

    recall, mrr = calculate_metrics(
        results,
        ["contact.md"],
        k=3,
    )

    assert recall == 0.0
    assert mrr == 0.0


def test_expected_source_outside_top_k_is_not_counted():
    results = [
        make_hit("delivery.md"),
        make_hit("returns.md"),
        make_hit("products.md"),
        make_hit("contact.md"),
    ]

    recall, mrr = calculate_metrics(
        results,
        ["contact.md"],
        k=3,
    )

    assert recall == 0.0
    assert mrr == 0.0
