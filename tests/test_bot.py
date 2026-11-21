import pytest

from support_bot import SupportBot


@pytest.fixture(scope="module")
def bot():
    return SupportBot(use_llm=False)


def test_answers_from_the_right_document(bot):
    reply = bot.ask("Is home delivery free?")
    assert reply.route == "answered"
    assert reply.sources[0] == "delivery.md"


def test_opening_hours_returns_the_actual_hours(bot):
    for question in ("What are your opening hours?", "what is open hours ?"):
        reply = bot.ask(question)
        assert reply.route == "answered"
        assert "contact.md" in reply.sources
        assert "8:30" in reply.text
        assert "21:00" in reply.text
        assert "Speaking to the pharmacist" not in reply.text


def test_named_product_availability_uses_products_document(bot):
    reply = bot.ask("Do you have paracetamol available?")
    assert reply.route == "answered"
    # The products document explains that live stock is not shown online.
    # The contact document is also relevant because it tells customers how
    # to reach the store/pharmacist for medicine availability questions.
    assert "products.md" in reply.sources
    assert "contact.md" in reply.sources
    assert "live stock levels" in reply.text.lower() or "available" in reply.text.lower()
    assert "Refrigerated products" not in reply.text


def test_refuses_medical_advice(bot):
    reply = bot.ask("What is the dosage of ibuprofen for a child?")
    assert reply.route == "medical_refusal"
    assert not reply.sources


def test_refuses_diagnosis_requests(bot):
    reply = bot.ask("Can you diagnose my illness?")
    assert reply.route == "medical_refusal"
    assert not reply.sources


def test_hands_off_unknown_topics(bot):
    assert bot.ask("Do you sell electric bikes?").route == "handoff"


def test_reranker_scores_are_set(bot):
    reply = bot.ask("Can I return medicines I bought?")
    assert reply.ranked and all(h.rerank is not None for h in reply.ranked)
