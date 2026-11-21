import pytest

from support_bot import SupportBot


@pytest.fixture(scope="module")
def bot():
    return SupportBot(use_llm=False)


def test_greeting_returns_a_welcome_without_retrieval(bot):
    for question in ("hello", "Hi!", "good morning"):
        reply = bot.ask(question)
        assert reply.route == "greeting"
        assert "Welcome to CareWell" in reply.text
        assert not reply.sources


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


def test_named_product_availability_explains_live_stock_limitation(bot):
    reply = bot.ask("Do you have paracetamol available?")
    assert reply.route == "answered"
    assert "products.md" in reply.sources
    assert "contact.md" in reply.sources
    assert "cannot confirm" in reply.text.lower()
    assert "live stock levels" in reply.text.lower()
    assert "+49 681 000111" in reply.text
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
