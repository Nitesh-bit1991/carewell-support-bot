import pytest

from support_bot.safety import asks_for_medical_advice


@pytest.mark.parametrize("q", [
    "What is the dosage of ibuprofen for a child?",
    "Can I take paracetamol with alcohol?",
    "What are the side effects of metformin?",
    "Which medicine is best for a headache?",
])
def test_medical_questions_are_flagged(q):
    assert asks_for_medical_advice(q)


@pytest.mark.parametrize("q", [
    "Is home delivery free?",
    "Can I use my health insurance?",
    "Do I need a prescription to order antibiotics?",
    "Can I return medicines I bought?",
])
def test_store_questions_are_not_flagged(q):
    assert not asks_for_medical_advice(q)
