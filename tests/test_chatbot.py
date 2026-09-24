import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from chatbot import MetricMedic


def test_chatbot_initialization():
    chatbot = MetricMedic()
    assert len(chatbot.knowledge_base) > 0


def test_power_bi_match():
    chatbot = MetricMedic()

    response = chatbot.generate_response(
        "My Power BI dashboard shows more revenue than my SQL query"
    )

    assert "Power BI" in response


def test_empty_input():
    chatbot = MetricMedic()

    response = chatbot.generate_response("")

    assert "describe" in response.lower()


def test_unknown_question():
    chatbot = MetricMedic()

    response = chatbot.generate_response(
        "How do I bake a chocolate cake?"
    )

    assert "not confident" in response.lower()