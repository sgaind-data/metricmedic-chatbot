import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).parent / "data" / "troubleshooting_guide.json"


class MetricMedic:
    """Retrieval-based chatbot for troubleshooting data and dashboard issues."""

    def __init__(self, data_path=DATA_PATH):
        self.knowledge_base = self.load_knowledge_base(data_path)

        self.documents = [
            f"{item['issue']} {item['keywords']}"
            for item in self.knowledge_base
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.document_vectors = self.vectorizer.fit_transform(self.documents)

    @staticmethod
    def load_knowledge_base(data_path):
        """Load troubleshooting scenarios from JSON."""
        with open(data_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def find_best_match(self, user_input):
        """Find the most similar troubleshooting scenario."""
        user_vector = self.vectorizer.transform([user_input])

        similarities = cosine_similarity(
            user_vector,
            self.document_vectors
        )[0]

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        return self.knowledge_base[best_index], best_score

    def generate_response(self, user_input):
        """Generate a chatbot response."""
        if not user_input.strip():
            return "Please describe the data issue you're experiencing."

        match, score = self.find_best_match(user_input)

        if score < 0.10:
            return (
                "I'm not confident that I recognize this issue yet. "
                "Try mentioning the dashboard, SQL, KPI, joins, duplicates, "
                "NULL values, filters, or refresh issue in more detail."
            )

        return (
            f"\nPossible issue: {match['issue']}\n\n"
            f"Recommended checks:\n{match['response']}"
        )


def main():
    chatbot = MetricMedic()

    print("=" * 55)
    print("MetricMedic - Data Troubleshooting Chatbot")
    print("=" * 55)
    print(
        "Describe a dashboard, SQL, KPI, or data-quality issue.\n"
        "Type 'exit' to close the chatbot."
    )

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("\nMetricMedic: Happy debugging!")
            break

        response = chatbot.generate_response(user_input)
        print(f"\nMetricMedic:{response}")


if __name__ == "__main__":
    main()