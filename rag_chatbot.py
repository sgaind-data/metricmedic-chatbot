from rag import MetricMedicRetriever
from llm import MetricMedicLLM


class MetricMedicRAG:
    """Full RAG pipeline: retrieval + grounded LLM generation."""

    def __init__(self):
        self.retriever = MetricMedicRetriever()
        self.llm = MetricMedicLLM()

    def answer(self, query, top_k=3):
        retrieved_results = self.retriever.retrieve(
            query,
            top_k=top_k,
        )

        answer = self.llm.generate_answer(
            query,
            retrieved_results,
        )

        return {
            "answer": answer,
            "sources": retrieved_results,
        }


if __name__ == "__main__":
    chatbot = MetricMedicRAG()

    query = (
        "My dashboard customer count doubled after joining two tables"
    )

    result = chatbot.answer(query)

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nRetrieved Issues:")

    for source in result["sources"]:
        print(
            f"- {source['issue']} "
            f"({source['score']:.4f})"
        )