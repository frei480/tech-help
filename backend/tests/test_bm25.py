from app.search.bm25 import rank_documents


def test_rank_documents_prefers_exact_architecture_terms() -> None:
    documents = [
        {
            "text": "T-FLEX DOCs provides an API for CAD integration.",
            "metadata": {"rel_path": "api.txt"},
        },
        {
            "text": (
                "В основе комплекса лежит трехуровневая архитектура. "
                "Данные обрабатывает сервер приложений."
            ),
            "metadata": {"rel_path": "architecture.txt"},
        },
        {
            "text": "The web server provides user pages.",
            "metadata": {"rel_path": "web.txt"},
        },
    ]

    hits = rank_documents(
        "Какая архитектура лежит в основе T-FLEX DOCs? "
        "трёхуровневая архитектура сервер приложений",
        documents,
        top_k=2,
    )

    assert hits[0]["metadata"]["rel_path"] == "architecture.txt"
    assert hits[0]["bm25_score"] > hits[1]["bm25_score"]


def test_rank_documents_returns_empty_for_no_matching_terms() -> None:
    hits = rank_documents(
        "неизвестный термин",
        [{"text": "Документ об архитектуре", "metadata": {"rel_path": "doc.txt"}}],
    )

    assert hits == []
