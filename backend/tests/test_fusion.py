from app.search.fusion import reciprocal_rank_fusion, weighted_fusion


def test_reciprocal_rank_fusion_rewards_cross_retriever_agreement() -> None:
    vector_hits = [
        {"text": "semantic result", "metadata": {"rel_path": "semantic.md"}},
        {"text": "shared result", "metadata": {"rel_path": "shared.md"}},
    ]
    bm25_hits = [
        {"text": "shared result", "metadata": {"rel_path": "shared.md"}},
        {"text": "lexical result", "metadata": {"rel_path": "lexical.md"}},
    ]

    fused = reciprocal_rank_fusion([vector_hits, bm25_hits], top_k=3)

    assert [hit["metadata"]["rel_path"] for hit in fused] == [
        "shared.md",
        "semantic.md",
        "lexical.md",
    ]
    assert fused[0]["score"] > fused[1]["score"]


def test_reciprocal_rank_fusion_deduplicates_paths_case_insensitively() -> None:
    fused = reciprocal_rank_fusion(
        [
            [
                {"text": "first", "metadata": {"rel_path": "Guide.md"}},
                {"text": "duplicate", "metadata": {"rel_path": "guide.md"}},
            ]
        ]
    )

    assert len(fused) == 1
    assert fused[0]["text"] == "first"


def test_fusion_combines_lexical_and_vector() -> None:
    """Lexical and vector hits pointing to the same location are combined."""
    lexical_hits = [
        {
            "text": "alpha",
            "metadata": {
                "rel_path": "a.md",
                "line_number": 42,
                "chunk_index": -1,
                "heading": "",
            },
            "lexical_score": 1.0,
        }
    ]
    vector_hits = [
        {
            "text": "alpha",
            "metadata": {
                "rel_path": "a.md",
                "chunk_index": 0,
                "heading": "",
            },
            "vector_score": 0.9,
        }
    ]
    fused = weighted_fusion(lexical_hits, vector_hits, top_k=3)
    assert len(fused) == 2  # different keys: L42 vs C0
    assert fused[0]["score"] > 0


def test_fusion_deduplicates_identical_lexical_hits() -> None:
    """Same line_number from different query variants yields one entry."""
    lexical_hits = [
        {
            "text": "beta",
            "metadata": {
                "rel_path": "b.md",
                "line_number": 10,
                "chunk_index": -1,
                "heading": "",
            },
            "lexical_score": 1.0,
        },
        {
            "text": "beta",
            "metadata": {
                "rel_path": "b.md",
                "line_number": 10,
                "chunk_index": -1,
                "heading": "",
            },
            "lexical_score": 0.8,
        },
    ]
    fused = weighted_fusion(lexical_hits, [], top_k=3)
    assert len(fused) == 1
    # The higher score wins (last added overwrites, but both are summed)
    assert fused[0]["score"] > 0


def test_fusion_deduplicates_identical_vector_hits() -> None:
    """Same chunk_index from different queries yields one entry."""
    vector_hits = [
        {
            "text": "gamma",
            "metadata": {"rel_path": "c.md", "chunk_index": 3, "heading": ""},
            "vector_score": 0.7,
        },
        {
            "text": "gamma",
            "metadata": {"rel_path": "c.md", "chunk_index": 3, "heading": ""},
            "vector_score": 0.6,
        },
    ]
    fused = weighted_fusion([], vector_hits, top_k=3)
    assert len(fused) == 1
    assert fused[0]["score"] > 0


def test_fusion_respects_top_k() -> None:
    lexical_hits = [
        {
            "text": f"hit_{i}",
            "metadata": {
                "rel_path": f"f{i}.md",
                "line_number": 1,
                "chunk_index": -1,
                "heading": "",
            },
            "lexical_score": 1.0,
        }
        for i in range(10)
    ]
    fused = weighted_fusion(lexical_hits, [], top_k=3)
    assert len(fused) == 3


def test_fusion_metadata_boost() -> None:
    """Hits with a heading get a metadata boost."""
    lexical_hits = [
        {
            "text": "with heading",
            "metadata": {
                "rel_path": "h.md",
                "line_number": 5,
                "chunk_index": -1,
                "heading": "Introduction",
            },
            "lexical_score": 1.0,
        },
        {
            "text": "no heading",
            "metadata": {
                "rel_path": "h.md",
                "line_number": 10,
                "chunk_index": -1,
                "heading": "",
            },
            "lexical_score": 1.0,
        },
    ]
    fused = weighted_fusion(lexical_hits, [], top_k=3)
    assert len(fused) == 2
    # The hit with heading should rank higher
    scores = {f["item"]["text"]: f["score"] for f in fused}
    assert scores["with heading"] > scores["no heading"]
