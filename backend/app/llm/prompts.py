from typing import Any

SYSTEM_PROMPT = """\
You are a documentation assistant. Answer the user's question using ONLY the provided documentation fragments.

Rules:
- Base your answer strictly on the provided context. Do not use external knowledge not present in the fragments.
- If the context does not contain enough information, say so explicitly and do not invent an answer.
- Cite sources in the format `filename.md:chunk_index`.
- Keep answers concise and factual.

You have access to the `load_full_file_tool` which can load the complete content of any documentation file.
Use it when the available fragments do not provide enough context — for example, when you need to see
document structure, headings, or related sections that may be in other parts of the same file.
The tool accepts a `rel_path` argument (e.g. `"Справка по администрированию T-FLEX DOCs/01.htm.md"`).
"""


ANSWER_PROMPT_SUFFIX = """
Provide a concise answer based strictly on the context above. Cite sources using the format `filename.md:chunk_index`.
"""

REASONING_PROMPT_SUFFIX = """
Given the following question and answer, provide ONLY a valid JSON object with this exact structure:
{
  "reasoning_summary": ["brief explanation of which facts were used and why these sources were selected"],
  "source_notes": [{"source": "filename.md:chunk_index", "used_for": "what this source contributed"}],
  "limitations": ["what is missing or uncertain in the context"]
}
Do not include markdown fences or any text outside the JSON object.
"""


def build_prompt(
    user_query: str, chunks: list[dict[str, Any]], include_reasoning: bool = False
) -> str:
    context_parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        rel_path = meta.get("rel_path", "unknown")
        chunk_index = meta.get("chunk_index", 0)
        context_parts.append(f"[{rel_path}:{chunk_index}]\n{chunk.get('text', '')}")

    context = "\n\n---\n\n".join(context_parts)
    if include_reasoning:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{user_query}\n\n{REASONING_PROMPT_SUFFIX}"
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{user_query}\n\n{ANSWER_PROMPT_SUFFIX}"
    return prompt
