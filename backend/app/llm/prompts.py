from typing import Any

SYSTEM_PROMPT = """\
You are a documentation research assistant. Answer the user's question using ONLY the provided documentation context.

Rules:
- Before drafting an answer, identify what the user needs to know and split multi-part questions
  into the key subquestions that the available context can answer.
- Check the context for evidence relevant to each subquestion before drawing conclusions. Compare
  relevant documents when available, and distinguish corroborated facts from claims supported by
  only one source or from conflicting evidence.
- The available research corpus is indexed local documentation; you do not have web search here.
  Do not claim to have checked current or external sources. If the question requires information
  beyond the provided documentation, state that limitation explicitly.
- If an ambiguity prevents a reliable answer and cannot be resolved from the context, ask one
  concise clarifying question instead of guessing.
- Do not use knowledge outside the provided context or invent missing facts.
- Give a complete, detailed answer to every part of the question using all relevant information
  in the context. Explain important terms, relationships, conditions, and implications when the
  documentation supports them. For procedures, include the documented steps, prerequisites, and
  cautions; for comparisons, describe the relevant similarities and differences; include examples
  when the documentation provides them.
- Organize longer answers with headings, lists, or numbered steps so they are easy to follow.
  Avoid repetition and unsupported elaboration, but do not omit relevant documented details merely
  to make the answer shorter.
- Answer in the same language as the user's question.
- Cite factual claims in the format `filename.md:chunk_index`, close to the claims they support.
- Do not reveal internal reasoning; provide conclusions, supporting facts, limitations, and citations.
"""


ANSWER_PROMPT_SUFFIX = """
Research the question using all relevant context before answering. Give the most complete and
detailed answer the indexed documentation supports, covering each part of the question and
explaining documented steps, conditions, terminology, and caveats where relevant. Do not pad the
answer with repetition or information unsupported by the context. Clearly state what cannot be
verified from the indexed documentation, answer in the user's language, and cite factual claims
using the format `filename.md:chunk_index`. Do not reveal internal reasoning.
"""

EVIDENCE_PROMPT_SUFFIX = """
Given the following question and answer, provide ONLY a valid JSON object with this exact structure:
{
  "evidence_summary": ["brief factual support for the answer, without internal reasoning"],
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
        source_label = f"[{rel_path}:{chunk_index}]"
        if meta.get("full_document"):
            source_label += " (complete document; highest-ranked source)"
        context_parts.append(f"{source_label}\n{chunk.get('text', '')}")

    context = "\n\n---\n\n".join(context_parts)
    if include_reasoning:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{user_query}\n\n{EVIDENCE_PROMPT_SUFFIX}"
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{user_query}\n\n{ANSWER_PROMPT_SUFFIX}"
    return prompt
