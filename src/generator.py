import json
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

def refine_query(query: str) -> str:
    """
    Uses Gemini to refine vague queries into clear, search-optimized queries.
    If the query is already clear, it is kept the same.
    """
    prompt = f"""
You are a search query refinement assistant. Your job is to improve a user's query so it can be effectively used to search a knowledge base.

Rules:
1. If the query is vague, short, or ambiguous (e.g., "AI?"), expand it into a clear, formal search question.
2. If the query is a clear, functional question, return it EXACTLY as is.
3. Output ONLY the refined query string. No explanations, no quotes.

Original Query: {query}
"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        refined = response.text.strip()
        if refined.startswith('"') and refined.endswith('"'):
            refined = refined[1:-1]
            
        # Logging: print original and refined query
        print(f"Original sequence: '{query}' => Refined: '{refined}'")
        logger.info(f"Original Query: '{query}' | Refined Query: '{refined}'")
        return refined
    except Exception as e:
        logger.error(f"Refinement API failure, falling back to original: {e}")
        # Logging: fallback
        print(f"Original sequence: '{query}' => Refined: '{query}' (fallback)")
        logger.info(f"Original Query: '{query}' | Refined Query: '{query}' (fallback)")
        return query

def _extract_json(text: str) -> dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def generate_response(query: str, context_chunks: list[str], level: str = "beginner", assistant_mode: str = "assist") -> dict[str, object]:
    """
    Generates a structured response using Gemini API based on the query, context, and user mode.
    """
    if not context_chunks:
        logger.warning("Empty context provided to generator. Cannot ground response.")
        return {
            "understanding": "No source content was available to answer this request.",
            "key_points": ["The system could not find relevant source material."],
            "explanation": "I am unable to ground the response in a source, so I can only provide general knowledge.",
            "real_world_example": "",
            "next_steps": ["Upload related content or ask a more specific question."],
        }

    context_text = "\n\n".join(context_chunks)

    if level == "expert":
        explanation_instructions = (
            "Provide a highly technical explanation suitable for an expert. "
            "Include detailed reasoning, architecture, and technical terms. "
            "Base your answer ONLY on the provided context."
        )
    elif level == "normal":
        explanation_instructions = (
            "Provide a balanced, clear explanation with moderate technical depth. "
            "Use real-world examples to make the answer relatable. "
            "Base your answer ONLY on the provided context."
        )
    else:
        explanation_instructions = (
            "Provide a very simple explanation suitable for a beginner. "
            "Use an analogy and say it like I'm 5. "
            "Include a short real-world example if appropriate. "
            "Base your answer ONLY on the provided context."
        )

    if assistant_mode == "teach":
        assistant_prompt = (
            "You are a developer learning assistant that teaches concepts step-by-step. "
            "Explain the topic in a structured way, and suggest a follow-up learning action."
        )
    else:
        assistant_prompt = (
            "You are a developer code assistant that gives practical help and troubleshooting advice. "
            "Focus on actionable guidance, code fixes, or steps the user can take next."
        )

    prompt = f"""
You are a domain-specific developer knowledge copilot. Use the following context to answer the user's request.

Context Information:
{context_text}

User Query: {query}

{assistant_prompt}

Instructions:
{explanation_instructions}

Output JSON only with these fields:
{{
  "understanding": "...",
  "key_points": ["..."],
  "explanation": "...",
  "real_world_example": "...",
  "next_steps": ["..."]
}}
"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        payload = _extract_json(response.text)
        if payload is not None:
            return {
                "understanding": str(payload.get("understanding", "")),
                "key_points": list(payload.get("key_points", [])),
                "explanation": str(payload.get("explanation", "")),
                "real_world_example": str(payload.get("real_world_example", "")),
                "next_steps": list(payload.get("next_steps", [])),
            }
        logger.warning("Generator returned non-JSON output, falling back to text parsing.")
        return {
            "understanding": query,
            "key_points": [],
            "explanation": response.text.strip(),
            "real_world_example": "",
            "next_steps": ["Review the response above for next steps."],
        }
    except Exception as e:
        logger.error(f"Generator API failure: {str(e)}")
        raise RuntimeError("Generator API failure")
