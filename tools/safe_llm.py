from typing import Any, List


def safe_llm_invoke(
    llm: Any,
    messages: List[Any],
    agent_name: str = "Agent",
    fallback_text: str = "LLM unavailable",
) -> str:
    try:
        response = llm.invoke(messages)

        if hasattr(response, "content"):
            return str(response.content).strip()

        return str(response).strip()

    except Exception as e:
        return f"[{agent_name} fallback] {fallback_text}. Reason: {str(e)}"
