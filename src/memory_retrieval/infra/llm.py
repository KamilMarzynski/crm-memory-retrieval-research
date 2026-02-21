import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 500
DEFAULT_TIMEOUT_S = 120


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> str:
    """Call the OpenRouter API and return the assistant message content.

    OpenRouter accepts messages in OpenAI chat format:
      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Args:
        api_key: OpenRouter API key (from OPENROUTER_API_KEY env var).
        model: OpenRouter model ID (e.g. "anthropic/claude-haiku-4.5").
        messages: Conversation messages in OpenAI chat format.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in the response.
        timeout_s: HTTP request timeout in seconds.

    Returns:
        The assistant's response text, stripped of leading/trailing whitespace.

    Raises:
        requests.HTTPError: On non-2xx API responses.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "memory-retrieval-research",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
