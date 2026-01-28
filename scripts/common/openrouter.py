"""
OpenRouter API client for LLM access.

This module provides a unified interface for calling LLM models through
OpenRouter's API. It handles authentication, request formatting, and
response parsing.

Constants:
    OPENROUTER_URL: API endpoint
    OPENROUTER_API_KEY_ENV: Environment variable name for API key

Functions:
    call_openrouter: Send chat completion request to OpenRouter
"""

from typing import Dict, List

import requests

# OpenRouter API configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

# Default request parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 500
DEFAULT_TIMEOUT_S = 120


def call_openrouter(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> str:
    """
    Call OpenRouter API to generate LLM response.

    Sends a chat completion request to OpenRouter's unified LLM API.

    Args:
        api_key: OpenRouter API key.
        model: Model identifier (e.g., "anthropic/claude-sonnet-4.5").
        messages: List of message dicts with "role" and "content" keys.
        temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.
        max_tokens: Maximum tokens in response.
        timeout_s: Request timeout in seconds.

    Returns:
        Response content as string.

    Raises:
        requests.HTTPError: If API returns error status.
        requests.Timeout: If request exceeds timeout.

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> response = call_openrouter(api_key, "anthropic/claude-sonnet-4.5", messages)
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
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()
