"""Factory helpers for LLM clients."""

from langchain_openai import ChatOpenAI

from src.config import ConfigLoader


def get_llm(
    config: ConfigLoader,
    timeout_default: int = 120,
    section: str = "llm",
) -> ChatOpenAI:
    """Return a configured ``ChatOpenAI`` client from shared config."""
    config_section = section if config.get(section, "url") is not None else "llm"

    llm_url = config.get(config_section, "url", default="http://localhost:8081")
    llm_model = config.get(config_section, "model_name", default="llama3")
    llm_temp = float(config.get(config_section, "temperature", default=0))
    llm_timeout = int(config.get(config_section, "timeout", default=timeout_default))
    base_url = f"{llm_url.rstrip('/')}/v1"

    return ChatOpenAI(
        base_url=base_url,
        api_key="not-needed",
        model=llm_model,
        temperature=llm_temp,
        timeout=llm_timeout,
    )
