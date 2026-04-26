import os
import yaml
from langchain_core.language_models.chat_models import BaseChatModel


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_llm(config: dict | None = None) -> BaseChatModel:
    if config is None:
        config = load_config()

    llm_cfg = config["llm"]
    provider = llm_cfg["provider"]
    model = llm_cfg["model"]
    temperature = llm_cfg.get("temperature", 0)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.environ["OPENAI_API_KEY"],
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    elif provider == "ollama":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=llm_cfg.get("base_url", "http://localhost:11434/v1"),
            api_key="ollama",
        )

    else:
        raise ValueError(f"Unknown provider: '{provider}'. Options: openai, anthropic, ollama")