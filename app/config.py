"""Application configuration loaded from environment / .env file.

Exposes a module-level singleton ``settings`` that other modules import.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


# Supported competitor data categories used across the application.
CATEGORIES: list[str] = ["news", "press_release", "review", "price", "job"]


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are read from environment variables (case-insensitive) and from a
    ``.env`` file if present. Unknown keys are ignored.
    """

    # --- LLM provider selection -------------------------------------------
    # Which provider to use for chat/analysis and for embeddings. These are
    # independent: you can mix-and-match (e.g. chat via Ollama, embeddings via
    # OpenAI, or vice-versa). One of: "openai" | "ollama".
    chat_provider: str = "openai"
    embedding_provider: str = "openai"

    # --- OpenAI-compatible LLM access -------------------------------------
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # --- Ollama (OpenAI-compatible API at {host}:11434/v1) ----------------
    # Ollama exposes an OpenAI-compatible endpoint, so we reuse the openai SDK.
    ollama_base_url: str = "http://ollama:11434/v1"
    # Ollama ignores the API key, but the OpenAI SDK requires a non-empty value.
    ollama_api_key: str = "ollama"
    # Chat model used when chat_provider == "ollama" (must be pulled first).
    ollama_model: str = "llama3.1"
    # Embedding model used when embedding_provider == "ollama".
    ollama_embedding_model: str = "nomic-embed-text"

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = "text-embedding-3-small"
    # IMPORTANT: embedding_dim MUST match the vector size produced by the chosen
    # embedding model:
    #   - text-embedding-3-small (OpenAI) -> 1536
    #   - nomic-embed-text       (Ollama) -> 768
    # The pgvector column dimension is fixed at schema-creation time, so changing
    # this value requires recreating the database (drop the pgdata volume).
    embedding_dim: int = 1536

    # --- Database ---------------------------------------------------------
    database_url: str = "postgresql+psycopg://ci_agent:ci_agent_pass@postgres:5432/ci_agent"

    # --- Search -----------------------------------------------------------
    searxng_url: str = "http://searxng:8080"

    # --- Langfuse tracing -------------------------------------------------
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://langfuse-web:3000"

    # --- Domain / collection ---------------------------------------------
    companies: str = "Apple,Microsoft"
    seed_days: int = 30
    collect_interval_hours: int = 6

    # --- Runtime ----------------------------------------------------------
    app_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def companies_list(self) -> list[str]:
        """Return the configured companies as a clean list of names."""
        return [c.strip() for c in self.companies.split(",") if c.strip()]

    @property
    def chat_provider_norm(self) -> str:
        """Normalized chat provider name (trimmed, lower-cased)."""
        return self.chat_provider.strip().lower()

    @property
    def embedding_provider_norm(self) -> str:
        """Normalized embedding provider name (trimmed, lower-cased)."""
        return self.embedding_provider.strip().lower()


# Module-level singleton imported throughout the app.
settings = Settings()
