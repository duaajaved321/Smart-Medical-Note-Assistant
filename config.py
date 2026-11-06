"""Configuration and Gemini client initialization."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-2.5-flash"


def get_api_config() -> str:
    """Return the configured Gemini API key or raise a clear error."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to a local .env file.")
    return api_key


def get_model() -> str:
    """Read the model name from the environment."""
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def create_llm_client() -> OpenAI:
    """Create an OpenAI-compatible client for the Gemini API."""
    return OpenAI(api_key=get_api_config(), base_url=BASE_URL, timeout=60.0, max_retries=0)


MODEL = get_model()
