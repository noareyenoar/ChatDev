"""Register built-in agent providers."""

from runtime.node.agent.providers.base import ProviderRegistry

from runtime.node.agent.providers.openai_provider import OpenAIProvider
from runtime.local_backends import SimulatedProvider

ProviderRegistry.register(
    "openai",
    OpenAIProvider,
    label="OpenAI",
    summary="OpenAI models via the official OpenAI SDK (responses API)",
)

# Register a local simulated provider for smoke testing and offline development
try:
    ProviderRegistry.register(
        "simulated",
        SimulatedProvider,
        label="Simulated Provider",
        summary="Local deterministic provider for offline development and smoke tests.",
    )
except Exception:
    # avoid breaking imports when registry registration fails
    print("Simulated provider registration skipped")

try:
    from runtime.node.agent.providers.gemini_provider import GeminiProvider
except ImportError:
    GeminiProvider = None

if GeminiProvider is not None:
    ProviderRegistry.register(
        "gemini",
        GeminiProvider,
        label="Google Gemini",
        summary="Google Gemini models via google-genai",
    )
else:
    print("Gemini provider not registered: google-genai library not found.")
