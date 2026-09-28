import os
from langchain_mistralai import ChatMistralAI
from langchain_core.rate_limiters import InMemoryRateLimiter

# Shared across every chain so all Mistral calls together stay under the API rate limit
# (free tier allows ~1 request/second).
RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=0.8,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def get_llm(temperature: float = 0.3) -> ChatMistralAI:
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "open-mistral-nemo"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
        rate_limiter=RATE_LIMITER,
        max_retries=6,  # retries network errors only, not 429s
    )
