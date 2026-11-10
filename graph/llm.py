"""
Gemini on Vertex AI, authenticated with gcloud application-default credentials.

  VERTEX_FAST_MODEL   intake, planner, critic        (default gemini-3.5-flash-lite)
  VERTEX_SMART_MODEL  claim mapping, report          (default gemini-3.8-flash)
  VERTEX_LOCATION     default "global" (these models are not served from us-central1)
  GOOGLE_CLOUD_PROJECT  default: the gcloud / ADC project
"""
import os
from functools import lru_cache

FAST_MODEL = os.environ.get("VERTEX_FAST_MODEL", "gemini-3.5-flash-lite")
SMART_MODEL = os.environ.get("VERTEX_SMART_MODEL", "gemini-3.8-flash")


@lru_cache(maxsize=None)
def get_llm(tier: str = "fast"):
    import google.auth
    from langchain_google_genai import ChatGoogleGenerativeAI

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or google.auth.default()[1]
    return ChatGoogleGenerativeAI(
        model=SMART_MODEL if tier == "smart" else FAST_MODEL,
        vertexai=True,
        project=project,
        location=os.environ.get("VERTEX_LOCATION", "global"),
        temperature=0,
        max_retries=3,
    )


def structured(schema, tier: str = "fast"):
    return get_llm(tier).with_structured_output(schema)
