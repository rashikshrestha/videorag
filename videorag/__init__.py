"""
videorag
~~~~~~~~
Multimodal RAG system for video temporal grounding.

Typical usage::

    from videorag.api import build_context, run_video_grounding

    ctx = build_context("config.yaml")
    result = run_video_grounding("Chandler Ross Joey watching ice hockey", ctx=ctx)
    print(result)
"""
from videorag.api import build_context, run_video_grounding  # noqa: F401

__version__ = "0.1.0"
__all__ = ["build_context", "run_video_grounding"]
