"""videorag.evaluation — temporal IoU metrics and harness."""
from videorag.evaluation.evaluation import (  # noqa: F401
    GOLD_QUERIES, evaluate_grounding, iou,
)
__all__ = ["GOLD_QUERIES","iou","evaluate_grounding"]
