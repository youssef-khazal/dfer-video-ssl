"""DFER metrics. The field reports both:

* **WAR** (weighted average recall) = plain accuracy, dominated by frequent classes.
* **UAR** (unweighted average recall) = mean per-class recall, every emotion counts equally.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def war(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_recall(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(support > 0, np.diag(cm) / support, np.nan)


def uar(y_true, y_pred, num_classes: int) -> float:
    """Mean recall over the classes present in ``y_true``."""
    return float(np.nanmean(per_class_recall(y_true, y_pred, num_classes)))
