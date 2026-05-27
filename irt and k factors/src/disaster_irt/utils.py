import ast
import json
import math
from typing import Any, Dict

import numpy as np
from scipy.special import expit, logit as scipy_logit


def parse_maybe_dict(x: Any) -> Dict[str, Any]:
    """Parse dict-like strings from CSV cells.

    The metadata CSV stores household_attributes / feature_key as Python literal strings.
    This function also accepts JSON strings or already parsed dictionaries.
    """
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    if isinstance(x, float) and math.isnan(x):
        return {}
    if not isinstance(x, str):
        return {}
    s = x.strip()
    if not s:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    return {}


def safe_logit(p, eps: float = 1e-4):
    return scipy_logit(np.clip(p, eps, 1 - eps))


def sigmoid(x):
    return expit(x)


def normalize_model_name_from_json_path(path):
    """Infer model name from file/folder names such as results_phi4_baseline_combined.json."""
    import re
    stem = path.stem
    m = re.match(r"results_(.*)_combined$", stem)
    if m:
        return m.group(1)
    return path.parent.name
