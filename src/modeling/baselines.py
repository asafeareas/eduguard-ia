"""Simple reference models. All are fitted on the training subset only.

Each exposes fit(X, y) / predict_proba(X) -> (n, 2), like sklearn classifiers,
so they go through exactly the same evaluation code and data split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class PrevalenceBaseline:
    """Constant probability = positive rate in the training data."""

    name = "baseline_prevalence"

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])

    def params(self):
        return {"train_rate": self.rate_}


class ModuleRateBaseline:
    """Historical withdrawal rate of the module (from training rows only).

    Modules with fewer than `min_count` training rows fall back to the global training rate.
    """

    name = "baseline_module_rate"

    def __init__(self, column: str = "ctx_code_module", min_count: int = 30):
        self.column, self.min_count = column, min_count

    def fit(self, X, y):
        df = pd.DataFrame({"g": X[self.column].to_numpy(), "y": np.asarray(y)})
        stats = df.groupby("g")["y"].agg(["mean", "size"])
        self.global_ = float(df["y"].mean())
        self.rates_ = stats.loc[stats["size"] >= self.min_count, "mean"].to_dict()
        return self

    def predict_proba(self, X):
        p = X[self.column].map(self.rates_).fillna(self.global_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def params(self):
        return {"rates": {str(k): round(v, 4) for k, v in self.rates_.items()}, "global": self.global_}


class InactivityRuleBaseline:
    """Rule: high risk if no VLE activity in the last N days observed (days_since_last_activity >= N).

    N is chosen on the VALIDATION subset (max F1). The two groups' probabilities are the
    positive rates observed in the TRAINING subset, so the rule also yields probabilities.
    Uses only the as-of recency feature (never the last activity day of the full module).
    """

    name = "baseline_inactivity_rule"
    column = "eng_days_since_last_activity"

    def __init__(self, n_days: int = 7):
        self.n_days = n_days

    def flag(self, X):
        return (X[self.column].to_numpy() >= self.n_days).astype(int)

    def fit(self, X, y):
        y = np.asarray(y)
        f = self.flag(X)
        overall = float(y.mean())
        self.p_flag_ = float(y[f == 1].mean()) if (f == 1).any() else overall
        self.p_noflag_ = float(y[f == 0].mean()) if (f == 0).any() else overall
        return self

    def predict_proba(self, X):
        p = np.where(self.flag(X) == 1, self.p_flag_, self.p_noflag_)
        return np.column_stack([1 - p, p])

    @classmethod
    def select_n(cls, X_train, y_train, X_val, y_val, candidates):
        """Pick N maximising F1 of the binary rule on validation; ties -> smaller N."""
        scores = {}
        for n in candidates:
            rule = cls(n).fit(X_train, y_train)
            scores[int(n)] = float(f1_score(y_val, rule.flag(X_val), zero_division=0))
        best = max(scores, key=lambda n: (scores[n], -n))
        return cls(best).fit(X_train, y_train), scores

    def params(self):
        return {"n_days": self.n_days, "p_flag": self.p_flag_, "p_noflag": self.p_noflag_}
