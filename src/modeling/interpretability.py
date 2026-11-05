"""Model interpretation (associations, NOT causes).

- Logistic regression: standardised coefficients + stability across student bootstrap refits.
- Any model: permutation importance on the VALIDATION subset (scoring = PR-AUC), per feature
  and per feature group (columns of a group permuted jointly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

SEED = 42

CUMULATIVE = ["eng_total_clicks", "eng_active_days", "eng_clicks_per_active_day", "eng_clicks_per_elapsed_day",
              "eng_distinct_sites", "eng_distinct_activity_types", "eng_activity_type_entropy",
              "eng_active_weeks", "eng_inactive_weeks", "eng_active_weeks_ratio", "eng_clicks_pct_in_presentation"]
RECENT = ["eng_clicks_last_14_days", "eng_active_days_last_14_days", "eng_days_since_last_activity",
          "eng_last_week_change", "eng_weekly_trend_slope", "eng_weekly_clicks_std", "eng_weekly_clicks_cv"]
PRE_START = ["eng_clicks_pre_start", "eng_active_days_pre_start", "eng_first_activity_day"]


def feature_groups(cols: list[str], meta: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(f"group:{meta['feature_groups'].get(c, 'other')}", []).append(c)
    weekly = sorted(c for c in cols if c.startswith("eng_clicks_week_"))
    groups["engagement:cumulative"] = [c for c in CUMULATIVE if c in cols]
    groups["engagement:recent"] = [c for c in RECENT if c in cols] + weekly[-1:]
    groups["engagement:pre_start"] = [c for c in PRE_START if c in cols]
    groups["engagement:weekly_trajectory"] = weekly
    return {k: v for k, v in groups.items() if v}


def lr_coefficients(pipe) -> pd.DataFrame:
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_.ravel()
    return pd.DataFrame({"feature": names, "coef": coef}).sort_values("coef", key=np.abs, ascending=False)


def lr_report(pipe, train_df, y_train, cols, meta, params, n_boot: int = 30) -> dict:
    """Coefficients of the fitted LR + sign stability over student-bootstrap refits of the same config."""
    from src.modeling.train import make_lr  # local import avoids a cycle
    base = lr_coefficients(pipe).set_index("feature")["coef"]
    rng = np.random.default_rng(SEED)
    students = train_df["student_id"].to_numpy()
    uniq = np.unique(students)
    rows_by = pd.Series(np.arange(len(students))).groupby(np.searchsorted(uniq, students)).apply(np.array).to_list()
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_by[i] for i in rng.integers(0, len(uniq), len(uniq))])
        m = make_lr(meta, cols, **params).fit(train_df.iloc[idx][cols], np.asarray(y_train)[idx])
        draws.append(lr_coefficients(m).set_index("feature")["coef"])
    D = pd.concat(draws, axis=1).reindex(base.index).fillna(0.0)
    out = pd.DataFrame({
        "coef": base, "boot_mean": D.mean(axis=1), "boot_ci_low": D.quantile(0.025, axis=1),
        "boot_ci_high": D.quantile(0.975, axis=1),
        "sign_agreement": (np.sign(D).eq(np.sign(base), axis=0)).mean(axis=1),
    })
    out["stable"] = (out["sign_agreement"] >= 0.9) & (out["coef"] != 0)
    out = out.reindex(out["coef"].abs().sort_values(ascending=False).index)
    return {"n_bootstrap": n_boot, "n_nonzero": int((out["coef"] != 0).sum()),
            "coefficients": out.reset_index().rename(columns={"index": "feature"}).round(4).to_dict("records")}


def permutation_report(model, X_val, y_val, cols, meta, n_repeats: int = 5) -> dict:
    y_val = np.asarray(y_val)
    base = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    pi = permutation_importance(model, X_val, y_val, scoring="average_precision", n_repeats=n_repeats,
                                random_state=SEED)
    per_feature = (pd.DataFrame({"feature": cols, "importance_mean": pi.importances_mean,
                                 "importance_std": pi.importances_std})
                   .sort_values("importance_mean", ascending=False).round(5))
    rng = np.random.default_rng(SEED)
    grp = {}
    for name, members in feature_groups(cols, meta).items():
        drops = []
        for _ in range(n_repeats):
            Xp = X_val.copy()
            perm = rng.permutation(len(Xp))
            for c in members:  # same permutation for every member: joint shuffle, dtypes preserved
                Xp[c] = X_val[c].to_numpy()[perm]
            drops.append(base - average_precision_score(y_val, model.predict_proba(Xp)[:, 1]))
        grp[name] = {"n_features": len(members), "pr_auc_drop_mean": float(np.mean(drops)),
                     "pr_auc_drop_std": float(np.std(drops))}
    return {"validation_pr_auc": float(base), "per_feature": per_feature.to_dict("records"),
            "per_group": dict(sorted(grp.items(), key=lambda kv: -kv[1]["pr_auc_drop_mean"]))}
