"""External-validation tooling: compatibility gate + evaluation WITHOUT retraining.

Rules encoded here:
- A trained model may be applied to external data only if EVERY predictor it uses has status
  `direct` in the correspondence matrix and is present in the data. `transform` features change
  meaning, so feeding them to the original model would not be an external validation of it.
- The saved pipeline/calibrator/threshold are applied as they are (no refit of anything).
- Distribution shift is reported (PSI for numeric features, unseen-category share for nominals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling import evaluation
from src.validation.feature_mapping import DIRECT, by_feature


def compatibility_report(predictors: list[str], available_fields: set[str] | None = None) -> dict:
    """Can a model with these predictors be transferred directly?"""
    mp = by_feature()
    rows, counts = [], {}
    for f in predictors:
        m = mp.get(f)
        status = m.status if m else "unmapped"
        reqs = set(m.requires) if m else set()
        computable = None if available_fields is None else reqs <= set(available_fields)
        rows.append({"feature": f, "status": status, "requires": sorted(reqs), "computable": computable,
                     "brazilian_equivalent": m.brazilian_equivalent if m else None})
        counts[status] = counts.get(status, 0) + 1
    direct_ok = all(r["status"] == DIRECT and r["computable"] is not False for r in rows)
    return {"n_predictors": len(predictors), "status_counts": counts,
            "direct_transfer_possible": bool(direct_ok and available_fields is not None), "features": rows,
            "decision": ("Transferência direta possível" if direct_ok and available_fields is not None else
                         "Transferência direta IMPOSSÍVEL: há preditores sem equivalência direta ou dados ausentes. "
                         "Qualquer modelo adaptado/retreinado NÃO é validação externa do modelo original.")}


def population_stability_index(reference, current, bins: int = 10) -> float:
    """PSI with quantile bins of the reference distribution (>0.25 = large shift, rule of thumb)."""
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def shift_report(reference: pd.DataFrame, current: pd.DataFrame, feature_types: dict) -> list[dict]:
    rows = []
    for col, t in feature_types.items():
        if col not in reference or col not in current:
            continue
        if t == "nominal":
            seen = set(reference[col].dropna().astype(str))
            unseen = (~current[col].dropna().astype(str).isin(seen)).mean() if current[col].notna().any() else 0.0
            rows.append({"feature": col, "type": t, "unseen_category_share": float(unseen)})
        else:
            psi = population_stability_index(reference[col], current[col])
            rows.append({"feature": col, "type": t, "psi": psi,
                         "shift": "grande" if psi > 0.25 else "moderada" if psi > 0.1 else "pequena"})
    return rows


def evaluate_external(bundle: dict, model_name: str, external: pd.DataFrame, target: str,
                      group_cols: list[str] | None = None, reference: pd.DataFrame | None = None,
                      feature_types: dict | None = None, require_compatibility: bool = True) -> dict:
    """Apply a saved model (pipeline + calibrator + threshold) to external labelled data, no refit."""
    predictors = bundle["predictors"]
    if require_compatibility:
        comp = compatibility_report(predictors, set(external.columns))
        missing = [c for c in predictors if c not in external.columns]
        if missing or not comp["direct_transfer_possible"]:
            raise ValueError(f"External data not compatible for direct transfer: {comp['decision']}; "
                             f"missing columns: {missing}")
    model, cal = bundle["models"][model_name], bundle["calibrators"][model_name]
    thr = bundle["thresholds"][model_name]
    p = cal.transform(model.predict_proba(external[predictors])[:, 1])
    y = external[target].to_numpy()
    out = {"model": model_name, "threshold_from_development": thr, "n": int(len(y)),
           "overall": evaluation.full_report(y, p, thr),
           "calibration_table": evaluation.calibration_table(y, p).to_dict("records")}
    for g in group_cols or []:
        res = []
        for level, idx in external.groupby(g).groups.items():
            yy, pp = y[external.index.get_indexer(idx)], p[external.index.get_indexer(idx)]
            r = evaluation.probability_metrics(yy, pp)
            r.update(level=str(level), **{k: v for k, v in evaluation.threshold_metrics(yy, pp, thr).items()
                                         if k in ("recall", "precision")})
            res.append(r)
        out[f"by_{g}"] = res
    if reference is not None and feature_types is not None:
        out["distribution_shift"] = shift_report(reference[predictors], external[predictors],
                                                 {c: feature_types[c] for c in predictors if c in feature_types})
    return out
