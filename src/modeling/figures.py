"""Figures for Phase 04 (saved under reports/modeling/figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
LABELS = {"baseline_prevalence": "Prevalência", "baseline_module_rate": "Taxa do módulo",
          "baseline_course_rate": "Taxa do curso", "baseline_inactivity_rule": "Regra de inatividade",
          "logistic_regression": "Regressão logística", "gradient_boosting": "Gradient boosting"}
COLORS = {"baseline_prevalence": "#95a5a6", "baseline_module_rate": "#7f8c8d", "baseline_course_rate": "#7f8c8d",
          "baseline_inactivity_rule": "#e67e22", "logistic_regression": "#2e86c1", "gradient_boosting": "#c0392b"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def pr_curves(pred: pd.DataFrame, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    y = pred["y"].to_numpy()
    for name in LABELS:
        col = f"p_{name}"
        if col not in pred:
            continue
        prec, rec, _ = precision_recall_curve(y, pred[col])
        ax.plot(rec, prec, color=COLORS[name], lw=2 if "baseline" not in name else 1.3, label=LABELS[name])
    ax.axhline(y.mean(), color="black", ls=":", lw=1, label=f"Prevalência no teste ({y.mean():.1%})")
    ax.set(title=title, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    return _save(fig, path)


def calibration_plot(tables: dict[str, list], pred: pd.DataFrame, title: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="black", ls=":", lw=1, label="Calibração perfeita")
    for name, rows in tables.items():
        t = pd.DataFrame(rows)
        ax.plot(t["mean_predicted"], t["observed_rate"], marker="o", color=COLORS[name], label=LABELS[name])
    ax.set(title="Curva de calibração (decis de probabilidade)", xlabel="Probabilidade média prevista",
           ylabel="Frequência observada de cancelamento", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    ax = axes[1]
    for name in tables:
        ax.hist(pred[f"p_{name}"], bins=np.linspace(0, 1, 41), alpha=0.55, color=COLORS[name], label=LABELS[name])
    ax.set(title="Distribuição das probabilidades previstas", xlabel="Probabilidade (calibrada)", ylabel="Registros")
    ax.legend(fontsize=8)
    fig.suptitle(title)
    return _save(fig, path)


def model_comparison(summary: pd.DataFrame, path: Path, title: str):
    """summary: week, model, pr_auc, ci_low, ci_high, prevalence."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    weeks = sorted(summary["week"].unique())
    models = [m for m in LABELS if m in set(summary["model"])]
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        s = summary[summary["model"] == m].set_index("week").reindex(weeks)
        x = np.arange(len(weeks)) + i * width
        ax.bar(x, s["pr_auc"], width, color=COLORS[m], label=LABELS[m])
        ax.errorbar(x, s["pr_auc"], yerr=[s["pr_auc"] - s["ci_low"], s["ci_high"] - s["pr_auc"]],
                    fmt="none", ecolor="black", lw=1, capsize=2)
    for j, w in enumerate(weeks):
        prev = summary[summary["week"] == w]["prevalence"].iloc[0]
        ax.hlines(prev, j - 0.05, j + 0.8, color="black", ls=":", lw=1)
    ax.set_xticks(np.arange(len(weeks)) + 0.4 - width / 2, [f"Semana {w}" for w in weeks])
    ax.set(title=title, ylabel="PR-AUC no teste 2014 (IC 95% bootstrap por estudante)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    return _save(fig, path)


def group_importance(df: pd.DataFrame, path: Path):
    """df: week, group, drop."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=df, y="group", x="drop", hue="week", palette="Blues", ax=ax)
    ax.set(title="Gradient boosting — queda de PR-AUC (validação) ao embaralhar cada grupo de features",
           xlabel="Queda média de PR-AUC (maior = mais informativo)", ylabel="")
    ax.legend(title="Semana", fontsize=8)
    return _save(fig, path)


def top_features(per_feature: list[dict], value: str, title: str, path: Path, n: int = 15):
    d = pd.DataFrame(per_feature).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#c0392b" if v > 0 else "#2e86c1" for v in d[value]]
    ax.barh(d["feature"], d[value], color=colors)
    ax.set(title=title, xlabel=value)
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, path)


def fairness_plot(audit: dict, title: str, path: Path):
    rows = []
    for g, levels in audit.items():
        for r in levels:
            for metric in ("recall", "false_positive_rate"):
                ci = r.get(f"{metric}_ci95", [np.nan, np.nan])
                rows.append({"grupo": f"{g}: {r['level']} (n={r['n_records']})", "métrica": metric,
                             "valor": r[metric], "lo": ci[0], "hi": ci[1], "ok": r["sufficient_sample"]})
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * d["grupo"].nunique() + 2), sharey=True)
    for ax, metric, color in zip(axes, ("recall", "false_positive_rate"), ("#2e86c1", "#c0392b")):
        s = d[d["métrica"] == metric]
        ax.barh(s["grupo"], s["valor"], color=[color if ok else "#cccccc" for ok in s["ok"]])
        ax.errorbar(s["valor"], s["grupo"], xerr=[s["valor"] - s["lo"], s["hi"] - s["valor"]], fmt="none",
                    ecolor="black", lw=1, capsize=2)
        ax.set(title={"recall": "Recall", "false_positive_rate": "Taxa de falsos positivos"}[metric], xlim=(0, 1))
        ax.invert_yaxis()
    fig.suptitle(title + " (cinza = amostra insuficiente; barras = IC 95%)", fontsize=10)
    return _save(fig, path)


def group_calibration(rows: list[dict], title: str, path: Path):
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 0.6], [0, 0.6], color="black", ls=":", lw=1, label="Calibração perfeita")
    for level, color in (("N", "#2e86c1"), ("Y", "#c0392b")):
        s = d[d["level"] == level]
        if len(s):
            ax.plot(s["mean_predicted"], s["observed_rate"], marker="o", color=color,
                    label=f"deficiência = {level} (n={int(s['n'].sum())})")
    ax.set(title=title, xlabel="Probabilidade média prevista", ylabel="Taxa observada de cancelamento",
           xlim=(0, 0.6), ylim=(0, 0.6))
    ax.legend(fontsize=8)
    return _save(fig, path)
