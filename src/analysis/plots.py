"""Figure builders for Phase 02. Each function draws one figure and saves it.

Conventions: titles in Portuguese (documentation language), variable names kept
as in the source data, consistent outcome colours across all figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: scripts and CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

OULAD_COLORS = {"Withdrawn": "#c0392b", "Fail": "#e67e22", "Pass": "#2e86c1", "Distinction": "#1e8449"}
UCI_COLORS = {"Dropout": "#c0392b", "Enrolled": "#e67e22", "Graduate": "#1e8449"}
CLASS_COLORS = {"PERMITTED": "#1e8449", "CONDITIONAL": "#e67e22", "PROHIBITED": "#c0392b"}

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def outcome_bars(counts: pd.Series, colors: dict, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    order = [c for c in colors if c in counts.index]
    vals = counts.reindex(order)
    bars = ax.bar(order, vals, color=[colors[c] for c in order])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:,}\n({v / vals.sum():.1%})", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set(title=title, xlabel="Resultado", ylabel="Número de registros")
    ax.set_ylim(0, vals.max() * 1.22)
    return _save(fig, path)


def outcome_share_by_group(df: pd.DataFrame, group: str, outcome: str, colors: dict,
                           title: str, path: Path) -> Path:
    share = pd.crosstab(df[group], df[outcome], normalize="index")[[c for c in colors if c in df[outcome].unique()]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    share.plot(kind="bar", stacked=True, color=[colors[c] for c in share.columns], ax=ax, width=0.8)
    ax.set(title=title, xlabel=group, ylabel="Proporção")
    ax.legend(title=outcome, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, path)


def withdrawal_timing(unreg_days: pd.Series, cutoffs: dict[str, int], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(unreg_days.dropna(), bins=np.arange(-140, 280, 7), color=OULAD_COLORS["Withdrawn"], alpha=0.85)
    ax.axvline(0, color="black", lw=1.2, label="Início da apresentação (dia 0)")
    for (label, day), ls in zip(cutoffs.items(), ["--", "-.", ":"]):
        ax.axvline(day, color="#34495e", ls=ls, lw=1.2, label=f"Corte {label} (dia {day})")
    ax.set(title="OULAD — Quando os cancelamentos acontecem (Withdrawn)",
           xlabel="date_unregistration (dias desde o início)", ylabel="Estudantes (bins de 7 dias)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def weekly_engagement(weekly_by_outcome: pd.DataFrame, cutoffs: dict[str, int], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for outcome, color in OULAD_COLORS.items():
        sub = weekly_by_outcome.loc[weekly_by_outcome["final_result"] == outcome]
        ax.plot(sub["week"], sub["mean"], color=color, lw=2, label=outcome)
    for (label, day), ls in zip(cutoffs.items(), ["--", "-.", ":"]):
        ax.axvline(day // 7 + 1, color="#34495e", ls=ls, lw=1, label=f"Fim da semana {label}")
    ax.set(title="OULAD — Média de cliques no AVA por semana (só estudantes ainda matriculados)",
           xlabel="Semana da apresentação (≤0 = antes do início)", ylabel="Cliques médios por estudante")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, path)


def disengagement_before_withdrawal(rel: pd.DataFrame, path: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(rel["rel_week"], rel["mean"], color=OULAD_COLORS["Withdrawn"], marker="o", lw=2,
             label="Cliques médios")
    ax1.set(xlabel="Semanas relativas ao cancelamento (0 = semana do cancelamento)",
            ylabel="Cliques médios por estudante",
            title="OULAD — Desengajamento progressivo antes do cancelamento")
    ax2 = ax1.twinx()
    ax2.plot(rel["rel_week"], rel["share_inactive"] * 100, color="#34495e", ls="--", marker="s",
             label="% sem nenhum clique na semana")
    ax2.set_ylabel("% de estudantes inativos na semana")
    ax2.grid(False)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, fontsize=8, loc="center left")
    return _save(fig, path)


def engagement_at_cutoffs(snaps: pd.DataFrame, path: Path) -> Path:
    df = snaps.assign(log_clicks=np.log10(snaps["total_clicks"] + 1))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=df, x="week_label", y="log_clicks", hue="final_result",
                hue_order=list(OULAD_COLORS), palette=OULAD_COLORS, showfliers=False, ax=ax)
    ax.set(title="OULAD — Cliques acumulados até o corte, por resultado final",
           xlabel="Momento da previsão", ylabel="log10(cliques acumulados + 1)")
    ax.legend(title="final_result", fontsize=8)
    return _save(fig, path)


def auc_by_cutoff(auc_df: pd.DataFrame, path: Path, reference: dict[str, float] | None = None) -> Path:
    """auc_df: columns indicator, week_label, separation."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=auc_df, y="indicator", x="separation", hue="week_label", ax=ax, palette="Blues")
    if reference:
        for i, (name, val) in enumerate(reference.items()):
            ax.axvline(val, color=CLASS_COLORS["PROHIBITED"], ls="--", lw=1.2,
                       label=f"{name} (PROIBIDA): {val:.2f}")
    ax.set(title="OULAD — Separação univariada Withdrawn × Pass/Distinction por corte\n"
                 "(|AUC−0,5|×2; estatística descritiva, não modelo)",
           xlabel="Força de separação (0 = nenhuma, 1 = perfeita)", ylabel="")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)


def rate_panels(rates: dict[str, pd.DataFrame], overall: float, title: str, path: Path) -> Path:
    n = len(rates)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows))
    for ax, (name, df) in zip(np.ravel(axes), rates.items()):
        d = df.reset_index()
        ax.barh(d.iloc[:, 0].astype(str), d["rate"], color=np.where(d["small_sample"], "#bbbbbb", "#c0392b"))
        ax.axvline(overall, color="black", ls="--", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Taxa de Withdrawn")
        ax.invert_yaxis()
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle(title + f" (linha tracejada = taxa geral {overall:.1%}; cinza = n<30)", fontsize=11)
    return _save(fig, path)


def score_distribution(df: pd.DataFrame, score_col: str, outcome_col: str, colors: dict,
                       title: str, xlabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for outcome, color in colors.items():
        vals = df.loc[df[outcome_col] == outcome, score_col].dropna()
        if len(vals):
            sns.kdeplot(vals, ax=ax, color=color, lw=2, label=f"{outcome} (n={len(vals):,})", cut=0)
    ax.set(title=title, xlabel=xlabel, ylabel="Densidade")
    ax.legend(fontsize=8)
    return _save(fig, path)


def activity_type_shares(share: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    share.plot(kind="barh", ax=ax, color=[OULAD_COLORS[c] for c in share.columns], width=0.8)
    ax.set(title="OULAD — Distribuição dos cliques por tipo de material até a semana 4",
           xlabel="Proporção dos cliques do grupo", ylabel="activity_type")
    ax.invert_yaxis()
    ax.legend(title="final_result", fontsize=8)
    return _save(fig, path)


def availability_over_weeks(avail: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label, color in [
        ("share_with_vle_activity", "Com ≥1 clique no AVA", "#2e86c1"),
        ("share_with_submission", "Com ≥1 avaliação entregue", "#1e8449"),
        ("share_presentations_with_due_assessment", "Apresentações com avaliação vencida", "#8e44ad"),
    ]:
        ax.plot(avail["week"], avail[col] * 100, marker="o", lw=2, label=label, color=color)
    ax.set(title="OULAD — Informação disponível ao fim de cada semana",
           xlabel="Semana (corte no último dia)", ylabel="%")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    return _save(fig, path)


def uci_semester_performance(df: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sem in zip(axes, ("1st", "2nd")):
        p = f"Curricular units {sem} sem "
        enrolled = df[p + "(enrolled)"]
        ratio = (df[p + "(approved)"] / enrolled.where(enrolled > 0)).clip(upper=1)
        for cls, color in UCI_COLORS.items():
            vals = ratio[df["Target"] == cls].dropna()
            ax.hist(vals, bins=np.linspace(0, 1, 11), alpha=0.55, color=color, label=cls, density=True)
        ax.set(title=f"Taxa de aprovação — {sem} semestre (fim do semestre)",
               xlabel="UCs aprovadas / UCs inscritas", ylabel="Densidade")
        ax.legend(fontsize=8)
    return _save(fig, path)


def uci_availability_separation(sep: pd.DataFrame, path: Path) -> Path:
    """sep: variable, separation, class (for 1st-semester cutoff)."""
    d = sep.dropna(subset=["separation"]).sort_values("separation", ascending=True)  # nominal codes excluded
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(d))))
    ax.barh(d["variable"], d["separation"], color=[CLASS_COLORS[c] for c in d["class"]])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS.values()]
    ax.legend(handles, ["PERMITIDA", "CONDICIONAL", "PROIBIDA"],
              title="Classe p/ semanas 2–6 do 1º semestre", fontsize=8, loc="lower right")
    ax.set(title="UCI 697 — Separação univariada Dropout × Graduate (variáveis ordinais/numéricas)\n"
                 "(|AUC−0,5|×2; descritivo, não causal; nominais avaliadas por Cramér's V)",
           xlabel="Força de separação", ylabel="")
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, path)


def uci_cohort_proxy(df: pd.DataFrame, path: Path) -> Path:
    key = df["Unemployment rate"].astype(str) + " | " + df["Inflation rate"].astype(str) + " | " + df["GDP"].astype(str)
    share = pd.crosstab(key, df["Target"], normalize="index")[list(UCI_COLORS)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    share.plot(kind="bar", stacked=True, color=list(UCI_COLORS.values()), ax=ax, width=0.8)
    ax.set(title="UCI 697 — Composição das classes por combinação macroeconômica (provável coorte)",
           xlabel="Desemprego | Inflação | PIB", ylabel="Proporção")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.legend(title="Target", bbox_to_anchor=(1.01, 1), loc="upper left")
    return _save(fig, path)


def correlation_heatmap(corr: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "Spearman ρ"}, xticklabels=True, yticklabels=True)
    ax.set_title(title)
    ax.tick_params(labelsize=7)
    return _save(fig, path)
