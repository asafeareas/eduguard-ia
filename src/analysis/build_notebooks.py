"""Generate and execute the Phase 02 notebooks.

The analysis logic lives in src/ (eda, temporal, leakage, plots); notebooks are
thin, reproducible narratives on top of it. Regenerate with:

    python -m src.analysis.build_notebooks           # build + execute
    python -m src.analysis.build_notebooks --no-run  # build only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from src.data.loaders import PROJECT_ROOT

NB_DIR = PROJECT_ROOT / "notebooks"

SETUP = """\
import sys, pathlib
ROOT = pathlib.Path.cwd().resolve()
ROOT = ROOT.parent if ROOT.name == "notebooks" else ROOT
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from IPython.display import Image, display
pd.set_option("display.max_columns", 30, "display.width", 160)
FIG = ROOT / "research" / "figures"
"""


def _nb(cells: list) -> nbformat.NotebookNode:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def oulad_notebook() -> nbformat.NotebookNode:
    md, code = new_markdown_cell, new_code_cell
    return _nb([
        md("# 01 — OULAD: análise exploratória\n\n"
           "Fase 02 do EduGuard-IA. **Nenhum modelo é treinado.** Toda a lógica está em `src/`; "
           "este notebook apenas a executa e mostra os resultados. Associações aqui são descritivas, "
           "não causais."),
        code(SETUP + "from src.data.loaders import load_oulad, load_student_vle, OULAD_KEY\n"
             "from src.data.quality import oulad_integrity\n"
             "from src.feature_engineering import temporal\n"
             "from src.analysis import eda, plots\n"
             "t = load_oulad(); sv = load_student_vle(); base = temporal.student_base(t)\n"
             "{k: v.shape for k, v in t.items()} | {'studentVle': sv.shape}"),
        md("## 1. Estrutura, granularidade e integridade das chaves\n\n"
           "| Tabela | Granularidade (chave) |\n|---|---|\n"
           "| courses | módulo × apresentação |\n| assessments | avaliação (`id_assessment`) |\n"
           "| vle | material do AVA (`id_site`) |\n| studentInfo / studentRegistration | estudante × módulo × apresentação |\n"
           "| studentAssessment | estudante × avaliação |\n| studentVle | estudante × material × dia (várias linhas por dia possíveis) |"),
        code("integ = oulad_integrity(t, sv)\n"
             "pd.DataFrame(integ['granularity']).T"),
        code("pd.Series(integ['orphans'], name='linhas órfãs')"),
        code("pd.Series(integ['consistency'], name='inconsistências')"),
        code("per_student = t['studentInfo'].groupby('id_student').size()\n"
             "print('estudantes em >1 apresentação:', (per_student > 1).sum(), 'de', per_student.size)\n"
             "print('linhas totalmente duplicadas em studentVle:', sv.duplicated().sum())"),
        md("## 2. Resultados acadêmicos"),
        code("plots.outcome_bars(base['final_result'].value_counts(), plots.OULAD_COLORS,\n"
             "                    'OULAD — Resultado final (32.593 matrículas estudante×módulo)', FIG / 'oulad_outcome_distribution.png')\n"
             "display(Image(FIG / 'oulad_outcome_distribution.png'))\n"
             "eda.distribution(base['final_result'])"),
        code("plots.outcome_share_by_group(base, 'code_module', 'final_result', plots.OULAD_COLORS,\n"
             "                              'OULAD — Resultado final por módulo', FIG / 'oulad_outcome_by_module.png')\n"
             "display(Image(FIG / 'oulad_outcome_by_module.png'))"),
        md("### Características de ingresso × Withdrawn"),
        code("withdrawn = base['final_result'] == 'Withdrawn'\n"
             "for col in ['highest_education', 'imd_band', 'age_band', 'disability']:\n"
             "    display(eda.rate_by_category(base, col, withdrawn))\n"
             "display(Image(FIG / 'oulad_withdrawal_rate_by_attribute.png'))"),
        md("### Desempenho: 1ª avaliação não-final\n\nSó estudantes ainda matriculados no prazo da avaliação."),
        code("display(Image(FIG / 'oulad_first_assessment_score.png'))"),
        md("### Qualidade do rótulo: `Fail` silencioso\n\n"
           "Diagnóstico do **rótulo** usando o período inteiro (nunca usado como feature)."),
        code("last = eda.last_activity_profile(sv, base)\n"
             "last.assign(before_half=last['last_day_ratio'] < 0.5).groupby('final_result')['before_half'].mean().round(3)"),
        md("## 3. Engajamento digital"),
        code("weekly = temporal.weekly_clicks(sv)\n"
             "wbo = eda.mean_weekly_clicks_by_outcome(weekly, base, range(-3, 40))\n"
             "wbo[wbo['week'].between(1, 6)].pivot(index='week', columns='final_result', values='mean').round(1)"),
        code("display(Image(FIG / 'oulad_weekly_clicks_by_outcome.png'))"),
        code("rel = eda.clicks_relative_to_withdrawal(weekly, base, weeks_before=8)\n"
             "display(rel.round(3)); display(Image(FIG / 'oulad_disengagement_before_withdrawal.png'))"),
        code("display(Image(FIG / 'oulad_activity_type_until_week4.png'))"),
        md("**Leitura:** Withdrawn e Fail já clicam ~40–45% menos que Pass na semana 1; o "
           "desengajamento antes do cancelamento é progressivo (inatividade semanal 41% → 65% "
           "nas 8 semanas anteriores)."),
    ])


def uci_notebook() -> nbformat.NotebookNode:
    md, code = new_markdown_cell, new_code_cell
    return _nb([
        md("# 02 — UCI 697: análise exploratória\n\n"
           "Predict Students' Dropout and Academic Success. **Nenhum modelo é treinado.**"),
        code(SETUP + "from src.data.loaders import load_uci\n"
             "from src.data.quality import uci_consistency\n"
             "from src.analysis import eda, leakage, plots\n"
             "df = load_uci(); df.shape"),
        code("pd.DataFrame(uci_consistency(df)).T"),
        md("## 1. Classes"),
        code("display(eda.distribution(df['Target'])); display(Image(FIG / 'uci_target_distribution.png'))"),
        md("## 2. Variáveis de ingresso e desempenho por classe"),
        code("df.groupby('Target')[['Admission grade', 'Age at enrollment', 'Curricular units 1st sem (approved)',\n"
             "                      'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (approved)']].median()"),
        code("df.groupby('Target')[['Debtor', 'Tuition fees up to date', 'Scholarship holder', 'Displaced', 'Gender']].mean().round(3)"),
        code("display(Image(FIG / 'uci_admission_grade_by_class.png')); display(Image(FIG / 'uci_semester_approval_by_class.png'))"),
        md("## 3. O problema do rótulo `Enrolled`\n\n"
           "`Enrolled` = ainda matriculado ao fim da duração normal: desfecho **censurado**. "
           "Seu perfil acadêmico fica entre Dropout e Graduate."),
        code("cohort = df['Unemployment rate'].astype(str) + '|' + df['Inflation rate'].astype(str) + '|' + df['GDP'].astype(str)\n"
             "display(pd.crosstab(cohort, df['Target'], normalize='index').round(3))\n"
             "display(Image(FIG / 'uci_cohort_proxy.png'))"),
        md("## 4. Regras que revelam o desfecho"),
        code("leakage.uci_outcome_revealing_rules(df)"),
        md("## 5. Associações (descritivas)"),
        code("sep = pd.read_csv(ROOT / 'research' / 'inspection' / 'uci_univariate_separation.csv')\n"
             "sep[['variable', 'separation', 'cramers_v_3class', 'class', 'class_sem2']]"),
        code("display(Image(FIG / 'uci_separation_by_availability.png')); display(Image(FIG / 'uci_spearman_correlation.png'))"),
    ])


def temporal_notebook() -> nbformat.NotebookNode:
    md, code = new_markdown_cell, new_code_cell
    return _nb([
        md("# 03 — Viabilidade temporal e auditoria de vazamento\n\n"
           "Convenção: dia 0 = início da apresentação; semana *k* = dias 7(k−1)…7k−1; "
           "previsão ao fim da semana *k* ⇒ `cutoff_day = 7k − 1`. Só entram eventos com data ≤ corte "
           "e estudantes ainda matriculados no corte."),
        code(SETUP + "from src.data.loaders import load_oulad, load_student_vle\n"
             "from src.feature_engineering import temporal\n"
             "from src.analysis import eda, leakage\n"
             "t = load_oulad(); sv = load_student_vle(); base = temporal.student_base(t)\n"
             "{w: temporal.cutoff_day_for_week(w) for w in (2, 4, 6)}"),
        md("## 1. Quando o cancelamento acontece"),
        code("display(Image(FIG / 'oulad_withdrawal_timing.png'))"),
        md("## 2. População e informação disponível em cada corte"),
        code("rows = []\n"
             "snaps = {}\n"
             "for w in (2, 4, 6):\n"
             "    c = temporal.cutoff_day_for_week(w)\n"
             "    s = temporal.snapshot(t, sv, c); snaps[w] = s\n"
             "    rows.append({'semana': w, 'cutoff_day': c, 'população': len(s),\n"
             "                 'Withdrawn restantes': int((s['final_result'] == 'Withdrawn').sum()),\n"
             "                 '% com entrega': round((s['n_submitted'] > 0).mean() * 100, 1),\n"
             "                 'max data VLE usada': int(sv.loc[sv['date'] <= c, 'date'].max())})\n"
             "pd.DataFrame(rows)"),
        code("display(Image(FIG / 'oulad_information_availability.png'))"),
        code("cols = ['total_clicks', 'active_days', 'clicks_last_7_days', 'days_since_last_activity', 'n_submitted', 'mean_score']\n"
             "for w, s in snaps.items():\n"
             "    print(f'Semana {w}'); display(s.groupby('final_result')[cols].median())"),
        code("display(Image(FIG / 'oulad_clicks_at_cutoffs.png'))"),
        md("## 3. Separação univariada (descritiva) vs. referência com vazamento"),
        code("auc = pd.read_csv(ROOT / 'research' / 'inspection' / 'oulad_univariate_separation_by_cutoff.csv')\n"
             "display(auc.pivot(index='indicator', columns='week_label', values='separation'))\n"
             "display(Image(FIG / 'oulad_separation_by_cutoff.png'))"),
        md("A referência **proibida** (agregados do módulo inteiro) separa muito mais — exatamente "
           "porque usa o futuro. Ver `docs/DATA_LEAKAGE_AUDIT.md`."),
        md("## 4. Catálogo de vazamento"),
        code("display(leakage.catalog_frame(leakage.OULAD_CATALOG)[['table', 'variable', 'class']])\n"
             "display(leakage.catalog_frame(leakage.UCI_CATALOG)[['variable', 'class_sem1_weeks_2_6', 'class_sem2_weeks_2_6']])"),
    ])


NOTEBOOKS = {
    "01_oulad_eda.ipynb": oulad_notebook,
    "02_uci_eda.ipynb": uci_notebook,
    "03_temporal_feasibility_and_leakage.ipynb": temporal_notebook,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 02 notebooks")
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args(argv)
    NB_DIR.mkdir(exist_ok=True)
    for name, builder in NOTEBOOKS.items():
        nb = builder()
        if not args.no_run:
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        path = NB_DIR / name
        nbformat.write(nb, path)
        print(f"notebook -> {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
