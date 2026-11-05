"""Registry of candidate data sources evaluated in Phase 01.

Each entry records provenance, licensing and acquisition policy. The
`auto_download` flag marks sources that are public, small, licence-compatible
and require no authentication; every other source needs explicit approval
from the project owner before being downloaded (see docs/DATASET_RESEARCH.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataSource:
    """Metadata describing one external dataset."""

    source_id: str
    name: str
    institution: str
    country: str
    landing_url: str
    download_url: str | None
    license: str
    citation: str
    compatibility: str  # "high" | "partial" | "low"
    auto_download: bool
    approx_download_size: str
    notes: str = ""
    expected_files: tuple[str, ...] = field(default_factory=tuple)


SOURCES: dict[str, DataSource] = {
    "uci_dropout_697": DataSource(
        source_id="uci_dropout_697",
        name="Predict Students' Dropout and Academic Success",
        institution="Instituto Politécnico de Portalegre (via UCI ML Repository)",
        country="Portugal",
        landing_url=(
            "https://archive.ics.uci.edu/dataset/697/"
            "predict+students+dropout+and+academic+success"
        ),
        download_url=(
            "https://archive.ics.uci.edu/static/public/697/"
            "predict+students+dropout+and+academic+success.zip"
        ),
        license="CC BY 4.0",
        citation=(
            "Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). "
            "Predict Students' Dropout and Academic Success [Dataset]. UCI Machine "
            "Learning Repository. https://doi.org/10.24432/C5MC89"
        ),
        compatibility="partial",
        auto_download=True,
        approx_download_size="520.8 KB",
        expected_files=("data.csv",),
    ),
    "oulad_349": DataSource(
        source_id="oulad_349",
        name="Open University Learning Analytics Dataset (OULAD)",
        institution="The Open University (KMi) via UCI ML Repository",
        country="United Kingdom",
        landing_url=(
            "https://archive.ics.uci.edu/dataset/349/"
            "open+university+learning+analytics+dataset"
        ),
        download_url=(
            "https://archive.ics.uci.edu/static/public/349/"
            "open+university+learning+analytics+dataset.zip"
        ),
        license="CC BY 4.0",
        citation=(
            "Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning "
            "Analytics dataset. Scientific Data 4, 170171. "
            "https://doi.org/10.1038/sdata.2017.171"
        ),
        compatibility="high",
        auto_download=False,  # 44.6 MB zip, ~450 MB extracted: needs approval
        approx_download_size="44.6 MB (zip); studentVle.csv alone 432.8 MB",
        notes="Download approved by project owner on 2026-11-05. Missing values encoded as '?'.",
    ),
    "ufrn_open_data": DataSource(
        source_id="ufrn_open_data",
        name="UFRN Dados Abertos: situacoes-de-discentes + matriculas-componentes",
        institution="Universidade Federal do Rio Grande do Norte (UFRN)",
        country="Brazil",
        landing_url="https://dados.ufrn.br/",
        download_url=None,  # many per-semester CSVs; see docs
        license="'Outra (Aberta)' per CKAN metadata — exact terms not verified",
        citation="UFRN. Portal de Dados Abertos. https://dados.ufrn.br/",
        compatibility="high",
        auto_download=False,  # ~65-77 MB per semester file, pseudonymised personal data
        approx_download_size="~65-77 MB per semester (matriculas); ~3 GB for all",
        notes=(
            "Pseudonymised IDs (hash). The 'discentes' dataset exposes real names "
            "and enrollment numbers and MUST NOT be used."
        ),
    ),
    "inep_censup": DataSource(
        source_id="inep_censup",
        name="Microdados do Censo da Educação Superior",
        institution="INEP",
        country="Brazil",
        landing_url=(
            "https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/"
            "microdados/censo-da-educacao-superior"
        ),
        download_url=(
            "https://download.inep.gov.br/microdados/"
            "microdados_censo_da_educacao_superior_2024.zip"
        ),
        license="Dados abertos governamentais (termos específicos não verificados)",
        citation="INEP. Microdados do Censo da Educação Superior.",
        compatibility="low",
        auto_download=False,  # aggregated, size unverified, host reset connections
        approx_download_size="unknown (download host reset connection during check)",
        notes="Public files are aggregated (course/IES level); no student-level labels.",
    ),
}


def get_source(source_id: str) -> DataSource:
    """Return the registered source or raise a helpful KeyError."""
    try:
        return SOURCES[source_id]
    except KeyError as exc:
        known = ", ".join(sorted(SOURCES))
        raise KeyError(f"Unknown source '{source_id}'. Known: {known}") from exc
