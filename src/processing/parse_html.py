"""
Parsing des documents 10-K (HTML) : extraction du texte propre et des tableaux.
"""
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

IGNORED_TAGS = ["script", "style", "head"]


def load_html(filepath) -> BeautifulSoup:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return BeautifulSoup(f.read(), "lxml")


def extract_clean_text(soup: BeautifulSoup) -> str:
    soup_copy = BeautifulSoup(str(soup), "lxml")
    for tag_name in IGNORED_TAGS:
        for tag in soup_copy.find_all(tag_name):
            tag.decompose()
    for table in soup_copy.find_all("table"):
        table.decompose()
    text = soup_copy.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def extract_tables(filepath) -> list:
    try:
        return pd.read_html(filepath)
    except ValueError:
        return []


def _table_is_meaningful(df: pd.DataFrame, min_cells: int = 4, min_fill_ratio: float = 0.4) -> bool:
    n_rows, n_cols = df.shape
    if n_rows < 2 or n_cols < 2:
        return False
    total_cells = n_rows * n_cols
    if total_cells < min_cells:
        return False
    filled = df.notna().sum().sum()
    return (filled / total_cells) >= min_fill_ratio


def extract_tables_as_text(filepath) -> list[str]:
    """Extrait les tableaux pertinents en CSV (pas to_string) : to_string aligne les
    colonnes avec des espaces et produit des blocs quasi vides sur des tableaux creux,
    ce qui noie les vrais chiffres. CSV reste compact quel que soit le remplissage."""
    tables = extract_tables(filepath)
    texts = []
    for df in tables:
        df_clean = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if not _table_is_meaningful(df_clean):
            continue
        texts.append(df_clean.to_csv(index=False, na_rep=""))
    return texts
