"""
Construit le corpus traité : parse chaque 10-K brut, extrait le texte narratif (découpé
par section Item 1/1A/7/8...) ET les tableaux (convertis en texte), découpe le tout en
chunks, et sauvegarde avec métadonnées dans data/processed/chunks.json.

Usage :
    python -m src.processing.build_corpus
"""
import json
import re
from pathlib import Path

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.processing.parse_html import load_html, extract_clean_text, extract_tables_as_text, extract_tables
from src.processing.chunking import chunk_text, split_into_sections


def parse_filename(filepath: Path) -> dict:
    match = re.match(r"(10-K|10-Q)_(\d{4}-\d{2}-\d{2})_", filepath.name)
    if not match:
        return {"form": "unknown", "filing_date": "unknown"}
    return {"form": match.group(1), "filing_date": match.group(2)}


def slugify(label: str) -> str:
    """Transforme un label de section en identifiant compact et sûr pour les IDs Chroma."""
    return re.sub(r"\W+", "_", label).strip("_")[:40]


def process_filing(filepath: Path, ticker: str) -> list[dict]:
    soup = load_html(filepath)
    text = extract_clean_text(soup)
    table_texts = extract_tables_as_text(filepath)
    total_tables_detected = len(extract_tables(filepath))

    meta = parse_filename(filepath)
    records = []

    # --- Texte narratif : découpé par section AVANT le chunking taille fixe ---
    sections = split_into_sections(text)
    text_chunk_count = 0
    for section_label, section_text in sections:
        section_chunks = chunk_text(section_text)
        for i, chunk in enumerate(section_chunks):
            # On préfixe chaque chunk avec le nom de sa section : ça aide la recherche
            # sémantique à faire le lien entre une question sur "les risques" et les
            # chunks qui viennent réellement de la section Risk Factors, même si le
            # mot "risque" n'apparaît pas explicitement dans le chunk lui-même.
            # Exception : si la détection de sections a échoué (repli "Document" faute
            # d'en-têtes "Item X." reconnus), on ne préfixe pas — ça n'apporterait aucune
            # info utile et polluerait inutilement l'embedding de chaque chunk.
            if section_label == "Document":
                prefixed_text = chunk
            else:
                prefixed_text = f"[Section : {section_label}]\n{chunk}"
            records.append({
                "ticker": ticker,
                "form": meta["form"],
                "filing_date": meta["filing_date"],
                "source_file": filepath.name,
                "chunk_type": "text",
                "section": section_label,
                "chunk_index": f"{slugify(section_label)}_{i}",
                "text": prefixed_text,
            })
            text_chunk_count += 1

    # --- Tableaux : pas de notion de section (extraits indépendamment du texte) ---
    table_chunk_count = 0
    for j, table_text in enumerate(table_texts):
        for k, sub_chunk in enumerate(chunk_text(table_text, chunk_size=2000, overlap=100)):
            records.append({
                "ticker": ticker,
                "form": meta["form"],
                "filing_date": meta["filing_date"],
                "source_file": filepath.name,
                "chunk_type": "table",
                "section": "Tableau",
                "chunk_index": f"table{j}_{k}",
                "text": sub_chunk,
            })
            table_chunk_count += 1

    print(
        f"  {filepath.name}: {len(sections)} sections -> {text_chunk_count} chunks texte, "
        f"{table_chunk_count} chunks tableaux "
        f"({len(table_texts)} tableaux retenus sur {total_tables_detected} détectés)"
    )
    return records


def main():
    raw_dir = Path(RAW_DATA_DIR)
    processed_dir = Path(PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for ticker_dir in sorted(raw_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        print(f"\n{ticker}")

        for filepath in sorted(ticker_dir.glob("*.htm")):
            records = process_filing(filepath, ticker)
            all_records.extend(records)

    out_path = processed_dir / "chunks.json"
    out_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2))
    print(f"\n✓ {len(all_records)} chunks au total, sauvegardés dans {out_path}")


if __name__ == "__main__":
    main()
