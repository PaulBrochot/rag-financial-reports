"""
Découpage du texte en chunks pour l'indexation RAG.

Deux fonctions :
- chunk_text        : découpage brut à taille fixe avec overlap (utilisé pour le texte
                       à l'intérieur d'une section, et pour les tableaux).
- split_into_sections : découpe un 10-K en sections basées sur ses en-têtes "Item X."
                       (Item 1A - Risk Factors, Item 7 - MD&A, etc.) avant chunking.
                       Sans ça, un chunk à taille fixe aveugle peut couper en plein
                       milieu d'une section, et une question sur "les risques" n'a
                       aucune garantie de tomber sur les bons passages.
"""
import re

ITEM_HEADER_RE = re.compile(r'^Item\s+(\d{1,2}[A-Z]?)\.\s+(.+)$', re.MULTILINE)


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Découpe un texte en chunks de `chunk_size` caractères avec un `overlap` entre chunks
    consécutifs (pour éviter de couper une info importante pile à la frontière)."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size doit être strictement supérieur à overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Découpe le texte d'un 10-K en sections basées sur ses en-têtes "Item X.".

    Les 10-K commencent par une table des matières qui répète les mêmes intitulés
    ("Item 1A.    Risk Factors") avant le vrai contenu. On ne garde donc que la
    DERNIÈRE occurrence de chaque item comme point de départ réel de sa section,
    ce qui évite de découper au mauvais endroit à cause de la table des matières.

    Retourne une liste de tuples (label_de_section, texte_de_la_section)."""
    matches = list(ITEM_HEADER_RE.finditer(text))
    if not matches:
        return [("Document", text)]

    last_occurrence = {}
    for m in matches:
        item_number = m.group(1)
        last_occurrence[item_number] = m  # écrase les occurrences précédentes (donc garde la dernière)

    boundaries = sorted(last_occurrence.values(), key=lambda m: m.start())

    sections = []
    if boundaries[0].start() > 0:
        sections.append(("Préambule", text[:boundaries[0].start()]))

    for i, m in enumerate(boundaries):
        label = f"Item {m.group(1)} - {m.group(2).strip()[:80]}"
        start = m.start()
        end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
        sections.append((label, text[start:end]))

    return sections
