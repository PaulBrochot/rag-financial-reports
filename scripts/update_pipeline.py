"""
Orchestre la mise à jour complète du pipeline : vérifie s'il y a de nouveaux filings SEC,
et si oui seulement, reconstruit le corpus et l'index automatiquement.

Pensé pour être lancé régulièrement (à la main ou via une tâche planifiée / GitHub
Actions) sans intervention manuelle.

Usage :
    python -m scripts.update_pipeline
"""
from src.ingestion.fetch_edgar import main as fetch_new_filings
from src.processing.build_corpus import main as build_corpus
from src.rag.build_index import main as build_index


def main():
    print("=== Étape 1 : vérification de nouveaux filings SEC EDGAR ===")
    new_count = fetch_new_filings()

    if new_count == 0:
        print("\nAucun nouveau filing détecté. Pipeline déjà à jour, rien à faire.")
        return

    print(f"\n{new_count} nouveau(x) filing(s) détecté(s). Mise à jour du corpus...")

    print("\n=== Étape 2 : reconstruction du corpus (parsing + chunking) ===")
    build_corpus()

    print("\n=== Étape 3 : reconstruction de l'index vectoriel ===")
    build_index()

    print("\n✓ Pipeline mis à jour avec succès.")


if __name__ == "__main__":
    main()
