"""
Recherche les chunks les plus pertinents pour une question donnée (retrieval seul,
sans appel à un LLM — utile pour tester la qualité du retrieval avant de brancher
la génération).

Usage :
    python -m src.rag.retrieve "Quel a été le chiffre d'affaires d'Apple en 2023 ?"
"""
import sys

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "financial_reports"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def main():
    if len(sys.argv) < 2:
        print('Usage : python -m src.rag.retrieve "ta question"')
        sys.exit(1)

    query = sys.argv[1]
    results = retrieve(query)

    print(f"\nQuestion : {query}\n")
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        print(f"--- Résultat {i} (distance={r['distance']:.3f}) ---")
        print(f"{m['ticker']} | {m['form']} | déposé le {m['filing_date']}")
        print(r["text"][:300] + "...")
        print()


if __name__ == "__main__":
    main()
