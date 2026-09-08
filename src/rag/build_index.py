"""
Génère les embeddings de tous les chunks et les indexe dans une base vectorielle Chroma.

Usage :
    python -m src.rag.build_index
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DATA_DIR

CHUNKS_PATH = Path(PROCESSED_DATA_DIR) / "chunks.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "financial_reports"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def main():
    print(f"Chargement des chunks depuis {CHUNKS_PATH}...")
    chunks = json.loads(CHUNKS_PATH.read_text())
    print(f"  {len(chunks)} chunks à indexer")

    print(f"\nChargement du modèle d'embeddings ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nCalcul des embeddings (peut prendre plusieurs minutes sur CPU)...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    print("\nIndexation dans Chroma...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids = [f"{c['ticker']}_{c['source_file']}_{c['chunk_index']}" for c in chunks]
    metadatas = [
        {
            "ticker": c["ticker"],
            "form": c["form"],
            "filing_date": c["filing_date"],
            "source_file": c["source_file"],
            "chunk_index": str(c["chunk_index"]),
        }
        for c in chunks
    ]

    BATCH_SIZE = 500
    for i in range(0, len(chunks), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"  {end}/{len(chunks)} indexés")

    print(f"\n✓ Index créé dans ./{CHROMA_DIR} ({collection.count()} chunks)")


if __name__ == "__main__":
    main()
