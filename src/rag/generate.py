"""
Pipeline RAG complet : retrieval + génération de réponse via l'API Mistral.

Usage :
    python -m src.rag.generate "Quel a été le chiffre d'affaires d'Apple en 2023 ?"
"""
import os
import sys
import time

from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.errors.sdkerror import SDKError

from src.rag.retrieve import retrieve

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY manquant dans .env")

MODEL = "mistral-medium-latest"

PROMPT_TEMPLATE = """Tu es un assistant qui répond à des questions sur des rapports financiers \
d'entreprises cotées (10-K). Réponds UNIQUEMENT à partir des extraits fournis ci-dessous. \
Si l'information n'y figure pas clairement, dis-le plutôt que d'inventer un chiffre. \
Cite le ticker et la date de dépôt de ta ou tes sources.

Extraits :
{context}

Question : {question}

Réponse :"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        m = c["metadata"]
        parts.append(f"[{m['ticker']} | {m['form']} déposé le {m['filing_date']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def call_with_retry(client, max_retries: int = 5, **kwargs):
    """Réessaie automatiquement en cas de 429, avec un délai qui augmente à chaque tentative
    (backoff exponentiel) — c'est la solution recommandée par la doc officielle Mistral,
    le tier gratuit ayant des limites basses même en usage normal."""
    for attempt in range(max_retries):
        try:
            return client.chat.complete(**kwargs)
        except SDKError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"  Rate limit atteint, nouvelle tentative dans {delay}s "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise


def answer(question: str, top_k: int = 3) -> str:
    chunks = retrieve(question, top_k=top_k)
    context = build_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    client = Mistral(api_key=MISTRAL_API_KEY)
    response = call_with_retry(
        client,
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print('Usage : python -m src.rag.generate "ta question"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"\nQuestion : {question}\n")
    print("Génération de la réponse...\n")
    print(answer(question))


if __name__ == "__main__":
    main()
