"""
Évaluation du pipeline RAG sur un jeu de test annoté.

Note méthodologique : la librairie `ragas` a des dépendances qui entrent en conflit
avec la version récente du SDK Mistral utilisée dans ce projet (instructor/langchain
attendent une ancienne structure de mistralai). On implémente donc nous-mêmes les
métriques clés qu'elle propose, en utilisant Mistral comme "juge" — même principe
méthodologique (LLM-as-judge), sans le problème de compatibilité.

Métriques :
- correctness   : la réponse générée correspond-elle à la réponse de référence ?
- faithfulness  : la réponse est-elle bien étayée par le contexte récupéré
                  (détecte les hallucinations) ?

Usage :
    python -m eval.evaluate
"""
import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral

from src.rag.generate import answer, build_context, call_with_retry, MODEL
from src.rag.retrieve import retrieve

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY manquant dans .env")

TEST_SET_PATH = Path("eval/test_set.json")
RESULTS_PATH = Path("eval/results.csv")
TOP_K = 3

CORRECTNESS_PROMPT = """Tu es un évaluateur strict. Compare la réponse donnée à la réponse de référence.
Réponds UNIQUEMENT par un chiffre : 1 si la réponse donnée est correcte et cohérente avec la référence \
(même si la formulation diffère), 0.5 si elle est partiellement correcte, 0 si elle est fausse ou ne répond pas à la question.

Question : {question}
Réponse de référence : {ground_truth}
Réponse donnée : {answer}

Score (0, 0.5 ou 1) :"""

FAITHFULNESS_PROMPT = """Tu es un évaluateur strict. Vérifie si la réponse ci-dessous est entièrement \
étayée par le contexte fourni (aucune information inventée ou absente du contexte).
Réponds UNIQUEMENT par un chiffre : 1 si tout est étayé, 0.5 si partiellement, 0 si des éléments \
importants ne sont pas dans le contexte.

Contexte :
{context}

Réponse à vérifier : {answer}

Score (0, 0.5 ou 1) :"""


def judge_score(client: Mistral, prompt: str) -> float:
    """Demande à Mistral d'évaluer et extrait le score numérique de sa réponse."""
    response = call_with_retry(
        client,
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    match = re.search(r"0\.5|[01]", text)
    return float(match.group()) if match else 0.0


def evaluate_question(client: Mistral, item: dict) -> dict:
    question = item["question"]
    ground_truth = item["ground_truth"]

    chunks = retrieve(question, top_k=TOP_K)
    context = build_context(chunks)
    generated_answer = answer(question, top_k=TOP_K)

    time.sleep(1)
    correctness = judge_score(client, CORRECTNESS_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=generated_answer
    ))
    time.sleep(1)
    faithfulness = judge_score(client, FAITHFULNESS_PROMPT.format(
        context=context, answer=generated_answer
    ))

    return {
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": generated_answer,
        "correctness": correctness,
        "faithfulness": faithfulness,
    }


def main():
    test_set = json.loads(TEST_SET_PATH.read_text())
    client = Mistral(api_key=MISTRAL_API_KEY)

    results = []
    for i, item in enumerate(test_set, 1):
        print(f"[{i}/{len(test_set)}] {item['question']}")
        result = evaluate_question(client, item)
        results.append(result)
        print(f"  -> correctness={result['correctness']} faithfulness={result['faithfulness']}")
        time.sleep(1)

    avg_correctness = sum(r["correctness"] for r in results) / len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)

    print("\n=== Résumé ===")
    print(f"Correctness moyenne  : {avg_correctness:.2f} / 1.0")
    print(f"Faithfulness moyenne : {avg_faithfulness:.2f} / 1.0")

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "ground_truth", "generated_answer", "correctness", "faithfulness"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nRésultats détaillés sauvegardés dans {RESULTS_PATH}")


if __name__ == "__main__":
    main()
