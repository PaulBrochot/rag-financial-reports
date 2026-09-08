[🇬🇧 English](README.md) | 🇫🇷 Français

# RAG sur rapports financiers (S&P 500)

Système de question-réponse basé sur les 10-K d'entreprises cotées (Apple, Microsoft,
JPMorgan Chase, Johnson & Johnson, Chevron, Coca-Cola, Walmart), avec une évaluation
rigoureuse de la qualité du retrieval et de la génération plutôt qu'un simple "ça marche".

Projet mené sur 12 mois pour valoriser une candidature stage/alternance en IA/data.

## Pipeline

```
SEC EDGAR (10-K bruts)
    -> Parsing HTML : texte narratif + tableaux financiers extraits séparément
    -> Chunking section-aware (Item 1 / 1A / 7 / 8...) + chunking des tableaux
    -> Embeddings (bge-small-en-v1.5) + indexation Chroma
    -> Retrieval + génération de réponse citée (Mistral)
    -> Évaluation LLM-as-judge (correctness, faithfulness) sur un jeu de test annoté
```

## Résultats de l'évaluation

Jeu de test de 11 questions (chiffres vérifiés manuellement), évalué avec des métriques
inspirées de RAGAS (implémentées à la main après un conflit de dépendances entre `ragas`
et le SDK Mistral récent — voir `eval/evaluate.py`) :

| Itération | Correctness | Faithfulness | Constat |
|---|---|---|---|
| Chunking taille fixe, tableaux ignorés | — | — | Échec total sur les questions chiffrées : les nombres sont dans des tableaux, supprimés du texte indexé |
| + Tableaux convertis en texte et indexés | 1.00 (7 questions simples) | 1.00 | Les questions factuelles simples (chiffre d'affaires, résultat net) passent toutes |
| + Questions qualitatives et comparatives ajoutées | 0.73 (11 questions) | 1.00 | Le score chute dès qu'on sort des lookups de chiffres uniques — l'évaluation initiale était trop facile |
| + Chunking section-aware (Item 1/1A/7/8) | 0.73 | 0.91 | Amélioration nulle sur la question en français malgré un chunking correct |

**Découverte clé** : le problème n'était ni le chunking ni le retrieval, mais un
**désalignement cross-lingue** entre les questions (français) et le modèle d'embeddings
(anglais, `bge-small-en-v1.5`). La même question posée en anglais ("What are the main
risk factors...") retrouve immédiatement la bonne section (Item 1A) et génère une
réponse correcte et bien citée. Un modèle d'embeddings multilingue (ex. `bge-m3`)
réglerait ce problème — piste identifiée pour la prochaine itération.

Point positif transversal : la **faithfulness reste élevée même en cas d'échec** — le
système reconnaît ne pas trouver l'information plutôt que d'inventer une réponse,
signe que le prompt de génération est bien conçu.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # renseigne SEC_USER_AGENT et MISTRAL_API_KEY
```

## Utilisation

```bash
# 1. Récupérer les 10-K depuis SEC EDGAR
python -m src.ingestion.fetch_edgar

# 2. Parser et chunker (texte + tableaux, découpage par section)
python -m src.processing.build_corpus

# 3. Générer les embeddings et indexer
python -m src.rag.build_index

# 4. Interroger le système
python -m src.rag.generate "Quel a été le chiffre d'affaires d'Apple en 2023 ?"

# 5. Lancer l'évaluation complète
python -m eval.evaluate
```

## Structure du projet

```
rag-financial-reports/
├── src/
│   ├── config.py              # Liste des entreprises et paramètres
│   ├── ingestion/
│   │   └── fetch_edgar.py     # Récupération des filings SEC EDGAR
│   ├── processing/
│   │   ├── parse_html.py      # Extraction texte + tableaux depuis le HTML
│   │   ├── chunking.py        # Chunking taille fixe + découpage par section
│   │   └── build_corpus.py    # Orchestration du parsing -> chunks.json
│   └── rag/
│       ├── build_index.py     # Embeddings + indexation Chroma
│       ├── retrieve.py        # Recherche des chunks pertinents
│       └── generate.py        # Pipeline RAG complet (retrieval + génération Mistral)
├── eval/
│   ├── test_set.json          # Jeu de test annoté (réponses de référence vérifiées)
│   ├── evaluate.py            # Évaluation LLM-as-judge (correctness, faithfulness)
│   └── results.csv            # Résultats détaillés de la dernière évaluation
└── data/                      # Corpus brut et traité (non versionné, régénérable)
```

## Stack technique

Python · BeautifulSoup/pandas (parsing) · sentence-transformers (embeddings locaux) ·
ChromaDB (vector store) · Mistral API (génération) · évaluation LLM-as-judge maison

## Prochaines pistes

- [ ] Modèle d'embeddings multilingue (`bge-m3`) pour corriger le désalignement français/anglais
- [ ] Retrieval multi-entités pour les questions de comparaison entre entreprises
- [ ] Re-ranking (cross-encoder) pour affiner la précision du retrieval
- [ ] Déploiement (API + interface) pour une démo accessible en ligne
