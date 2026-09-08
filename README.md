🇬🇧 English | [🇫🇷 Français](README.fr.md)

# RAG on Financial Reports (S&P 500)

A question-answering system built on 10-K filings from publicly traded companies
(Apple, Microsoft, JPMorgan Chase, Johnson & Johnson, Chevron, Coca-Cola, Walmart),
with a rigorous evaluation of retrieval and generation quality — not just "it works".

Built over 12 months as a portfolio project for internship/apprenticeship applications
in AI/data.

## Pipeline

```
SEC EDGAR (raw 10-K filings)
    -> HTML parsing: narrative text and financial tables extracted separately
    -> Section-aware chunking (Item 1 / 1A / 7 / 8...) + table chunking
    -> Embeddings (bge-small-en-v1.5) + Chroma indexing
    -> Retrieval + cited answer generation (Mistral)
    -> LLM-as-judge evaluation (correctness, faithfulness) on an annotated test set
```

## Evaluation results

Test set of 11 questions (manually verified reference answers), evaluated with metrics
inspired by RAGAS (implemented by hand after a dependency conflict between `ragas` and
the current Mistral SDK — see `eval/evaluate.py`) :

| Iteration | Correctness | Faithfulness | Finding |
|---|---|---|---|
| Fixed-size chunking, tables ignored | — | — | Complete failure on numeric questions: figures live in tables, stripped from the indexed text |
| + Tables converted to text and indexed | 1.00 (7 simple questions) | 1.00 | All simple factual questions (revenue, net income) pass |
| + Qualitative and comparative questions added | 0.73 (11 questions) | 1.00 | Score drops as soon as questions go beyond single-number lookups — the initial test set was too easy |
| + Section-aware chunking (Item 1/1A/7/8) | 0.73 | 0.91 | No improvement on the French-language question despite correct chunking |

**Key finding**: the issue was neither the chunking nor the retrieval logic, but a
**cross-lingual mismatch** between the questions (French) and the embedding model
(English-only, `bge-small-en-v1.5`). The same question asked in English ("What are the
main risk factors...") immediately retrieves the correct section (Item 1A) and produces
a correct, well-cited answer. A multilingual embedding model (e.g. `bge-m3`) would fix
this — identified as the next iteration.

A positive cross-cutting signal: **faithfulness stays high even on failures** — the
system acknowledges it can't find the answer rather than inventing one, indicating the
generation prompt is well designed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # fill in SEC_USER_AGENT and MISTRAL_API_KEY
```

## Usage

```bash
# 1. Fetch 10-K filings from SEC EDGAR
python -m src.ingestion.fetch_edgar

# 2. Parse and chunk (text + tables, section-based splitting)
python -m src.processing.build_corpus

# 3. Generate embeddings and index
python -m src.rag.build_index

# 4. Query the system
python -m src.rag.generate "What was Apple's revenue in 2023?"

# 5. Run the full evaluation
python -m eval.evaluate
```

## Project structure

```
rag-financial-reports/
├── src/
│   ├── config.py              # Company list and parameters
│   ├── ingestion/
│   │   └── fetch_edgar.py     # SEC EDGAR filing retrieval
│   ├── processing/
│   │   ├── parse_html.py      # Text + table extraction from HTML
│   │   ├── chunking.py        # Fixed-size chunking + section-based splitting
│   │   └── build_corpus.py    # Parsing orchestration -> chunks.json
│   └── rag/
│       ├── build_index.py     # Embeddings + Chroma indexing
│       ├── retrieve.py        # Relevant chunk search
│       └── generate.py        # Full RAG pipeline (retrieval + Mistral generation)
├── eval/
│   ├── test_set.json          # Annotated test set (verified reference answers)
│   ├── evaluate.py            # LLM-as-judge evaluation (correctness, faithfulness)
│   └── results.csv            # Detailed results from the latest evaluation run
└── data/                      # Raw and processed corpus (not versioned, regenerable)
```

## Tech stack

Python · BeautifulSoup/pandas (parsing) · sentence-transformers (local embeddings) ·
ChromaDB (vector store) · Mistral API (generation) · custom LLM-as-judge evaluation

## Next steps

- [ ] Multilingual embedding model (`bge-m3`) to fix the French/English mismatch
- [ ] Multi-entity retrieval for cross-company comparison questions
- [ ] Re-ranking (cross-encoder) to improve retrieval precision
- [ ] Deployment (API + UI) for an online-accessible demo
