"""
Récupère les filings 10-K des entreprises listées dans src/config.py depuis SEC EDGAR.

Usage :
    python -m src.ingestion.fetch_edgar
"""
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.config import COMPANIES, FORM_TYPES, FILINGS_PER_COMPANY, RAW_DATA_DIR

load_dotenv()

USER_AGENT = os.getenv("SEC_USER_AGENT")
if not USER_AGENT:
    raise RuntimeError(
        "SEC_USER_AGENT manquant. Copie .env.example en .env et renseigne ton nom/email "
        "(la SEC bloque les requêtes sans User-Agent identifiable)."
    )

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{filename}"


def get_ticker_to_cik_map() -> dict:
    """Télécharge le mapping ticker -> CIK (identifiant SEC des entreprises)."""
    resp = requests.get(TICKERS_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # Le JSON est indexé par des entiers ; chaque valeur = {"cik_str":..., "ticker":..., "title":...}
    return {entry["ticker"]: str(entry["cik_str"]).zfill(10) for entry in data.values()}


def get_filings(cik: str) -> list[dict]:
    """Récupère la liste des filings récents pour une entreprise (par CIK)."""
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    filings = []
    for i in range(len(recent["form"])):
        filings.append({
            "form": recent["form"][i],
            "accessionNumber": recent["accessionNumber"][i],
            "primaryDocument": recent["primaryDocument"][i],
            "filingDate": recent["filingDate"][i],
            "reportDate": recent["reportDate"][i],
        })
    return filings


def download_filing(cik: str, filing: dict, ticker: str) -> Path:
    """Télécharge le document principal d'un filing et le sauvegarde localement."""
    cik_no_zeros = str(int(cik))
    accession_no_dashes = filing["accessionNumber"].replace("-", "")
    url = ARCHIVES_URL.format(
        cik_no_zeros=cik_no_zeros,
        accession_no_dashes=accession_no_dashes,
        filename=filing["primaryDocument"],
    )

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    out_dir = Path(RAW_DATA_DIR) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filing['form']}_{filing['filingDate']}_{filing['primaryDocument']}"
    out_path.write_bytes(resp.content)
    return out_path


def main():
    print("Récupération du mapping ticker -> CIK...")
    ticker_to_cik = get_ticker_to_cik_map()

    for ticker in COMPANIES:
        cik = ticker_to_cik.get(ticker)
        if not cik:
            print(f"  ⚠ Ticker {ticker} introuvable, skip.")
            continue

        print(f"\n{ticker} (CIK {cik})")
        time.sleep(0.2)  # respecte le rate limit SEC (~10 req/s max)

        filings = get_filings(cik)
        matching = [f for f in filings if f["form"] in FORM_TYPES][:FILINGS_PER_COMPANY]

        if not matching:
            print(f"  ⚠ Aucun filing {FORM_TYPES} trouvé pour {ticker}.")
            continue

        for filing in matching:
            time.sleep(0.2)
            path = download_filing(cik, filing, ticker)
            print(f"  ✓ {filing['form']} déposé le {filing['filingDate']} -> {path}")


if __name__ == "__main__":
    main()
