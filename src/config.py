"""Config centrale du projet — modifie cette liste pour ajouter/retirer des entreprises."""

# Tickers de départ (diversifiés par secteur, cf. plan de projet)
COMPANIES = [
    "AAPL",   # Apple — Tech
    "MSFT",   # Microsoft — Tech
    "JPM",    # JPMorgan Chase — Finance
    "JNJ",    # Johnson & Johnson — Santé
    "CVX",    # Chevron — Énergie
    "WMT",    # Walmart — Distribution
    "KO",     # Coca-Cola — Consommation
]

# Formulaires à récupérer (10-K = annuel, 10-Q = trimestriel)
FORM_TYPES = ["10-K"]

# Combien de filings récents par entreprise
FILINGS_PER_COMPANY = 2

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
