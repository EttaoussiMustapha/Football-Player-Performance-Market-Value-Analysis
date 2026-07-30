# ⚽ Football Player Performance & Market Value Analysis

An end-to-end Machine Learning and Data Engineering pipeline to scrape, clean, analyze, and predict the market value of professional football players using real-world data from **Transfermarkt**.

---

## 📌 Project Overview

Predicting market value in professional football is a complex task shaped by on-field performance, physical attributes, positional role, and market biases.

This project implements a complete data science workflow:

1. **Automated Web Scraping** — collected data for ~2,000+ professional players from Transfermarkt with a custom retry/backoff mechanism.
2. **Data Cleaning & Feature Engineering** — deduplication, missing-value handling, currency parsing, log transformation of the target, nationality indicators, one-hot encoding, and IQR-based outlier detection.
3. **Exploratory Data Analysis (EDA)** — descriptive statistics, distribution plots, and correlation analysis to identify the core drivers of market value.
4. **Predictive Modeling** — compared and cross-validated Linear Regression, KNN, Decision Tree, and Random Forest regressors, then tuned Random Forest with `GridSearchCV`.
5. **Unsupervised Clustering** — segmented player profiles with K-Means to surface hidden market dynamics.
6. **Inference & Scenario Analysis** — sensitivity tests on synthetic player profiles to probe model behavior (e.g. effect of nationality, effect of age).

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Scraping & Parsing:** `requests`, `beautifulsoup4`
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`

---

## 📂 Repository Structure

```text
├── Collecte_donnees.ipynb        # Web scraping (Transfermarkt)
├── nettoyage.ipynb               # Data cleaning & feature engineering
├── Analyse_exploratoire.ipynb    # Exploratory Data Analysis (EDA)
├── Machine_Learning.ipynb        # Modeling, tuning, clustering & inference
├── README.md
```

---

## 🕸️ A. Data Collection (Web Scraping)

**Source:** [Transfermarkt.com](https://www.transfermarkt.com) — "Most Valuable Players" ranking, filtered by position (`spielerposition_id`) to get past the 500-player cap on the global ranking. The player table is present directly in the server-rendered HTML, so no headless browser is needed — `requests` + `BeautifulSoup` is enough.

**Fields collected:** name, position, age, nationality, club, market value, matches played, goals, own goals, assists, yellow/red cards, and the player's profile link.

**Reliability features:**
- Retry with growing backoff (5s, 10s, 15s...) on `502` / `503` / `429` responses
- Per-row `try/except` so one malformed row doesn't stop the whole scrape
- Progressive save to CSV after every page — no data lost if the site blocks the scraper mid-run
- Deduplication on the player's profile URL

```python
def build_page_url(page_num, position_id):
    base = ("https://www.transfermarkt.com/spieler-statistik/wertvollstespieler/marktwertetop"
            f"?land_id=0&ausrichtung=alle&spielerposition_id={position_id}&altersklasse=alle"
            "&jahrgang=0&kontinent_id=0&jahr=2025&plus=1")
    return base if page_num == 1 else f"{base}&page={page_num}"

POSTES = [
    ("Goalkeeper", 1), ("Centre-Back", 3), ("Left-Back", 4), ("Right-Back", 5),
    ("Defensive Midfield", 6), ("Central Midfield", 7), ("Attacking Midfield", 10),
    ("Left Winger", 11), ("Right Winger", 12), ("Centre-Forward", 14),
]
```

Ten positions × up to 20 pages × 25 players/page → several thousand raw rows collected across the full run.

---

## 🧹 B. Data Cleaning & Feature Engineering

- **Duplicates & missing values:** dropped rows with no name, deduplicated on `(nom, club)`.
- **Currency parsing:** `€90.00m` / `€500k` style strings converted to raw euros (`valeur_marchande_eur`).
- **Log transform:** market value is heavily right-skewed (a handful of €200M stars vs. most players at €1–20M), so `valeur_marchande_log = log1p(valeur_marchande_eur)` is used as the modeling target to avoid outliers dominating the fit.
- **Nationality indicators:** one binary column per major football nation (`nat_Brazil`, `nat_England`, `nat_Spain`, `nat_France`, `nat_Argentina`, `nat_Germany`) instead of a broad continent grouping.
- **Categorical encoding:** one-hot encoding of `poste` (position) into `poste_*` columns.
- **Outlier detection:** IQR method flagged on `valeur_marchande_eur`, `matches_played`, `goals`, `assists`, and `age`.

```python
def convertir_valeur(valeur):
    if pd.isna(valeur):
        return np.nan
    valeur = str(valeur).replace("€", "").strip()
    if "m" in valeur.lower():
        return float(valeur.lower().replace("m", "")) * 1_000_000
    elif "k" in valeur.lower():
        return float(valeur.lower().replace("k", "")) * 1_000
    try:
        return float(valeur)
    except ValueError:
        return np.nan

df["valeur_marchande_eur"] = df["valeur_marchande"].apply(convertir_valeur)
df["valeur_marchande_log"] = np.log1p(df["valeur_marchande_eur"])
```

---

## 📊 C. Exploratory Data Analysis

Key findings from the correlation analysis against `valeur_marchande_log`:

| Feature | Correlation |
| :--- | :---: |
| `valeur_marchande_eur` (raw) | 0.72 |
| `matches_played` | 0.37 |
| `goals` | 0.27 |
| `assists` | 0.19 |
| `yellow_cards` | 0.18 |
| `nat_England` | 0.15 |
| `nat_France` | 0.12 |
| `age` | **-0.13** |

**Takeaways:**
- Playing time (`matches_played`) is a stronger driver of value than raw goals or assists.
- Age has a mild *negative* correlation with value — younger players tend to carry higher valuations (growth potential priced in).
- Nationality has a small but visible effect, with English and French players trending slightly higher, all else equal.

The notebook also produces histograms, boxplots (value by position, value by nationality), and a full correlation heatmap, all saved as PNGs (`histogrammes.png`, `boxplot_poste.png`, `boxplot_nationalite.png`, `matrice_correlation.png`, `variables_importantes.png`, `goals_vs_valeur.png`).

---

## 🤖 D. Modeling

### Model comparison (test set, R² on log market value)

| Model | MAE | RMSE | R² |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 0.623 | 0.807 | **0.751** |
| Random Forest | 0.636 | 0.838 | 0.731 |
| KNN | 0.657 | 0.868 | 0.711 |
| Decision Tree | 0.712 | 0.953 | 0.652 |

### 5-fold cross-validation (R²)

| Model | Mean R² |
| :--- | :---: |
| **Linear Regression** | **0.786** (± 0.019) |
| Random Forest | 0.766 (± 0.019) |
| KNN | 0.740 (± 0.022) |
| Decision Tree | 0.722 (± 0.025) |

### Hyperparameter tuning

`GridSearchCV` over Random Forest (`n_estimators`, `max_depth`, `min_samples_split`) found `{'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 200}`, reaching **R² = 0.744** — still slightly below plain Linear Regression's R² = 0.751. **Linear Regression was selected as the final model**: it's simpler, more interpretable, and matched or beat the tuned ensemble.

### Final model performance

| Metric | Value |
| :--- | :---: |
| MAE | 0.623 |
| RMSE | 0.807 |
| R² | 0.751 |

---

## 🧩 E. Unsupervised Clustering (K-Means)

Clustering on `age`, `matches_played`, `goals`, `assists`, and `valeur_marchande_log` (standardized), with *k* chosen via the elbow method and silhouette score.

**k = 3** was selected (silhouette score ≈ 0.214):

| Cluster | Age | Matches | Goals | Assists | Log Value | Players |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 25.5 | 26.5 | 2.4 | 2.6 | 13.7 | 575 |
| 1 | 24.6 | 45.1 | 10.9 | 8.2 | 16.7 | 466 |
| 2 | 22.7 | 33.4 | 2.8 | 2.6 | 16.4 | 951 |

**Interpretation:**
- **Cluster 0** — squad/rotation players: low minutes, low output, lower value.
- **Cluster 1** — key attacking performers: high minutes, high goal contribution, highest value.
- **Cluster 2** — young high-value prospects: moderate output but high value, likely driven by potential rather than current production.

---

## 🔮 F. Inference & Scenario Analysis

Using the final Linear Regression model, synthetic player profiles were fed through to test sensitivity to nationality and age (all else held constant: age 24–35, 35 matches, 18 goals, 10 assists, 4 yellow cards, Centre-Forward):

| Scenario | Predicted Value |
| :--- | :---: |
| Nationality: France | €48.7M |
| Nationality: none of the major nations | €37.0M |
| Nationality: England | €51.2M |
| Same profile, age raised to 35 | €39.8M |

This confirms the EDA signal: nationality (England/France in particular) adds a measurable premium, and age has a real depressing effect on predicted value even with identical performance stats.

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/EttaoussiMustapha/Football-Player-Performance-Market-Value-Analysis.git
cd football-market-value
```

### 2. Install Dependencies

```bash
pip install requests beautifulsoup4 pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Pipeline

Run the notebooks in order:

1. `Collecte_donnees.ipynb` — scrapes Transfermarkt and produces the raw dataset (this step is slow due to rate-limiting delays; a pre-scraped CSV can be substituted to skip it)
2. `nettoyage.ipynb` — cleans the raw data into `players_clean.csv`
3. `Analyse_exploratoire.ipynb` — EDA and visualizations
4. `Machine_Learning.ipynb` — model training, tuning, clustering, and inference

---

## 🛡️ Future Enhancements

- [ ] Add more seasons of historical data to model value trends over time.
- [ ] Try gradient boosting models (XGBoost/LightGBM) for comparison.
- [ ] Package the final model behind a simple API or Gradio demo for interactive predictions.
- [ ] Investigate injury history and contract length as additional features.

---

## 📜 License & Data Source

Data collected from [Transfermarkt.com](https://www.transfermarkt.com) for academic/educational purposes. This project is for educational use; Transfermarkt data is not redistributed as part of this repository.
