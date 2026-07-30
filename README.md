# Football Player Performance & Market Value Analysis

An end-to-end Machine Learning and Data Engineering pipeline designed to scrape, clean, analyze, and predict the market value of professional football players using real-world data from Transfermarkt.

---

## 📌 Project Overview

Predicting market values in professional football is a complex task influenced by on-field performance metrics, physical attributes, positional roles, and market biases. 

This project implements a complete data science workflow:
1. **Automated Web Scraping:** Extracted data for over 2,000 professional players from Transfermarkt using custom retry/backoff mechanisms.
2. **Data Engineering & Preprocessing:** Handled missing values, encoded categorical features, transformed target distributions, and eliminated data leakage.
3. **Exploratory Data Analysis (EDA):** Identified core drivers of player market values through correlation analysis and visual distribution studies.
4. **Predictive Modeling:** Evaluated and cross-validated multiple regression models (Linear Regression, Random Forest, KNN, Decision Trees) and performed hyperparameter tuning (`GridSearchCV`).
5. **Unsupervised Clustering:** Segmented player profiles using $K$-Means clustering to discover hidden market dynamics.
6. **Inference & Scenario Analysis:** Conducted sensitivity tests on unseen synthetic player profiles to evaluate model behavior and bias.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Scraping & Parsing:** `requests`, `beautifulsoup4`
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Pipeline:** `scikit-learn`

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw_players.csv         # Raw scraped dataset (~2,159 records)
│   └── cleaned_players.csv     # Preprocessed dataset (~1,992 records)
├── notebooks/
│   ├── 01_scraping.ipynb       # Web scraping script & logic
│   ├── 02_preprocessing_EDA.ipynb # Data cleaning, leakage prevention, and EDA
│   └── 03_modeling_clustering.ipynb # Model training, evaluation, tuning & KMeans
├── report/
│   └── project_report.pdf      # Detailed academic/project report
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies
