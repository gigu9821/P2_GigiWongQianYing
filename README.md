# P2_GigiWongQianYing
P2
# Financial News Sentiment → Market Signal (FIQA + FNSPID)

This repository contains an end-to-end research pipeline that transforms financial news headlines into quantitative market signals using Large Language Models (LLMs), and evaluates the predictive value of sentiment against DJIA returns.

## Project Structure

- `01_fiqa_preprocessing.ipynb`
  - Loads FIQA-2018, standardizes fields, injects sector metadata (Yahoo Finance + manual mapping),
  - Produces: `fiqa_standardized.csv`.

- `02_fiqa_experiment_and_evaluation.ipynb`
  - Runs FIQA sentiment classification using 9 prompt strategies (ZS / RP / CoT) with Gemini,
  - Saves inference outputs to CSV incrementally (supports sample run + full run),
  - Evaluates Accuracy / Precision / Recall / F1-score by prompt_id,
  - Produces: `fiqa_results.csv`, `fiqa_metrics_summary.csv`.

- `03_fnspid_experiment_and_results.ipynb`
  - Runs selected prompts on a DJIA news dataset (FNSPID-like pipeline),
  - Includes batch execution, retry handling, failure logging, and resume logic,
  - Constructs Daily Market Sentiment Index (DMSI) and computes Information Coefficient (IC),
  - Performs deeper validation (Granger test, T+2 IC safety check),
  - Produces: `fnspid_full_sentiment_results.csv`, `Final_Leaderboard.csv`, `Final_Dataset_Predictive_Best_*.csv`,
  - Includes a simple long/short backtest vs buy-and-hold DJIA.

## Data Requirements

You will need the following CSV files available in your working directory (or Google Drive if running in Colab):

- FIQA:
  - `fiqa_1.csv` (FIQA dataset exported to CSV)
  - Output: `fiqa_standardized.csv`

- DJIA / FNSPID:
  - `djia_news_cleaned_no_duplicates.csv` (news headlines with at least: date, ticker, title)
  - `DJIA_2021_2023.csv` (DJIA price history with at least: date, close)
  - `final_ticker_sector*.csv` (ticker → sector mapping)

## Environment

Recommended (Colab-friendly):
- Python 3.10+
- pandas, numpy, tqdm, scikit-learn, statsmodels, matplotlib
- google-genai SDK

Install (example):
```bash
pip install -U pandas numpy tqdm scikit-learn statsmodels matplotlib google-genai
