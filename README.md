# P2_GigiWongQianYing
P2
# LLM-Based Financial Sentiment and Market Predictability

This repository contains the full experimental code for a master's-level research project that investigates whether Large Language Models (LLMs) can extract predictive market sentiment from financial news headlines.

The pipeline evaluates prompt-based sentiment inference on FIQA-2018 and applies the best-performing strategies to a DJIA news dataset to construct a Daily Market Sentiment Index (DMSI) and test its predictive power.

---

## Repository Structure

### 01_fiqa_preprocessing.ipynb
- Loads and cleans the FIQA-2018 dataset
- Constructs sentiment labels from continuous scores
- Injects sector information using Yahoo Finance with manual correction
- Outputs a standardized dataset for prompt experiments

### 02_fiqa_experiment_and_evaluation.ipynb
- Runs sentiment classification using 9 prompt strategies (Zero-Shot, Role-Playing, Chain-of-Thought)
- Uses Google Gemini (Flash) via prompt engineering
- Supports sample runs and full-scale execution with retry handling
- Evaluates Accuracy, Precision, Recall, and F1-score for each prompt

### 03_fnspid_experiment_and_results.ipynb
- Applies selected prompts to DJIA financial news
- Constructs the Daily Market Sentiment Index (DMSI)
- Evaluates predictability using Information Coefficient (IC)
- Performs Granger causality tests and T+2 safety checks
- Includes a simple long–short backtesting simulation

---

## Environment

- Python 3.10+
- pandas, numpy, tqdm, scikit-learn, statsmodels, matplotlib
- google-genai SDK

The notebooks request a Google GenAI API key via secure input and do not store credentials in code.

---

## Execution Order

1. Run `01_fiqa_preprocessing.ipynb`
2. Run `02_fiqa_experiment_and_evaluation.ipynb`
3. Run `03_fnspid_experiment_and_results.ipynb`

Intermediate and final results are generated automatically during execution and are not included in this repository.

# Data Directory

This folder contains the cleaned and standardized datasets used in this research.
All datasets have been preprocessed to ensure consistency, reproducibility, and direct usability in the experimental pipelines.

---

## 1. djia_news_cleaned_2021_2023.csv

**Description**  
Cleaned financial news data extracted from the FNSPID dataset, filtered to include only Dow Jones Industrial Average (DJIA) constituent companies between 2021 and 2023.

**Key preprocessing steps**
- Removed records with missing dates, tickers, or headlines  
- Filtered by DJIA constituent tickers  
- Standardized column names (`date`, `title`, `ticker`)  
- Preserved duplicate headlines across different tickers to reflect real-world news coverage

**Usage**
- Input dataset for FNSPID sentiment inference experiments  
- Used to construct the Daily Market Sentiment Index (DMSI)

---

## 2. DJIA_2021_2023.csv

**Description**  
Historical price data for the Dow Jones Industrial Average (DJIA) index covering the period from 2021 to 2023.

**Key features**
- Daily open, high, low, close, and volume information  
- Aligned to trading days for return and predictive analysis  

**Usage**
- Used as the market benchmark  
- Supports return calculation, Information Coefficient (IC), Granger causality tests, and backtesting analysis

---

## 3. fiqa_1.csv

**Description**  
Balanced and deduplicated FIQA-2018 dataset with sentiment labels derived from continuous sentiment scores.

**Key preprocessing steps**
- Merged train, validation, and test splits  
- Converted continuous sentiment scores into three classes (Positive, Neutral, Negative) using a neutral band threshold  
- Performed class balancing via stratified sampling  
- Removed fully duplicated rows after balancing  

**Usage**
- Benchmark dataset for prompt strategy evaluation  
- Used to select the optimal prompt configuration before applying it to the FNSPID dataset

# Streamlit Dashboard (Deployment)

This Streamlit app is the interactive research interface for the thesis project:
LLM Sentiment Inference → DMSI Construction → Quant Strategy Evaluation.

## Features
- Strategy Evaluation dashboard (IC leaderboard, cumulative returns, alpha, win-rate, max drawdown)
- Live DMSI Experimental Lab (Gemini-based sentiment inference on headlines + T+1 market validation)
- Academic disclaimer: theoretical gross returns (no transaction costs/slippage)

## Project Structure
app/
app.py
requirements.txt
assets/
data/


## How to Run Locally
1) Create environment & install dependencies:
```bash
pip install -r requirements.txt


Run Streamlit:
streamlit run app.py
