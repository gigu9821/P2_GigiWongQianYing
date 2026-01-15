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
