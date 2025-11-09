# Phishing Detection

## A project for the USU Natural Language Processing course

By Paul Harrison, Ethan Gee, Spencer Hardy

## A Deep Learning Approach to Phishing Email Classification

Phishing is a prevalent social engineering scam where attackers impersonate trusted organizations to extract sensitive
information (credentials, payment details, personal data). Since its first appearance in the 1990s, phishing has grown
more sophisticated—especially with the rise of large language models (LLMs) that enable scalable, personalized
“spear‑phishing” campaigns. These emails closely mimic legitimate messages in grammar, tone, and layout, making them
difficult to detect with traditional, keyword‑centric NLP methods. This project explores a deep learning
approach—fine‑tuning transformer models—to classify phishing emails from content and contextual signals, comparing
against strong traditional baselines.

## What is phishing and how it works

- Impersonation: Messages mimic brands, domains, and workflows to gain trust.
- Deception: Urgency, authority, or fear prompts users to click links, open attachments, or share data.
- Exploitation: Fake login pages harvest credentials; malware or fraudulent payments may follow.

Common indicators:

- Urgent calls to action (verify now, payment overdue)
- Look‑alike domains and spoofed senders
- Credential or MFA requests
- Links and attachments positioned as “security checks” or “invoices”
- Subtle stylistic anomalies despite otherwise polished language

## Why NLP for content‑based phishing detection

Traditional defenses (blacklists/whitelists, sender reputation, SPF/DKIM/DMARC, URL/attachment analysis) struggle with
zero‑day domains, compromised legitimate infrastructure, and obfuscated links. Content‑based NLP complements these by:

- Capturing semantics and intent (credential harvesting, financial requests)
- Modeling stylistic and discourse cues beyond simple keywords
- Remaining resilient when surface indicators (URLs, headers) are hidden
- Generalizing from patterns seen across varied phishing campaigns

However, simple keyword models are insufficient because modern phishing mirrors legitimate content. This motivates
transformer‑based models (e.g., BERT variants) that encode context and semantics to distinguish malicious intent from
benign communication.

## Methodology overview

1. Data collection and preprocessing
    - Source publicly available phishing/legitimate email datasets (e.g., CEAS_08.csv).
    - Clean text (remove HTML/artifacts), normalize casing/whitespace.
    - Segment metadata (subject, body, URLs, sender) for separate handling.
2. Feature representation
    - Baselines: TF‑IDF on subject/body with simple metadata features (e.g., URL presence/count).
    - Deep learning: Transformer embeddings over subject/body; optionally incorporate structured metadata.
3. Models
    - Baselines: Dummy (majority class), Random Forest, SVM.
    - Deep model (in development): Fine‑tuned transformer classifier.
4. Evaluation
    - Stratified train/test split; metrics: Accuracy, Precision, Recall, F1.
    - Emphasize Recall to minimize false negatives (missed phishing).

## Experiments (planned)

- Stage 1: Preprocess data; train/evaluate Dummy, Random Forest, SVM on TF‑IDF + metadata features.
- Stage 2: Fine‑tune transformer on subject/body (+ optional metadata); tune learning rate, batch size, epochs.
- Stage 3: Compare metrics across all models; perform error and recall‑focused analysis.

## Source data

- Primary file: data/CEAS_08.csv
- Data source: [naserabdullahalam on Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?resource=download&select=CEAS_08.csv)
- Expected columns:
  - subject: email subject line text
  - body: main email content text
  - urls: binary or count feature indicating presence/number of URLs
  - label: phishing vs. legitimate
- Notes:
  - Ensure UTF‑8 encoding and consistent headers.
  - If your CSV uses different column names or encodings, adjust preprocessing accordingly.
  - Be mindful of data leakage (e.g., headers that trivially reveal labels) and duplicates.

## Repository setup

1. Place dataset
    - Put CEAS_08.csv in data/.
2. Create and activate a virtual environment

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows (Powershell)

```bash
python -m venv .venv
.venv/scripts/activate.bat

```

3. Install requirements

```bash
pip install -r requirements.txt
```

## Execution

To train and test the model, run
```bash
python -m src.main
```
See output on standard out and model evaluation metrics in the `artifacts/` directory.

## Typical workflow

- Load and clean CEAS_08.csv; stratified train/test split.
- Vectorize:
  - Baseline: TF‑IDF (unigrams/bigrams) on subject/body + simple metadata features.
  - Deep: Tokenize for transformer; build inputs from subject/body (+ optional metadata).
- Train/evaluate:
  - Baselines: Dummy, Random Forest, SVM.
  - Deep: Fine‑tuned transformer classifier.
- Report Accuracy, Precision, Recall, F1; analyze confusion matrix and hard errors.

## Notes and considerations

- Class imbalance is common; use stratification and recall‑oriented evaluation.
- Validate robustness with time‑split or out-of-distribution tests to handle drift.
- Ethics: Handle potentially sensitive content responsibly; anonymize when appropriate.
