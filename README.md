# Buy-or-Wait Financial Agent

An AI-assisted financial affordability project that evaluates whether a user can safely afford a requested expense using cash-flow forecasting, recurring transaction analysis, payment-plan optimization, and deterministic financial safety checks.

> Built for the HackerRank Orchestrate September 2026 **Buy or Wait?** challenge.

## What it does

For each purchase request, the agent produces:

- maximum amount safe to pay today
- affordability status
- recommended payment method
- payment plan
- earliest date for full payment
- spending changes needed, when applicable
- concise decision explanation

The core decision engine is deliberately deterministic: financial safety decisions are made from transaction data, dates, supplied payment options, and explicit constraints rather than from unconstrained text generation.

## Architecture

```text
Input CSVs / linked images
          │
          ▼
   Data normalization
          │
          ├── Currency conversion
          ├── Missing-amount extraction (OCR + controlled fallback)
          ├── Message/event interpretation
          └── Transaction status filtering
          │
          ▼
 Historical recurring-pattern detection
          │
          ▼
      90-day forecast
          │
          ▼
 Candidate payment plans
          │
          ├── Full payment
          ├── Partial payment
          ├── Installments
          ├── Spending-change plans
          └── Wait
          │
          ▼
 Deterministic safety simulation
          │
          ▼
 Constraint-based ranking
          │
          ▼
             output.csv
```

## Key engineering ideas

### 1. 90-day cash-flow safety

The engine builds a daily balance timeline over the forecast horizon and tests candidate payments against the user's minimum safe balance.

### 2. Recurring expense/income detection

Historical settled transactions are grouped by event type, category, and direction. Recurrence is inferred from repeated dates and interval consistency, with conservative thresholds to avoid treating isolated transactions as recurring cash flow.

### 3. Payment-plan optimization

Candidate plans are generated and evaluated against the deadline and cash-flow constraints. Feasible plans are ranked using the challenge's decision priorities, including completion by deadline, avoiding spending changes, total paid, start date, number of payments, and payment-option ID.

### 4. Spending-change optimization

Only eligible flexible recurring expenses are considered for reduction or stopping. The search is bounded to at most three changes and evaluates combinations systematically.

### 5. Image and message processing

When a financial event has a missing amount, the linked image can be processed with OCR. Structured message parsing extracts supported financial facts such as salary changes, invoice information, and dates. The financial engine remains deterministic after extraction.

## Technology

- Python 3
- Pandas
- NumPy
- Pillow
- pytesseract (optional OCR)
- deterministic rule/constraint engine
- time-series cash-flow simulation
- combinatorial candidate-plan search

## AI usage

ChatGPT was used during development for architecture brainstorming, edge-case analysis, documentation, and interview preparation. The submitted runtime does **not** make a live LLM API call. Financial decisions are computed by the deterministic engine, while OCR and structured parsing handle the supported unstructured inputs.

A production extension would add a lightweight LLM/classifier for more robust message interpretation while keeping the financial safety engine deterministic and independently verifiable.

See [`docs/AI_USAGE.md`](docs/AI_USAGE.md) for the boundary between development-time AI assistance and runtime computation.

## Repository structure

```text
buy-or-wait-financial-agent/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── src/
│   └── main.py
├── evaluation/
│   ├── main.py
│   └── usage_report.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FINANCIAL_ENGINE.md
│   ├── PAYMENT_PLANNING.md
│   ├── VALIDATION.md
│   └── AI_USAGE.md
└── results/
    └── output.csv
```

The original competition dataset is intentionally not included in this public repository.

## Running the solution

Place the competition dataset in a local `dataset/` directory, then run:

```bash
pip install -r requirements.txt
python src/main.py
```

The solution expects the challenge dataset layout and writes the requested output CSV.

## Evaluation

The `evaluation/` directory contains the validation helper and runtime usage report used during development/submission preparation.

## Disclaimer

This is a competition/portfolio project, not a personal financial advisory service. It should not be used to make real financial decisions without independent verification.
