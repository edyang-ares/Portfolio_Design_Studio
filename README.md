# Portfolio Toolkit

A portfolio construction and simulation toolkit for modeling Ares-style fund cashflows, converting them into portfolio-ready formats, and running scenario-based portfolio analyses from either structured inputs or free-form natural language.

## What this toolkit does

This toolkit supports a workflow that goes from fund-level assumptions to portfolio-level scenario analysis:

1. Load fund assumptions from an MFR-style source file
2. Model gross fund cashflows
3. Apply management fees and performance fees
4. Convert modeled cashflows into the format expected by the portfolio engine
5. Construct and run portfolio scenarios
6. Optionally parse free-form user requests into model inputs using the OpenAI API

The current primary entry points are:
- `run_master_demo_from_names(...)`
- `run_master_demo_from_text(...)`

The main orchestration lives in `master_demo.py`, and the natural-language parsing layer lives in `openai_master_demo_parser.py`. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## Main workflow

### Structured-input workflow
Use `run_master_demo_from_names(...)` when you already know:
- which funds you want
- coinvest multipliers
- split mode
- commitment bounds
- commitment mode
- portfolio size and horizon

This path is implemented in `master_demo.py`. It loads MFR data, generates fund cashflows, optionally creates separate co-invest sleeves, aggregates portfolio results, and returns a dictionary of cases. :contentReference[oaicite:2]{index=2}

### Natural-language workflow
Use `run_master_demo_from_text(...)` when you want to type something like:

> Run a set and forget portfolio with SDL IV (Levered) and Pathfinder III, equal split, 0 and 0.5 coinvest multipliers, min 50 million, max 300 million, 1 billion portfolio size, 10-year life.

This path is implemented in `openai_master_demo_parser.py`. It:
1. calls the OpenAI Responses API with structured output
2. normalizes and validates the parsed fields
3. passes them into `run_master_demo_from_names(...)` :contentReference[oaicite:3]{index=3}

---

## Suggested folder structure

```text
portfolio_toolkit/
├── README.md
├── requirements.txt
├── .env.example
│
├── core/
│   ├── master_demo.py
│   ├── openai_master_demo_parser.py
│   ├── openai_client.py
│   ├── fund_deployment_model.py
│   ├── fund_deployment_model_runner.py
│   ├── adapters.py
│   ├── data.py
│   └── GenPortvN.py
│
├── experimental/
│   ├── portfolio_constructor.py
│   └── GenPortSimulator.py
│
├── legacy/
│   └── analytics.py
│
├── notebooks/
│   └── openai_demo.ipynb
│
├── inputs/
│   ├── MFR_PYTHON_2026.04.csv
│   └── LIBOR SOFR Forward.xlsx
│
└── outputs/