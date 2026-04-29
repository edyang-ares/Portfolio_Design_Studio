"""
openai_integrated_portfolio_parser.py
======================================

Natural-language interface to ``run_integrated_portfolio_simulation`` powered
by the OpenAI Responses API with structured JSON output.

This module mirrors the design of ``openai_master_demo_parser.py`` but targets
the richer ``integrated_portfolio_simulation`` engine which supports:

* MFR-generated **and** Preqin / Burgiss-selected private funds
* Multiple public-asset sleeves (not a single hard-coded "Public Sleeve")
* Commitment pacing (one-time, annual, or custom schedule)
* Full timecycle_drawdown simulation controls

Three public functions
----------------------
``parse_integrated_portfolio_request(user_text)``
    LLM parse → raw dict
``normalize_integrated_portfolio_inputs(parsed)``
    Fill defaults, coerce types, validate → clean dict
``run_integrated_portfolio_simulation_from_text(user_text, **data_sources)``
    End-to-end: parse → normalize → run simulation → return results

Examples
--------
>>> from openai_integrated_portfolio_parser import (
...     run_integrated_portfolio_simulation_from_text,
... )
>>> output = run_integrated_portfolio_simulation_from_text(
...     "Run a 10-year MFR portfolio with SDL IV (Levered) and Pathfinder III, "
...     "equal split, annual commitments, 1B size, 40% cash, 20% equity at 7% "
...     "return, quarterly, no rebalance.",
...     mfr_source="Data Files/MFR_PYTHON_2026.04.csv",
...     public_returnstreams={"SPY": spy_df},  # optional historical returns
... )
>>> output["parsed_inputs"]   # dict
>>> output["results"]          # simulation output
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .openai_client import make_openai_client
from .integrated_portfolio_simulation import (
    run_integrated_portfolio_simulation,
    PublicAssetConfig,
    PrivateGenConfig,
    PrivateSelConfig,
)

# ---------------------------------------------------------------------------
# OpenAI client (lazy singleton — instantiated once on first use)
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Return a cached OpenAI client instance."""
    global _client
    if _client is None:
        _client = make_openai_client()
    return _client


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-5.4"


# ═══════════════════════════════════════════════════════════════════════════
# JSON Schema for structured output
# ═══════════════════════════════════════════════════════════════════════════

_PUBLIC_ASSET_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Display name for this public asset "
                "(e.g. 'Cash', 'US Equity', 'SPY', 'Fixed Income')."
            ),
        },
        "target_weight": {
            "type": "number",
            "description": (
                "Target portfolio weight for this asset as a decimal "
                "(e.g. 0.20 for 20%).  Must be between 0 and 1."
            ),
        },
        "price0": {
            "type": "number",
            "description": "Starting unit price (default 1.0).",
        },
        "a_return": {
            "type": "number",
            "description": (
                "Expected periodic return (quarterly if periodicity=Q). "
                "Convert annual returns to quarterly: quarterly ≈ annual/4. "
                "Default 0.0."
            ),
        },
        "volatility": {
            "type": "number",
            "description": (
                "Standard deviation of periodic returns (quarterly). "
                "Default 0.0."
            ),
        },
        "a_income": {
            "type": "number",
            "description": (
                "Expected periodic income yield (quarterly). Default 0.0."
            ),
        },
        "income_volatility": {
            "type": "number",
            "description": "Std dev of income yield. Default 0.0.",
        },
        "asset_class": {
            "type": "string",
            "description": (
                "Asset class label (e.g. 'Cash', 'Equity', 'Fixed Income', "
                "'Real Estate'). Default 'Generic'."
            ),
        },
        "reinvestment_rate": {
            "type": "number",
            "description": (
                "Fraction of income reinvested into this asset (0-1). "
                "Default 1.0 (100% reinvested)."
            ),
        },
        "liquidity": {
            "type": "integer",
            "description": (
                "How often this asset can be liquidated in periods. "
                "1 = every period. Default 1."
            ),
        },
        "sub_period": {
            "type": "integer",
            "description": (
                "How often subscriptions into this asset are allowed. "
                "Default 1."
            ),
        },
        "prorate": {
            "type": "number",
            "description": (
                "Fraction of value realisable in a redemption event (0-1). "
                "Default 1.0."
            ),
        },
        "periodicity": {
            "type": "string",
            "enum": ["Q"], #["D", "M", "Q", "Y"],
            "description": "Must match portfolio periodicity. Default 'Q'.",
        },
        "returnstream_name": {
            "type": ["string", "null"],
            "description": (
                "Optional name / ticker of a historical return stream. "
                "null if not specified (use simulated returns)."
            ),
        },
    },
    "required": [
        "name", "target_weight", "price0", "a_return", "volatility",
        "a_income", "income_volatility", "asset_class", "reinvestment_rate",
        "liquidity", "sub_period", "prorate", "periodicity",
        "returnstream_name",
    ],
    "additionalProperties": False,
}


INTEGRATED_PORTFOLIO_RESPONSE_SCHEMA: dict = {
    "type": "json_schema",
    "name": "integrated_portfolio_inputs",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            # ── Source & mode ────────────────────────────────────────────
            "private_source": {
                "type": "string",
                "enum": ["mfr", "preqin", "burgiss"],
                "description": (
                    "Private fund data source. "
                    "'mfr' for Ares MFR-modelled funds, "
                    "'preqin' for Preqin SQL Server, "
                    "'burgiss' for Burgiss Excel."
                ),
            },
            "private_build_mode": {
                "type": "string",
                "enum": ["generated", "selected"],
                "description": (
                    "'generated' = model cashflows from MFR inputs; "
                    "'selected' = draw historical cashflows via fund_selector."
                ),
            },

            # ── Portfolio basics ─────────────────────────────────────────
            "portfolio_name": {
                "type": ["string", "null"],
                "description": "Human-readable portfolio label. Default null.",
            },
            "ptf_size": {
                "type": "number",
                "description": (
                    "Total portfolio size in dollars. "
                    "Convert shorthand: 1B=1000000000, 500M=500000000."
                ),
            },
            "ptf_life": {
                "type": "integer",
                "description": "Portfolio life in years. Default 10.",
            },
            "periodicity": {
                "type": "string",
                "enum": ["D", "M", "Q", "Y"],
                "description": "Simulation periodicity. Default 'Q' (quarterly).",
            },
            "startdate": {
                "type": ["string", "null"],
                "description": (
                    "Simulation start date as YYYY-MM-DD string. "
                    "null = auto-generate next quarter-end."
                ),
            },
            "historical": {
                "type": "boolean",
                "description": (
                    "Use historical return streams for public assets. "
                    "Default false."
                ),
            },

            # ── Simulation controls ──────────────────────────────────────
            "rebalance_method": {
                "type": "string",
                "enum": ["No Rebalance", "Priority", "Pro-Rata"],
                "description": "Rebalancing method. Default 'No Rebalance'.",
            },
            "rebal_periodicity": {
                "type": "integer",
                "description": (
                    "Rebalance every N periods. "
                    "Default 999999999 (effectively never)."
                ),
            },
            "red_base": {
                "type": "string",
                "enum": ["Fixed", "NAV", "Dist"],
                "description": "Redemption base. Default 'Fixed'.",
            },
            "redemption_rate": {
                "type": ["number", "null"],
                "description": (
                    "Periodic redemption rate as a decimal. "
                    "null or 0 = no redemptions. "
                    "E.g. 0.02 for 2% quarterly redemptions."
                ),
            },
            "redemption_years": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Which simulation years have redemptions. "
                    "Empty list = every year (if redemption_rate > 0)."
                ),
            },
            "subscription_amount": {
                "type": ["number", "null"],
                "description": (
                    "Dollar amount of new subscriptions per period. "
                    "This is an absolute cash amount, NOT a rate. "
                    "null or 0 = no subscriptions. "
                    "E.g. 5000000 for $5M per period."
                ),
            },
            "subscription_years": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Which simulation years have subscriptions. "
                    "Empty list = none."
                ),
            },
            "red_max": {
                "type": "boolean",
                "description": "Cap redemptions to max_pct of NAV. Default true.",
            },
            "max_pct": {
                "type": "number",
                "description": (
                    "Max redemption as fraction of NAV (if red_max=true). "
                    "Default 0.05."
                ),
            },
            "sub_max": {
                "type": "boolean",
                "description": "Cap subscriptions. Default false.",
            },
            "smax_pct": {
                "type": "number",
                "description": "Max subscription fraction of NAV. Default 0.0.",
            },
            "earlybreak": {
                "type": "boolean",
                "description": (
                    "Stop simulation on unmet capital calls / redemptions. "
                    "Default true."
                ),
            },
            "growfirst": {
                "type": "boolean",
                "description": "Apply growth in the first period. Default false.",
            },
            "random_seed": {
                "type": ["integer", "null"],
                "description": "Random seed for reproducibility. null if not set.",
            },
            "commitment_pacing": {
                "type": "string",
                "enum": ["one_time", "annual", "schedule"],
                "description": (
                    "'one_time' = set-and-forget; "
                    "'annual' = recommit each year; "
                    "'schedule' = use explicit year-by-fund commitments. "
                ),
            },

            # ── MFR-generated private config ─────────────────────────────
            "fund_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "MFR Strategy Short Names (e.g. 'SDL IV (Levered)'). "
                    "Required when private_source='mfr'."
                ),
            },
            "split_mode": {
                "type": "string",
                "enum": ["equal", "explicit"],
                "description": (
                    "Fund weight allocation mode. "
                    "'equal' for equal-weight; 'explicit' for custom weights."
                ),
            },
            "explicit_fund_weights": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "fund_name": {"type": "string"},
                        "weight": {"type": "number"},
                    },
                    "required": ["fund_name", "weight"],
                    "additionalProperties": False,
                },
                "description": (
                    "List of {fund_name, weight} when split_mode='explicit'. "
                    "null when split_mode='equal'."
                ),
            },
            "coinvest_multipliers": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "fund_name": {"type": "string"},
                        "multiplier": {"type": "number"},
                    },
                    "required": ["fund_name", "multiplier"],
                    "additionalProperties": False,
                },
                "description": (
                    "Per-fund co-invest multipliers. "
                    "null or empty if no co-invest."
                ),
            },
            "min_commitment": {
                "type": "number",
                "description": "Minimum per-fund commitment in dollars. Default 0.",
            },
            "max_commitment": {
                "type": ["number", "null"],
                "description": (
                    "Maximum per-fund commitment in dollars. null = no cap."
                ),
            },
            "start_year": {
                "type": ["integer", "null"],
                "description": (
                    "Override vintage / start year for generated funds. "
                    "null = use MFR file defaults."
                ),
            },

            "commitment_schedule": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "year_offset": {
                            "type": "integer",
                            "description": "Simulation year offset, where 0 is the start year."
                        },
                        "fund_name": {
                            "type": "string",
                            "description": "MFR fund name."
                        },
                        "commitment": {
                            "type": "number",
                            "description": "Dollar commitment for this fund in this simulation year."
                        }
                    },
                    "required": ["year_offset", "fund_name", "commitment"],
                    "additionalProperties": False,
                },
                "description": (
                    "Explicit commitment schedule. Use only when commitment_pacing='schedule'. "
                    "Each row is one fund/year commitment."
                ),
            },

            # ── Selected private config (Preqin / Burgiss) ──────────────
            "init_funds": {
                "type": "integer",
                "description": (
                    "Number of funds to randomly select. "
                    "Used when private_build_mode='selected'. Default 5."
                ),
            },
            "target_base": {
                "type": "string",
                "enum": ["Percentage","Target", "Dollar"],
                "description": (
                    "'Percentage', 'Target' or 'Dollar' for sizing the private allocation."
                ),
            },
            "target_private": {
                "type": "number",
                "description": (
                    "If target_base='Target': fraction of portfolio (e.g. 0.30). "
                    "If target_base='Dollar': absolute dollar amount."
                ),
            },
            "init_age": {
                "type": "integer",
                "description": "Fund age in years at selection. Default 0.",
            },
            "select_year": {
                "type": "boolean",
                "description": (
                    "Select funds from a specific vintage year. Default true."
                ),
            },
            "d_year": {
                "type": ["integer", "null"],
                "description": (
                    "Vintage year for fund selection. "
                    "Required if select_year=true."
                ),
            },
            "replacement": {
                "type": "boolean",
                "description": "Sample with replacement. Default true.",
            },
            "overcommit_pct": {
                "type": "number",
                "description": "Overcommitment as fraction of NAV. Default 0.",
            },
            "sel_strategy": {
                "type": ["string", "null"],
                "description": (
                    "Strategy filter for Preqin/Burgiss selection "
                    "(e.g. 'Infrastructure', 'Buyout'). null = no filter."
                ),
            },
            "sel_currency": {
                "type": ["string", "null"],
                "description": (
                    "Currency filter for selection (e.g. 'USD'). null = no filter."
                ),
            },

            # ── Public assets ────────────────────────────────────────────
            "public_assets": {
                "type": "array",
                "items": _PUBLIC_ASSET_SCHEMA,
                "description": (
                    "Array of public / liquid asset configurations. "
                    "Each asset is independent — do NOT collapse multiple "
                    "assets into one sleeve."
                ),
            },
            "cash_weight": {
                "type": "number",
                "description": (
                    "Target weight for a cash sleeve. "
                    "If > 0 and no asset named 'Cash' is in public_assets, "
                    "a default cash asset will be added. Default 0."
                ),
            },

            # ── Line of credit ───────────────────────────────────────────
            "line_of_credit_max": {
                "type": ["number", "null"],
                "description": (
                    "Maximum credit facility balance. null = no facility."
                ),
            },
            "line_of_credit_rate": {
                "type": ["number", "null"],
                "description": "Annual interest rate on the facility. Default 0.05.",
            },
        },
        "required": [
            "private_source", "private_build_mode",
            "portfolio_name", "ptf_size", "ptf_life",
            "periodicity", "startdate", "historical",
            "rebalance_method", "rebal_periodicity",
            "red_base", "redemption_rate", "redemption_years",
            "subscription_amount", "subscription_years",
            "red_max", "max_pct", "sub_max", "smax_pct",
            "earlybreak", "growfirst", "random_seed",
            "commitment_pacing",
            "fund_names", "split_mode", "explicit_fund_weights",
            "coinvest_multipliers", "min_commitment", "max_commitment",
            "start_year", "commitment_schedule",
            "init_funds", "target_base", "target_private",
            "init_age", "select_year", "d_year", "replacement",
            "overcommit_pct", "sel_strategy", "sel_currency",
            "public_assets", "cash_weight",
            "line_of_credit_max", "line_of_credit_rate",
        ],
        "additionalProperties": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# System prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a structured-data extraction assistant for an integrated portfolio
simulation tool that models private funds and public assets together.

Your job: read the user's free-form request and return a single JSON object
whose fields match the schema exactly.  Follow these rules:

1. **Extract ONLY the fields defined in the schema.**

2. **Choose reasonable defaults when the user omits a value:**
   - private_source → "mfr"
   - private_build_mode → "generated" (if source is "mfr"), "selected" (otherwise)
   - portfolio_name → null
   - ptf_size → 1000000000  (1 billion)
   - ptf_life → 10
   - periodicity → "Q"
   - startdate → null  (auto-generate)
   - historical → false
   - rebalance_method → "No Rebalance"
   - rebal_periodicity → 999999999
   - red_base → "Fixed"
   - redemption_rate → null  (no redemptions)
   - redemption_years → []
   - subscription_amount → null
   - subscription_years → []
   - red_max → true
   - max_pct → 0.05
   - sub_max → false
   - smax_pct → 0.0
   - earlybreak → true
   - growfirst → false
   - random_seed → null
   - commitment_pacing → "one_time"
   - commitment_schedule → null
   - fund_names → []
   - split_mode → "equal"
   - explicit_fund_weights → null
   - coinvest_multipliers → null
   - min_commitment → 0
   - max_commitment → null
   - start_year → null
   - init_funds → 5
   - target_base → "Target"
   - target_private → 0.30
   - init_age → 0
   - select_year → true
   - d_year → null
   - replacement → true
   - overcommit_pct → 0.0
   - sel_strategy → null
   - sel_currency → null
   - cash_weight → 0.0
   - line_of_credit_max → null
   - line_of_credit_rate → null

3. **Interpret natural-language synonyms:**
   - "set and forget", "one-time", "single commitment" → commitment_pacing = "one_time"
   - "annual commitments", "re-up every year", "annual recommit" → commitment_pacing = "annual"
   - "custom schedule", "commit in years 0 and 2", "first and third years", "commit $X to fund Y in year Z" → commitment_pacing = "schedule"
   - "no rebalance" → rebalance_method = "No Rebalance"
   - "priority rebalance" → rebalance_method = "Priority"
   - "pro-rata rebalance", "pro rata" → rebalance_method = "Pro-Rata"
   - "historical public returns", "real returns" → historical = true
   - "MFR", "Ares", "modelled" → private_source = "mfr", private_build_mode = "generated"
   - "Preqin" → private_source = "preqin", private_build_mode = "selected"
   - "Burgiss" → private_source = "burgiss", private_build_mode = "selected"
   - "selected funds", "historical funds" → private_build_mode = "selected"
   - "equal split", "split evenly" → split_mode = "equal"
   - A user mentioning specific fund names with no Preqin/Burgiss → private_source = "mfr"

4. **Dollar amounts:** convert shorthand:
   - "50M" → 50000000, "1B" → 1000000000, "500K" → 500000
   - "20%" → 0.20 (for weights or rates)

5. **Return parameters:** Convert annual returns/income to quarterly by dividing
   by 4.  For example, "7% annual return" → a_return = 0.0175.
   Convert annual volatility to quarterly by dividing by 2 (approx sqrt(4)).
   For example, "15% annual volatility" → volatility = 0.075.

6. **Public assets:**
   - Each distinct public asset gets its own entry in the public_assets array.
   - If the user mentions "cash" or "liquid reserve", add it as a separate asset
     with asset_class="Cash", low return, low volatility, and set cash_weight too.
   - If the user mentions a ticker (SPY, AGG, etc.) but provides no return
     assumptions, use reasonable defaults:
       SPY: a_return=0.02, volatility=0.08, a_income=0.005, asset_class="Equity"
       AGG: a_return=0.01, volatility=0.02, a_income=0.008, asset_class="Fixed Income"
   - If the user gives an annual return like "7%", convert to quarterly (~1.75%).
   - Do NOT collapse multiple assets into a single entry.

7. **Source/mode compatibility:**
   - private_source="mfr" MUST use private_build_mode="generated"
   - private_source="preqin" or "burgiss" MUST use private_build_mode="selected"

8. **Subscriptions vs redemptions:**
   - `redemption_rate` is a per-period RATE (e.g. 0.02 for 2%).
   - `subscription_amount` is a per-period DOLLAR AMOUNT (e.g. 5000000 for $5M).
   - Do NOT use a rate for subscriptions — always use an absolute amount.
   - If the user says "$10M subscriptions in years 3-5", set subscription_amount=10000000,
     subscription_years=[3,4,5].

9. **Historical return streams:**
   - If the user says "historical returns" or "real return data", set historical=true.
   - Set `returnstream_name` on public assets to a ticker (e.g. "SPY", "AGG") when
     the user references one.
   - The caller must supply actual return DataFrames separately; the LLM only names them.

10. Return ONLY schema-conforming values — no commentary, no markdown.

11. For commitment_schedule:
- Use year_offset where 0 is the simulation start year.
- "first year" means year_offset=0.
- "second year" means year_offset=1.
- "third year" means year_offset=2.
- Each row should be one fund/year commitment.

"""


# ═══════════════════════════════════════════════════════════════════════════
# 1) LLM Parse
# ═══════════════════════════════════════════════════════════════════════════

def parse_integrated_portfolio_request(user_text: str) -> dict:
    """Send *user_text* to OpenAI and return a structured dict of inputs.

    Uses the Responses API with ``text.format = json_schema`` so the model
    is forced to return only schema-valid JSON.

    Parameters
    ----------
    user_text : str
        Free-form natural-language description of a portfolio scenario.

    Returns
    -------
    dict
        Parsed fields matching the integrated portfolio simulation schema.

    Raises
    ------
    RuntimeError
        If the model refuses or returns unparseable output.
    """
    client = _get_client()

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        text={"format": INTEGRATED_PORTFOLIO_RESPONSE_SCHEMA},
    )

    raw_text = response.output_text

    if not raw_text:
        raise RuntimeError(
            "OpenAI returned an empty response. "
            "The model may have refused the request."
        )

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse model JSON output: {exc}\nRaw text: {raw_text}"
        ) from exc

    return parsed


# ═══════════════════════════════════════════════════════════════════════════
# 2) Normalize & Validate
# ═══════════════════════════════════════════════════════════════════════════

def normalize_integrated_portfolio_inputs(parsed: dict) -> dict:
    """Fill defaults, coerce types, and validate the parsed dict.

    This function is **deterministic** — no LLM calls.  It takes the raw
    dict produced by ``parse_integrated_portfolio_request`` and returns a
    cleaned dict that can be unpacked into the downstream builders.

    The returned dict has two sub-dicts (``"generated_config_kwargs"`` and
    ``"selected_config_kwargs"``) plus top-level simulation parameters,
    ready for ``run_integrated_portfolio_simulation``.

    Parameters
    ----------
    parsed : dict
        Raw output from :func:`parse_integrated_portfolio_request`.

    Returns
    -------
    dict
        Cleaned, validated parameter dict.

    Raises
    ------
    ValueError
        On any invalid or inconsistent combination of inputs.
    """
    out: Dict[str, Any] = {}

    # ── Source & mode ────────────────────────────────────────────────────
    src = parsed.get("private_source", "mfr")
    if src not in ("mfr", "preqin", "burgiss"):
        raise ValueError(
            f"private_source must be 'mfr', 'preqin', or 'burgiss' — got '{src}'."
        )
    out["private_source"] = src

    mode = parsed.get("private_build_mode", "generated" if src == "mfr" else "selected")
    if mode not in ("generated", "selected"):
        raise ValueError(
            f"private_build_mode must be 'generated' or 'selected' — got '{mode}'."
        )
    # Enforce compatibility
    if src == "mfr" and mode == "selected":
        raise ValueError(
            "private_source='mfr' only supports private_build_mode='generated'. "
            "Use 'preqin' or 'burgiss' for 'selected' mode."
        )
    if src in ("preqin", "burgiss") and mode == "generated":
        raise ValueError(
            f"private_source='{src}' only supports private_build_mode='selected'. "
            "Use 'mfr' for 'generated' mode."
        )
    out["private_build_mode"] = mode

    # ── Portfolio basics ─────────────────────────────────────────────────
    out["portfolio_name"] = parsed.get("portfolio_name") or "Integrated Portfolio"

    ptf_size = float(parsed.get("ptf_size", 1_000_000_000) or 1_000_000_000)
    if ptf_size <= 0:
        raise ValueError(f"ptf_size must be > 0, got {ptf_size}.")
    out["ptf_size"] = ptf_size

    out["ptf_life"] = int(parsed.get("ptf_life", 10) or 10)
    if out["ptf_life"] < 1:
        raise ValueError(f"ptf_life must be >= 1, got {out['ptf_life']}.")

    period = parsed.get("periodicity", "Q") or "Q"
    if period != "Q": #not in ("D", "M", "Q", "Y"):
        period = "Q"
    out["periodicity"] = period

    out["startdate"] = parsed.get("startdate")  # None is fine
    out["historical"] = bool(parsed.get("historical", False))

    # ── Simulation controls ──────────────────────────────────────────────
    rebal = parsed.get("rebalance_method", "No Rebalance") or "No Rebalance"
    if rebal not in ("No Rebalance", "Priority", "Pro-Rata"):
        raise ValueError(
            f"rebalance_method must be 'No Rebalance', 'Priority', or "
            f"'Pro-Rata' — got '{rebal}'."
        )
    out["rebalance_method"] = rebal

    out["rebal_periodicity"] = int(
        parsed.get("rebal_periodicity", 999999999) or 999999999
    )

    red_base = parsed.get("red_base", "Fixed") or "Fixed"
    if red_base not in ("Fixed", "NAV", "Dist"):
        red_base = "Fixed"
    out["red_base"] = red_base

    out["red_max"] = bool(parsed.get("red_max", True))
    out["max_pct"] = float(parsed.get("max_pct", 0.05) or 0.05)
    out["sub_max"] = bool(parsed.get("sub_max", False))
    out["smax_pct"] = float(parsed.get("smax_pct", 0.0) or 0.0)
    out["earlybreak"] = bool(parsed.get("earlybreak", True))
    out["growfirst"] = bool(parsed.get("growfirst", False))

    rs = parsed.get("random_seed")
    out["random_seed"] = int(rs) if rs is not None else None

    # ── Commitment pacing ────────────────────────────────────────────────
    pacing = parsed.get("commitment_pacing", "one_time") or "one_time"
    if pacing == "annual_target":
        pacing = "annual"

    if pacing not in ("one_time", "annual", "schedule"):
        pacing = "one_time"

    out["commitment_pacing"] = pacing

    # ── Redemptions / subscriptions → dict mapping ───────────────────────
    out["redemptions"] = _build_redemption_dict(
        rate=parsed.get("redemption_rate"),
        years=parsed.get("redemption_years", []),
        ptf_life=out["ptf_life"],
        periodicity=out["periodicity"],
    )

    # Subscription handling: use subscription_amount (dollar amount)
    # Backward compat: if caller passed deprecated 'subscription_rate',
    # emit a warning and refuse.
    if parsed.get("subscription_rate") is not None:
        sr = parsed["subscription_rate"]
        if sr != 0 and sr is not None:
            raise ValueError(
                "'subscription_rate' is deprecated and no longer supported. "
                "Use 'subscription_amount' (a dollar amount) instead. "
                f"Received subscription_rate={sr}."
            )

    out["subscriptions"] = _build_subscription_dict(
        amount=parsed.get("subscription_amount"),
        years=parsed.get("subscription_years", []),
        ptf_life=out["ptf_life"],
        periodicity=out["periodicity"],
    )

    # ── MFR-generated private config ─────────────────────────────────────
    gen_kwargs: Optional[Dict[str, Any]] = None

    if mode == "generated":
        fund_names = parsed.get("fund_names") or []
        if not fund_names:
            raise ValueError(
                "fund_names is required when private_build_mode='generated'. "
                "Please specify at least one MFR fund name."
            )

        # Weights
        split = parsed.get("split_mode", "equal") or "equal"
        fund_weights: Optional[Dict[str, float]] = None
        if split == "explicit":
            efw = parsed.get("explicit_fund_weights")
            if not efw or not isinstance(efw, list):
                raise ValueError(
                    "split_mode='explicit' requires explicit_fund_weights "
                    "to be a non-empty list of {fund_name, weight} objects."
                )
            fund_weights = {}
            for item in efw:
                fn = item.get("fund_name", "")
                wt = float(item.get("weight", 0))
                fund_weights[fn] = wt
            wt_sum = sum(fund_weights.values())
            if not math.isclose(wt_sum, 1.0, abs_tol=0.02):
                raise ValueError(
                    f"explicit_fund_weights must sum to 1.0 (got {wt_sum:.4f}). "
                    "Please provide weights that sum to 1.0."
                )
        
        commitment_amounts = None
        min_commitment = parsed.get("min_commitment")
        max_commitment = parsed.get("max_commitment")

        if min_commitment is not None and max_commitment is not None:
            min_commitment = float(min_commitment)
            max_commitment = float(max_commitment)

            if abs(min_commitment - max_commitment) < 1e-9:
                fixed_commit = min_commitment
                if split == "equal":
                    commitment_amounts = {fn: fixed_commit for fn in fund_names}
                elif split == "explicit" and fund_weights is not None:
                    # If explicit weights are given, still treat equal min=max as per-fund fixed commitment
                    commitment_amounts = {fn: fixed_commit for fn in fund_names}
        
        commitment_schedule_df = None
        schedule_rows = parsed.get("commitment_schedule")

        if pacing == "schedule":
            if not schedule_rows:
                raise ValueError(
                    "commitment_pacing='schedule' requires commitment_schedule rows."
                )

            schedule_long = pd.DataFrame(schedule_rows)

            required_cols = {"year_offset", "fund_name", "commitment"}
            missing = required_cols - set(schedule_long.columns)
            if missing:
                raise ValueError(
                    f"commitment_schedule is missing columns: {sorted(missing)}"
                )

            commitment_schedule_df = (
                schedule_long
                .pivot_table(
                    index="year_offset",
                    columns="fund_name",
                    values="commitment",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .sort_index()
            )

            commitment_schedule_df.index = commitment_schedule_df.index.astype(int)

            # Make sure fund_names covers all scheduled funds
            scheduled_funds = list(commitment_schedule_df.columns)
            fund_names = list(dict.fromkeys(list(fund_names) + scheduled_funds))
            
        # Co-invest
        ci_raw = parsed.get("coinvest_multipliers")
        coinvest_map: Optional[Dict[str, float]] = None
        if ci_raw and isinstance(ci_raw, list) and len(ci_raw) > 0:
            coinvest_map = {}
            for item in ci_raw:
                if isinstance(item, dict):
                    coinvest_map[item.get("fund_name", "")] = float(
                        item.get("multiplier", 0)
                    )
                elif isinstance(item, (int, float)):
                    # Flat list → apply same multiplier to all funds
                    for fn in fund_names:
                        coinvest_map[fn] = float(item)
                    break

        sy = parsed.get("start_year")
        gen_kwargs = {
            "fund_names": list(fund_names),
            "fund_weights": fund_weights,
            "coinvest_multipliers": coinvest_map,
            "commitment_amounts": commitment_amounts,
            "start_year_override": int(sy) if sy is not None else None,
            "commitment_schedule": commitment_schedule_df,
        }
    
    target_base_raw = parsed.get("target_base", "Target") or "Target"
    if target_base_raw not in {"Percentage", "Target", "Dollar"}:
        raise ValueError(
            f"target_base must be 'Percentage', 'Target', or 'Dollar' — got '{target_base_raw}'."
    )
    target_base = target_base_raw


    out["generated_config_kwargs"] = gen_kwargs

    # ── Selected private config (Preqin / Burgiss) ───────────────────────
    sel_kwargs: Optional[Dict[str, Any]] = None
    if mode == "selected":
        sel_kwargs = {
            "init_funds": int(parsed.get("init_funds", 5) or 5),
            "target_base": target_base,
            "target_private": float(parsed.get("target_private", 0.30) or 0.30),
            "init_age": int(parsed.get("init_age", 0) or 0),
            "select_year": bool(parsed.get("select_year", True)),
            "d_year": (
                int(parsed["d_year"]) if parsed.get("d_year") is not None else None
            ),
            "replacement": bool(parsed.get("replacement", True)),
            "overcommit_pct": float(parsed.get("overcommit_pct", 0.0) or 0.0),
            "strategy": parsed.get("sel_strategy"),
            "currency": parsed.get("sel_currency"),
        }
    out["selected_config_kwargs"] = sel_kwargs

    # ── Public assets ────────────────────────────────────────────────────
    pa_raw = parsed.get("public_assets") or []
    public_cfgs: List[Dict[str, Any]] = []
    for item in pa_raw:
        cfg = _normalize_public_asset(item, out["periodicity"])
        public_cfgs.append(cfg)

    # Auto-add cash sleeve if cash_weight > 0 and no cash asset present
    cash_w = float(parsed.get("cash_weight", 0.0) or 0.0)
    cash_names = {c["name"].lower() for c in public_cfgs}
    if cash_w > 0 and "cash" not in cash_names and "liquid reserve" not in cash_names:
        public_cfgs.insert(0, {
            "name": "Cash",
            "target_weight": cash_w,
            "price0": 1.0,
            "a_return": 0.01,
            "volatility": 0.001,
            "a_income": 0.005,
            "income_volatility": 0.0,
            "asset_class": "Cash",
            "reinvestment_rate": 1.0,
            "liquidity": 1,
            "sub_period": 1,
            "prorate": 1.0,
            "periodicity": out["periodicity"],
            "deviation": 0.0,
            "liquidity_rank": 1,
        })

    # Validate total public weight
    total_pub = sum(c.get("target_weight", 0) for c in public_cfgs)
    if total_pub > 1.0 + 1e-6:
        raise ValueError(
            f"Public-asset target weights sum to {total_pub:.4f}, which exceeds 1.0. "
            "Reduce weights or let the engine normalise by setting them proportionally."
        )
    if not public_cfgs:
        raise ValueError(
            "At least one public / liquid asset is required "
            "(e.g. a cash sleeve) to absorb capital calls and distributions."
        )

    out["public_assets"] = public_cfgs

    # ── Line of credit ───────────────────────────────────────────────────
    loc_max = parsed.get("line_of_credit_max")
    if loc_max is not None and float(loc_max) > 0:
        loc_rate = parsed.get("line_of_credit_rate")
        out["line_of_credit"] = {
            "name": "Credit Facility",
            "balance": 0.0,
            "interest_rate": float(loc_rate) if loc_rate else 0.05,
            "max_balance": float(loc_max),
            "liquidity": 1,
            "periodicity": out["periodicity"],
            "policy": "last_resort",
        }
    else:
        out["line_of_credit"] = None

    # Convenience keys for caller introspection
    out["loc_size"] = float(loc_max) if loc_max is not None else 0.0
    out["loc_rate"] = (
        out["line_of_credit"]["interest_rate"]
        if out["line_of_credit"] is not None
        else 0.0
    )

    # Expose fund_names at top level for easy access
    if out["generated_config_kwargs"]:
        out["fund_names"] = out["generated_config_kwargs"]["fund_names"]
    elif out["selected_config_kwargs"]:
        out["fund_names"] = []  # selected mode doesn't use named funds
    else:
        out["fund_names"] = []

    return out


# ── Private helpers ──────────────────────────────────────────────────────

def _normalize_public_asset(raw: dict, default_period: str) -> dict:
    """Normalize a single public-asset config from the LLM parse."""
    period = raw.get("periodicity", default_period) or default_period
    if period not in ("D", "M", "Q", "Y"):
        period = default_period

    # Assign liquidity rank heuristic based on asset class
    ac = (raw.get("asset_class") or "Generic").lower()
    name = (raw.get("name") or "").lower()
    is_cash = ac == "cash" or "cash" in name

    if is_cash:
        liq_rank = 1
        volatility = 0.0
        income_volatility = 0.0
    elif ac in ("equity", "fixed income", "bonds"):
        liq_rank = 2
        volatility = float(raw.get("volatility", 0.0) or 0.0)
        income_volatility = float(raw.get("income_volatility", 0.0) or 0.0)
    else:
        liq_rank = 3
        volatility = float(raw.get("volatility", 0.0) or 0.0)
        income_volatility = float(raw.get("income_volatility", 0.0) or 0.0)


    return {
        "name": raw.get("name", "Public Asset"),
        "target_weight": float(raw.get("target_weight", 0)),
        "price0": float(raw.get("price0", 1.0) or 1.0),
        "a_return": float(raw.get("a_return", 0.0) or 0.0),
        "volatility": volatility,
        "a_income": float(raw.get("a_income", 0.0) or 0.0),
        "income_volatility": income_volatility,
        "asset_class": raw.get("asset_class", "Generic") or "Generic",
        "reinvestment_rate": float(raw.get("reinvestment_rate", 1.0) or 1.0),
        "liquidity": int(raw.get("liquidity", 1) or 1),
        "sub_period": int(raw.get("sub_period", 1) or 1),
        "prorate": float(raw.get("prorate", 1.0) or 1.0),
        "periodicity": period,
        "deviation": 0.0,
        "liquidity_rank": liq_rank,
        "returnstream_name": raw.get("returnstream_name"),
    }


def _build_redemption_dict(
    rate: Optional[float],
    years: list,
    ptf_life: int,
    periodicity: str,
) -> dict:
    """Convert a redemption *rate* + year list into ``{period_index: rate}``.

    Redemptions are rates (e.g. 0.02 for 2%) because ``red_base``
    determines the dollar base (Fixed, NAV, Dist).

    If *rate* is 0 or None, returns ``{}`` (no redemptions).
    If *years* is empty and rate is non-zero, applies to **all** years.
    """
    if rate is None or rate == 0:
        return {}

    rate = float(rate)
    period_map = {"Q": 4, "M": 12, "Y": 1, "D": 365}
    periods_per_year = period_map.get(periodicity, 4)

    if not years:
        years = list(range(1, ptf_life + 1))

    result: Dict[int, float] = {}
    for yr in years:
        for q in range(periods_per_year):
            period_idx = (yr - 1) * periods_per_year + q
            result[period_idx] = rate
    return result


def _build_subscription_dict(
    amount: Optional[float],
    years: list,
    ptf_life: int,
    periodicity: str,
) -> dict:
    """Convert a subscription *amount* + year list into ``{period_index: amount}``.

    Subscriptions are **absolute dollar amounts**, NOT rates.
    ``timecycle_drawdown`` expects the subscription dict values to be
    cash amounts injected into the portfolio each period.

    If *amount* is 0 or None, returns ``{}`` (no subscriptions).
    If *years* is empty, returns ``{}`` (subscriptions require explicit years).
    """
    if amount is None or amount == 0:
        return {}
    if not years:
        return {}

    amount = float(amount)
    period_map = {"Q": 4, "M": 12, "Y": 1, "D": 365}
    periods_per_year = period_map.get(periodicity, 4)

    result: Dict[int, float] = {}
    for yr in years:
        for q in range(periods_per_year):
            period_idx = (yr - 1) * periods_per_year + q
            result[period_idx] = amount
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Return-stream resolution
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_returnstream(
    returnstream_name: Optional[str],
    public_returnstreams: Optional[Dict[str, pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    """Look up a historical return DataFrame by name.

    Parameters
    ----------
    returnstream_name : str | None
        Ticker or key emitted by the LLM (e.g. ``"SPY"``, ``"AGG"``).
    public_returnstreams : dict[str, pd.DataFrame] | None
        Caller-supplied mapping of names to DataFrames.  Each DataFrame
        should have columns ``Date``, ``Growth``, ``Income`` (matching
        the ``GenPortvN.Asset`` expectation).

    Returns
    -------
    pd.DataFrame | None
        The resolved DataFrame, or ``None`` if unavailable.

    Examples
    --------
    >>> spy_df = pd.DataFrame({"Date": [...], "Growth": [...], "Income": [...]})
    >>> streams = {"SPY": spy_df, "AGG": agg_df}
    >>> _resolve_returnstream("SPY", streams)
    <spy_df>
    >>> _resolve_returnstream("BND", streams)  # not provided
    None
    """
    if not returnstream_name or not public_returnstreams:
        return None
    return public_returnstreams.get(returnstream_name)


# ═══════════════════════════════════════════════════════════════════════════
# 3) End-to-end runner
# ═══════════════════════════════════════════════════════════════════════════

def run_integrated_portfolio_simulation_from_text(
    user_text: str,
    *,
    mfr_source: Optional[Union[str, pd.DataFrame]] = None,
    preqin_engine: Any = None,
    burgiss_path: Optional[str] = None,
    public_returnstreams: Optional[Dict[str, pd.DataFrame]] = None,
    verbose_override: Optional[bool] = None,
) -> dict:
    """Parse free-form text and execute the integrated portfolio simulation.

    This is the convenience end-to-end function: it calls the LLM parser,
    normalises the output, builds the appropriate config objects, and runs
    ``run_integrated_portfolio_simulation``.

    Parameters
    ----------
    user_text : str
        Natural-language portfolio scenario description.
    mfr_source : str | pd.DataFrame | None
        Path to MFR CSV or pre-loaded DataFrame (forwarded to the engine
        when ``private_source="mfr"``).
    preqin_engine : sqlalchemy.Engine | None
        Engine for Preqin SQL Server.
    burgiss_path : str | None
        Path to Burgiss Excel file.
    public_returnstreams : dict[str, pd.DataFrame] | None
        Mapping of ticker/name → historical return DataFrame.  When
        the LLM emits a ``returnstream_name`` on a public asset and this
        dict contains a matching key, that DataFrame is attached as the
        asset's ``returnstream``.  Example::

            {"SPY": spy_returns_df, "AGG": agg_returns_df}

        Each DataFrame should have columns ``Date``, ``Growth``, ``Income``.
        Assets without a matching entry fall back to simulated returns.
    verbose_override : bool | None
        If not None, overrides the verbose flag from the parsed inputs.

    Returns
    -------
    dict
        ``{"parsed_inputs": <normalised dict>,
          "normalized_inputs": <kwargs sent to engine>,
          "results": <simulation output>}``

    Raises
    ------
    RuntimeError
        If the LLM parse fails.
    ValueError
        If the normalised inputs fail validation.
    """
    # ── Step 1: LLM extraction ───────────────────────────────────────────
    raw_parsed = parse_integrated_portfolio_request(user_text)
    logger.info("LLM raw parse:\n%s", json.dumps(raw_parsed, indent=2))

    # ── Step 2: Normalize & validate ─────────────────────────────────────
    inputs = normalize_integrated_portfolio_inputs(raw_parsed)
    logger.info(
        "Normalized inputs:\n%s",
        json.dumps(inputs, indent=2, default=str),
    )

    verbose = False
    if verbose_override is not None:
        verbose = verbose_override

    # ── Step 3: Build config objects ─────────────────────────────────────
    gen_config: Optional[PrivateGenConfig] = None
    sel_config: Optional[PrivateSelConfig] = None

    if inputs["private_build_mode"] == "generated":
        gk = inputs["generated_config_kwargs"]
        gen_config = PrivateGenConfig(
            fund_names=gk["fund_names"],
            commitment_amounts=gk.get("commitment_amounts"),
            fund_weights=gk.get("fund_weights"),
            coinvest_multipliers=gk.get("coinvest_multipliers"),
            start_year_override=gk.get("start_year_override"),
            commitment_schedule=gk.get("commitment_schedule"),
        )

    if inputs["private_build_mode"] == "selected":
        sk = inputs["selected_config_kwargs"]
        sel_config = PrivateSelConfig(
            init_funds=sk["init_funds"],
            target_base=sk["target_base"],
            target_private=sk["target_private"],
            init_age=sk["init_age"],
            select_year=sk["select_year"],
            d_year=sk.get("d_year"),
            replacement=sk["replacement"],
            overcommit_pct=sk.get("overcommit_pct", 0.0),
            strategy=sk.get("strategy"),
            currency=sk.get("currency"),
        )

    # Build PublicAssetConfig objects
    # NOTE: historical=True requires the caller to supply actual return
    # DataFrames via public_returnstreams.  The LLM only emits a
    # returnstream_name (ticker); we resolve it here.
    pub_configs = []
    for pa in inputs["public_assets"]:
        rs_name = pa.get("returnstream_name")
        resolved_rs = _resolve_returnstream(rs_name, public_returnstreams)
        if rs_name and resolved_rs is None:
            logger.warning(
                "Public asset '%s' has returnstream_name='%s' but no "
                "matching entry was found in public_returnstreams. "
                "Falling back to simulated returns.",
                pa["name"], rs_name,
            )
        pub_configs.append(
            PublicAssetConfig(
                name=pa["name"],
                price0=pa["price0"],
                target_weight=pa["target_weight"],
                a_return=pa["a_return"],
                volatility=pa["volatility"],
                a_income=pa["a_income"],
                income_volatility=pa["income_volatility"],
                asset_class=pa["asset_class"],
                reinvestment_rate=pa["reinvestment_rate"],
                liquidity=pa["liquidity"],
                sub_period=pa["sub_period"],
                prorate=pa["prorate"],
                periodicity=pa["periodicity"],
                returnstream=resolved_rs,
                deviation=pa.get("deviation", 0.0),
                liquidity_rank=pa.get("liquidity_rank"),
            )
        )

    # ── Step 4: Assemble kwargs for the engine ───────────────────────────
    run_kwargs: Dict[str, Any] = {
        "private_source": inputs["private_source"],
        "private_build_mode": inputs["private_build_mode"],
        "mfr_source": mfr_source,
        "preqin_engine": preqin_engine,
        "burgiss_path": burgiss_path,
        "portfolio_name": inputs["portfolio_name"],
        "ptf_size": inputs["ptf_size"],
        "ptf_life": inputs["ptf_life"],
        "periodicity": inputs["periodicity"],
        "startdate": inputs["startdate"],
        "generated_config": gen_config,
        "selected_config": sel_config,
        "commitment_pacing": inputs["commitment_pacing"],
        "public_assets": pub_configs,
        "historical": inputs["historical"],
        "rebalance_method": inputs["rebalance_method"],
        "rebal_periodicity": inputs["rebal_periodicity"],
        "redemptions": inputs["redemptions"],
        "subscriptions": inputs["subscriptions"],
        "red_base": inputs["red_base"],
        "red_max": inputs["red_max"],
        "max_pct": inputs["max_pct"],
        "sub_max": inputs["sub_max"],
        "smax_pct": inputs["smax_pct"],
        "earlybreak": inputs["earlybreak"],
        "growfirst": inputs["growfirst"],
        "line_of_credit": inputs["line_of_credit"],
        "random_seed": inputs["random_seed"],
        "verbose": verbose,
    }

    # ── Step 5: Run simulation ───────────────────────────────────────────
    results = run_integrated_portfolio_simulation(**run_kwargs)

    return {
        "parsed_inputs": inputs,
        "normalized_inputs": {
            k: (str(v) if isinstance(v, (PrivateGenConfig, PrivateSelConfig)) else v)
            for k, v in run_kwargs.items()
            if k not in ("mfr_source", "preqin_engine")  # don't serialise engines
        },
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Example usage
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ── Example 1: MFR-generated, natural-language ───────────────────────
    user_text = (
        "Run a 10-year portfolio with MFR-generated private funds using "
        "SDL IV (Levered) and Pathfinder III, equal split, annual commitments, "
        "1 billion size, 40% cash sleeve, 20% US equity at 7% annual return "
        "and 1.5% annual income, quarterly simulation, no rebalance."
    )

    print("User request:")
    print(f"  {user_text}\n")

    # Step 1: Parse
    raw = parse_integrated_portfolio_request(user_text)
    print("── Raw LLM parse ──────────────────────────────────────────")
    print(json.dumps(raw, indent=2))

    # Step 2: Normalize
    clean = normalize_integrated_portfolio_inputs(raw)
    print("\n── Normalized inputs ──────────────────────────────────────")
    print(json.dumps(clean, indent=2, default=str))

    # Step 3: End-to-end (commented — requires MFR CSV on disk)
    # To use historical returns, supply a public_returnstreams dict:
    #   spy_df = pd.DataFrame({"Date": [...], "Growth": [...], "Income": [...]})
    #   agg_df = pd.DataFrame({"Date": [...], "Growth": [...], "Income": [...]})
    #
    # output = run_integrated_portfolio_simulation_from_text(
    #     user_text,
    #     mfr_source="Data Files/MFR_PYTHON_2026.04.csv",
    #     public_returnstreams={"SPY": spy_df, "AGG": agg_df},
    #     verbose_override=True,
    # )
    # print("\nSimulation result keys:", list(output["results"].keys()))

    # ── Example 2: Burgiss-selected ──────────────────────────────────────
    user_text_2 = (
        "Use Burgiss selected funds, start in 2015, target 25% private, "
        "5 initial funds, annual commitments, infrastructure strategy, "
        "USD currency, 500M portfolio, 12-year life, 5% cash, "
        "pro-rata rebalance every 4 quarters, 2% quarterly redemptions."
    )

    print("\n\n── Example 2 ──────────────────────────────────────────────")
    print(f"  {user_text_2}\n")

    raw2 = parse_integrated_portfolio_request(user_text_2)
    print(json.dumps(raw2, indent=2))

    clean2 = normalize_integrated_portfolio_inputs(raw2)
    print("\n── Normalized ─────────────────────────────────────────────")
    print(json.dumps(clean2, indent=2, default=str))
