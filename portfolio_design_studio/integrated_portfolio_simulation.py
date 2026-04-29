"""
integrated_portfolio_simulation.py — Unified Portfolio Simulation Orchestrator

Provides a single production-ready entry-point (`run_integrated_portfolio_simulation`)
that can:

1. Source private fund data from **MFR** (Ares-modelled), **Preqin** (SQL Server),
   or **Burgiss** (Excel).
2. Construct private funds in two modes:
     - ``"generated"``: model cashflows via the Ares MFR pipeline
       (create_cashflow_schedule → management_fee → carry_calculations →
        ares_cashflow_convert) and wrap the result as GenPort Fund objects.
     - ``"selected"``: draw historical cashflows from Preqin / Burgiss via
       ``fund_selector`` and wrap them as GenPort Fund objects.
3. Construct public / liquid Asset objects from caller-supplied configs.
4. Build a GenPortvN.Portfolio, attach positions with target weights,
   and drive simulation via ``Portfolio.timecycle_drawdown``.
5. Support commitment pacing (one-time or annual recommitment), redemptions,
   subscriptions, line-of-credit, rebalancing, and periodicity controls.
6. Return structured outputs with analytics (TVPI, IRR, TWR, etc.).

This module does **not** rely on ``run_master_demo_from_names`` or
``GenPortSimulator`` for orchestration — it builds a clean, transparent
layer on top of the existing building blocks.

Key differences between MFR-generated and Preqin/Burgiss-selected funds
------------------------------------------------------------------------
* **MFR-generated** funds use deterministic deployment and fee models from
  ``fund_deployment_model.py``.  The caller specifies fund names from the
  MFR CSV together with commitment amounts and optional co-invest
  multipliers.  Cashflows are *created* from scratch, converted via
  ``ares_cashflow_convert``, and then wrapped in ``GenPortvN.Fund``.
  Because these are forward-looking projections they always start at the
  configured start-date and have consistent quarterly cadence.

* **Preqin/Burgiss-selected** funds use historical cashflow curves drawn
  randomly by ``fund_selector``.  The caller specifies how many funds to
  draw, the target private allocation, vintage year, and other selection
  parameters.  Cashflows come from real funds, so they have idiosyncratic
  timing, and ``fund_selector`` consolidates and scales them for you.
  The result is also wrapped in ``GenPortvN.Fund``.

Integration points
------------------
* ``data.py``                      — get_engine, load_cashflows, load_burgiss
* ``fund_deployment_model.py``     — create_cashflow_schedule, management_fee,
                                      carry_calculations
* ``fund_deployment_model_runner.py`` — data_setup, load_fund_data_from_mfr
* ``adapters.py``                  — ares_cashflow_convert
* ``GenPortvN.py``                 — Portfolio, Fund, Asset, fund_selector,
                                      combine_cashflows, Line_of_Credit
* ``analytics.py``                 — aggregate_simulation_results, calculate_tvpi,
                                      calculate_dpi, calculate_irr
"""
from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ares fund modelling pipeline
# ---------------------------------------------------------------------------
from .fund_deployment_model_runner import (
    data_setup,
    load_fund_data_from_mfr,
)
from .fund_deployment_model import (
    create_cashflow_schedule,
    management_fee,
    carry_calculations,
)
from .adapters import ares_cashflow_convert

# ---------------------------------------------------------------------------
# GenPort portfolio simulation objects
# ---------------------------------------------------------------------------
from .GenPortvN import (
    Portfolio,
    Fund as GenPortFund,
    Asset,
    fund_selector,
    combine_cashflows,
    Line_of_Credit,
)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
from . import data as _data

# ---------------------------------------------------------------------------
# Analytics (optional)
# ---------------------------------------------------------------------------
try:
    from . import analytics as _analytics
except ImportError:
    _analytics = None

# ---------------------------------------------------------------------------
# Scenario growth rates to be used in growth_df
# ---------------------------------------------------------------------------
#Unused for now
#DEFAULT_MFR_GROWTH_RATES_PATH = "Data Files/USDL_Growth_Rates.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] integrated_sim: %(message)s"))
    logger.addHandler(_h)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PublicAssetConfig:
    """Configuration for a single public / liquid asset sleeve.

    All return and volatility parameters should match the chosen periodicity
    (e.g. quarterly returns if ``periodicity="Q"``).

    Parameters
    ----------
    name : str
        Display name (e.g. ``"Cash"``, ``"US Equity"``).
    price0 : float
        Starting unit price (default 1.0).
    quantity : float
        Number of units held at inception (sets the dollar value together
        with ``price0``).  Alternatively, use ``target_weight`` and let the
        builder calculate the initial quantity.
    target_weight : float
        Target portfolio weight for this asset during rebalancing (0–1).
    a_return : float
        Expected periodic return (mean of the growth distribution).
    volatility : float
        Standard deviation of the periodic growth distribution.
    a_income : float
        Expected periodic income yield.
    income_volatility : float
        Standard deviation of income yield.
    returnstream : pd.DataFrame | None
        Historical return series (columns ``Date``, ``Growth``, ``Income``).
        If ``None`` simulated returns are used.
    asset_class : str
        Asset-class label (used for grouping / reporting).
    reinvestment_rate : float
        Fraction of income reinvested into this asset (0–1).
    liquidity : int
        How often this asset can be liquidated (in periods).  ``1`` = every period.
    sub_period : int
        How often subscriptions into this asset are allowed (in periods).
    prorate : float
        Fraction of the asset's value that can actually be realised in a
        redemption event.
    periodicity : str
        Must match the portfolio simulation periodicity (``"Q"``).
    deviation : float
        Allowable deviation from ``target_weight`` before rebalancing is
        triggered.
    liquidity_rank : int | None
        Manually set the liquidity rank.  If ``None``, one is assigned
        automatically (lower = more liquid).
    """

    name: str = "Public Sleeve"
    price0: float = 1.0
    quantity: float = 0.0
    target_weight: float = 0.0
    a_return: float = 0.0
    volatility: float = 0.0
    a_income: float = 0.0
    income_volatility: float = 0.0
    returnstream: Optional[pd.DataFrame] = None
    asset_class: str = "Generic"
    reinvestment_rate: float = 1.0
    liquidity: int = 1
    sub_period: int = 1
    prorate: float = 1.0
    periodicity: str = "Q"
    deviation: float = 0.0
    liquidity_rank: Optional[int] = None


@dataclass
class PrivateGenConfig:
    """Configuration for MFR-generated private funds.

    Used when ``private_build_mode == "generated"``.

    Parameters
    ----------
    fund_names : list[str]
        MFR *Strategy Short Names* to include.
    commitment_amounts : dict[str, float] | None
        Fund-name → dollar commitment.  If ``None``  the commitment is
        sized from ``fund_weights`` × (total private allocation).
    fund_weights : dict[str, float] | None
        Explicit fund weights (must sum to 1.0).  If ``None`` weights are equal.
    coinvest_multipliers : dict[str, float] | None
        Fund-name → co-invest multiplier.  ``None`` or ``{}`` ⇒ no co-invest.
    start_year_override : int | None
        Override Vintage / Investment Start Date for all funds.
    fee_overrides : dict[str, dict] | None
        Per-fund fee input dict replacements.
    """

    fund_names: List[str] = field(default_factory=list)
    commitment_amounts: Optional[Dict[str, float]] = None
    fund_weights: Optional[Dict[str, float]] = None
    coinvest_multipliers: Optional[Dict[str, float]] = None
    start_year_override: Optional[int] = None
    fee_overrides: Optional[Dict[str, dict]] = None

    # NEW: optional MFR growth-vector inputs
    growth_curves: Optional[Dict[str, pd.DataFrame]] = None
    growth_start_dates: Optional[Dict[str, Any]] = None
    growth_date_col: str = "Date"
    growth_rate_col: str = "Quarterly Growth"
    require_full_growth_history: bool = True

    #NEW: optional explicit commitment pacing schedule
    commitment_schedule: Optional[pd.DataFrame] = None


@dataclass
class SimulationCache:
    """Lightweight cache for Monte Carlo / multi-scenario runs.

    Avoids redundant ``data_setup`` calls and re-generation of identical
    fund cashflows across scenarios.

    Attributes
    ----------
    mfr_data : tuple[dict, dict] | None
        Cached ``(fund_data_all, fund_fee_all)`` from ``data_setup``.
    generated_cashflows : dict[tuple, pd.DataFrame]
        Cache of raw Ares cashflow DataFrames keyed by a frozen tuple of
        all inputs that affect the generated output.

    Example
    -------
    >>> cache = SimulationCache()
    >>> for scenario in scenarios:
    ...     result = run_integrated_portfolio_simulation(
    ...         ...,
    ...         simulation_cache=cache,
    ...         summary_only=True,
    ...         compute_irr_series=False,
    ...         compute_tvpi_series=False,
    ...     )
    """

    mfr_data: Optional[Tuple[dict, dict]] = None
    generated_cashflows: Dict[tuple, pd.DataFrame] = field(default_factory=dict)


@dataclass
class PrivateSelConfig:
    """Configuration for Preqin / Burgiss fund selection.

    Used when ``private_build_mode == "selected"``.

    Parameters
    ----------
    init_funds : int
        Number of funds to randomly select.
    target_base : str
        ``"Percentage"``, ``"Target"``, or ``"Dollar"``.
    target_private : float
        If ``target_base == "Percentage"``  → fraction of portfolio allocated
        to private.  If ``target_base == "Target"`` → target allocation.  If ``target_base == "Dollar"`` → absolute dollar amount.
    init_age : int
        Fund age in years at selection.
    select_year : bool
        Whether to select from a specific vintage year.
    d_year : int | None
        Vintage year for selection (required if ``select_year`` is True).
    replacement : bool
        Allow sampling with replacement.
    year_limit : int
        Minimum years of data required (0 = no limit).
    final_liquidation : bool
        Set remaining NAV to 0 at end.
    first_quarter : bool
        Ensure cashflows start on 3/31.
    scale_by_contrib : bool
        Scale cashflows by total cumulative contributions.
    vintage : int | list[int] | None
        Vintage filter for data loading.
    strategy : str | list[str] | None
        Strategy filter for data loading.
    geography : str | list[str] | None
        Geography filter.
    currency : str | list[str] | None
        Currency filter.
    new_commits : list[dict] | None
        List of dicts describing additional annual commitments.
        Each dict should contain ``{ "year": int, "n_funds": int, ... }``.
    overcommit_pct : float
        Overcommitment as a fraction of NAV.
    """

    init_funds: int = 5
    target_base: str = "Percentage"
    target_private: float = 0.30
    init_age: int = 0
    select_year: bool = True
    d_year: Optional[int] = None
    replacement: bool = True
    year_limit: int = 0
    final_liquidation: bool = True
    first_quarter: bool = True
    scale_by_contrib: bool = False
    vintage: Optional[Any] = None
    strategy: Optional[Any] = None
    geography: Optional[Any] = None
    currency: Optional[Any] = None
    new_commits: Optional[List[dict]] = None
    overcommit_pct: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

_VALID_SOURCES = {"mfr", "preqin", "burgiss"}
_VALID_BUILD_MODES = {"generated", "selected"}
_VALID_COMMITMENT_PACING = {"one_time", "annual","schedule"}
_VALID_PERIODICITIES = {"Q"} #{"D", "M", "Q", "Y"}
_VALID_REBALANCE = {"No Rebalance", "Priority", "Pro-Rata"}
_VALID_RED_BASES = {"Fixed", "NAV", "Dist"}


def _validate_inputs(
    private_source: str,
    private_build_mode: str,
    ptf_size: float,
    ptf_life: int,
    periodicity: str,
    rebalance_method: str,
    red_base: str,
    public_assets: list,
    commitment_pacing: str,
) -> None:
    """Raise ``ValueError`` on invalid / inconsistent inputs."""

    if private_source not in _VALID_SOURCES:
        raise ValueError(
            f"private_source must be one of {_VALID_SOURCES}, got '{private_source}'."
        )

    if private_build_mode not in _VALID_BUILD_MODES:
        raise ValueError(
            f"private_build_mode must be one of {_VALID_BUILD_MODES}, "
            f"got '{private_build_mode}'."
        )

    # Source / build-mode compatibility
    if private_source == "mfr" and private_build_mode == "selected":
        raise ValueError(
            "private_source='mfr' only supports private_build_mode='generated'. "
            "MFR data describes Ares-modelled funds, not historical cashflow curves "
            "suitable for fund_selector.  Use 'preqin' or 'burgiss' with 'selected'."
        )
    if private_source in {"preqin", "burgiss"} and private_build_mode == "generated":
        raise ValueError(
            f"private_source='{private_source}' only supports "
            f"private_build_mode='selected'.  The 'generated' mode requires "
            "MFR data to model cashflows from scratch."
        )

    if ptf_size <= 0:
        raise ValueError(f"ptf_size must be > 0, got {ptf_size}.")
    if ptf_life < 1:
        raise ValueError(f"ptf_life must be >= 1, got {ptf_life}.")
    if periodicity not in _VALID_PERIODICITIES:
        raise ValueError(
            f"periodicity must be one of {_VALID_PERIODICITIES}, got '{periodicity}'."
        )
    if rebalance_method not in _VALID_REBALANCE:
        raise ValueError(
            f"rebalance_method must be one of {_VALID_REBALANCE}, "
            f"got '{rebalance_method}'."
        )
    if red_base not in _VALID_RED_BASES:
        raise ValueError(
            f"red_base must be one of {_VALID_RED_BASES}, got '{red_base}'."
        )
    if commitment_pacing not in _VALID_COMMITMENT_PACING:
        raise ValueError(
            f"commitment_pacing must be one of {_VALID_COMMITMENT_PACING}, "
            f"got '{commitment_pacing}'."
        )

    # Public-asset weight sanity
    if public_assets:
        total_pub = sum(a.target_weight for a in public_assets)
        if total_pub > 1.0 + 1e-6:
            raise ValueError(
                f"Public-asset target weights sum to {total_pub:.4f} which exceeds 1.0."
            )


# ═══════════════════════════════════════════════════════════════════════════
# Private-data loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_private_data(
    private_source: str,
    mfr_source: Optional[Union[str, pd.DataFrame]],
    preqin_engine: Any,
    burgiss_path: Optional[str],
    sel_cfg: Optional[PrivateSelConfig],
    verbose: bool,
) -> pd.DataFrame:
    """Load the raw private fund cashflow data for selection-based modes.

    Returns
    -------
    pd.DataFrame
        Cashflow data ready for ``fund_selector``.
    """
    if private_source == "preqin":
        if preqin_engine is None:
            raise ValueError(
                "preqin_engine is required when private_source='preqin'. "
                "Create one with data.get_engine(conn_str)."
            )
        kw: Dict[str, Any] = {}
        if sel_cfg is not None:
            if sel_cfg.vintage is not None:
                kw["vintage"] = sel_cfg.vintage
            if sel_cfg.strategy is not None:
                kw["strategy"] = sel_cfg.strategy
            if sel_cfg.geography is not None:
                kw["geography"] = sel_cfg.geography
            if sel_cfg.currency is not None:
                kw["currency"] = sel_cfg.currency
        df = _data.load_cashflows(preqin_engine, **kw)
        logger.info("Loaded %d rows from Preqin.", len(df))
        return df

    if private_source == "burgiss":
        if burgiss_path is None:
            burgiss_path = "Data Files/Burgiss_Cashflowsv4.xlsx"
        kw = {}
        if sel_cfg is not None:
            if sel_cfg.vintage is not None:
                kw["vintage"] = sel_cfg.vintage
            if sel_cfg.strategy is not None:
                kw["strategy"] = sel_cfg.strategy
            if sel_cfg.geography is not None:
                kw["geography"] = sel_cfg.geography
            if sel_cfg.currency is not None:
                kw["currency"] = sel_cfg.currency
        df = _data.load_burgiss(burgiss_path, **kw)
        logger.info("Loaded %d rows from Burgiss (%s).", len(df), burgiss_path)
        return df

    raise ValueError(f"Cannot load selection data for private_source='{private_source}'.")


# ═══════════════════════════════════════════════════════════════════════════
# MFR data + cashflow caching helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_mfr_data(
    mfr_source: Union[str, pd.DataFrame],
    verbose: bool = False,
    simulation_cache: Optional[SimulationCache] = None,
) -> Tuple[dict, dict]:
    """Load MFR fund/fee data, using *simulation_cache* when available."""
    if simulation_cache is not None:
        if simulation_cache.mfr_data is None:
            simulation_cache.mfr_data = data_setup(mfr_source, verbose=verbose)
        return simulation_cache.mfr_data
    return data_setup(mfr_source, verbose=verbose)


def _freeze_mapping(d: Optional[dict]) -> Optional[tuple]:
    """Recursively convert a dict to a hashable nested-tuple."""
    if not d:
        return None
    return tuple(
        sorted(
            (k, _freeze_mapping(v) if isinstance(v, dict) else v)
            for k, v in d.items()
        )
    )


def _growth_df_fingerprint(
    df: Optional[pd.DataFrame],
    date_col: str = "Date",
    rate_col: str = "Quarterly Growth",
) -> Optional[tuple]:
    """Cheap fingerprint for a growth-curve DataFrame (avoids hashing all rows)."""
    if df is None:
        return None
    if len(df) == 0:
        return ("empty",)
    return (
        tuple(df.shape),
        str(pd.to_datetime(df[date_col]).min()) if date_col in df.columns else None,
        str(pd.to_datetime(df[date_col]).max()) if date_col in df.columns else None,
        float(df[rate_col].iloc[0]) if rate_col in df.columns and len(df) > 0 else None,
        float(df[rate_col].iloc[-1]) if rate_col in df.columns and len(df) > 0 else None,
    )


def _make_cf_cache_key(
    fund_name: str,
    commitment: float,
    start_year_override: Optional[int],
    fee_overrides: Optional[dict],
    growth_df: Optional[pd.DataFrame],
    growth_start_date: Optional[Any],
    growth_date_col: str,
    growth_rate_col: str,
    require_full_growth_history: bool,
) -> tuple:
    """Build a hashable cache key for ``_generate_single_fund_cf`` inputs."""
    return (
        fund_name,
        round(float(commitment), 2),
        start_year_override,
        _freeze_mapping(fee_overrides),
        _growth_df_fingerprint(growth_df, growth_date_col, growth_rate_col),
        str(growth_start_date) if growth_start_date is not None else None,
        growth_date_col,
        growth_rate_col,
        require_full_growth_history,
    )

def prepare_growth_curve_from_quarter_index(
    df: pd.DataFrame,
    growth_col: str | None = None,
    output_date_col: str = "Date",
    output_growth_col: str = "Quarterly Growth",
    base_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
    """
    Convert a growth-rate DataFrame indexed by quarters into the format expected
    by integrated_portfolio_simulation.PrivateGenConfig.

    Expected input examples
    -----------------------
    1. String quarter index:
        index = ["2010Q1", "2010Q2", "2010Q3"]
        column = "Growth"

    2. PeriodIndex:
        index = PeriodIndex(["2010Q1", "2010Q2"], freq="Q")
        column = "Growth"

    3. Numeric quarter offsets:
        index = [0, 1, 2, 3]
        column = "Growth"
        base_date must be provided, e.g. "2010-03-31"

    Returns
    -------
    pd.DataFrame with columns:
        Date
        Quarterly Growth
    """
    if df is None or df.empty:
        raise ValueError("Growth curve DataFrame is empty.")

    out = df.copy()

    # If no growth column is specified, use the first column.
    if growth_col is None:
        if out.shape[1] != 1:
            raise ValueError(
                "growth_col must be specified when the DataFrame has multiple columns."
            )
        growth_col = out.columns[0]

    if growth_col not in out.columns:
        raise ValueError(f"growth_col '{growth_col}' not found in DataFrame.")

    q_index = out.index

    # Case 1: PeriodIndex, e.g. PeriodIndex(['2010Q1', '2010Q2'], freq='Q')
    if isinstance(q_index, pd.PeriodIndex):
        dates = q_index.to_timestamp(how="end").normalize()

    # Case 2: Numeric quarter offsets, e.g. 0, 1, 2, 3
    elif pd.api.types.is_numeric_dtype(q_index):
        if base_date is None:
            raise ValueError(
                "base_date is required when the index contains numeric quarter offsets."
            )
        base_date = pd.Timestamp(base_date)
        dates = pd.DatetimeIndex([
            base_date + pd.offsets.QuarterEnd(int(q))
            for q in q_index
        ])

    # Case 3: String quarter labels, e.g. '2010Q1', '2010-Q1', '2010 Q1'
    else:
        quarter_labels = (
            pd.Series(q_index.astype(str), index=q_index)
            .str.upper()
            .str.replace(" ", "", regex=False)
            .str.replace("-", "", regex=False)
        )

        try:
            dates = (
                pd.PeriodIndex(quarter_labels, freq="Q")
                .to_timestamp(how="end")
                .normalize()
            )
        except Exception as exc:
            raise ValueError(
                "Could not parse quarter index. Expected values like "
                "'2010Q1', '2010-Q1', '2010 Q1', a PeriodIndex, or numeric offsets."
            ) from exc

    result = pd.DataFrame({
        output_date_col: pd.to_datetime(dates),
        output_growth_col: pd.to_numeric(out[growth_col].values, errors="coerce"),
    })

    if result[output_growth_col].isna().any():
        raise ValueError(
            f"Some values in '{growth_col}' could not be converted to numeric growth rates."
        )

    result = result.sort_values(output_date_col).reset_index(drop=True)
    return result

# ═══════════════════════════════════════════════════════════════════════════
# MFR-generated cashflow helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_zero_fee_input(base_fee_input: dict) -> dict:
    """Deep-copy of ``base_fee_input`` with all fee rates zeroed (for co-invest)."""
    ff = copy.deepcopy(base_fee_input)
    for k in ("Primary Standard Mgmt Fee", "Secondary Standard Mgmt Fee",
              "Primary Post-Inv Period Mgmt Fee"):
        if k in ff:
            ff[k] = 0.0
    for k in ("First Close Discount", "1st Tier Size Discount",
              "2nd Tier Size Discount", "3rd Tier Size Discount",
              "Partnership Discount"):
        if k in ff:
            ff[k] = 0.0
    for k in ("First Close Discount Applied?", "Size Discount Applied?"):
        if k in ff:
            ff[k] = False
    for k in ("Performance Fee", "Hurdle Rate", "GP Catchup"):
        if k in ff:
            ff[k] = 0.0
    return ff


def _generate_single_fund_cf(
    fund_name: str,
    fund_data_all: dict,
    fund_fee_all: dict,
    commitment: float,
    start_year_override: Optional[int],
    fee_overrides: Optional[dict],
    verbose: bool,
    growth_df: Optional[pd.DataFrame] = None,
    growth_start_date: Optional[Any] = None,
    growth_date_col: str = "Date",
    growth_rate_col: str = "Quarterly Growth",
    require_full_growth_history: bool = True,
    simulation_cache: Optional[SimulationCache] = None,
) -> pd.DataFrame:
    """Run the Ares cashflow pipeline for one fund and return the raw DF.

    Calls ``create_cashflow_schedule → management_fee → carry_calculations``.

    If *simulation_cache* is provided, the result is stored (deep-copied)
    and subsequent calls with identical inputs return a deep copy from cache.
    """
    # ── Cache lookup ─────────────────────────────────────────────────────
    cache_key = None
    if simulation_cache is not None:
        cache_key = _make_cf_cache_key(
            fund_name=fund_name,
            commitment=commitment,
            start_year_override=start_year_override,
            fee_overrides=fee_overrides,
            growth_df=growth_df,
            growth_start_date=growth_start_date,
            growth_date_col=growth_date_col,
            growth_rate_col=growth_rate_col,
            require_full_growth_history=require_full_growth_history,
        )
        if cache_key in simulation_cache.generated_cashflows:
            if verbose:
                logger.info(
                    "  Cache hit for '%s'  commitment=$%,.0f", fund_name, commitment
                )
            return simulation_cache.generated_cashflows[cache_key].copy(deep=True)

    # ── Generate (cache miss or no cache) ────────────────────────────────
    fd = copy.deepcopy(fund_data_all[fund_name])
    ff = copy.deepcopy(fee_overrides) if fee_overrides else copy.deepcopy(fund_fee_all[fund_name])

    if commitment > 0:
        fd["Original Equity Commitment"] = commitment

    if start_year_override is not None:
        fd["Vintage"] = start_year_override
        fd["Investment Start Date"] = f"{start_year_override}-03-31"

    if verbose:
        logger.info(
            "  Generating cashflows for '%s'  commitment=$%,.0f",
            fund_name, fd["Original Equity Commitment"],
        )

    cf = create_cashflow_schedule(
        fd,
        growth_df=growth_df,
        growth_start_date=growth_start_date,
        growth_date_col=growth_date_col,
        growth_rate_col=growth_rate_col,
        require_full_growth_history=require_full_growth_history,
    )
    cf = management_fee(cf, ff)
    cf = carry_calculations(cf, ff)
    cf["Vintage"] = fd.get("Vintage")
    cf["Geography"] = fd.get("Geography", "Unknown")
    cf["Strategy"] = fd.get("Strategy", "Ares")
    cf["Currency"] = fd.get("Currency", "USD")

    # ── Store in cache ───────────────────────────────────────────────────
    if cache_key is not None:
        simulation_cache.generated_cashflows[cache_key] = cf.copy(deep=True)

    return cf


def _build_generated_private_funds(
    gen_cfg: PrivateGenConfig,
    mfr_source: Union[str, pd.DataFrame],
    ptf_size: float,
    startdate: str,
    verbose: bool,
    simulation_cache: Optional[SimulationCache] = None,
) -> Tuple[List[GenPortFund], List[float], List[float], Dict[str, pd.DataFrame]]:
    """Build GenPort Fund objects from MFR-generated cashflows.

    Returns
    -------
    funds : list[GenPortFund]
        Fund objects ready for ``Portfolio.add_asset`` / ``set_assets``.
    target_weights : list[float]
        Target weights for each fund (always 0 for funds — no rebalancing into them).
    deviations : list[float]
        Allowed deviations (always 0 for funds).
    raw_cashflows : dict[str, pd.DataFrame]
        Original Ares-model cashflow DFs keyed by label.
    """
    # Load MFR (cached if simulation_cache provided)
    fund_data_all, fund_fee_all = _get_mfr_data(
        mfr_source, verbose=verbose, simulation_cache=simulation_cache,
    )

    # Validate requested fund names
    available = set(fund_data_all.keys())
    missing = set(gen_cfg.fund_names) - available
    if missing:
        raise ValueError(
            f"Fund(s) not found in MFR source: {missing}. "
            f"Available: {sorted(available)}"
        )

    # Compute weights
    n = len(gen_cfg.fund_names)
    if gen_cfg.fund_weights is not None:
        weights = gen_cfg.fund_weights
        wt_sum = sum(weights.get(f, 0) for f in gen_cfg.fund_names)
        if abs(wt_sum - 1.0) > 1e-6:
            raise ValueError(
                f"gen_cfg.fund_weights must sum to 1.0 (got {wt_sum:.6f})."
            )
    else:
        weights = {f: 1.0 / n for f in gen_cfg.fund_names}

    # Compute per-fund commitment amounts
    coinvest_map = gen_cfg.coinvest_multipliers or {}

    funds: List[GenPortFund] = []
    tw: List[float] = []
    devs: List[float] = []
    raw_cfs: Dict[str, pd.DataFrame] = {}

    for fname in gen_cfg.fund_names:

        growth_df = None
        if gen_cfg.growth_curves is not None:
            growth_df = gen_cfg.growth_curves.get(fname)

        growth_start_date = None
        if gen_cfg.growth_start_dates is not None:
            growth_start_date = gen_cfg.growth_start_dates.get(fname)

        w = weights[fname]
        if gen_cfg.commitment_amounts and fname in gen_cfg.commitment_amounts:
            core_commit = gen_cfg.commitment_amounts[fname]
        else:
            core_commit = w * ptf_size

        co_mult = coinvest_map.get(fname, 0.0)

        # Generate core fund cashflows
        cf = _generate_single_fund_cf(
            fund_name=fname,
            fund_data_all=fund_data_all,
            fund_fee_all=fund_fee_all,
            commitment=core_commit,
            start_year_override=gen_cfg.start_year_override,
            fee_overrides=(gen_cfg.fee_overrides or {}).get(fname),
            verbose=verbose,
            growth_df=growth_df,
            growth_start_date=growth_start_date,
            growth_date_col=gen_cfg.growth_date_col,
            growth_rate_col=gen_cfg.growth_rate_col,
            require_full_growth_history=gen_cfg.require_full_growth_history,
            simulation_cache=simulation_cache,
        )

        raw_cfs[fname] = cf
        genport_cf = _ares_to_genport_df(cf, startdate, scale_to_commitment=None)
        fund_obj = GenPortFund(
            name=fname,
            cashflows=genport_cf,
            asset_class=cf["Strategy"].iloc[0] if "Strategy" in cf.columns else "Private",
            periodicity="Q",
        )
        funds.append(fund_obj)
        tw.append(0.0)  # funds cannot be rebalanced into
        devs.append(0.0)

        # Co-invest vehicle
        if co_mult > 0:
            coinvest_commit = core_commit * co_mult
            ci_label = f"{fname} [Co-Invest]"
            ci_ff = _make_zero_fee_input(fund_fee_all[fname])
            cf_ci = _generate_single_fund_cf(
                fund_name=fname,
                fund_data_all=fund_data_all,
                fund_fee_all=fund_fee_all,
                commitment=coinvest_commit,
                start_year_override=gen_cfg.start_year_override,
                fee_overrides=ci_ff,
                verbose=verbose,
                growth_df=growth_df,
                growth_start_date=growth_start_date,
                growth_date_col=gen_cfg.growth_date_col,
                growth_rate_col=gen_cfg.growth_rate_col,
                require_full_growth_history=gen_cfg.require_full_growth_history,
                simulation_cache=simulation_cache,
            )

            cf_ci["fund_name"] = ci_label
            raw_cfs[ci_label] = cf_ci
            genport_ci = _ares_to_genport_df(cf_ci, startdate, scale_to_commitment=None)
            ci_obj = GenPortFund(
                name=ci_label,
                cashflows=genport_ci,
                asset_class="Co-Investment",
                periodicity="Q",
            )
            funds.append(ci_obj)
            tw.append(0.0)
            devs.append(0.0)

    return funds, tw, devs, raw_cfs


def _ares_to_genport_df(
    cf: pd.DataFrame,
    startdate: str,
    scale_to_commitment: Optional[float] = None,
) -> pd.DataFrame:
    """Convert Ares-model cashflow DF to the schema expected by ``GenPortvN.Fund``.

    ``GenPortvN.Fund`` expects a DataFrame indexed by date with columns:
    ``Age, quarter_distribution, quarter_contribution, nav_eoq,
      unfunded_eoq, committed_not_called, committed,
      Management Fees Paid, GP Distributions this Period``.

    The Ares DF has columns like ``Date, Gross Equity NAV, Committed,
    Cumulative Contributions, Cumulative LP Received, ...``.

    We use ``ares_cashflow_convert`` as the first step and then reshape.
    """
    # For MFR-generated funds in the integrated simulation, preserve actual
    # commitment dollars by default. Standardized scaling (e.g. to 10MM) is
    # only for legacy normalized workflows.
    # Guard missing columns
    required = [
        "Date", "Quarter", "Gross Equity NAV", "Committed",
        "Unfunded", "Cumulative Contributions", "Cumulative LP Received",
        "Vintage", "Strategy", "Geography", "Currency",
    ]
    missing = [c for c in required if c not in cf.columns]
    if missing:
        warnings.warn(
            f"_ares_to_genport_df: cashflow DF missing columns {missing}; "
            "attempting best-effort conversion."
        )

    # Use the adapter that normalises column names and scales
    converted = ares_cashflow_convert(cf, scale_to_commitment=scale_to_commitment)

    # Build the GenPort-compatible DF
    out = pd.DataFrame()
    out["quarter_end"] = pd.to_datetime(converted["quarter_end"])
    out["nav_eoq"] = converted["nav_eoq"].values
    out["committed"] = converted["committed"].values
    out["unfunded_eoq"] = converted.get("unfunded_eoq", 0).values
    out["cum_contributions_eoq"] = converted["cum_contributions_eoq"].values
    out["cum_distributions_eoq"] = converted["cum_distributions_eoq"].values

    if "Management Fees Paid" in converted.columns:
        out["Management Fees Paid"] = converted["Management Fees Paid"].values
    else:
        out["Management Fees Paid"] = 0.0

    if "GP Distributions this Period" in converted.columns:
        out["GP Distributions this Period"] = converted["GP Distributions this Period"].values
    else:
        out["GP Distributions this Period"] = 0.0

    # Derive quarterly flows
    out = out.sort_values("quarter_end").reset_index(drop=True)
    out["quarter_contribution"] = out["cum_contributions_eoq"].diff().fillna(
        out["cum_contributions_eoq"].iloc[0]
    )
    out["quarter_distribution"] = out["cum_distributions_eoq"].diff().fillna(
        out["cum_distributions_eoq"].iloc[0]
    )

    # Committed not called (contributions are negative by convention after convert)
    out["committed_not_called"] = (out["committed"] + out["cum_contributions_eoq"]).clip(lower=0)

    out["Age"] = range(len(out))
    out = out.set_index("quarter_end")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Annual pacing validation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _assert_fund_starts_on_or_after(
    fund_df: pd.DataFrame,
    expected_year: int,
    label: str,
) -> None:
    """Validate that a fund's cashflows start in or after *expected_year*.

    For MFR-generated annual commitments we set ``start_year_override``
    so the resulting cashflows **must** begin in the intended vintage year.
    If they don't, something went wrong in the Ares pipeline or the
    MFR data has an overridden Investment Start Date.

    Raises
    ------
    ValueError
        If the earliest cashflow date is before *expected_year*.
    """
    if fund_df is None or fund_df.empty:
        return

    idx = fund_df.index
    if hasattr(idx, "min"):
        earliest = pd.to_datetime(idx.min())
    else:
        return

    if earliest.year < expected_year:
        raise ValueError(
            f"Annual pacing fund '{label}': cashflows start in "
            f"{earliest.year} but expected_year={expected_year}.  "
            f"Earliest date is {earliest.date()}.  Check start_year_override "
            f"and MFR data for this fund."
        )


def _warn_selected_fund_vintage(
    consolidated: pd.DataFrame,
    expected_year: int,
    label: str,
) -> None:
    """Warn if selected-fund cashflows are wildly inconsistent with the target year.

    Selected funds use historical curves, so the cashflow dates reflect the
    original vintage.  ``fund_selector`` with ``d_year`` should select from
    the right vintage, but we emit a warning if the earliest index date is
    more than 2 years away from the target.
    """
    if consolidated is None or consolidated.empty:
        return

    idx = consolidated.index
    if hasattr(idx, "min"):
        earliest = pd.to_datetime(idx.min())
    else:
        return

    gap = abs(earliest.year - expected_year)
    if gap > 2:
        warnings.warn(
            f"Annual pacing ({label}): selected cashflows start in "
            f"{earliest.year}, which is {gap} years from target year "
            f"{expected_year}.  Activation may be delayed or premature.  "
            f"Verify d_year and data vintage filters.",
            stacklevel=2,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Selected-fund helpers (Preqin / Burgiss)
# ═══════════════════════════════════════════════════════════════════════════

def _build_selected_private_funds(
    sel_cfg: PrivateSelConfig,
    private_data: pd.DataFrame,
    ptf_size: float,
    startdate: str,
) -> Tuple[List[GenPortFund], List[float], List[float], pd.DataFrame]:
    """Select and wrap private funds from Preqin / Burgiss data.

    Uses ``GenPortvN.fund_selector`` and wraps the result in a single
    ``GenPortFund`` object (fund_selector already aggregates by quarter).

    ``overcommit_pct`` is applied as a sizing-layer adjustment: the
    effective private target passed to ``fund_selector`` is scaled up by
    ``(1 + overcommit_pct)``.  This means the portfolio will select
    commitments *larger* than the nominal target, simulating
    overcommitment.  ``fund_selector`` itself doesn't know about
    overcommitment; we simply give it a bigger target.

    Returns
    -------
    funds : list[GenPortFund]
    target_weights : list[float]
    deviations : list[float]
    consolidated_cashflows : pd.DataFrame
    """
    d_year = sel_cfg.d_year
    if d_year is None:
        # Default to start-date year
        d_year = pd.to_datetime(startdate).year

    # Apply overcommit_pct as a sizing-layer adjustment.
    # fund_selector does not accept overcommit_pct directly, so we
    # inflate the target to achieve the same effect.
    effective_target = sel_cfg.target_private
    if sel_cfg.overcommit_pct > 0:
        effective_target = sel_cfg.target_private * (1 + sel_cfg.overcommit_pct)
        if sel_cfg.target_base in {"Percentage", "Target"}:
            effective_target = min(effective_target, 1.0)
        logger.info(
            "Overcommit %.1f%%: effective target_private %.4f (was %.4f)",
            sel_cfg.overcommit_pct * 100,
            effective_target,
            sel_cfg.target_private,
        )

    consolidated, sample_ids = fund_selector(
        init_funds=sel_cfg.init_funds,
        ptf_size=ptf_size,
        target_base=sel_cfg.target_base,
        target_private=effective_target,
        init_age=sel_cfg.init_age,
        data=private_data.copy(),
        select_year=sel_cfg.select_year,
        d_year=d_year,
        final_liquidation=sel_cfg.final_liquidation,
        first_quarter=sel_cfg.first_quarter,
        scale_by_contrib=sel_cfg.scale_by_contrib,
        year_limit=sel_cfg.year_limit,
        replacement=sel_cfg.replacement,
    )

    # Wrap in a Fund object
    fund_obj = GenPortFund(
        name="Private Sleeve (Selected)",
        cashflows=consolidated,
        asset_class="Private",
        periodicity="Q",
    )

    return [fund_obj], [0.0], [0.0], consolidated


# ═══════════════════════════════════════════════════════════════════════════
# Public / liquid asset builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_public_assets(
    configs: List[PublicAssetConfig],
    ptf_size: float,
    total_private_weight: float,
) -> Tuple[List[Asset], List[float], List[float], List[int]]:
    """Build Asset objects from public-asset config dicts.

    If a config's ``quantity`` is 0 and ``target_weight`` is > 0, the
    quantity is computed so that ``quantity × price0 = target_weight × ptf_size``.

    Returns
    -------
    assets : list[Asset]
    target_weights : list[float]
    deviations : list[float]
    liquidity_ranks : list[int]
    """
    assets: List[Asset] = []
    t_weights: List[float] = []
    devs: List[float] = []
    liq_ranks: List[int] = []

    for idx, cfg in enumerate(configs):
        qty = cfg.quantity
        if qty == 0 and cfg.target_weight > 0 and cfg.price0 > 0:
            qty = (cfg.target_weight * ptf_size) / cfg.price0

        rs = cfg.returnstream if cfg.returnstream is not None else pd.DataFrame()

        asset = Asset(
            name=cfg.name,
            price0=cfg.price0,
            quantity=qty,
            a_return=cfg.a_return,
            volatility=cfg.volatility,
            a_income=cfg.a_income,
            income_volatility=cfg.income_volatility,
            returnstream=rs,
            asset_class=cfg.asset_class,
            reinvestment_rate=cfg.reinvestment_rate,
            liquidity=cfg.liquidity,
            sub_period=cfg.sub_period,
            prorate=cfg.prorate,
            periodicity=cfg.periodicity,
        )
        assets.append(asset)
        t_weights.append(cfg.target_weight)
        devs.append(cfg.deviation)

        if cfg.liquidity_rank is not None:
            liq_ranks.append(cfg.liquidity_rank)
        else:
            liq_ranks.append(idx + 1)

    return assets, t_weights, devs, liq_ranks


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio construction
# ═══════════════════════════════════════════════════════════════════════════

def _build_portfolio(
    portfolio_name: str,
    fund_objs: List[GenPortFund],
    fund_tws: List[float],
    fund_devs: List[float],
    asset_objs: List[Asset],
    asset_tws: List[float],
    asset_devs: List[float],
    asset_liq_ranks: List[int],
    line_of_credit_cfg: Optional[dict],
) -> Portfolio:
    """Assemble a ``Portfolio`` from fund and asset components.

    Target weights across all positions must sum to 1.0. Fund target
    weights are always 0 (can't rebalance into them); asset target
    weights will be rescaled if the sum is slightly off.
    """
    all_positions = list(asset_objs) + list(fund_objs)
    all_weights = list(asset_tws) + list(fund_tws)
    all_devs = list(asset_devs) + list(fund_devs)

    # Build liquidity ranks: assets first (user-supplied), funds after
    max_asset_rank = max(asset_liq_ranks) if asset_liq_ranks else 0
    fund_ranks = list(range(max_asset_rank + 1, max_asset_rank + 1 + len(fund_objs)))
    all_ranks = list(asset_liq_ranks) + fund_ranks

    # Normalise weights to sum to 1.0
    total_w = sum(all_weights)
    if abs(total_w - 1.0) > 1e-6:
        if total_w == 0:
            raise ValueError(
                "Total target weights are 0.  At least one asset must have a "
                "non-zero target_weight."
            )
        # Rescale asset weights proportionally (fund weights stay 0)
        asset_w_sum = sum(asset_tws)
        if asset_w_sum > 0:
            scale = (1.0 - sum(fund_tws)) / asset_w_sum
            all_weights = [w * scale for w in asset_tws] + list(fund_tws)
        else:
            warnings.warn(
                f"Target weights sum to {total_w:.4f}; rescaling all weights."
            )
            all_weights = [w / total_w for w in all_weights]

    ptf = Portfolio(portfolio_name)
    ptf.set_assets(
        positions=all_positions,
        target_weights=all_weights,
        asset_deviations=all_devs,
        liquidity_ranks=all_ranks,
    )

    if line_of_credit_cfg is not None:
        loc = Line_of_Credit(
            name=line_of_credit_cfg.get("name", "Credit Facility"),
            balance=line_of_credit_cfg.get("balance", 0.0),
            interest_rate=line_of_credit_cfg.get("interest_rate", 0.05),
            max_balance=line_of_credit_cfg.get("max_balance", 0.0),
            liquidity=line_of_credit_cfg.get("liquidity", 1),
            periodicity=line_of_credit_cfg.get("periodicity", "Q"),
        )
        ptf.set_line_of_credit(
            loc,
            policy=line_of_credit_cfg.get("policy", "last_resort"),
        )

    return ptf

# ═══════════════════════════════════════════════════════════════════════════
# Explicit commitment pacing
# ═══════════════════════════════════════════════════════════════════════════
def _add_scheduled_funds_generated(
    ptf: Portfolio,
    year_offset: int,
    start_yr: int,
    gen_cfg: PrivateGenConfig,
    mfr_source: Union[str, pd.DataFrame],
    startdate: str,
    verbose: bool,
    simulation_cache: Optional[SimulationCache] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Add MFR-generated funds for a specific simulation year based on
    gen_cfg.commitment_schedule.

    commitment_schedule:
        index = year offsets, e.g. 0, 1, 2, 3
        columns = fund names
        values = dollar commitments
    """
    schedule = gen_cfg.commitment_schedule
    if schedule is None or year_offset not in schedule.index:
        return {}

    target_yr = start_yr + year_offset
    raw_cfs = {}

    for fund_name, commitment in schedule.loc[year_offset].items():
        commitment = float(commitment or 0)
        if commitment <= 0:
            continue

        cfg_copy = copy.deepcopy(gen_cfg)
        cfg_copy.fund_names = [fund_name]
        cfg_copy.commitment_amounts = {fund_name: commitment}
        cfg_copy.fund_weights = None
        cfg_copy.start_year_override = target_yr

        new_funds, _, _, fund_raw_cfs = _build_generated_private_funds(
            gen_cfg=cfg_copy,
            mfr_source=mfr_source,
            ptf_size=commitment,
            startdate=startdate,
            verbose=verbose,
            simulation_cache=simulation_cache,
        )

        for f in new_funds:
            f.name = f"{f.name}_{target_yr}"
            ptf.add_asset(
                f,
                target_weight=0.0,
                asset_deviation=0.0,
                liquidity_rank=len(ptf.positions) + 1,
            )

        raw_cfs.update(fund_raw_cfs)

    return raw_cfs

# ═══════════════════════════════════════════════════════════════════════════
# Annual pacing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _add_annual_funds_generated(
    ptf: Portfolio,
    year: int,
    gen_cfg: PrivateGenConfig,
    mfr_source: Union[str, pd.DataFrame],
    ptf_size: float,
    startdate: str,
    verbose: bool,
    simulation_cache: Optional[SimulationCache] = None,
) -> Dict[str, pd.DataFrame]:
    """Add a new vintage of MFR-generated funds to the portfolio."""
    # Override start year to current year
    cfg_copy = copy.deepcopy(gen_cfg)
    cfg_copy.start_year_override = year

    new_funds, _, _, raw_cfs = _build_generated_private_funds(
        gen_cfg=cfg_copy,
        mfr_source=mfr_source,
        ptf_size=ptf_size,
        startdate=startdate,
        verbose=verbose,
        simulation_cache=simulation_cache,
    )

    for f in new_funds:
        f.name = f"{f.name}_{year}"
        ptf.add_asset(f, target_weight=0.0, asset_deviation=0.0,
                       liquidity_rank=len(ptf.positions) + 1)

    return raw_cfs


def _add_annual_funds_selected(
    ptf: Portfolio,
    year: int,
    sel_cfg: PrivateSelConfig,
    private_data: pd.DataFrame,
    ptf_size: float,
    startdate: str,
) -> pd.DataFrame:
    """Add a new vintage of selected funds to the portfolio."""
    cfg_copy = copy.deepcopy(sel_cfg)
    cfg_copy.d_year = year

    new_funds, _, _, consolidated = _build_selected_private_funds(
        sel_cfg=cfg_copy,
        private_data=private_data,
        ptf_size=ptf_size,
        startdate=startdate,
    )
    for f in new_funds:
        f.name = f"{f.name} ({year})"
        ptf.add_asset(f, target_weight=0.0, asset_deviation=0.0,
                       liquidity_rank=len(ptf.positions) + 1)
    return consolidated


# ═══════════════════════════════════════════════════════════════════════════
# Analytics helper
# ═══════════════════════════════════════════════════════════════════════════

def _compute_analytics(
    asset_values: dict,
    privatecashflows: dict,
    compute_irr_series: bool = True,
    compute_tvpi_series: bool = True,
) -> dict:
    """Compute analytics from timecycle_drawdown outputs.

    Parameters
    ----------
    compute_irr_series : bool
        If True (default) compute the full IRR time-series.
        If False, compute only the final scalar IRR (cheaper).
    compute_tvpi_series : bool
        If True (default) compute the full TVPI time-series.
        If False, only the scalar TVPI is included.

    Best-effort: never raises, stores errors in the result dict.
    """
    analytics: Dict[str, Any] = {}

    try:
        df = pd.DataFrame(asset_values)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        # Total NAV
        if "Total" in df.columns and len(df) > 0:
            analytics["total_nav"] = float(df["Total"].iloc[-1])

        # Private NAV
        if "Total Private NAV" in df.columns and len(df) > 0:
            analytics["private_nav"] = float(df["Total Private NAV"].iloc[-1])

        # Cumulative contributions / distributions
        if "Total Period Contributions" in df.columns:
            analytics["cumulative_contributions"] = float(
                df["Total Period Contributions"].sum()
            )
        if "Total Period Distributions" in df.columns:
            analytics["cumulative_distributions"] = float(
                df["Total Period Distributions"].sum()
            )

        # TVPI
        cum_c = analytics.get("cumulative_contributions", 0)
        cum_d = analytics.get("cumulative_distributions", 0)
        nav = analytics.get("private_nav", 0)
        if cum_c > 0:
            analytics["tvpi"] = (cum_d + nav) / cum_c
        else:
            analytics["tvpi"] = np.nan

        # IRR via analytics module
        if _analytics is not None and "Date" in df.columns:
            try:
                irr_df = _analytics.calculate_irr(
                    df,
                    date_col="Date",
                    contrib_col="Total Period Contributions",
                    dist_col="Total Period Distributions",
                    nav_col="Total Private NAV",
                    out_col="Private ITD IRR",
                )

                if "Private ITD IRR" in irr_df.columns and len(irr_df) > 0:
                    if compute_irr_series:
                        analytics["irr_series"] = irr_df["Private ITD IRR"].tolist()
                    analytics["irr"] = float(irr_df["Private ITD IRR"].iloc[-1])

            except Exception as e:
                analytics["irr_error"] = str(e)

        # TVPI over time
        if _analytics is not None and compute_tvpi_series:
            try:
                df2 = df.copy()
                if "Cumulative Contributions" not in df2.columns:
                    df2["Cumulative Contributions"] = df2["Total Period Contributions"].cumsum()
                if "Cumulative Distributions" not in df2.columns:
                    df2["Cumulative Distributions"] = df2["Total Period Distributions"].cumsum()
                if (
                    "Total Private NAV" in df2.columns
                    and "Cumulative Contributions" in df2.columns
                    and "Cumulative Distributions" in df2.columns
                ):
                    denom = df2["Cumulative Contributions"].replace(0, np.nan)
                    df2["Private TVPI"] = (
                        df2["Total Private NAV"] + df2["Cumulative Distributions"]
                    ) / denom

                    analytics["tvpi_series"] = df2["Private TVPI"].tolist()
            except Exception as e:
                analytics["tvpi_series_error"] = str(e)
        
        # Deployment / Private Percentage over time
        # Total NAV
        if "Total" in df.columns and len(df) > 0:
            analytics["total_nav"] = float(df["Total"].iloc[-1])

        # Private NAV
        if "Total Private NAV" in df.columns and len(df) > 0:
            analytics["private_nav"] = float(df["Total Private NAV"].iloc[-1])

        # Private NAV exposure %
        if "Total Private NAV" in df.columns and "Total" in df.columns and len(df) > 0:
            denom = df["Total"].replace(0, np.nan)
            private_nav_pct_series = df["Total Private NAV"] / denom

            analytics["private_nav_pct"] = float(private_nav_pct_series.iloc[-1])
            analytics["private_nav_pct_series"] = private_nav_pct_series.tolist()

        # Additional analytics can be added here...
    except Exception as e:
        analytics["error"] = str(e)

    return analytics


# ═══════════════════════════════════════════════════════════════════════════
# Simulation runner
# ═══════════════════════════════════════════════════════════════════════════

def _run_simulation(
    ptf: Portfolio,
    time_periods: int,
    rebalance_method: str,
    rebal_periodicity: int,
    redemptions: dict,
    subscriptions: dict,
    red_base: str,
    red_max: bool,
    max_pct: float,
    sub_max: bool,
    smax_pct: float,
    historical: bool,
    startdate: str,
    periodicity: str,
    earlybreak: bool,
    growfirst: bool,
    verbose: bool,
) -> dict:
    """Run ``Portfolio.timecycle_drawdown`` and package outputs.

    Returns
    -------
    dict with keys:
        asset_values, warnings, subs, reds, privatecashflows,
        earlybreakflag, line_of_credit_usage
    """
    result = ptf.timecycle_drawdown(
        time_periods=time_periods,
        rebalance_method=rebalance_method,
        rebal_periodicity=rebal_periodicity,
        redemptions=redemptions,
        red_base=red_base,
        subscriptions=subscriptions,
        red_max=red_max,
        max_pct=max_pct,
        sub_max=sub_max,
        smax_pct=smax_pct,
        verbose=verbose,
        historical=historical,
        startdate=startdate,
        periodicity=periodicity,
        earlybreak=earlybreak,
        growfirst=growfirst,
    )

    if result is None:
        raise RuntimeError(
            "timecycle_drawdown returned None — likely a configuration "
            "error (e.g. periodicity mismatch between assets)."
        )

    (asset_values, sim_warnings, subs, reds,
     privatecashflows, earlybreakflag, loc_usage) = result

    return {
        "asset_values": asset_values,
        "warnings": sim_warnings,
        "subs": subs,
        "reds": reds,
        "privatecashflows": privatecashflows,
        "earlybreakflag": earlybreakflag,
        "line_of_credit_usage": loc_usage,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY-POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_integrated_portfolio_simulation(
    # ── Data source selection ────────────────────────────────────────────
    private_source: str,
    private_build_mode: str,
    mfr_source: Optional[Union[str, pd.DataFrame]] = None,
    preqin_engine: Any = None,
    burgiss_path: Optional[str] = None,
    # ── Portfolio basics ─────────────────────────────────────────────────
    portfolio_name: str = "Integrated Portfolio",
    ptf_size: float = 1_000_000_000.0,
    ptf_life: int = 10,
    periodicity: str = "Q",
    startdate: Optional[str] = None,
    # ── Private-side config ──────────────────────────────────────────────
    generated_config: Optional[PrivateGenConfig] = None,
    selected_config: Optional[PrivateSelConfig] = None,
    commitment_pacing: str = "one_time",
    # ── Public-side config ───────────────────────────────────────────────
    public_assets: Optional[List[PublicAssetConfig]] = None,
    # ── Simulation controls ──────────────────────────────────────────────
    historical: bool = False, #for return streams on public assets
    rebalance_method: str = "No Rebalance",
    rebal_periodicity: int = 999999999,
    redemptions: Optional[dict] = None,
    subscriptions: Optional[dict] = None,
    red_base: str = "Fixed",
    red_max: bool = True,
    max_pct: float = 0.05,
    sub_max: bool = False,
    smax_pct: float = 0.0,
    earlybreak: bool = True,
    growfirst: bool = False,
    # ── Line of credit (optional) ────────────────────────────────────────
    line_of_credit: Optional[dict] = None,
    # ── Misc ─────────────────────────────────────────────────────────────
    random_seed: Optional[int] = None,
    verbose: bool = False,
    # ── Phase-1 performance options ──────────────────────────────────────
    simulation_cache: Optional[SimulationCache] = None,
    summary_only: bool = False,
    compute_irr_series: bool = True,
    compute_tvpi_series: bool = True,
) -> dict:
    """Run an integrated portfolio simulation mixing private funds and public assets.

    This is the single high-level entry-point for the portfolio simulation
    toolkit.  It supports:

    * **MFR-generated** private funds (``private_source="mfr"``,
      ``private_build_mode="generated"``): modelled via the Ares pipeline
      and converted to GenPort Fund objects.
    * **Preqin/Burgiss-selected** private funds (``private_source="preqin"``
      or ``"burgiss"``, ``private_build_mode="selected"``): historical
      cashflows drawn via ``fund_selector``.
    * Any number of public/liquid Asset sleeves.
    * One-time or annual commitment pacing.
    * Full per-period simulation via ``Portfolio.timecycle_drawdown``.

    Parameters
    ----------
    private_source : str
        ``"mfr"``, ``"preqin"``, or ``"burgiss"``.
    private_build_mode : str
        ``"generated"`` (model from MFR) or ``"selected"`` (draw from database).
    mfr_source : str | pd.DataFrame | None
        Path to MFR CSV or pre-loaded DataFrame (required for ``"mfr"``).
    preqin_engine : sqlalchemy.Engine | None
        Engine for Preqin SQL Server (required for ``"preqin"``).
    burgiss_path : str | None
        Path to Burgiss Excel file (optional for ``"burgiss"``).
    portfolio_name : str
        Human-readable label.
    ptf_size : float
        Total portfolio size in dollars.
    ptf_life : int
        Portfolio life in years.
    periodicity : str
        Simulation periodicity: ``"Q"`` (quarterly, default), ``"M"``, ``"Y"``.
    startdate : str | None
        Simulation start date (e.g. ``"2027-03-31"``).  Defaults to next
        quarter-end if ``None``.
    generated_config : PrivateGenConfig | None
        Required when ``private_build_mode == "generated"``.
    selected_config : PrivateSelConfig | None
        Required when ``private_build_mode == "selected"``.
    commitment_pacing : str
        ``"one_time"``, ``"annual"``, ``"schedule"``.
    public_assets : list[PublicAssetConfig] | None
        Configs for public/liquid asset sleeves.
    historical : bool
        Use historical return streams (requires ``returnstream`` on assets).
    rebalance_method : str
        ``"No Rebalance"``, ``"Priority"``, ``"Pro-Rata"``.
    rebal_periodicity : int
        Rebalance every *n* periods.
    redemptions : dict | None
        ``{period_index: amount}``.
    subscriptions : dict | None
        ``{period_index: amount}``.
    red_base : str
        ``"Fixed"``, ``"NAV"``, ``"Dist"``.
    red_max : bool
        Cap redemptions to ``max_pct`` of NAV.
    max_pct : float
        Maximum redemption % if ``red_max`` is True.
    sub_max : bool
        Cap subscriptions.
    smax_pct : float
        Max subscription % of NAV.
    earlybreak : bool
        Stop simulation on unmet capital calls / redemptions.
    growfirst : bool
        Apply growth in the first period.
    line_of_credit : dict | None
        Optional credit facility config.  Keys: ``name``, ``balance``,
        ``interest_rate``, ``max_balance``, ``liquidity``, ``periodicity``,
        ``policy``.
    random_seed : int | None
        For reproducibility.
    verbose : bool
        Emit progress messages.
    simulation_cache : SimulationCache | None
        Optional cache object.  When provided, MFR data is loaded once and
        generated cashflows are cached across calls.  Pass the same
        ``SimulationCache`` instance to every scenario in a Monte Carlo run.
    summary_only : bool
        If True, return a compact dict (``config``, ``analytics``,
        ``earlybreakflag``, ``positions_summary``) without the heavy
        portfolio / simulation_results / cashflow record objects.
    compute_irr_series : bool
        If False, skip the expensive per-period IRR series and compute
        only the final scalar IRR.
    compute_tvpi_series : bool
        If False, skip the per-period TVPI series.

    Returns
    -------
    dict
        Structured output with keys:
        ``config``, ``portfolio``, ``private_source``, ``private_build_mode``,
        ``positions_summary``, ``simulation_results``,
        ``private_cashflow_records``, ``public_asset_records``, ``analytics``.
        When ``summary_only=True``, only ``config``, ``analytics``,
        ``earlybreakflag``, and ``positions_summary`` are returned.

    Example (Monte Carlo)
    ---------------------
    >>> cache = SimulationCache()
    >>> for scenario in scenarios:
    ...     result = run_integrated_portfolio_simulation(
    ...         ...,
    ...         simulation_cache=cache,
    ...         summary_only=True,
    ...         compute_irr_series=False,
    ...         compute_tvpi_series=False,
    ...     )
    """

    # ── Defaults ─────────────────────────────────────────────────────────
    if public_assets is None:
        public_assets = []
    if redemptions is None:
        redemptions = {}
    if subscriptions is None:
        subscriptions = {}
    if startdate is None:
        from dateutil.relativedelta import relativedelta as _rd
        from datetime import datetime as _dt
        now = _dt.now()
        # Next quarter-end
        q = ((now.month - 1) // 3 + 1) * 3
        y = now.year
        if q > 12:
            q = 3
            y += 1
        startdate = pd.Timestamp(y, q, 1) + pd.offsets.MonthEnd(0)
        startdate = startdate.strftime("%Y-%m-%d")

    # ── Validation ───────────────────────────────────────────────────────
    _validate_inputs(
        private_source=private_source,
        private_build_mode=private_build_mode,
        ptf_size=ptf_size,
        ptf_life=ptf_life,
        periodicity=periodicity,
        rebalance_method=rebalance_method,
        red_base=red_base,
        public_assets=public_assets,
        commitment_pacing=commitment_pacing,
    )

    if private_build_mode == "generated" and generated_config is None:
        raise ValueError(
            "generated_config (PrivateGenConfig) is required when "
            "private_build_mode='generated'."
        )
    if private_build_mode == "selected" and selected_config is None:
        raise ValueError(
            "selected_config (PrivateSelConfig) is required when "
            "private_build_mode='selected'."
        )
    if private_source == "mfr" and mfr_source is None:
        raise ValueError(
            "mfr_source is required when private_source='mfr'."
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    # ── Determine time periods ───────────────────────────────────────────
    period_map = {"Q": 4, "M": 12, "Y": 1}
    time_periods = ptf_life * period_map.get(periodicity, 4)

    if verbose:
        logger.info("=" * 60)
        logger.info("Integrated Portfolio Simulation")
        logger.info("  source=%s  mode=%s  pacing=%s",
                     private_source, private_build_mode, commitment_pacing)
        logger.info("  size=$%,.0f  life=%d yr  periods=%d (%s)",
                     ptf_size, ptf_life, time_periods, periodicity)
        logger.info("=" * 60)

    # ── Build private-side ───────────────────────────────────────────────
    private_raw_cfs: Dict[str, Any] = {}

    if private_build_mode == "generated":
        if commitment_pacing == "schedule":
            fund_objs = []
            fund_tws = []
            fund_devs = []
            private_raw_cfs = {}
        else:
            fund_objs, fund_tws, fund_devs, private_raw_cfs = (
                _build_generated_private_funds(
                    gen_cfg=generated_config,
                    mfr_source=mfr_source,
                    ptf_size=ptf_size,
                    startdate=startdate,
                    verbose=verbose,
                    simulation_cache=simulation_cache,
                )
            )
    else:
        # Load base selection data for initial + annual pacing
        private_data = _load_private_data(
            private_source=private_source,
            mfr_source=mfr_source,
            preqin_engine=preqin_engine,
            burgiss_path=burgiss_path,
            sel_cfg=selected_config,
            verbose=verbose,
        )
        fund_objs, fund_tws, fund_devs, consolidated = (
            _build_selected_private_funds(
                sel_cfg=selected_config,
                private_data=private_data,
                ptf_size=ptf_size,
                startdate=startdate,
            )
        )
        private_raw_cfs["initial_selected"] = consolidated

    # ── Build public-side ────────────────────────────────────────────────
    total_fund_weight = sum(fund_tws)  # will be 0 for funds
    asset_objs, asset_tws, asset_devs, asset_ranks = _build_public_assets(
        configs=public_assets,
        ptf_size=ptf_size,
        total_private_weight=total_fund_weight,
    )

    if verbose:
        logger.info("Private funds: %d   Public assets: %d",
                     len(fund_objs), len(asset_objs))

    # ── Build portfolio ──────────────────────────────────────────────────
    ptf = _build_portfolio(
        portfolio_name=portfolio_name,
        fund_objs=fund_objs,
        fund_tws=fund_tws,
        fund_devs=fund_devs,
        asset_objs=asset_objs,
        asset_tws=asset_tws,
        asset_devs=asset_devs,
        asset_liq_ranks=asset_ranks,
        line_of_credit_cfg=line_of_credit,
    )

    if verbose:
        logger.info("Portfolio '%s' constructed with %d positions.",
                     ptf.name, len(ptf.positions))
        for p in ptf.positions:
            logger.info("  %s (%s) → weight=%.4f",
                         p.name, p.type, ptf.target_weights.get(p, 0))

    # ── Annual pacing: pre-add future vintage funds ──────────────────────
    # For annual pacing we add Fund objects for each future year *before*
    # running the simulation.  Each Fund's cashflow index determines when
    # it "activates":
    #
    #   ``timecycle_drawdown()`` only starts processing a Fund once
    #   ``fund.cashflows.index.min() <= current_date``
    #
    # Therefore:
    #   - MFR-generated funds: we set ``start_year_override`` to the
    #     intended commitment year so the Ares pipeline emits cashflows
    #     starting in that year.  ``_assert_fund_starts_on_or_after``
    #     validates this.
    #   - Selected (Preqin/Burgiss) funds: ``d_year`` is set to the
    #     target vintage year so ``fund_selector`` draws curves from that
    #     vintage.  ``_warn_selected_fund_vintage`` checks consistency.
    #
    # All annual Funds are added before the simulation loop starts.
    # They remain dormant until their activation date.
    if commitment_pacing == "annual":
        start_yr = pd.to_datetime(startdate).year
        for yr_offset in range(1, ptf_life):
            target_yr = start_yr + yr_offset
            if verbose:
                logger.info("Adding annual commitment for year %d", target_yr)

            if private_build_mode == "generated":
                new_cfs = _add_annual_funds_generated(
                    ptf=ptf,
                    year=target_yr,
                    gen_cfg=generated_config,
                    mfr_source=mfr_source,
                    ptf_size=ptf_size,
                    startdate=startdate,
                    verbose=verbose,
                    simulation_cache=simulation_cache,
                )
                # Validate: each generated fund must start in target_yr
                for label, cf_df in new_cfs.items():
                    genport_df = _ares_to_genport_df(cf_df, startdate, scale_to_commitment=None)
                    _assert_fund_starts_on_or_after(
                        genport_df, target_yr, label,
                    )
                private_raw_cfs[f"annual_{target_yr}"] = new_cfs
            else:
                new_consolidated = _add_annual_funds_selected(
                    ptf=ptf,
                    year=target_yr,
                    sel_cfg=selected_config,
                    private_data=private_data,
                    ptf_size=ptf_size,
                    startdate=startdate,
                )
                # Warn if selected cashflows are far from the target year
                _warn_selected_fund_vintage(
                    new_consolidated, target_yr,
                    f"annual_{target_yr}",
                )
                private_raw_cfs[f"annual_{target_yr}"] = new_consolidated

    if commitment_pacing == "schedule":
        if private_build_mode != "generated":
            raise ValueError(
                "commitment_pacing='schedule' is currently only supported "
                "when private_build_mode='generated'."
            )
        if generated_config is None or generated_config.commitment_schedule is None:
            raise ValueError(
                "commitment_pacing='schedule' requires "
                "generated_config.commitment_schedule."
            )

        start_yr = pd.to_datetime(startdate).year

        for year_offset in generated_config.commitment_schedule.index:
            year_offset = int(year_offset)
            target_yr = start_yr + year_offset
            
            new_cfs = _add_scheduled_funds_generated(
                ptf=ptf,
                year_offset=year_offset,
                start_yr=start_yr,
                gen_cfg=generated_config,
                mfr_source=mfr_source,
                startdate=startdate,
                verbose=verbose,
                simulation_cache=simulation_cache,
            )

            # Validate that the generated fund starts in or after the scheduled year.
            for label, cf_df in new_cfs.items():
                genport_df = _ares_to_genport_df(
                    cf_df,
                    startdate,
                    scale_to_commitment=None,
                )
                _assert_fund_starts_on_or_after(
                    genport_df,
                    target_yr,
                    label,
                )
            private_raw_cfs[f"scheduled_{target_yr}"] = new_cfs


    # ── Run simulation ───────────────────────────────────────────────────
    if verbose:
        logger.info("Running simulation for %d periods …", time_periods)

    sim = _run_simulation(
        ptf=ptf,
        time_periods=time_periods,
        rebalance_method=rebalance_method,
        rebal_periodicity=rebal_periodicity,
        redemptions=redemptions,
        subscriptions=subscriptions,
        red_base=red_base,
        red_max=red_max,
        max_pct=max_pct,
        sub_max=sub_max,
        smax_pct=smax_pct,
        historical=historical,
        startdate=startdate,
        periodicity=periodicity,
        earlybreak=earlybreak,
        growfirst=growfirst,
        verbose=verbose,
    )

    # ── Analytics ────────────────────────────────────────────────────────
    analytics = _compute_analytics(
        asset_values=sim["asset_values"],
        privatecashflows=sim["privatecashflows"],
        compute_irr_series=compute_irr_series,
        compute_tvpi_series=compute_tvpi_series,
    )

    # ── Build positions summary ──────────────────────────────────────────
    positions_summary = []
    for p in ptf.positions:
        positions_summary.append({
            "name": p.name,
            "type": p.type,
            "asset_class": p.asset_class,
            "target_weight": ptf.target_weights.get(p, 0),
            "periodicity": p.periodicity,
        })

    # ── Public asset records ─────────────────────────────────────────────
    public_asset_records = []
    for p in ptf.positions:
        if p.type == "Asset":
            nav_key = p.name
            if nav_key in sim["asset_values"]:
                public_asset_records.append({
                    "name": p.name,
                    "values": sim["asset_values"][nav_key],
                })

    # ── Package config for output ────────────────────────────────────────
    config_out = {
        "private_source": private_source,
        "private_build_mode": private_build_mode,
        "portfolio_name": portfolio_name,
        "ptf_size": ptf_size,
        "ptf_life": ptf_life,
        "periodicity": periodicity,
        "startdate": startdate,
        "commitment_pacing": commitment_pacing,
        "rebalance_method": rebalance_method,
        "rebal_periodicity": rebal_periodicity,
        "red_base": red_base,
        "red_max": red_max,
        "max_pct": max_pct,
        "earlybreak": earlybreak,
        "growfirst": growfirst,
        "random_seed": random_seed,
        "n_public_assets": len(public_assets),
        "n_private_funds": sum(1 for p in ptf.positions if p.type == "Fund"),
    }
    if generated_config is not None:
        gen_cfg_out = asdict(generated_config)

        if generated_config.growth_curves is not None:
            gen_cfg_out["growth_curves"] = list(generated_config.growth_curves.keys())

        if generated_config.commitment_schedule is not None:
            gen_cfg_out["commitment_schedule"] = (
                generated_config.commitment_schedule.to_dict()
            )

        config_out["generated_config"] = gen_cfg_out
    if selected_config is not None:
        config_out["selected_config"] = asdict(selected_config)

    if verbose:
        early = sim.get("earlybreakflag", 0)
        logger.info("Simulation complete.  Early-break flag = %d", early)
        logger.info("Analytics: %s", {k: v for k, v in analytics.items()
                                       if not k.endswith("_series")})

    # ── Return ────────────────────────────────────────────────────────────
    if summary_only:
        return {
            "config": config_out,
            "analytics": analytics,
            "earlybreakflag": sim.get("earlybreakflag", 0),
            "positions_summary": positions_summary,
        }

    return {
        "config": config_out,
        "portfolio": ptf,
        "private_source": private_source,
        "private_build_mode": private_build_mode,
        "positions_summary": positions_summary,
        "simulation_results": sim,
        "private_cashflow_records": private_raw_cfs,
        "public_asset_records": public_asset_records,
        "analytics": analytics,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Example usage
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Integrated Portfolio Simulation — Example Configs")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Example 1: MFR-generated private funds + cash + public equity
    # ------------------------------------------------------------------
    print("\n--- Example 1: MFR-generated (one_time) ---\n")

    MFR_CSV = "Data Files/MFR_PYTHON_2026.04.csv"

    gen_cfg = PrivateGenConfig(
        fund_names=["SDL IV (Levered)", "Pathfinder III"],
        fund_weights={"SDL IV (Levered)": 0.6, "Pathfinder III": 0.4},
        coinvest_multipliers={"SDL IV (Levered)": 0.0, "Pathfinder III": 0.5},
    )

    cash_sleeve = PublicAssetConfig(
        name="Cash",
        price0=1.0,
        target_weight=0.40,
        a_return=0.01,        # ~4% annual in quarterly terms
        volatility=0.001,
        a_income=0.005,       # quarterly income yield
        income_volatility=0.0,
        asset_class="Cash",
        reinvestment_rate=1.0,
        liquidity=1,
        sub_period=1,
        prorate=1.0,
        periodicity="Q",
        deviation=0.05,
        liquidity_rank=1,
    )

    equity_sleeve = PublicAssetConfig(
        name="US Equity",
        price0=100.0,
        target_weight=0.20,
        a_return=0.02,        # quarterly
        volatility=0.08,
        a_income=0.005,
        income_volatility=0.002,
        asset_class="Equity",
        reinvestment_rate=0.5,
        liquidity=1,
        sub_period=1,
        prorate=1.0,
        periodicity="Q",
        deviation=0.05,
        liquidity_rank=2,
    )

    # NOTE: Private funds claim 0% target_weight because you can't
    # rebalance into them. The remaining 40% (= 1 - 0.40 - 0.20) is
    # implied by the dollar amount of private commitments.

    try:
        result = run_integrated_portfolio_simulation(
            private_source="mfr",
            private_build_mode="generated",
            mfr_source=MFR_CSV,
            portfolio_name="Example MFR Portfolio",
            ptf_size=1_000_000_000,
            ptf_life=10,
            periodicity="Q",
            startdate="2027-03-31",
            generated_config=gen_cfg,
            commitment_pacing="one_time",
            public_assets=[cash_sleeve, equity_sleeve],
            rebalance_method="Priority",
            rebal_periodicity=4,
            earlybreak=True,
            growfirst=False,
            random_seed=42,
            verbose=True,
        )
        print("\nPositions:")
        for p in result["positions_summary"]:
            print(f"  {p['name']:35s}  type={p['type']:5s}  weight={p['target_weight']:.4f}")
        print(f"\nAnalytics: {result['analytics']}")
    except Exception as exc:
        print(f"Example 1 failed (expected if MFR file is missing): {exc}")

    # ------------------------------------------------------------------
    # Example 2: Burgiss-selected funds with annual commitments
    # ------------------------------------------------------------------
    print("\n--- Example 2: Burgiss-selected (annual pacing) ---\n")

    sel_cfg = PrivateSelConfig(
        init_funds=5,
        target_base="Percentage",
        target_private=0.30,
        init_age=0,
        select_year=True,
        d_year=2015,
        replacement=True,
        strategy="Infrastructure",
        currency="USD",
    )

    cash_only = PublicAssetConfig(
        name="Liquid Reserve",
        price0=1.0,
        target_weight=1.0,
        a_return=0.01,
        volatility=0.001,
        a_income=0.005,
        income_volatility=0.0,
        asset_class="Cash",
        reinvestment_rate=1.0,
        liquidity=1,
        sub_period=1,
        prorate=1.0,
        periodicity="Q",
        deviation=0.0,
        liquidity_rank=1,
    )

    # This example requires the Burgiss file on disk.
    # Uncomment and adjust the path to run:
    #
    # result2 = run_integrated_portfolio_simulation(
    #     private_source="burgiss",
    #     private_build_mode="selected",
    #     burgiss_path="Data Files/Burgiss_Cashflowsv4.xlsx",
    #     portfolio_name="Burgiss Annual Pacing Example",
    #     ptf_size=500_000_000,
    #     ptf_life=12,
    #     periodicity="Q",
    #     startdate="2015-03-31",
    #     selected_config=sel_cfg,
    #     commitment_pacing="annual",
    #     public_assets=[cash_only],
    #     earlybreak=False,
    #     growfirst=False,
    #     random_seed=123,
    #     verbose=True,
    # )

    print("(Burgiss example is commented out — requires the Excel file on disk.)")

    # ------------------------------------------------------------------
    # Schema notes
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCHEMA / COMPATIBILITY NOTES")
    print("=" * 70)
    print("""
1. ares_cashflow_convert hard-codes the reference year to 2026 when
   computing 'vint years from today'.  If the simulation uses a different
   reference year, downstream quartile filters may be slightly off.

2. fund_selector requires 'Management Fees Paid' and
   'GP Distributions this Period' columns.  If these are absent from
   the source data, they default to 0.

3. GenPortvN.Fund.get_current_period_data() looks up by the 'Age' column.
   Ensure 'Age' starts at 0 and increments by 1 each period.

4. When using Burgiss data, the scale_factor defaults to 100,000
   (Burgiss units are 100, whereas Preqin uses 10,000,000).

5. The timecycle_drawdown engine checks
   ``asset.cashflows.index.min() <= current_date`` before processing a
   Fund.  For annual pacing, new vintage Fund objects are added to the
   Portfolio before the simulation starts, and they 'activate' when the
   sim date reaches their first cashflow date.
   - MFR-generated funds: start_year_override is set per vintage so
     cashflows begin in the intended year.  _assert_fund_starts_on_or_after
     validates this at build time.
   - Selected funds: d_year is set per vintage.  _warn_selected_fund_vintage
     emits a warning if cashflow dates are wildly inconsistent.

6. overcommit_pct is applied as a sizing-layer adjustment in selected-fund
   mode.  The effective private target passed to fund_selector is
   target_private * (1 + overcommit_pct), clamped at 1.0 for Percentage
   base.  fund_selector itself does not know about overcommitment.

7. subscriptions (timecycle_drawdown 'subscriptions' dict) must contain
   absolute dollar amounts, NOT rates.  Redemptions use rates because
   red_base determines the dollar base.
""")
    print("=" * 70)
    print("Module loaded successfully.  Import run_integrated_portfolio_simulation.")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1 Performance: SimulationCache usage snippet
    # ------------------------------------------------------------------
    # cache = SimulationCache()
    # scenarios = [{"random_seed": i} for i in range(100)]
    # results = []
    # for s in scenarios:
    #     r = run_integrated_portfolio_simulation(
    #         private_source="mfr",
    #         private_build_mode="generated",
    #         mfr_source=MFR_CSV,
    #         generated_config=gen_cfg,
    #         public_assets=[cash_sleeve, equity_sleeve],
    #         ptf_size=1_000_000_000,
    #         ptf_life=10,
    #         periodicity="Q",
    #         startdate="2027-03-31",
    #         commitment_pacing="one_time",
    #         random_seed=s["random_seed"],
    #         simulation_cache=cache,      # MFR loaded once, cashflows cached
    #         summary_only=True,           # compact output
    #         compute_irr_series=False,    # skip expensive series
    #         compute_tvpi_series=False,
    #     )
    #     results.append(r["analytics"])
    # print(f"Ran {len(results)} scenarios, cache has "
    #       f"{len(cache.generated_cashflows)} cached cashflow entries.")
