"""
Data Module for Portfolio Simulation Toolkit

This module provides data-loading and data-preparation functions for the portfolio
simulation toolkit. All functions are designed to be importable without side effects
(no global connections or file reads at import time).

Functions:
    - get_engine: Create a SQLAlchemy engine from a connection string
    - load_cashflows: Load Preqin cashflows from SQL Server
    - load_irr: Load Preqin IRR performance data from SQL Server
    - load_burgiss: Load Burgiss cashflows from an Excel file
    - calculate_quartiles: Compute IRR quartile labels for fund-level data
"""
"""
Import necesary data like SOFR curve
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from pathlib import Path



# =============================================================================
# Exceptions
# =============================================================================

class DataLoadError(Exception):
    """Raised when data loading fails due to missing columns or invalid data."""
    pass


# =============================================================================
# Engine Factory
# =============================================================================

def get_engine(conn_str: str | None = None) -> Engine | None:
    """
    Create a SQLAlchemy engine from a connection string.

    If conn_str is None, returns None (does not create a global engine by default).

    Parameters
    ----------
    conn_str : str | None, optional
        A SQLAlchemy-compatible connection string, e.g.:
        'mssql+pyodbc://server/database?trusted_connection=yes&driver=SQL+Server'
        If None, the function returns None without creating an engine.

    Returns
    -------
    sqlalchemy.Engine | None
        A SQLAlchemy Engine instance if conn_str is provided, otherwise None.

    Examples
    --------
    >>> engine = get_engine("mssql+pyodbc://server/db?trusted_connection=yes&driver=SQL+Server")
    >>> engine is not None
    True
    >>> get_engine(None) is None
    True
    """
    if conn_str is None:
        return None
    return create_engine(conn_str)


# =============================================================================
# SQL Loaders
# =============================================================================

# Default query for load_cashflows matching GenPortvN.py behavior
_DEFAULT_CASHFLOWS_QUERY = "SELECT * FROM [FundMetrics].[dbo].[Preqin_CF_Curves]"

# Strategy mapping used for filtering (same as GenPortvN.py)
STRATEGY_MAP: dict[str | None, list[str] | None] = {
    "Buyout": ["Buyout"],
    "Turnaround": ["Turnaround"],
    "Private Debt": [
        "Mezzanine", "Distressed Debt", "Direct Lending - Senior Debt", "Special Situations",
        "Venture Debt", "Direct Lending - Blended / Opportunistic Debt",
        "Direct Lending - Unitranche Debt", "Direct Lending",
        "Direct Lending - Junior / Subordinated Debt", "Direct Lending Credit Strategies",
        "Private Debt Fund of Funds"
    ],
    "Growth": ["Growth"],
    "Venture": ["Venture (General)", "Expansion / Late Stage", "Early Stage", "Early Stage: Start-up"],
    "Balanced": ["Balanced"],
    "Real Estate": [
        "Real Estate Opportunistic", "Real Estate Debt", "Real Estate Value Added",
        "Real Estate Secondaries", "Real Estate Core", "Real Estate Core-Plus",
        "Real Estate Co-Investment", "Real Estate Fund of Funds", "Real Estate Distressed"
    ],
    "Infrastructure": [
        "Infrastructure Core Plus", "Infrastructure Core", "Infrastructure Value Added",
        "Infrastructure Secondaries", "Infrastructure Opportunistic", "Infrastructure Debt",
        "Infrastructure Fund of Funds"
    ],
    "Natural Resources": ["Natural Resources", "Timber"],
    "Co-Investment": ["Co-Investment", "Co-Investment Multi-Manager"],
    "Secondaries": ["Secondaries", "Direct Secondaries"],
    "Fund of Funds": ["Fund of Funds"],
    None: None
}


def _normalize_filter(value: Any, default: Any = None) -> list | None:
    """
    Normalize a filter value to a list.

    Parameters
    ----------
    value : Any
        The filter value (str, int, list, or None).
    default : Any, optional
        The default value if value is None.

    Returns
    -------
    list | None
        A list of values or None if value is None.
    """
    if value is None:
        return default
    return [value] if isinstance(value, (str, int)) else list(value)


def load_cashflows(
    engine: Engine,
    query: str | None = None,
    *,
    vintage: int | list[int] | None = None,
    strategy: str | None = None,
    geography: str | list[str] | None = None,
    currency: str | list[str] | None = None,
    fund_status: str | list[str] | None = None,
    industries: str | list[str] | None = None,
    reference_year: int = 2024,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Load Preqin Cashflows from SQL Server.

    Columns returned:
        fund_id, fund_name, firm_name, vintage, geography, strategy, committed,
        currency, fund_status, industries, fund_quarterly_age, quarter_start,
        quarter_end, nav_boq, cum_contributions_boq, cum_distributions_boq,
        unfunded_boq, contributions, distributions, nav_eoq, cum_contributions_eoq,
        cum_distributions_eoq, unfunded_eoq, max_fund_age, vint years from today

    Parameters
    ----------
    engine : sqlalchemy.Engine
        A SQLAlchemy engine connected to the FundMetrics database.
    query : str | None, optional
        Custom SQL query. If None, uses the default query that pulls from
        [FundMetrics].[dbo].[Preqin_CF_Curves].
    vintage : int | list[int] | None, optional
        Filter by vintage year(s). Options: 1983-2022.
    strategy : str | None, optional
        Filter by strategy (uses STRATEGY_MAP for grouping).
        Options: Buyout, Turnaround, Private Debt, Growth, Venture, Balanced,
        Real Estate, Infrastructure, Natural Resources, Co-Investment,
        Secondaries, Fund of Funds.
    geography : str | list[str] | None, optional
        Filter by geography (e.g., US, Europe, UK, Asia).
    currency : str | list[str] | None, optional
        Filter by currency (e.g., USD, EUR, GBP).
    fund_status : str | list[str] | None, optional
        Filter by fund status (e.g., Liquidated, Closed).
    industries : str | list[str] | None, optional
        Filter by industries (e.g., Healthcare, Diversified).
    reference_year : int, optional
        Reference year for calculating 'vint years from today' (default: 2024).
    **kwargs : Any
        Additional keyword arguments (ignored, for forward compatibility).

    Returns
    -------
    pd.DataFrame
        DataFrame with cashflow data, filtered and enriched with max_fund_age
        and 'vint years from today' columns.

    Raises
    ------
    DataLoadError
        If engine is None or required columns are missing from the query result.

    Examples
    --------
    >>> engine = get_engine("mssql+pyodbc://server/FundMetrics?trusted_connection=yes&driver=SQL+Server")
    >>> cf = load_cashflows(engine, vintage=2015, strategy="Buyout")
    """
    if engine is None:
        raise DataLoadError("Engine cannot be None. Please provide a valid SQLAlchemy engine.")

    # Build filters using the strategy map
    strategy_values = STRATEGY_MAP.get(strategy, None) if strategy in STRATEGY_MAP else [strategy]

    filters = {
        "vintage": _normalize_filter(vintage),
        "strategy": _normalize_filter(strategy_values) if strategy is not None else None,
        "geography": _normalize_filter(geography),
        "currency": _normalize_filter(currency),
        "fund_status": _normalize_filter(fund_status),
        "industries": _normalize_filter(industries),
    }

    # Build WHERE clause with parameterized queries
    conditions = []
    params: list[Any] = []
    for key, value in filters.items():
        if value is not None:
            if len(value) == 1:
                conditions.append(f"{key} = ?")
                params.append(value[0])
            else:
                conditions.append(f"{key} IN ({','.join('?' * len(value))})")
                params.extend(value)

    # Use custom query or default
    base_query = query if query is not None else _DEFAULT_CASHFLOWS_QUERY
    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    # Execute query
    filtered_data = pd.read_sql(sql=base_query, con=engine, params=tuple(params) if params else None)

    # Validate required columns
    required_cols = ["fund_id", "fund_quarterly_age", "vintage"]
    missing = [col for col in required_cols if col not in filtered_data.columns]
    if missing:
        raise DataLoadError(f"Missing required columns: {missing}")

    # Calculate derived columns (same logic as GenPortvN.py)
    filtered_data["max_fund_age"] = filtered_data.groupby("fund_id")["fund_quarterly_age"].transform("max")
    filtered_data["vint years from today"] = reference_year - filtered_data["vintage"]

    # Exclude funds with insufficient data
    # Vintage <= 2015: must have >= 20 quarters (5 years)
    # Vintage > 2015: must have at least (n - 3) * 4 quarters
    mask = (
        ((filtered_data["max_fund_age"] >= 20) & (filtered_data["vintage"] <= 2015))
        | (
            (filtered_data["vintage"] > 2015)
            & (filtered_data["max_fund_age"] >= (filtered_data["vint years from today"] - 3) * 4)
        )
    )
    filtered_data = filtered_data[mask]

    return filtered_data


# Default query for load_irr matching GenPortvN.py behavior
_DEFAULT_IRR_QUERY = """
SELECT f.Fund_Name
      ,p.[Fund_ID]
      ,p.[Date_Reported]
      ,p.[Measure]
      ,p.[Value]
FROM [FundMetrics].[dbo].[PreqinPerformance] p 
INNER JOIN [FundMetrics].[dbo].[PreqinFund] f
    on p.Fund_ID = f.Fund_ID
where p.Measure = 'IRR'
order by p.Date_Reported desc, f.Fund_Name 
"""


def load_irr(
    engine: Engine,
    query: str | None = None,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Load Preqin IRR Performance Data from SQL Server.

    Columns returned:
        fund_name, fund_id, Date_Reported, Measure, Value

    Parameters
    ----------
    engine : sqlalchemy.Engine
        A SQLAlchemy engine connected to the FundMetrics database.
    query : str | None, optional
        Custom SQL query. If None, uses the default query that joins
        PreqinPerformance with PreqinFund and filters for IRR measure.
    **kwargs : Any
        Additional keyword arguments (ignored, for forward compatibility).

    Returns
    -------
    pd.DataFrame
        DataFrame with IRR performance data. Columns renamed to:
        fund_name, fund_id, Date_Reported, Measure, Value.

    Raises
    ------
    DataLoadError
        If engine is None or required columns are missing from the query result.

    Examples
    --------
    >>> engine = get_engine("mssql+pyodbc://server/FundMetrics?trusted_connection=yes&driver=SQL+Server")
    >>> irr_df = load_irr(engine)
    """
    if engine is None:
        raise DataLoadError("Engine cannot be None. Please provide a valid SQLAlchemy engine.")

    # Use custom query or default
    sql_query = query if query is not None else _DEFAULT_IRR_QUERY

    # Execute query
    irr_data = pd.read_sql(sql=sql_query, con=engine)

    # Validate required columns (before rename)
    required_cols = ["Fund_ID", "Fund_Name"]
    missing = [col for col in required_cols if col not in irr_data.columns]
    if missing:
        raise DataLoadError(f"Missing required columns: {missing}")

    # Rename columns to match downstream expectations (same as GenPortvN.py)
    irr_data = irr_data.rename(columns={"Fund_ID": "fund_id", "Fund_Name": "fund_name"})

    # Convert Date_Reported to datetime if present
    if "Date_Reported" in irr_data.columns:
        irr_data["Date_Reported"] = pd.to_datetime(irr_data["Date_Reported"], errors="coerce")

    return irr_data


# =============================================================================
# Excel Loaders
# =============================================================================

def load_burgiss(
    path: str | Path = """Data Files/Burgiss_Cashflowsv4.xlsx""",
    *,
    vintage: int | list[int] | None = None,
    strategy: str | list[str] | None = None,
    geography: str | list[str] | None = None,
    currency: str | list[str] | None = None,
    reference_year: int = 2025,
    scale_factor: float = 100_000.0
) -> pd.DataFrame:
    """
    Load Burgiss Cashflows from an Excel file.

    Columns returned (after processing):
        fund_id, fund_name, vintage, geography, strategy, committed, currency,
        fund_status, fund_quarterly_age, quarter_end, nav_eoq, cum_contributions_eoq,
        cum_distributions_eoq, unfunded_eoq, max_fund_age, vint years from today

    Parameters
    ----------
    path : str | Path
        Path to the Burgiss Excel file.
    vintage : int | list[int] | None, optional
        Filter by vintage year(s).
    strategy : str | list[str] | None, optional
        Filter by strategy (e.g., "Infrastructure").
    geography : str | list[str] | None, optional
        Filter by geography.
    currency : str | list[str] | None, optional
        Filter by currency (e.g., "USD").
    reference_year : int, optional
        Reference year for calculating 'vint years from today' (default: 2024).
    scale_factor : float, optional
        Factor to scale monetary columns (default: 100,000).
        Burgiss Cashflows are based on units of 100. Preqin cashflows are based on units of 10,000,000

    Returns
    -------
    pd.DataFrame
        DataFrame with Burgiss cashflow data, filtered and scaled.

    Raises
    ------
    DataLoadError
        If the file does not exist or required columns are missing.
    FileNotFoundError
        If the specified path does not exist.

    Examples
    --------
    >>> df = load_burgiss("Burgiss_Cashflowsv4.xlsx", strategy="Infrastructure", currency="USD")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Burgiss file not found: {path}")

    # Read the Excel file
    data = pd.read_excel(path)

    # Validate required columns
    required_cols = ["fund_id", "vintage", "fund_quarterly_age", "committed",
                     "cum_contributions_eoq", "cum_distributions_eoq", "nav_eoq", "unfunded_eoq"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise DataLoadError(f"Missing required columns in Burgiss file: {missing}")

    filtered_data = data.copy()

    # Apply filters
    if vintage is not None:
        vintage_list = [vintage] if isinstance(vintage, int) else list(vintage)
        filtered_data = filtered_data[filtered_data["vintage"].isin(vintage_list)]
    if strategy is not None:
        strategy_list = [strategy] if isinstance(strategy, str) else list(strategy)
        filtered_data = filtered_data[filtered_data["strategy"].isin(strategy_list)]
    if geography is not None:
        geography_list = [geography] if isinstance(geography, str) else list(geography)
        filtered_data = filtered_data[filtered_data["geography"].isin(geography_list)]
    if currency is not None:
        currency_list = [currency] if isinstance(currency, str) else list(currency)
        filtered_data = filtered_data[filtered_data["currency"].isin(currency_list)]

    # Calculate derived columns (same as GenPortvN.py)
    filtered_data["max_fund_age"] = filtered_data.groupby("fund_id")["fund_quarterly_age"].transform("max")
    filtered_data["vint years from today"] = reference_year - filtered_data["vintage"]

    # Exclude funds with insufficient data (same logic as GenPortvN.py)
    mask = (
        ((filtered_data["max_fund_age"] >= 20) & (filtered_data["vintage"] <= 2015))
        | (
            (filtered_data["vintage"] > 2015)
            & (filtered_data["max_fund_age"] >= (filtered_data["vint years from today"] - 3) * 4)
        )
    )
    filtered_data = filtered_data[mask]

    # Scale monetary columns (same as GenPortvN.py)
    scale_cols = ["committed", "cum_contributions_eoq", "cum_distributions_eoq", "nav_eoq", "unfunded_eoq"]
    for col in scale_cols:
        if col in filtered_data.columns:
            filtered_data[col] = filtered_data[col] * scale_factor

    return filtered_data.reset_index(drop=True)


# =============================================================================
# Quartile Calculation
# =============================================================================

def calculate_quartiles(
    irr_df: pd.DataFrame,
    by_cols: list[str] | None = None,
    irr_column: str = "IRR",
    quartile_labels: list[str] | None = None
) -> pd.DataFrame:
    """
    Compute quartile labels for fund-level rows based on IRR.

    Groups the data by the specified columns (e.g., Vintage Year, Strategy)
    and assigns quartile labels (Q1-Q4) within each group.

    Parameters
    ----------
    irr_df : pd.DataFrame
        DataFrame containing IRR data. Must have the column specified by
        `irr_column` and (optionally) the grouping columns.
    by_cols : list[str] | None, optional
        Columns to group by before computing quartiles.
        Default: ["Vintage Year", "Strategy"] if present, otherwise no grouping.
    irr_column : str, optional
        Name of the column containing IRR values (default: "IRR").
    quartile_labels : list[str] | None, optional
        Labels for quartiles from lowest to highest (default: ["Q1", "Q2", "Q3", "Q4"]).

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'quartile' column containing quartile labels.
        If there's insufficient data for quartiles, the label will be 'Not enough data'.

    Raises
    ------
    DataLoadError
        If the IRR column is missing from the DataFrame.

    Examples
    --------
    >>> irr_df = pd.DataFrame({
    ...     "fund_id": [1, 2, 3, 4, 5, 6, 7, 8],
    ...     "IRR": [0.15, 0.10, 0.20, 0.05, 0.12, 0.18, 0.08, 0.25],
    ...     "vintage": [2015, 2015, 2015, 2015, 2016, 2016, 2016, 2016]
    ... })
    >>> result = calculate_quartiles(irr_df, by_cols=["vintage"])
    """
    if irr_column not in irr_df.columns:
        raise DataLoadError(f"IRR column '{irr_column}' not found in DataFrame. "
                          f"Available columns: {list(irr_df.columns)}")

    # Default labels
    if quartile_labels is None:
        quartile_labels = ["Q1", "Q2", "Q3", "Q4"]

    # Default grouping columns (check if they exist)
    if by_cols is None:
        default_by_cols = ["Vintage Year", "Strategy"]
        by_cols = [col for col in default_by_cols if col in irr_df.columns]

    result = irr_df.copy()

    def _compute_quartile(group: pd.DataFrame) -> pd.DataFrame:
        """Compute quartiles for a single group."""
        group = group.copy()
        irr_values = group[irr_column].dropna()

        # Check for sufficient data
        if len(group) == 0 or irr_values.empty:
            group["quartile"] = "Not enough data"
        elif len(irr_values) < 4:
            group["quartile"] = "Not enough data"
        else:
            try:
                # pd.qcut assigns labels from lowest to highest
                group["quartile"] = pd.qcut(
                    group[irr_column].dropna(),
                    q=4,
                    labels=quartile_labels
                )
            except ValueError:
                # Not enough distinct values for quartiles
                group["quartile"] = "Not enough data"

        return group

    if by_cols:
        # Validate grouping columns exist
        missing_group_cols = [col for col in by_cols if col not in result.columns]
        if missing_group_cols:
            raise DataLoadError(f"Grouping columns not found: {missing_group_cols}")

        result = result.groupby(by_cols, group_keys=False).apply(_compute_quartile)
    else:
        result = _compute_quartile(result)

    # Fill any NaN quartiles (if IRRs were originally NaN)
    # Convert to string first to avoid categorical issues
    if "quartile" in result.columns:
        result["quartile"] = result["quartile"].astype(str).replace("nan", "Not enough data")

    return result


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Portfolio Simulation Toolkit - Data Module Demo")
    print("=" * 70)

    # Demo 1: Engine creation
    print("\n[Demo 1] Creating an engine (not connecting)")
    print("-" * 50)
    
    # Example connection string (not actually connecting here)
    demo_conn_str = "mssql+pyodbc://njd-p-lmksql01/FundMetrics?trusted_connection=yes&driver=SQL+Server"
    
    print(f"Connection string: {demo_conn_str}")
    print("To create an engine, call:")
    print("  engine = get_engine(conn_str)")
    print("  engine = get_engine(None)  # Returns None, no side effects")
    
    # Demo: None returns None
    null_engine = get_engine(None)
    print(f"\nget_engine(None) returned: {null_engine}")

    # Demo 2: Show expected usage patterns
    print("\n[Demo 2] Usage patterns for data loaders")
    print("-" * 50)
    print("""
# Load cashflows with filters:
engine = get_engine("mssql+pyodbc://server/FundMetrics?...")
cf_df = load_cashflows(
    engine,
    vintage=2015,
    strategy="Buyout",
    currency="USD"
)

# Load IRR data:
irr_df = load_irr(engine)

# Load Burgiss data from Excel:
burgiss_df = load_burgiss(
    "path/to/Burgiss_Cashflowsv4.xlsx",
    strategy="Infrastructure",
    currency="USD"
)

# Calculate quartiles:
irr_with_quartiles = calculate_quartiles(
    irr_df,
    by_cols=["vintage", "strategy"],
    irr_column="Value"  # or "IRR" depending on data
)
""")

    # Demo 3: Quartile calculation with sample data
    print("\n[Demo 3] Quartile calculation example with sample data")
    print("-" * 50)
    
    sample_irr_df = pd.DataFrame({
        "fund_id": list(range(1, 17)),
        "fund_name": [f"Fund_{i}" for i in range(1, 17)],
        "vintage": [2015] * 8 + [2016] * 8,
        "IRR": [0.15, 0.10, 0.20, 0.05, 0.12, 0.18, 0.08, 0.25,
                0.11, 0.14, 0.09, 0.22, 0.16, 0.07, 0.19, 0.13]
    })
    
    print("Sample IRR DataFrame:")
    print(sample_irr_df.to_string(index=False))
    
    result = calculate_quartiles(
        sample_irr_df,
        by_cols=["vintage"],
        irr_column="IRR"
    )
    
    print("\nWith quartiles calculated (grouped by vintage):")
    print(result[["fund_id", "fund_name", "vintage", "IRR", "quartile"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("Demo complete. Import this module to use in your portfolio simulation.")
    print("=" * 70)

def read_sofr_curve(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported SOFR curve file type: {path.suffix}")
