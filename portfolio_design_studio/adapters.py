"""
Adapt cashflows
"""
import pandas as pd

def ares_cashflow_convert(
    cashflow_df,
    scale_to_commitment=None,
):
    """
    Convert Ares-model cashflows to GenPort-style columns.

    Parameters
    ----------
    cashflow_df : pd.DataFrame
        Raw Ares cashflow dataframe.
    scale_to_commitment : float | None
        If provided, rescale the fund so max committed equals this amount.
        If None, preserve actual dollar sizes.
    """
    new_cashflows = cashflow_df.copy()

    new_cashflows["quarter_end"] = pd.to_datetime(new_cashflows["Date"])
    new_cashflows["nav_eoq"] = new_cashflows["Gross Equity NAV"]
    new_cashflows["committed"] = new_cashflows["Committed"]
    new_cashflows["unfunded_eoq"] = new_cashflows["Unfunded"]
    new_cashflows["cum_contributions_eoq"] = -new_cashflows["Cumulative Contributions"]
    new_cashflows["cum_distributions_eoq"] = new_cashflows["Cumulative LP Received"]

    if "Management Fees Paid" not in new_cashflows.columns:
        new_cashflows["Management Fees Paid"] = 0.0
    if "GP Distributions this Period" not in new_cashflows.columns:
        new_cashflows["GP Distributions this Period"] = 0.0

    if scale_to_commitment is not None:
        max_committed = float(new_cashflows["committed"].max())
        if max_committed > 0:
            scale_factor = max_committed / float(scale_to_commitment)
            scaled_columns = [
                "nav_eoq",
                "committed",
                "unfunded_eoq",
                "cum_contributions_eoq",
                "cum_distributions_eoq",
                "Management Fees Paid",
                "GP Distributions this Period",
            ]
            for col in scaled_columns:
                if col in new_cashflows.columns:
                    new_cashflows[col] = new_cashflows[col] / scale_factor

    keep_cols = [
        "quarter_end",
        "nav_eoq",
        "committed",
        "unfunded_eoq",
        "cum_contributions_eoq",
        "cum_distributions_eoq",
        "Management Fees Paid",
        "GP Distributions this Period",
    ]
    return new_cashflows[keep_cols]