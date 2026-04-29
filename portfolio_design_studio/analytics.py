"""
Analytics.py

Takes outputs from the timecycle_drawdown function and generates analytics such as TVPI, DPI, RVPI, IRR, PME, etc. for the portfolio and individual funds. 
Also generates charts such as TVPI over time, cash flow waterfalls, etc. 
"""

import pandas as pd
from datetime import date
from pyxirr import xirr
import numpy as np


def aggregate_simulation_results(simulation_results):
    """
    Output columns = 'Total', 'Subscription', 'Redemption', 'Date',
       'Total Period Contributions', 'Total Period Distributions',
       'Total Private NAV', 'Total Com. Not Called', 
       any NAV or Unfunded columns for public / private assets added to portfolio
       'Cumulative Contributions', 'Cumulative Distributions'
    """
    quarters = list(simulation_results.keys())
    combined_results = pd.DataFrame(simulation_results[quarters[0]][0])
    for year in range(len(quarters)):
        if year > 0:
            combined_results = pd.concat([combined_results, pd.DataFrame(simulation_results[quarters[year]][0])], ignore_index=True)

    combined_results['Cumulative Contributions'] = combined_results['Total Period Contributions'].cumsum()
    combined_results['Cumulative Distributions'] = combined_results['Total Period Distributions'].cumsum()

    return combined_results

def calculate_tvpi(cashflows):
    #Calculate TVPI over time 
    cashflows['TVPI'] = (cashflows['Total'] + cashflows['Cumulative Distributions']) / (cashflows['Cumulative Contributions'])
    return cashflows

def calculate_dpi(cashflows):
    #Calculate DPI over time 
    cashflows['DPI'] = cashflows['Cumulative Distributions'] / (cashflows['Cumulative Contributions'])
    return cashflows

def calculate_irr(
    cashflows: pd.DataFrame,
    date_col: str = "Date",
    contrib_col: str = "Total Period Contributions",
    dist_col: str = "Total Period Distributions",
    nav_col: str = "Total Private NAV",
    out_col: str = "ITD IRR",
) -> pd.DataFrame:
    """
    Calculate ITD IRR for each period using pyxirr.xirr.
    For each row t:
      cashflows = (distributions - contributions) from 0..t
      terminal value = total nav at t (added to last cashflow)
    """
    req = [date_col, contrib_col, dist_col, nav_col]
    missing = [c for c in req if c not in cashflows.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    x = cashflows.copy()
    x[date_col] = pd.to_datetime(x[date_col])
    x = x.sort_values(date_col).reset_index(drop=True)

    net_cf = (x[dist_col].fillna(0.0) - x[contrib_col].fillna(0.0)).astype(float)
    nav = x[nav_col].fillna(0.0).astype(float)

    irr_vals = []
    for i in range(len(x)):
        dts = x.loc[:i, date_col].tolist()
        amts = net_cf.iloc[:i + 1].tolist()
        amts[-1] += nav.iat[i]  # add terminal NAV at period i

        # xirr needs at least one negative and one positive cashflow
        if not (any(a < 0 for a in amts) and any(a > 0 for a in amts)):
            irr_vals.append(np.nan)
            continue

        try:
            irr_vals.append(xirr(dts, amts))
        except Exception:
            irr_vals.append(np.nan)

    x[out_col] = irr_vals
    return x