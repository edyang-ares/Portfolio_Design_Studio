"""
Generalized Portfolio Model
"""

############ Dependencies #######################################################################################################################################

# %pip install keyring artifacts-keyring
# %pip install time_series --extra-index-url https://pkgs.dev.azure.com/landmarkpartners/_packaging/Landmark/pypi/simple/
# %pip install pip-system-certs

import numpy as np
#We must use pandas 2.1.4
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine

# Import data loaders from data.py (no side effects at import time)
from . import data as _data

import threading
import warnings
warnings.filterwarnings('ignore')

#multiprocessing
# import concurrent.futures
# import asyncio
# import os
# import multiprocessing

import copy
import argparse as arg
import logging as log
import logging.handlers as loghand
from typing import List
#import pydantic
from mpl_toolkits.mplot3d import Axes3D
from .Lmk_Irr import BG_IRR_array
import numpy_financial as npf
#import requests
import pip_system_certs
#from requests_ntlm import HttpNtlmAuth
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import itertools
# %matplotlib qt

import yfinance as yf
from scipy.stats import norm

###################################################################################################################################################

############################### DEPRECATED: Use data.py instead ##################################################################################
# Legacy connection string for backward compatibility (do NOT create engine at import time)
DEFAULT_CONN_STR = 'mssql+pyodbc://njd-p-lmksql01/FundMetrics?trusted_connection=yes&driver=SQL+Server'

# DEPRECATED: set_engine is no longer used. Pass engine directly to load functions.
def set_engine(new_engine):
    """DEPRECATED: This function is no longer used. Pass engine directly to load_cashflows/load_irr."""
    import warnings
    warnings.warn(
        "set_engine() is deprecated. Pass engine directly to load_cashflows() or load_irr().",
        DeprecationWarning,
        stacklevel=2
    )

###################################################################################################################################################

def normalize_filter(value, default=None):
    """normalize_filter takes in a value and returns a list of values
    if value is None, return None
    """
    if value is None:
        return default
    return [value] if isinstance(value, (str, int)) else list(value)

def load_cashflows(engine=None, conn_str=None, vintage=None, strategy=None, geography=None, currency=None, fund_status=None, industries=None):
    """Thin wrapper around data.load_cashflows for backward compatibility.
    
    Code to pull Preqin Cashflows from SQL Server.
    
    Parameters
    ----------
    engine : sqlalchemy.Engine, optional
        SQLAlchemy engine to use. If None, creates one from conn_str.
    conn_str : str, optional
        Connection string. If None and engine is None, uses DEFAULT_CONN_STR.
    vintage, strategy, geography, currency, fund_status, industries : optional
        Filters passed to data.load_cashflows.
    
    Returns
    -------
    pd.DataFrame
        Preqin cashflow data.
    """
    if engine is None:
        if conn_str is None:
            conn_str = DEFAULT_CONN_STR
        engine = _data.get_engine(conn_str)
    
    return _data.load_cashflows(
        engine,
        vintage=vintage,
        strategy=strategy,
        geography=geography,
        currency=currency,
        fund_status=fund_status,
        industries=industries
    ) 

# def load_burgiss(path='Burgiss_Cashflowsv4.xlsx', vintage=None, strategy=None, geography=None, currency=None):
#     """Thin wrapper around data.load_burgiss for backward compatibility.
    
#     Code to pull Cashflows from an Excel file with Burgiss data.
    
#     Parameters
#     ----------
#     path : str, optional
#         Path to the Burgiss Excel file. Default: 'Burgiss_Cashflowsv4.xlsx'
#     vintage, strategy, geography, currency : optional
#         Filters passed to data.load_burgiss.
    
#     Returns
#     -------
#     pd.DataFrame
#         Burgiss cashflow data.
#     """
#     return _data.load_burgiss(
#         path,
#         vintage=vintage,
#         strategy=strategy,
#         geography=geography,
#         currency=currency
#     )

def load_researchdatabase(vintage=None, strategy=None, geography=None, currency=None):
    data = pd.read_excel("""Data Files/investment_cfs_withnavs.xlsx""")
    filtered_data = data.copy()
    if vintage is not None:
        filtered_data = filtered_data[filtered_data['vintage'].isin([vintage])]
    if strategy is not None:
        filtered_data = filtered_data[filtered_data['strategy'].isin([strategy])]
    if geography is not None:
        filtered_data = filtered_data[filtered_data['geography'].isin([geography])]
    if currency is not None:
        filtered_data = filtered_data[filtered_data['currency'].isin([currency])]
    return filtered_data

def load_irr(engine=None, conn_str=None):
    """Thin wrapper around data.load_irr for backward compatibility.
    
    Pulls the Preqin Performance Data.
    
    Parameters
    ----------
    engine : sqlalchemy.Engine, optional
        SQLAlchemy engine to use. If None, creates one from conn_str.
    conn_str : str, optional
        Connection string. If None and engine is None, uses DEFAULT_CONN_STR.
    
    Returns
    -------
    pd.DataFrame
        IRR performance data with columns: fund_name, fund_id, Date_Reported, Measure, Value.
    """
    if engine is None:
        if conn_str is None:
            conn_str = DEFAULT_CONN_STR
        engine = _data.get_engine(conn_str)
    
    return _data.load_irr(engine)


def calculate_quartiles(group):
    """Groups the data by IRR quartile. The data passed should be a set (e.g., all funds in a specific vintage year).
    
    This is the original function signature for backward compatibility with groupby().apply() usage.
    For the new data.calculate_quartiles with more options, use data.calculate_quartiles directly.
    """
    # Check if group is empty or all IRRs are NaN
    if len(group) == 0 or group['IRR'].dropna().empty:
        group['quartile'] = 'Not enough data'
    elif len(group) < 4:
        group['quartile'] = 'Not enough data'
    else:
        try:
            group['quartile'] = pd.qcut(group['IRR'].dropna(), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        except ValueError:
            group['quartile'] = 'Not enough data'
    # Fill NaN quartiles (if any IRRs were NaN)
    #group['quartile'] = group['quartile'].fillna('Not enough data',inplace=True)
    return group

def fund_selector(init_funds, ptf_size, target_base, target_private, init_age, data,select_year=True,d_year =None, final_liquidation=True,first_quarter=True,scale_by_contrib=False,year_limit=0,replacement=True):
    """
    Selects equally sized funds for the portfolio and scales cashflows to the desired size.
    init_funds = number of funds to select
    ptf_size = size of total portfolio
    target_base = either a $ dollar amount or a percentage of total portfolio : allowed values are "Percentage" or "Dollar"
    target_private = percentage of total portfolio that is private, or dollar value
    init_age = age of the funds to select (e.g., selecting 2009 Vintage funds would be start_year = 2009, init_age = 0)
    data = fund cashflow data
    select_year = select a specific year if TRUE, otherwise select all years if FALSE 
    d_year = the year to select if select_year is TRUE, otherwise a dummy year if select_year is FALSE
    final_liquidation = set any remaining NAV and unfunded to 0 at the end of the series, add remaining NAV to distributions
    first_quarter = if True, then the first quarter will be 3/31 of the d_year, otherwise it will be the first quarter of the data
    scale_by_contrib = if True, then scale the cashflows by total cumulative contributions (since some funds call more than the initial commitment)
                        e.g., if a fund commitment is 100, but it actually calls 150, then we overwrite the commitment to be 100/150 = 66.67
                        if a fund commitment is 100, but it calls 80, then we keep the commitment as 100
    year_limit = the min number of years of data, if 0 then no limit
    replacement = if True, then funds can be selected with replacement, otherwise no replacement
    """
    # Randomly select seed funds
    if select_year==False:
        data['fund_age_at_start'] = (data['fund_quarterly_age']/4).apply(np.floor)
        correct_age = data[data['fund_age_at_start'] >= init_age].reset_index(drop=True)
    else:
        data['fund_age_at_start'] = data['vintage'] - d_year
        correct_age = data[data['fund_age_at_start'] == -init_age].reset_index(drop=True)
    # Remove funds that don't have sufficient data
    if year_limit != 0:
        correct_age = correct_age[correct_age['max_fund_age'] >= (year_limit-init_age)*4].reset_index(drop=True)
    funds = correct_age['fund_id'].unique()

    # if select_year == True:
    #     #Remove funds that don't have sufficient data
    #     funds = correct_age[correct_age['max_fund_age'] >= 24]['fund_id'].unique()
    
    if replacement == True:
        sample_funds = np.random.choice(funds, init_funds, replace=True)
    if replacement == False:
        init_funds = min(init_funds, len(funds))  # Ensure we don't select more funds than available
        if init_funds <= 0:
            print("No funds available for the selected criteria.")
        sample_funds = np.random.choice(funds, init_funds, replace=False)

    # Seed cashflows
    seed_cashflows = pd.DataFrame(columns = data.columns)
    for fund in sample_funds:
        temp = data[data['fund_id'] == fund].sort_values(by=['quarter_end']).reset_index(drop=True)
        if init_age != 0:
            temp['cum_contributions_eoq'].iat[init_age*4] = temp['cum_contributions_eoq'].iat[init_age*4]-temp['cum_contributions_eoq'].iat[init_age*4-1]-temp['nav_eoq'].iat[init_age*4]
            temp['cum_distributions_eoq'].iat[init_age*4] = temp['cum_distributions_eoq'].iat[init_age*4]-temp['cum_distributions_eoq'].iat[init_age*4-1]
        temp = temp[temp['fund_quarterly_age'] >= init_age*4].reset_index(drop=True)
        seed_cashflows = pd.concat([seed_cashflows, temp], ignore_index=True)
        #seed_cashflows = data[data['fund_id'].isin(sample_funds)]

    # Scale cashflows by 10,000,000 because Preqin has scaled cashflows to be 10,000,000 size
    #Burgiss scales to 100, so in load_burgiss, we multiplied by 100_000
    # Some of the cum_contributions exceed 10,000,000
    # Funds are equally sized
    if init_funds != 0:
        if target_base == "Percentage" or target_base == "Target":
            scale_factors = ((ptf_size * target_private) / init_funds) / 10000000
        elif target_base == "Dollar":
            scale_factors = (target_private) / (init_funds * 10000000)
    else:
        scale_factors = 0
    to_keep = ['fund_id', 'fund_name', 'quarter_end', 'fund_quarterly_age', 'nav_eoq', 'committed', 'cum_contributions_eoq', 'cum_distributions_eoq',
               'unfunded_eoq','Management Fees Paid','GP Distributions this Period']
    
    #Check if we have actual management fees and carry:
    if not ('Management Fees Paid' in data.columns and 'GP Distributions this Period' in data.columns):
        seed_cashflows['Management Fees Paid'] = 0
        seed_cashflows['GP Distributions this Period'] = 0

    consolidated_cashflows = seed_cashflows.copy()[to_keep]
    consolidated_cashflows[['committed', 'cum_contributions_eoq', 'cum_distributions_eoq', 'nav_eoq', 'unfunded_eoq', 'Management Fees Paid', 'GP Distributions this Period']] *= scale_factors
    #sort by fund_id and quarter_end
    consolidated_cashflows['quarter_end'] = pd.to_datetime(consolidated_cashflows['quarter_end'])
    consolidated_cashflows = consolidated_cashflows.sort_values(by=['fund_id', 'quarter_end']).reset_index(drop=True)

    cc_1 = pd.DataFrame(columns=['fund_id', 'committed', 'quarter_end', 'cum_contributions_eoq', 'cum_distributions_eoq', 'nav_eoq', 'unfunded_eoq', 'fund_quarter','Management Fees Paid','GP Distributions this Period'])
    #Make sure missing data don't appear as zeros in the nav_eoq column
    for fund in sample_funds:
        temp = consolidated_cashflows[consolidated_cashflows['fund_id'] == fund].drop_duplicates().reset_index(drop=True)
        for i in range(len(temp)):
            if i == 0:
                pass
            else:
                temp['nav_eoq'] = temp['nav_eoq'].mask(temp['nav_eoq'].eq(0)).ffill().fillna(0)
        #create a column to hold fund quarter of life
        temp['fund_quarter'] = np.arange(0, len(temp))
        cc_1 = pd.concat([cc_1, temp], ignore_index=True)
    consolidated_cashflows = cc_1.copy()
    
    if final_liquidation:
        max_quarter = consolidated_cashflows['fund_quarter'].max()
        cc_1t = pd.DataFrame(columns=['fund_id', 'committed', 'quarter_end', 'cum_contributions_eoq', 'cum_distributions_eoq', 'nav_eoq', 'unfunded_eoq', 'fund_quarter','Management Fees Paid','GP Distributions this Period', 'realizations'])
        for fund in sample_funds:
            temp = consolidated_cashflows[consolidated_cashflows['fund_id'] == fund].drop_duplicates().reset_index(drop=True)
            temp['realizations'] = 0  # Create a column for realizations
            if temp['fund_quarter'].max() < max_quarter:
                # Fill in missing quarters with zeros for contributions and distributions
                tqm = temp['fund_quarter'].max()
                for i in range(max_quarter - tqm):
                    new_row = {'fund_id': fund, 
                            'committed': temp['committed'].iat[-1], 
                            'quarter_end': temp['quarter_end'].iat[-1] + relativedelta(months=3),
                            'cum_contributions_eoq': temp['cum_contributions_eoq'].iat[-1],
                            'cum_distributions_eoq': temp['cum_distributions_eoq'].iat[-1],  # Add NAV to distributions for missing quarters
                            'nav_eoq': 0,  # Assuming NAV is 0 for missing quarters
                            'unfunded_eoq': 0,
                            'fund_quarter': temp['fund_quarter'].iat[-1] + 1,
                            'Management Fees Paid': 0,
                            'GP Distributions this Period': 0,
                            'realizations': temp['nav_eoq'].iat[-1]}  # Add NAV to realizations for missing quarters
                    temp.loc[len(temp)] = new_row
            temp['cum_distributions_eoq'] = temp['cum_distributions_eoq'].cummax()
            temp['cum_contributions_eoq'] = temp['cum_contributions_eoq'].cummin()
            temp['quarter_contribution'] = temp['cum_contributions_eoq'].diff()
            temp['quarter_distribution'] = temp['cum_distributions_eoq'].diff()
            cc_1t = pd.concat([cc_1t, temp], ignore_index=True)
        consolidated_cashflows = cc_1t.copy()
    
    if final_liquidation:
        for fund in sample_funds:
            temp = consolidated_cashflows[consolidated_cashflows['fund_id'] == fund].drop_duplicates().reset_index(drop=True)
            max_age = temp['quarter_end'].max()
            last_row = temp[temp['quarter_end'] == max_age]
            # Assuming last_row is the DataFrame containing the last row
            new_date = max_age + relativedelta(months=3)
            
            final_row = {'fund_id' : fund, 
                        'committed' : last_row['committed'].iat[0], 
                        'quarter_end' : new_date,
                        'cum_contributions_eoq' : last_row['cum_contributions_eoq'].iat[0],
                        'cum_distributions_eoq' : last_row['cum_distributions_eoq'].iat[0],
                        'nav_eoq' : 0,
                        'unfunded_eoq' : 0,
                        'fund_quarter' : last_row['fund_quarter'].iat[0] + 1,
                        'Management Fees Paid' : 0,
                        'GP Distributions this Period' : 0,
                        'realizations' : last_row['nav_eoq'].iat[0]}                       
            consolidated_cashflows.loc[len(consolidated_cashflows)] = final_row

    # Sort and consolidate cashflows by quarter end
    if select_year == True:
        consolidated_cashflows = consolidated_cashflows.sort_values(by=['quarter_end'])
        consolidated_cashflows = consolidated_cashflows.groupby('quarter_end').sum()
    else:
        consolidated_cashflows.drop(columns=['quarter_end'], inplace=True)
        consolidated_cashflows = consolidated_cashflows.sort_values(by=['fund_quarter'])
        consolidated_cashflows = consolidated_cashflows.groupby('fund_quarter').sum()
        start_date = '3/31/'+str(d_year)
        consolidated_cashflows['quarter_end'] = pd.date_range(start=start_date, periods=len(consolidated_cashflows), freq='Q')
        #set index to quarter_end
        consolidated_cashflows.set_index('quarter_end', inplace=True)

    # Ensure cumulative contributions and distributions are non-decreasing
    consolidated_cashflows['cum_contributions_eoq'] = consolidated_cashflows['cum_contributions_eoq'].cummin()
    consolidated_cashflows['cum_distributions_eoq'] = consolidated_cashflows['cum_distributions_eoq'].cummax()

    # Calculate quarterly contributions and distributions
    consolidated_cashflows['quarter_contribution'] = consolidated_cashflows['cum_contributions_eoq'].diff()
    consolidated_cashflows['quarter_distribution'] = consolidated_cashflows['cum_distributions_eoq'].diff()

    # Set the first quarter's contributions and distributions
    consolidated_cashflows['quarter_contribution'].iat[0] = consolidated_cashflows['cum_contributions_eoq'].iat[0]
    consolidated_cashflows['quarter_distribution'].iat[0] = consolidated_cashflows['cum_distributions_eoq'].iat[0]

    # Replace NaN values with prior value and add Age column
    #consolidated_cashflows['nav_eoq'] = consolidated_cashflows['nav_eoq'].fillna(method='ffill')
    consolidated_cashflows = consolidated_cashflows.ffill()


    # Check to see if the first quarter starts with 3/31, if not then insert a row with 3/31
    if first_quarter:
        while consolidated_cashflows.index[0].month != 3:
            first_row = {'fund_id' : 0, 
                        'committed' : 0,
                        'cum_contributions_eoq' : 0,
                        'cum_distributions_eoq' : 0,
                        'nav_eoq' : 0,
                        'unfunded_eoq' : 0,
                        'quarter_contribution' : 0,
                        'quarter_distribution' : 0,
                        'Management Fees Paid' : 0,
                        'GP Distributions this Period' : 0,
                        'realizations' : 0}
            indexdate = consolidated_cashflows.index[0] - relativedelta(months=3)
            consolidated_cashflows = pd.concat([pd.DataFrame(first_row, index=[indexdate]), consolidated_cashflows])

    consolidated_cashflows['Age'] = range(0, len(consolidated_cashflows))
    consolidated_cashflows.sort_index(inplace=True)
    consolidated_cashflows['Vintage'] = d_year - init_age
    #overwrite committed with correct values
    if target_base == "Percentage":
        consolidated_cashflows['committed'] = ptf_size * target_private
    elif target_base == "Dollar":
        consolidated_cashflows['committed'] = target_private

    #Make sure missing data don't appear as zeros in the nav_eoq column
    for i in range(len(consolidated_cashflows)):
        if i == 0:
            pass
        else:
            consolidated_cashflows['nav_eoq'] = consolidated_cashflows['nav_eoq'].mask(consolidated_cashflows['nav_eoq'].eq(0)).ffill().fillna(0)
    
    #add the realizations to the cum_distributions_eoq and the quarter_distribution columns
    #rolling sum up the realizations column
    consolidated_cashflows['cum_realizations_eoq'] = consolidated_cashflows['realizations'].cumsum() 
    consolidated_cashflows['cum_distributions_eoq'] = consolidated_cashflows['cum_distributions_eoq'] + consolidated_cashflows['cum_realizations_eoq']
    consolidated_cashflows['quarter_distribution'] = consolidated_cashflows['quarter_distribution'] + consolidated_cashflows['realizations']

    if final_liquidation:
        max_age = consolidated_cashflows.index.max()
        last_row = consolidated_cashflows.loc[max_age]
        # Assuming last_row is the DataFrame containing the last row
        new_date = max_age + relativedelta(months=3)
        
        final_row = {'fund_id' : 0, 
                    'committed' : last_row['committed'], 
                    'cum_contributions_eoq' : last_row['cum_contributions_eoq'],
                    'cum_distributions_eoq' : last_row['cum_distributions_eoq'] + last_row['nav_eoq'],
                    'nav_eoq' : 0,
                    'unfunded_eoq' : 0,
                    'quarter_contribution' : 0,
                    'quarter_distribution' : last_row['nav_eoq'],
                    'Age' : last_row['Age'] + 1,
                    'Vintage' : last_row['Vintage'],
                    'Management Fees Paid' : 0,
                    'GP Distributions this Period' : 0,
                    'realizations' : last_row['nav_eoq'],
                    'cum_realizations_eoq' : last_row['cum_realizations_eoq']+ last_row['nav_eoq']}
        
        consolidated_cashflows = pd.concat([consolidated_cashflows, pd.DataFrame(final_row, index=[new_date])])

    if scale_by_contrib == True:
        try: 
            sc_factor = min(-max(consolidated_cashflows['committed'])/sum(consolidated_cashflows['quarter_contribution']),1)
        except:
            sc_factor = 1
        consolidated_cashflows[['cum_contributions_eoq', 'cum_distributions_eoq', 'nav_eoq', 'unfunded_eoq',
                                'quarter_contribution','quarter_distribution','realizations','cum_realizations_eoq','Management Fees Paid','GP Distributions this Period']] *= sc_factor
    #print(sc_factor)
    #cum_contributions is a negative
    consolidated_cashflows['committed_not_called'] = consolidated_cashflows['committed'] + consolidated_cashflows['cum_contributions_eoq']
    consolidated_cashflows['committed_not_called'].clip(lower=0, inplace=True)

    return consolidated_cashflows, sample_funds

def volatility_calculator(stock, days):
    """Calculate the annualized volatility of a stock over a specified number of days.
    """
    #get the stock data
    stock_data = yf.Ticker(stock)
    stock_data = stock_data.history(period="1d", start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
    stock_data['log_return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    #set first return period as 0
    stock_data['log_return'].iloc[0] = 0
    stock_data['volatility'] = stock_data['log_return'].rolling(window=days).std() * np.sqrt(252)
    return stock_data['log_return'].std() * np.sqrt(252)

def get_weekdays(start_date, num_days):
    """Get a list of weekdays (Monday to Friday) starting from a given date.
    """
    weekdays = []
    current_date = pd.to_datetime(start_date)
    while len(weekdays) < num_days:
        if current_date.weekday() < 5:
            weekdays.append(current_date)
        current_date += timedelta(days=1)
    return weekdays

def mini_twr(cashflows,navs,periodicity="Q"):
    """Calculates the Time-Weighted Rate of Return (TWRR) for a given set of cashflows and net asset values (NAVs). (Can be used for private funds as well as public assets)
    cashflows: list of net cashflows (distributions - contributions) for each period
    navs: list of net asset values (NAVs) for each period
    periodicity: the periodicity of the cashflows, can be "Q" for quarterly, "M" for monthly, or "Y" for yearly
    """
    #Check if cashflows and navs are the same length
    if len(cashflows) != len(navs):
        raise ValueError("Cashflows and NAVs must be the same length")
    
    period = {"Q":4,"M":12,"Y":1}
    twr = {"hp_return": []}
    for i in range(len(cashflows)):
        if i > 0:
            try:
                hp_return = (navs[i]+cashflows[i])/(navs[i-1])-1
            except:
                hp_return = 0
        else:
            hp_return = 0
        twr["hp_return"].append(hp_return)
    twr = pd.DataFrame(twr)
    twr['TWRR'] = (1 + twr['hp_return']).cumprod()
    
    for i in range(len(twr)):
        if i == 0:
            twr['TWRR'][i] = 0
        else:
            twr['TWRR'][i] = twr['TWRR'][i] ** (period[periodicity]/(i+1)) - 1 
    return twr

def mini_irr(cashflows,navs,dates,method,periodicity = "Q"):
    """Calculates the Internal Rate of Return (IRR) for a given set of cashflows and net asset values (NAVs).
    cashflows: list of net cashflows (distributions - contributions) for each period
    navs: list of net asset values (NAVs) for each period
    dates: list of dates for each period
    method: the method of calculating the IRR, can be "LMK" for the LKM method or "Simple" for the simple IRR method
    periodicity: the periodicity of the cashflows, can be "Q" for quarterly, "M" for monthly, or "Y" for yearly
    """
    #calculate the IRR for a given set of cashflows and navs
    #cashflows is a list of cashflows
    #navs is a list of navs
    #dates is a list of dates
    #method is the method of calculating the IRR
    period = {"Q":4,"M":12,"Y":1}
    irr = {}
    for i in range(len(dates)):
        if method == "LMK":
            if i > 0:
                irr[dates[i]] = BG_IRR_array(pd.Series(cashflows[:i+1]),pd.Series(dates[:i+1]),nav = navs[i])
            else:
                irr[dates[0]] = 0
        elif method == "Simple":
            cashflows[i] += navs[i]
            irr[dates[i]] = (1+npf.irr(cashflows[:i+1]))**period[periodicity]-1
            #irr[dates[i]] = npf.irr(cashflows[:i+1])
            cashflows[i] -= navs[i]
        else:
            return
    return irr

def next_quarter_end(date):
    """
    Given a date, return the end date of the next calendar quarter.
    """
    # Ensure input is a pandas Timestamp
    date = pd.to_datetime(date)
    # Find current quarter
    current_quarter = ((date.month - 1) // 3)
    # Next quarter
    next_quarter = current_quarter + 1
    next_year = date.year
    if next_quarter > 4:
        next_quarter = 1
        next_year += 1
    # Quarter end months: 3, 6, 9, 12
    quarter_end_month = next_quarter * 3
    # Get last day of that month
    quarter_end = pd.Timestamp(next_year, quarter_end_month, 1) + pd.offsets.MonthEnd(0)
    return quarter_end

################################# The Portfolio Object ##############################################################################################

class Portfolio:
    """Model a portfolio of assets that can be rebalanced and have subscriptions and redemptions

    Create a class for the portfolio, which will hold the assets
    We first need to define the target weights for each asset
    #The portfolio can be impacted in five different ways:
        #1. The price of the assets can change
        #2. Adding an asset
        #3. setting the target weights
        #4. Adding subscriptions
        #5. Adding redemptions
        #6. Adding Line of Credit to meet capital calls and redemptions (if necessary)
    """    
    def __init__(self, name):
        """Create basic logging information
        Args:
            name (str): The name of the portfolio.
            positions (list): An initial empty list of Asset objects that are in the portfolio.
            asset_classes (list): A list to hold the unique asset classes of the assets in the portfolio.
            target_weights (dict): A dictionary to hold the target weights of each asset in the portfolio. Funds should be 0 since we cannot rebalance into them, Assets should be nonnegative
                                    Key is the asset object, value is the target weight.
            asset_deviation (dict): A dictionary to hold the allowed deviation from the target weight for each asset.
            liquidity_rank (dict): A dictionary to hold the liquidity rank of each asset in the portfolio (lower is more liquid).
            age (int): The age of the portfolio, starting at 0.
        """
        self.logger = log.getLogger("portfolio")
        log.basicConfig(filename= "portfolio_debug1.log", encoding="utf-8", level=log.INFO)

        self.name = name
        self.positions = []
        self.asset_classes = []
        self.target_weights = {}
        self.asset_deviation = {}
        self.liquidity_rank = {}
        self.age = 0
        self.line_of_credit = None
        self.loc_policy = None
        self.logger.info("Portfolio created")
    
    def add_asset(self, position,target_weight,asset_deviation,liquidity_rank):
        """
        Add an asset to the portfolio.
        
        Args:
            position (Asset/Fund): The asset to add.
            target_weight (float): The target weight of the asset in the portfolio.
            asset_deviation (float): The allowed deviation from the target weight.
            liquidity_rank (int): The liquidity rank of the asset (lower is more liquid).
        """
        self.positions.append(position)
        scalefactor = 1-target_weight
        for i in self.target_weights:
            self.target_weights[i] = self.target_weights[i]*scalefactor
        self.target_weights[position] = target_weight
        self.asset_deviation[position] = asset_deviation
        for i in self.liquidity_rank:
            if self.liquidity_rank[i] >= liquidity_rank:
                self.liquidity_rank[i] += 1
        self.liquidity_rank[position] = liquidity_rank

    def remove_asset(self, position):
        """
        Remove an asset from the portfolio.
        
        Args:
            position (Asset/Fund): The asset to remove.
        """
        if position in self.positions:
            self.positions.remove(position)
            del self.target_weights[position]
            del self.asset_deviation[position]
            removed_liquidity_rank = self.liquidity_rank[position]
            del self.liquidity_rank[position]
            for i in self.liquidity_rank:
                if self.liquidity_rank[i] > removed_liquidity_rank:
                    self.liquidity_rank[i] -= 1
        else:
            print("Asset not found in portfolio")
    
    def set_line_of_credit(self, line_of_credit, policy = "last_resort"):
        """Set a line of credit for the portfolio to meet capital calls and redemptions if necessary.
        The default will be that it will be used as last resort.
        Args:
            line_of_credit (float): The amount of the line of credit.
            policy (str): The policy for using the line of credit. Default is "last_resort". Also takes "first".
        """
        self.line_of_credit = line_of_credit
        self.loc_policy = policy

    def set_assets(self, positions, target_weights,asset_deviations,liquidity_ranks):
        """set a number of Assets / Funds at one time
        Args:
            positions (list): A list of Asset/Fund objects that are in the portfolio.
            target_weights (list): A list of target weights for each asset in the portfolio.
            asset_deviations (list): A list of allowed deviations from the target weight for each asset.
            liquidity_ranks (list): A list of liquidity ranks for each asset in the portfolio (lower is more liquid).
        """
        
        #Checks to make sure the parameters are correct
        params = [len(positions), len(target_weights), len(asset_deviations), len(liquidity_ranks)]
        if len(set(params))>1:
            #Log error in portfolio_debug log
            #self.logger.error("Error, incorrect dimensions for parameters")
            print("Error, incorrect dimensions for parameters")
            return
        
        if round(sum(target_weights),0)!=1:
            #self.logger.error("Error, target weights incorrect")
            print("Error, target weights incorrect")
            return
        
        for i in range(len(target_weights)):
            if asset_deviations[i] > target_weights[i]:
                #self.logger.error("Error, asset deviations incorrect")
                print("Error, asset deviations incorrect")
                return
            
        if (len(set(liquidity_ranks)) != len(liquidity_ranks)) or (min(liquidity_ranks) != 1) or (max(liquidity_ranks) != len(liquidity_ranks)):
            #self.logger.error("Error, liquidity ranking incorrect")
            print("Error, liquidity ranking incorrect")
            return
        
        self.positions = positions
        
        for i in range(len(target_weights)):
            self.target_weights[positions[i]] =  target_weights[i]
            self.asset_deviation[positions[i]] = asset_deviations[i]
            self.liquidity_rank[positions[i]] = liquidity_ranks[i]

        for i in range(len(self.positions)):
            self.asset_classes.append(self.positions[i].asset_class)
            self.asset_classes = list(set(self.asset_classes))  # Ensure unique asset classes

    def sell_asset(self, asset, quantity):
        """Sell a certain quantity of an asset for most liquid (cash/publics).
        If an asset is not currently in the portfolio (i.e., opening a short position), it will be added with a negative quantity.
        """
        #if asset is not in the portfolio, then set negative quantity
        if asset not in self.positions:
            self.add_asset(asset,0,0,len(self.positions)+1)
            asset.quantity = -quantity
        else:
            asset.quantity -= quantity
        #add the proceeds to the most liquid balance
        most_liquid = self.get_most_liquid_asset()
        most_liquid.quantity += quantity*asset.price/most_liquid.price
    
    def buy_asset(self, asset, quantity):
        """Buy a certain quantity of an asset, adding it to the portfolio if it is not already present.
        """
        if asset not in self.positions:
            self.add_asset(asset,0,0,len(self.positions)+1)
            asset.quantity = quantity
        else:
            asset.quantity += quantity
        #subtract the cost from the most liquid balance
        most_liquid = self.get_most_liquid_asset()
        most_liquid.quantity -= quantity*asset.price/most_liquid.price

    def add_subscription(self,time_period,subscription):
        """Adding a subscription to the portfolio at a certain time period in the portfolio life. 
        Args:
            time_period (int): Period of the portfolio life. Subscriptions into certain Assets/Funds may be subject to liquidity constraints.
            subscription (float): The amount of money to add to the portfolio.
        """
        #Record action of subscriptions
        subs = []
        subs.append("Subscription of %s" % np.round(subscription,4))
        
        #Calculate proforma portfolio and then allocate first to most liquid
        proforma = self.calculate_portfolio_value()+subscription
        most_liquid = self.get_most_liquid_asset()
        #Check if most liquid asset is below the target weight, if it is, then add the subscription to it until it hits the target weight
        if np.round(most_liquid.calculate_value()/proforma,2) < np.round(self.target_weights[most_liquid],2):
            liquid_needed = self.target_weights[most_liquid]*proforma - most_liquid.calculate_value()
            most_liquid.quantity += min(subscription,liquid_needed)/most_liquid.price
            self.logger.info("Adding %s to %s" % (min(subscription,liquid_needed), most_liquid.name))
            subs.append("Adding %s to %s" % (min(subscription,liquid_needed), most_liquid.name))
            # print("Adding %s to %s" % (min(subscription,liquid_needed), most_liquid.name))
            subscription -= min(subscription,liquid_needed)
        
        #Then loop through the other assets and calculate how much to allocate to them, if they can accept subscriptions
        if subscription > 0:
            sub_weights = {}
            for asset in self.positions:
                #sub_weights holds the "distance" that the asset is from its target
                sub_weights[asset.name] = max(self.target_weights[asset]-(asset.calculate_value()/proforma),0)*asset.sub_ability(time_period)
            total_sub = sum(sub_weights.values())
            
            #If all assets are at target, distribute the remaining subscription to most liquid asset that has nonzero targets
            if total_sub == 0:
                sorted_assets = sorted(self.positions, key=lambda x: self.liquidity_rank[x])
                for asset in sorted_assets:
                    if np.round(self.target_weights[asset],2)*asset.sub_ability(time_period) > 0:
                        asset.quantity += subscription/asset.price
                        subs.append("Adding %s to %s" % (subscription, asset.name))
                        return subs

            #If assets are not at target, allocate the subscription to them based on their target weights
            check = 0
            for asset in self.positions:
                if asset.type == 'Asset':
                    asset.quantity += subscription*(sub_weights[asset.name]/total_sub)/asset.price
                    self.logger.info("Adding %s to %s" % (subscription*sub_weights[asset.name]/total_sub, asset.name))
                    subs.append("Adding %s to %s" % (subscription*sub_weights[asset.name]/total_sub, asset.name))
                    check += subscription*(sub_weights[asset.name]/total_sub)
            if np.round(check,2) != np.round(subscription,2):
                self.logger.error("Error, subscription not allocated correctly")  
        return subs

    
    def add_redemption(self, time_period,redemption):
        """
        Adding a redemption to the portfolio at a certain time period in the portfolio life.
        Args:
            time_period (int): Period of the portfolio life. Redemptions from certain Assets/Funds may be subject to liquidity constraints.
            redemption (float): The amount of money to redeem from the portfolio.
        """
    
        #Track warnings and actions for redemptions
        warnings = []
        actions = []

        #Check if line of credit can be used to meet redemption, if it is set and the policy is "first"
        if self.line_of_credit is not None and self.loc_policy == "first":
            to_draw = min(redemption, self.line_of_credit.availability())
            self.line_of_credit.draw(to_draw)
            redemption -= to_draw

        #Use the most liquid asset first (cash/publics), then sell the assets in order of liquidity rank
        most_liquid = self.get_most_liquid_asset()
        #cashfirst is the value that we want to redeem
        cashfirst = min(redemption, most_liquid.calculate_value()) * most_liquid.liq_ability(time_period)
        most_liquid.quantity -= cashfirst / most_liquid.price
        self.logger.info("Redemption for %s" % np.round(redemption,4))
        actions.append("Redemption for %s" % np.round(redemption,4))
        redemption -= cashfirst
        self.logger.info("Redeeming %s from %s" % (np.round(cashfirst,4), most_liquid.name))
        actions.append("Redeeming %s from %s" % (np.round(cashfirst,4), most_liquid.name))
        if redemption > 0.00001:
            warnings.append("Insufficient cash/liquids to cover redemption")
            self.logger.warning("Insufficient cash/liquids to cover redemption")

        sorted_assets = sorted(self.positions, key=lambda x: self.liquidity_rank[x])
        #Then loop through the assets and rebalance them
        for asset in sorted_assets:
            if asset.name != most_liquid.name:
                if redemption == 0:
                    break
                else:
                    if asset.type != 'Fund':
                        sell_value = min(redemption, asset.calculate_value()*asset.liq_ability(time_period))
                        sell_value = sell_value*asset.prorate
                        self.logger.info("Selling %s %s" % (asset.name ,np.round(sell_value,4)))
                        actions.append("Selling %s %s" % (asset.name ,np.round(sell_value,4)))
                        redemption -= sell_value
                        asset.quantity -= sell_value/asset.price
        
        #Check if line of credit can be used to meet remaining redemption, if it is set and the policy is "last_resort"
        if self.line_of_credit is not None and self.loc_policy == "last_resort" and redemption > 0.00001:
            to_draw = min(redemption, self.line_of_credit.availability())
            self.line_of_credit.draw(to_draw)
            redemption -= to_draw
        
        if redemption > 0.00001:
            self.logger.warning("Warning, redemption not met")
            warnings.append("Redemption not met")
                #return "Redemption not met"
        return warnings, actions

    def calculate_portfolio_weights(self):
        """Calculate and return the weights of each position in the portfolio.
        """
        total_value = self.calculate_portfolio_value()
        if total_value == 0:
            self.logger.error("Error, portfolio value is 0")
            return 0
        else:
            weights = {}
            for asset in self.positions:
                weights[asset.name] = asset.calculate_value() / total_value
            return weights
    
    def calculate_portfolio_value(self):
        """Calculate and return the total value of the portfolio.
        """
        #Calculate and return the value of the entire portfolio
        total_value = 0
        for asset in self.positions:
            total_value += asset.calculate_value()
        return total_value

    def calculate_portfolio_value_active_funds_only(self, current_date):
        """
        Calculate portfolio value, but only include Fund positions that have
        reached their activation date (first cashflow date <= current_date).
        """
        total_value = 0.0
        current_date = pd.to_datetime(current_date)

        for asset in self.positions:
            if asset.type != 'Fund':
                total_value += asset.calculate_value()
            else:
                if asset.cashflows.index.min() <= current_date:
                    total_value += asset.calculate_value()

        return total_value
    
    def calculate_private_exposure(self):
        """Calculate the private exposure of the portfolio. Exposure includes NAV + Unfunded
        """
        #Calculate and return the value of the private exposure
        total_value = 0
        for asset in self.positions:
            if asset.type == 'Fund':
                total_value += asset.calculate_exposure()
        return total_value / self.calculate_portfolio_value()
    
    def calculate_asset_classes_exposure(self):
        """Calculate the exposure of each asset class in the portfolio, irrespective of Asset/Fund
        """
        asset_class_exposure = {}
        for asset in self.positions:
            if asset.asset_class not in asset_class_exposure:
                asset_class_exposure[asset.asset_class] = 0
            asset_class_exposure[asset.asset_class] += asset.calculate_exposure()
        
        total_value = self.calculate_portfolio_value()
        for key in asset_class_exposure:
            asset_class_exposure[key] /= total_value
        
        return asset_class_exposure
    
    def calculate_asset_classes_value(self):
        """Calculate the NAV of each asset class in the portfolio, irrespective of Asset/Fund
        """
        asset_class_value = {}
        for asset in self.positions:
            if asset.asset_class not in asset_class_value:
                asset_class_value[asset.asset_class] = 0
            asset_class_value[asset.asset_class] += asset.calculate_value()

        total_value = self.calculate_portfolio_value()
        for key in asset_class_value:
            asset_class_value[key] /= total_value

        return asset_class_value

    def calculate_asset_values(self):
        """A generator that yields the value of each asset in the portfolio.
        """
        for asset in self.positions:
            yield asset.calculate_value()
    
    def calculate_liquid_value(self):
        """Calculates the sum of liquid value (e.g., of Assets only) in the portfolio"""
        liquid_value = 0
        for asset in self.positions:
            if asset.type == 'Asset':
                liquid_value += asset.calculate_value()
        return liquid_value
    
    def calculate_liquid_weights(self):
        """Calculate and return the weights of each liquid position (Assets only) in the portfolio.
        """
        lv = self.calculate_liquid_value()
        if lv <= 0:
            return {p.name: 0.0 for p in self.positions if p.type == "Asset"}
        return {p.name: p.calculate_value() / lv for p in self.positions if p.type == "Asset"}

    def get_most_liquid_asset(self):
        """Get the most liquid asset in the portfolio based on liquidity rank.
        """
        return min(
            (a for a in self.positions if a.type == "Asset"),
            key=lambda a: self.liquidity_rank[a],)

    def rebalance_liquidity(self,time_period,verbose=False):
        """Rebalance the portfolio to ensure that the most liquid asset is at the target weight.
        Args:
            time_period (int): The current period in the portfolio life, used to determine liquidity constraints.
            verbose (bool): If True, print warnings and actions taken during rebalancing.
        """
        #First calculate the portfolio weights
        weights = self.calculate_portfolio_weights()
        #If Portfolio value == 0 or if most liquid asset target == 0, then return
        most_liquid = self.get_most_liquid_asset()
        if (weights[most_liquid.name] < 1e-4) or (self.target_weights[most_liquid] < 1e-6):
            return

        #Track warnings and actions
        warnings = []
        actions = []
        
        #Then calculate the difference between most liquid asset target weights and current weights
        #calculate the $ value of liquid sleeve assets needed
        liquid_needed = max((self.target_weights[most_liquid]-weights[most_liquid.name])*self.calculate_portfolio_value(),0)
        
        #Then sort the assets by the liquidity rank
        #When rebalancing into most liquid asset, we are allowed to use the entire balance of other assets
        sorted_assets = sorted(self.positions, key=lambda x: self.liquidity_rank[x])
        #Then loop through the assets and rebalance them
        for asset in sorted_assets:
            if asset.name != most_liquid.name and asset.type != 'Fund':
                if liquid_needed == 0:
                    break
                else:
                    #We only allow the asset to be sold if it's liquidity flag in the period is 1    
                    sell_value = min(liquid_needed, asset.calculate_value()*asset.liq_ability(time_period))
                    sell_value = sell_value*asset.prorate
                    self.logger.info("Selling %s %s" % (asset.name, np.round(sell_value,4)))
                    actions.append("Selling %s %s" % (asset.name, np.round(sell_value,4)))
                    self.logger.info("Buying %s %s" % (most_liquid.name, np.round(sell_value,4)))
                    actions.append("Buying %s %s" % (most_liquid.name, np.round(sell_value,4)))
                    liquid_needed -= sell_value
                    asset.quantity -= sell_value/asset.price
                    most_liquid.quantity += sell_value/most_liquid.price
        
        if liquid_needed > 0.00001:
            self.logger.warning("Warning, liquidity rebalancing not completed")
            if verbose:
                #print("Warning, liquidity rebalancing not completed")
                warnings.append("Liquidity rebalancing not completed")
                #return "Liquidity rebalancing not completed"
        return warnings, actions

    def rebalance(self, method,time_period,verbose=False):
        """Rebalances the portfolio according to the specified method.
        Args:
            method (str): The method of rebalancing, can be 'Priority', 'Pro-Rata', or 'No Rebalance'.
            time_period (int): The current period in the portfolio life, used to determine liquidity constraints.
            verbose (bool): If True, print warnings and actions taken during rebalancing.
        """
        allowed_methods = ['Priority','Pro-Rata','No Rebalance']
        if method not in allowed_methods:
            self.logger.error("Error, method not allowed")
            return
        
        if method == 'No Rebalance':
            self.logger.info("No rebalancing")
            return

        elif method == 'Priority' or method == 'Pro-Rata':
            #track warnings and actions
            warnings = []
            actions = []
            self.logger.info("%s rebalancing" % method)
            #Rebalance all assets
            #First calculate the portfolio weights
            weights = self.calculate_liquid_weights()
            if weights == 0:
                return
            #Then calculate the difference between the target weights and the current weights
            difference = {}
            for asset in self.positions:
                #We can only rebalance assets, not drawdown funds
                if asset.type == 'Asset':
                    if self.target_weights[asset] - weights[asset.name] > self.asset_deviation[asset]:
                        difference[asset.name] = weights[asset.name] - self.target_weights[asset]
                    else:
                        if weights[asset.name] - self.target_weights[asset] >= 0:
                            difference[asset.name] = weights[asset.name] - self.target_weights[asset]
                        else:
                            difference[asset.name] = weights[asset.name] - self.target_weights[asset] + self.asset_deviation[asset]
            
            #maximum pullable from each asset
            max_pull = {}
            for asset in self.positions:
                if asset.type == 'Asset':
                    #We only allow the asset to be sold if it's liquidity flag in the period is true and according to a certain prorate percent
                    max_pull[asset.name] = max(difference[asset.name],0)*self.calculate_liquid_value()*asset.liq_ability(time_period)*asset.prorate
            
            #need investment into each asset
            need_rebalancing = {}
            for asset in self.positions:
                if asset.type == 'Asset':
                    need_rebalancing[asset.name] = -1*min(difference[asset.name],0)*self.calculate_liquid_value()*asset.sub_ability(time_period)
            totalneed = sum(need_rebalancing.values())
            
            if totalneed > sum(max_pull.values())+.00001:
                self.logger.warning("Warning, full rebalancing not possible")
            
            #Then calculate the total amount that can to be rebalanced
            total_rebal = min(totalneed,sum(max_pull.values()))

            if totalneed != 0:
                for i in need_rebalancing:
                    #scaling factor
                    need_rebalancing[i] = need_rebalancing[i]/totalneed*total_rebal

            prorata = {}
            prorata_d = 0
            for asset in self.positions:
                if asset.type == 'Asset':
                    if need_rebalancing[asset.name] > 0:
                        prorata_d += self.target_weights[asset]
            for asset in self.positions:
                if asset.type == 'Asset':
                    if need_rebalancing[asset.name] > 0:
                        prorata[asset.name] = self.target_weights[asset]/prorata_d
                else:
                    prorata[asset.name] = 0

            #Then sort the assets by the liquidity rank
            sorted_assets_liq = sorted(self.liquidity_rank, key=lambda x: self.liquidity_rank[x])
            #Then loop through the assets and rebalance them
            if total_rebal > 0:
                for asset in sorted_assets_liq:
                    if asset.type == 'Asset':
                        if max_pull[asset.name] > 0:
                            #sell the asset
                            to_pull = min(max_pull[asset.name], total_rebal)*asset.liq_ability(time_period)
                            self.logger.info("Selling %s %s" % (asset.name,np.round(to_pull,4)))
                            #print("Selling %s %s" % (asset.name,to_pull))
                            actions.append("Selling %s %s" % (asset.name,np.round(to_pull,4)))
                            total_rebal -= to_pull
                            asset.quantity -= to_pull/asset.price  
                    
                        if method == 'Priority':
                            if need_rebalancing[asset.name] > 0:
                                #buy the asset
                                to_push = need_rebalancing[asset.name]*asset.sub_ability(time_period)
                                self.logger.info("Buying %s %s" % (asset.name, np.round(to_push,4)))
                                actions.append("Buying %s %s" % (asset.name,np.round(to_push,4)))
                                #print("Buying %s %s" % (asset.name, to_push))
                                asset.quantity += to_push/asset.price

                        if method == 'Pro-Rata':
                            if prorata[asset.name] > 0:
                                #buy the asset
                                to_push = prorata[asset.name] * total_rebal*asset.sub_ability(time_period)
                                self.logger.info("Buying %s %s" % (asset.name,np.round(to_push,4)))
                                actions.append("Buying %s %s" % (asset.name,np.round(to_push,4)))
                                asset.quantity += to_push/asset.price
            if total_rebal > 0.00001:
                self.logger.warning("Warning, rebalancing not completed")
                if verbose:
                    #print("Warning, rebalancing not completed")
                    warnings.append("Rebalancing not completed")
            return warnings, actions
    
    def time_cycle(self, time_periods, rebalance_method="No Rebalance", rebal_periodicity=999999999, redemptions={}, subscriptions={},
                   red_max=False, max_pct=0.0, sub_max=False, smax_pct=0.0, verbose=False, historical=False, startdate=None, calc_irr="None",
                   periodicity="M", earlybreak=False, growfirst=False):
        #This function will simulate the portfolio over a given number of time periods (currently set up as months)
        #Due to the rebalancing logic, we must adjust the time periods to be daily, monthly, quarterly, etc
        #For example, if periodicity is set to D, we can only rebalance every 30 periods. If periodicity is M, we can rebalance every period. If periodicity is Q, Y, we rebalance every period 
        #It will return the asset values at the end of each time period
        #Redemptions/Subscriptions should be a dictionary that contains the time periods and the amount to be redeemed
        #The default is no redemptions or subscriptions
        
        """
        Simulate the portfolio over time periods
        
        Args:
            time_periods (int): Number of time periods to simulate.
            rebalance_method (str): Method of rebalancing ('Priority', 'Pro-Rata', 'No Rebalance').
            rebal_periodicity (int): Rebalancing frequency. 1 = rebalance every period, 2 = every other period, etc.
            redemptions (dict): Dictionary of redemptions by time period.
            subscriptions (dict): Dictionary of subscriptions by time period.
            red_max (bool): Limit redemptions allowed to a certain % of prior period NAV
            max_pct (float): Maximum redemption percentage of NAV.
            sub_max (bool): Whether to cap subscriptions as a percentage of NAV.
            smax_pct (float): Maximum subscription percentage of NAV.
            verbose (bool): Whether to log detailed information.
            historical (bool): Whether to use historical data.
            startdate (str): Start date for historical data.
            calc_irr (str): Method of calculating IRR ('LMK', 'Simple', 'None').
            periodicity (str): Time period frequency ('D', 'M', 'Q', 'Y').
            earlybreak (bool): Whether to stop simulation if redemption cannot be met.
            growfirst (bool): Is there growth in the first period?
        
        Returns:
            asset_values: asset values after rebalancing each period
            av_prerebal: asset values pre-rebalancing each period
            warnings: warnings per period
            subs: subscriptions per period
            reds: redemptions per period
            rebal: actions taken to rebalance each period
            irr: IRR
            h_returns: historical returns

        """
        #establish most liquid asset
        most_liquid = self.get_most_liquid_asset()

        #establish periodicity of time cycle and rebalancing. we need to adjust everything to be the smallest period
        periods = {"D":30,"M":1,"Q":1/3,"Y":1/12}
        
        #All assets must have the same period
        for asset in self.positions:
            if asset.periodicity != periodicity:
                print("Error, asset %s does not have the same period as the portfolio" % asset.name)
                #self.logger.error("Error, asset %s does not have the same period as the portfolio" % asset.name)
                return

        #First check that the redemptions and subscriptions are in the correct format
        if not isinstance(redemptions, dict):
            print("Error, redemptions not in correct format")
            #self.logger.error("Error, redemptions not in correct format")
            return
        if not isinstance(subscriptions, dict):
            print("Error, subscriptions not in correct format")
            #self.logger.error("Error, subscriptions not in correct format")
            return

        #Track Asset Values pre- and post- rebalancing
        commonkeys = ['Total','Subscription','Redemption','Date']
        asset_values = {key:list() for key in commonkeys}
        av_prerebal = {key:list() for key in commonkeys}

        for asset in self.positions:
            if asset.type == 'Asset':
                asset_values[asset.name] = []
                av_prerebal[asset.name] = []
            elif asset.type == 'Fund':
                asset_values[asset.name+' NAV'] = []
                asset_values[asset.name+' Com. Not Called'] = []
                av_prerebal[asset.name+' NAV'] = []
                av_prerebal[asset.name+' Com. Not Called'] = []

        privatecashflows = {'Date':[],'Contributions':[], 'Distributions':[], 'NAV':[], 'Com. Not Called':[]}
        #track all warnings, subscriptions, redemptions, rebalancing actions
        warnings = {}
        subs = {}
        reds = {}
        rebal = {}
        irr = {'Subscription':[],'Redemption':[],'Total NAV':[],'IRR':[]}

        #if we are using historical data, then create the historical data track record
        #if we are not using historical data, generate the dates separately based on periodicity
        if historical:
            if startdate == None:
                self.logger.error("Error, need a start date for historical data, using simulated data instead")
                pass
            else:
                h_returns = self.historicalrates(startdate=startdate,time_periods=time_periods,periodicity=periodicity)
                asset_values['Date'] = []
                av_prerebal['Date'] = []
                irr['Date'] = []
                privatecashflows['Date'] = []                
        else:
            if periodicity == "D":
                asset_values['Date'] = get_weekdays(startdate, time_periods)
                av_prerebal['Date'] = get_weekdays(startdate, time_periods)
                irr['Date'] = get_weekdays(startdate, time_periods)
                privatecashflows['Date'] = get_weekdays(startdate,time_periods)
            elif periodicity == "M":
                asset_values['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='MS')
                av_prerebal['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='MS')
                irr['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='MS')
                privatecashflows['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='MS')
            elif periodicity == "Q":
                asset_values['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='QS')
                av_prerebal['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='QS')
                irr['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='QS')
                privatecashflows['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='QS')
            elif periodicity == "Y":
                asset_values['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='YS')
                av_prerebal['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='YS')
                irr['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='YS')
                privatecashflows['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='YS')
            else:
                print(time_periods)
                asset_values['Date'] = range(time_periods)
                av_prerebal['Date'] = range(time_periods)
                irr['Date'] = range(time_periods)
                privatecashflows['Date'] = range(time_periods)

        for i in range(time_periods):
            if verbose:
                self.logger.info("Time Period %s" % i)
            warnings[i] = []
            subs[i] = []
            reds[i] = []
            rebal[i] = []

            if historical:
                asset_values['Date'].append(h_returns['Date'][i])
                av_prerebal['Date'].append(h_returns['Date'][i])
                irr['Date'].append(h_returns['Date'][i])
                privatecashflows['Date'].append(h_returns['Date'][i])
            
            #Apply subscriptions
            if i == 0:
                #First time period, count the initial investment as a subscription
                asset_values['Subscription'].append(self.calculate_portfolio_value())
                av_prerebal['Subscription'].append(self.calculate_portfolio_value())
            else:
                if i in subscriptions:
                    if sub_max:
                        #Calculate appropriate subscription based off prior period's NAV
                        subscriptions[i] = min(subscriptions[i],asset_values['Total'][i-1]*smax_pct)
                    
                    temp = self.add_subscription(i,subscriptions[i])
                    subs[i].append(temp)
                    #Record the subscription amount
                    asset_values['Subscription'].append(subscriptions[i])
                    av_prerebal['Subscription'].append(subscriptions[i])
                else:
                    asset_values['Subscription'].append(0)
                    av_prerebal['Subscription'].append(0)

            #Simulate the income and growth of each non Fund Asset. If growfirst == False, no growth in first period
            if growfirst == False and i == 0:
                pass
            else: 
                self.age += 1
                for asset in self.positions:
                    if (asset.type == 'Asset' or asset.type == 'Option'):
                        #Asset generates income, with the income being reinvested or going to most liquid asset
                        #Asset grows, with the price being increased by a random distribution
                        if historical:
                            income = h_returns[asset.name + ' Income'][i] * asset.quantity * asset.price
                            if asset.type == 'Asset':
                                pr = h_returns[asset.name + ' Growth'][i]
                                asset.growth(p_return=pr)
                            elif asset.type == 'Option':
                                asset.underlyingprice = asset.underlyingprice*(1+h_returns[asset.name + ' Growth'][i])
                                asset.growth()
                        else:
                            income = asset.income()
                            if asset.type == 'Option':
                                asset.underlying_growth()
                            asset.growth()
                            
                        #Add income to quantity, based on reinvestment rate. Amounts not reinvested will be added to most liquid asset
                        if asset.type == 'Asset':
                            asset.quantity += income*asset.reinvestment_rate/asset.price
                            most_liquid.quantity += income * (1-asset.reinvestment_rate)/most_liquid.price #most liquid sleeve
            
            earlybreakflag = 0
            #Calculate the Intra-fund cashflows
            #Contributions from Assets into Funds and then distributions from Funds into Assets
            callablecapital = sum(asset.calculate_value() * asset.liq_ability(i) for asset in self.positions if asset.type == 'Asset')
            period_called = 0
            period_distributed = 0
            nav = 0
            comnotcalled = 0
            for assets in self.positions:
                if assets.type == 'Fund':
                    #check to make sure the dates are aligned
                    if assets.cashflows.index.min() <= asset_values['Date'][i]:
                        call_needed = assets.call_needed()
                        if call_needed > 0:
                            period_called += min(call_needed, callablecapital)
                            if call_needed > callablecapital:
                                call_needed = callablecapital
                                warnings[i].append("Unable to Call Capital Fully")
                                #Break Early if capital not fully called (if earlybreak is activated)
                                earlybreakflag = 1
                            callablecapital -= call_needed
                        
                        #negative distributions not allowed
                        period_distributed += max(0,assets.income())
                        nav += assets.calculate_nav()
                        comnotcalled += assets.calculate_cnc()
            
            privatecashflows['Contributions'].append(period_called)
            privatecashflows['Distributions'].append(period_distributed)
            privatecashflows['NAV'].append(nav)
            privatecashflows['Com. Not Called'].append(comnotcalled)
            
            #remove called capital from the liquid sleeves, add distributed capital to the liquid sleeves
            #Record the asset values
            self.add_subscription(i,period_distributed)
            self.add_redemption(i,period_called)
            for asset in self.positions:
                if asset.type == 'Fund':
                    if asset.cashflows.index.min() <= asset_values['Date'][i]:
                        asset_values[asset.name+' NAV'].append(asset.calculate_nav())
                        asset_values[asset.name+' Com. Not Called'].append(asset.calculate_cnc())
                    else:
                        asset_values[asset.name+' NAV'].append(0)
                        asset_values[asset.name+' Com. Not Called'].append(0)
            
            #Increase the age of the funds
            for assets in self.positions:
                if assets.type == 'Fund':
                    if assets.cashflows.index.min() <= asset_values['Date'][i]:
                        assets.age += 1
            
            #Apply redemptions
            if i in redemptions:
                if red_max:
                    #calculate max redemption allowed based off prior period's NAV
                    redemptions[i] = min(redemptions[i],asset_values['Total'][i-1]*max_pct)
                asset_values['Redemption'].append(redemptions[i])
                av_prerebal['Redemption'].append(redemptions[i])
                temp = self.add_redemption(i,redemptions[i])
                warnings[i].extend(temp[0])
                reds[i].extend(temp[1])
                #break early if redemption not met
                if "Redemption not met" in temp[0]:
                    earlybreakflag = 1
            else:
                asset_values['Redemption'].append(0)
                av_prerebal['Redemption'].append(0)

            #Record the asset values before rebalancing
            for asset in self.positions:
                if asset.type != 'Fund':
                    av_prerebal[asset.name].append(asset.calculate_value())
                if asset.type == 'Fund':
                    if asset.cashflows.index.min() <= av_prerebal['Date'][i]:
                        av_prerebal[asset.name+' NAV'].append(asset.calculate_nav())
                        av_prerebal[asset.name+' Com. Not Called'].append(asset.calculate_cnc())
                    else:
                        av_prerebal[asset.name+' NAV'].append(0)
                        av_prerebal[asset.name+' Com. Not Called'].append(0)
            av_prerebal['Total'].append(self.calculate_portfolio_value())

            #Rebalance the portfolio
            #Rebalancing takes place on month end, which means we cannot rebalance every period if periodicity is daily unless rebal period is daily
            #in order to take into account liquidity, we convert everything to Asset periodicity
            if i % rebal_periodicity == 0:
                if rebalance_method != 'No Rebalance':
                    temp = self.rebalance_liquidity(time_period=i,verbose=verbose)
                    if type(temp) != None:
                        warnings[i].append(temp[0])
                        rebal[i].append(temp[1])
                if rebalance_method != 'No Rebalance':
                    temp = self.rebalance(rebalance_method,time_period=i,verbose=verbose)     
                    if type(temp) != None:
                        warnings[i].append(temp[0])
                        rebal[i].append(temp[1])
            
            #age the options based on the periodicity
            for asset in self.positions:
                if asset.type == 'Option':
                    asset.dte -= (1/periods[periodicity])*30
            
            #Record the asset values after rebalancing
            for asset in self.positions:
                asset_values[asset.name].append(asset.calculate_value())    
            asset_values['Total'].append(self.calculate_portfolio_value())
                
            #Calculate the IRR
            #first calculate the NAV
            #Calc IRR: None, Simple, LMK
            irr['Total NAV'] = asset_values['Total']
            irr['Subscription'] = asset_values['Subscription']
            irr['Redemption'] = asset_values['Redemption']
            irr['Cashflows'] = [x - y for x, y in zip(irr['Redemption'], irr['Subscription'])]

            if earlybreak:
                if earlybreakflag == 1:
                    break
            
        if calc_irr != "None":    
            if (calc_irr == "LMK"):
                for j in range(len(irr['Cashflows'])):
                    tempirr = BG_IRR_array(pd.Series(irr['Cashflows'][:j+1]),pd.Series(irr['Date'][:j+1]),nav = irr['Total NAV'][j])
                    irr['IRR'].append(tempirr)
            elif calc_irr == "Simple":
                for j in range(len(irr['Cashflows'])):
                    tempcfs = irr['Cashflows'][:j+1]
                    tempcfs[-1] += irr['Total NAV'][j]
                    tempirr = npf.irr(tempcfs)
                    tempirr = (1+tempirr)**12-1
                    irr['IRR'].append(tempirr)

        if historical:
            return asset_values, av_prerebal, warnings, subs, reds, rebal, irr, h_returns, privatecashflows
        else:
            return asset_values, av_prerebal, warnings, subs, reds, rebal, irr, privatecashflows

    def period_cycle(self, rebalance_method, redemptions=0.0, subscriptions=0.0,red_max=True,max_pct=0.05,verbose=False,historical=False,h_returns=None):
        #A version of time cycle that is more flexible
        #need to run this function for each time period
        
        #establish most liquid asset
        most_liquid = self.get_most_liquid_asset()

        #add subscriptions
        self.add_subscription(time_period=self.age, subscription=subscriptions)

        #Simulate the income and growth of each asset
        for asset in self.positions:
            #Asset generates income, with the income being reinvested or going to most liquid asset
            if historical:
                #h_returns is a dataframe of historical MONTHLY growth and income returns for each asset, could have only one row
                income = h_returns[asset.name + ' Income'] * asset.quantity * asset.price
            else:
                income = asset.income()
            #Asset grows, with the price being increased by a random distribution
            if historical:
                pr = h_returns[asset.name + ' Growth']
                asset.growth(p_return=pr)
            else:
                asset.growth()
            #Add income to quantity
            asset.quantity += income*asset.reinvestment_rate/asset.price
            most_liquid.quantity += income * (1-asset.reinvestment_rate)/most_liquid.price #cash
        
        #Apply redemptions
        if redemptions > 0:
            if red_max:
                redemptions = max(self.calculate_portfolio_value()*max_pct,redemptions)
            self.add_redemption(0,redemptions)
        
        #Rebalance the portfolio, starting with liquidity
        self.rebalance_liquidity(time_period=self.age,verbose=verbose)

        #Rebalance the portfolio
        if rebalance_method != 'No Rebalance':
            self.rebalance(rebalance_method,time_period=self.age,verbose=verbose)
        
    def timecycle_drawdown(self, time_periods, rebalance_method="No Rebalance",rebal_periodicity=999999999,redemptions={}, red_base="Fixed", subscriptions={}, red_max=True, max_pct=0.05, 
                            sub_max=True, smax_pct=.2, verbose=False, historical=False, startdate=None, periodicity="Q",earlybreak=True,growfirst=False):
        #need to calculate the drawdowns for each asset and the portfolio
        #the drawdown portfolio has Funds, does not rebalance, and is on quarterly periodicity
        #All assets must have the same period

        """
        Simulate the portfolio over time periods, but with the inclusion of close ended structures.
        
        Args:
            time_periods (int): Number of time periods to simulate.
            redemptions (dict): Dictionary of redemptions by time period.
            red_base (str): Redemption base; either "Fixed", "NAV", "Dist"
            subscriptions (dict): Dictionary of subscriptions by time period.
            red_max (bool): This red_max is technically not a maximum, but rather functions as a set value, if red_max is active then we will have fixed % redemptions
            max_pct (float): Maximum redemption percentage of NAV.
            sub_max (bool): Whether to cap subscriptions as a percentage of NAV.
            smax_pct (float): Maximum subscription percentage of NAV.
            verbose (bool): Whether to log detailed information.
            periodicity (str): Time period frequency ('D', 'M', 'Q', 'Y').
            earlybreak (bool): Whether to stop simulation if redemption cannot be met.
            growfirst (bool): Is there growth in the first period?
        
        Returns:
            dict: Asset values over time. Intra-fund cashflows, private NAV, committed not called.
            dict: Warnings generated during simulation.
            dict: Subscription actions.
            dict: Redemption actions.
            dict: Line of credit usage (if applicable).
        """
        #Establish most liquid asset
        most_liquid = self.get_most_liquid_asset()

        #if we are using historical data, then create the historical data track record
        if historical:
            if startdate == None:
                self.logger.error("Error, need a start date for historical data, using simulated data instead")
                pass
            else:
                h_returns = self.historicalrates(startdate=startdate,time_periods=time_periods,periodicity=periodicity)

        for asset in self.positions:
            if asset.periodicity != periodicity:
                print("Error, asset %s does not have the same period as the portfolio" % asset.name)
                #self.logger.error("Error, asset %s does not have the same period as the portfolio" % asset.name)
                return
        #First check that the redemptions and subscriptions are in the correct format
        if not isinstance(redemptions, dict):
            print("Error, redemptions not in correct format")
            #self.logger.error("Error, redemptions not in correct format")
            return
        if not isinstance(subscriptions, dict):
            print("Error, subscriptions not in correct format")
            #self.logger.error("Error, subscriptions not in correct format")
            return 
        
        #Initialize data structures
        #Track asset values post redemptions
        asset_values = {'Total': [], 'Subscription': [], 'Redemption': [], 'Date': [],
                        'Total Period Contributions': [], 'Total Period Distributions': [], 'Total Private NAV': [], 'Total Com. Not Called': [],
                        'Management Fees Total': [], 'Performance Fees Total': []}
        for asset in self.positions:
            if asset.type == 'Asset':
                asset_values[asset.name] = []
            elif asset.type == 'Fund':
                asset_values[asset.name+' NAV'] = []
                asset_values[asset.name+' Com. Not Called'] = []
                asset_values[asset.name+' Management Fees'] = []
                asset_values[asset.name+' Performance Fees'] = []

        #track all warnings, subscriptions, redemptions, rebalancing actions
        warnings = {}
        subs = {}
        reds = {}

        privatecashflows = {
            'Date': [],
            'Contributions': [],
            'Distributions': [],
            'NAV': [],
            'Com. Not Called': [],
            'Management Fees': [],
            'Performance Fees': []}
        
        #track line of credit usage
        loc_flag = False
        if self.line_of_credit != None:
            loc_flag = True
        line_of_credit_usage = {'Balance': [], 'Available': [], 'Accrued Interest': [], 'Drawn': [],'Repayment':[],'Interest Payment':[]}

        if periodicity == "Q":
            asset_values['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='Q')
            privatecashflows['Date'] = pd.date_range(start=startdate, periods=time_periods, freq='Q')

        earlybreakflag = 0
        for i in range(time_periods):
            warnings[i] = []
            subs[i] = []
            reds[i] = []
            assets = [p for p in self.positions if p.type == "Asset"]
            funds = [p for p in self.positions if p.type == "Fund"]

            #Apply line of credit interest if applicable
            if loc_flag:
                self.line_of_credit.accrue_interest()
                interest = self.line_of_credit.accrued_interest
                line_of_credit_usage['Accrued Interest'].append(np.round(interest,2))
                line_of_credit_usage['Balance'].append(np.round(self.line_of_credit.balance,2))
                line_of_credit_usage['Available'].append(np.round(self.line_of_credit.availability(),2))

            #First apply subscriptions
            if i == 0:
                asset_values['Subscription'].append(0)
            else: 
                if i in subscriptions:
                    subscription_amount = subscriptions[i]
                    if sub_max:
                        #calculate appropriate subscription based off prior period's NAV
                        subscription_amount = min(subscription_amount, asset_values['Total'][i-1]*smax_pct)
                    subs[i].append(self.add_subscription(i, subscription_amount))
                    asset_values['Subscription'].append(subscription_amount)                    
                else:
                    asset_values['Subscription'].append(0)
                

            #Simulate the income and growth of each asset
            if growfirst == False and i == 0:
                pass
            else: #no income or growth on the first period
                for asset in assets:
                    #Asset generates income, with the income being reinvested or going to most liquid 
                    if historical:
                        income = h_returns[asset.name + ' Income'][i] * asset.quantity * asset.price
                    else:
                        income = asset.income()
                    #Asset grows, with the price being increased by a random distribution
                    if historical:
                        pr = h_returns[asset.name + ' Growth'][i]
                        asset.growth(p_return=pr)
                    else:
                        asset.growth()
                    #Add income to quantity
                    asset.quantity += income*asset.reinvestment_rate/asset.price
                    most_liquid.quantity += income * (1-asset.reinvestment_rate)/most_liquid.price #cash
                
            #Calculate the Intra-fund cashflows
            #starting with contributions
            callablecapital = sum(asset.calculate_value() * asset.liq_ability(i) for asset in assets)
            if loc_flag:
                callablecapital += self.line_of_credit.availability()
            period_called = 0
            for fund in funds:
                if fund.cashflows.index.min() <= asset_values['Date'][i]:
                    call_needed = fund.call_needed()
                    if call_needed > 0:
                        period_called += min(call_needed, callablecapital)
                        if call_needed > callablecapital:
                            call_needed = callablecapital
                            warnings[i].append("Unable to Call Capital Fully")
                            #break early if capital not fully called
                            earlybreakflag = 1
                        callablecapital -= call_needed

            #next distributions
            period_distributed0 = 0
            nav = 0
            comnotcalled = 0
            for fund in funds:
                if fund.cashflows.index.min() <= asset_values['Date'][i]:
                    #negative distributions not allowed
                    period_distributed0 += max(0,fund.income())
                    nav += fund.calculate_nav()
                    comnotcalled += fund.calculate_cnc()
            #period_distributed = sum(asset.income() for asset in self.positions if asset.type == 'Fund')
            #nav = sum(asset.calculate_nav() for asset in self.positions if asset.type == 'Fund')

            #Calculate aggregate management fees and performance fees for the period
            total_mgmt_fees = sum(fund.calculate_mgmt_fee() for fund in funds)
            total_perf_fees = sum(fund.calculate_perf_fee() for fund in funds)

            asset_values['Total Period Contributions'].append(period_called)
            asset_values['Total Period Distributions'].append(period_distributed0)
            asset_values['Total Private NAV'].append(nav)
            asset_values['Total Com. Not Called'].append(comnotcalled)
            asset_values['Management Fees Total'].append(total_mgmt_fees)
            asset_values['Performance Fees Total'].append(total_perf_fees)

            privatecashflows['Contributions'].append(period_called)
            privatecashflows['Distributions'].append(period_distributed0)
            privatecashflows['NAV'].append(nav)
            privatecashflows['Com. Not Called'].append(comnotcalled)
            privatecashflows['Management Fees'].append(total_mgmt_fees)
            privatecashflows['Performance Fees'].append(total_perf_fees)
            
            #remove called capital from the liquid sleeves and line of credit (if applicable), add distributed capital to the liquid sleeves / pay down loc
            #Record the asset values
            
            #Rebalance Liquidity if necessary
            if np.round(self.target_weights[most_liquid],2) > 0:
                self.rebalance_liquidity(time_period=i,verbose=verbose)
            else:
                pass

            #then use the liquid sleeve to feed capital calls or absorb distributions based on deviation allowed
            period_distributed = period_distributed0
            #First pay down interest on line of credit with distributions, and then repay the line
            if loc_flag:
                int_pmt = min(period_distributed, self.line_of_credit.accrued_interest)
                self.line_of_credit.pay_interest(int_pmt)
                period_distributed -= int_pmt
                line_of_credit_usage['Interest Payment'].append(np.round(int_pmt,2))

                paydown_amount = min(period_distributed, self.line_of_credit.balance)
                self.line_of_credit.repay(paydown_amount)
                period_distributed -= paydown_amount
                #Record repayment
                line_of_credit_usage['Repayment'].append(np.round(paydown_amount,2))

            self.add_subscription(i,period_distributed)
            self.add_redemption(i,period_called)
            
            #Apply redemptions
            redemption_amount = 0.0
            if i in redemptions:
                rate_or_amount = redemptions[i]

                if red_base == "Fixed":
                    redemption_amount = rate_or_amount

                elif red_base == "Dist":
                    redemption_amount = period_distributed0 * rate_or_amount

                elif red_base == "NAV":
                    nav_base = (
                        asset_values["Total"][i - 1]
                        if i > 0 and len(asset_values["Total"]) > 0
                        else self.calculate_portfolio_value_active_funds_only(asset_values["Date"][i])
                    )
                    redemption_amount = nav_base * rate_or_amount

                if red_max and red_base != "Fixed":
                    nav_base = (
                        asset_values["Total"][i - 1]
                        if i > 0 and len(asset_values["Total"]) > 0
                        else self.calculate_portfolio_value_active_funds_only(asset_values["Date"][i])
                    )
                    redemption_amount = min(redemption_amount, nav_base * max_pct)

            asset_values["Redemption"].append(redemption_amount)

            if redemption_amount > 0:
                redemption_warnings, redemption_actions = self.add_redemption(i, redemption_amount)
                warnings[i].extend(redemption_warnings)
                reds[i].extend(redemption_actions)

                if "Redemption not met" in redemption_warnings:
                    earlybreakflag = 1
            
            #Record the asset values
            asset_values['Total'].append(
                self.calculate_portfolio_value_active_funds_only(asset_values['Date'][i])
            )

            for asset in assets:
                asset_values[asset.name].append(asset.calculate_value())
            for fund in funds:
                if fund.cashflows.index.min() <= asset_values['Date'][i]:
                    asset_values[fund.name+' NAV'].append(fund.calculate_nav())
                    asset_values[fund.name+' Com. Not Called'].append(fund.calculate_cnc())
                    asset_values[fund.name+' Management Fees'].append(fund.calculate_mgmt_fee())
                    asset_values[fund.name+' Performance Fees'].append(fund.calculate_perf_fee())
                else:
                    asset_values[fund.name+' NAV'].append(0)
                    asset_values[fund.name+' Com. Not Called'].append(0)
                    asset_values[fund.name+' Management Fees'].append(0)
                    asset_values[fund.name+' Performance Fees'].append(0)

            #Increase the age of the funds
            for fund in funds:
                if fund.cashflows.index.min() <= asset_values['Date'][i]:
                    fund.age += 1

            if earlybreak:
                if earlybreakflag == 1:
                    break
        return asset_values, warnings, subs, reds, privatecashflows, earlybreakflag, line_of_credit_usage

    def historicalrates(self,startdate,time_periods,periodicity):
        #Ensure each asset has appropriate length of historical data, if not, then default to the income and return given. 
        # Returns a dataframe of historical growth and income for each asset
        startdate = pd.to_datetime(startdate)  # Convert startdate to datetime format
        #Periodicity must be D, M, Q, Y
        dates = pd.date_range(start=startdate, periods=time_periods, freq=periodicity)
            
        df = pd.DataFrame({'Date': dates})

        for i in self.positions:
            if i.type != 'Fund':
                if len(i.returnstream) >0:
                    i.return_clean()
                if (len(i.returnstream) < time_periods) or (min(i.returnstream['Date']) > startdate):
                    self.logger.error("Error, asset %s does not have enough historical data" % i.name)
                    df[i.name + ' Growth'] = i.a_return
                    df[i.name + ' Income'] = i.a_income
                else:
                    temp = i.returnstream[i.returnstream['Date']>=startdate].reset_index(drop=True).copy()
                    temp = temp[:time_periods]
                    df[i.name + ' Growth'] = temp['Growth']
                    df[i.name + ' Income'] = temp['Income']

        return df

################################# The Portfolio Object End ##############################################################################################

################################# Asset, Fund, Option Objects ###########################################################################################

class Position:
    #Unused for now
    def __init__(self, asset, quantity):
        self.asset = asset
        self.quantity = quantity

    def calculate_value(self):
        return self.asset.price * self.quantity
    
    def get_price(self):
        return self.asset.price

class Asset:
    def __init__(self, name, price0, quantity,a_return,volatility,a_income,income_volatility,returnstream,asset_class= "Generic",reinvestment_rate=1,liquidity=1,sub_period=1,prorate=1,periodicity ="M"):
        """Initialize an Asset object.
        """
        self.type = 'Asset'
        self.name = name
        self.price = price0
        self.quantity = quantity
        self.a_return =  a_return #periodicity must match, default is monthly
        self.volatility = volatility #periodicity must match, default is monthly
        self.a_income = a_income #periodicity must match, default is monthly
        self.income_volatility = income_volatility #periodicity must match, default is monthly
        self.returnstream = returnstream #is a dataframe of historical growth and income returns
                                        #['Date','Growth','Income'], can be NA, periodicity must match the time cycle
        self.asset_class = asset_class #used to aggregate assets, can be used to group assets by strategy
        self.reinvestment_rate = reinvestment_rate #how much of the income is reinvested, default is 100%
        self.liquidity = liquidity #how many periods it takes to sell the asset, default is 1, can be liquidated any periods
        self.sub_period = sub_period #how many periods it takes to subscribe into the asset, default is 1, can be subscribed any periods
        self.liq_flag = 0
        self.sub_flag = 0
        self.prorate = prorate #how much of the asset's redemption can actually be realized. Prorate is used in the rebalancing and redemption functions
        self.periodicity = periodicity #are cashflows monthly, quarterly, etc ("D","M","Q","Y" for monthly, quarterly, yearly)
        
        self.price0 = price0
        self.quantity0 = quantity
    def income(self):
        # Simulate a random income from the given mean and standard deviation, returns the income
        mean = self.a_income
        std_dev = self.income_volatility
        random_value = np.random.normal(mean, std_dev)
        random_value = max(random_value,-.9999)
        return random_value * self.quantity * self.price
    
    def growth(self,p_return=None):
        # Simulate a random price return from the given mean and standard deviation, then increases the asset price
        if p_return == None:
            mean = self.a_return
            std_dev = self.volatility
            random_value = np.random.normal(mean, std_dev)
            p_return = max(random_value,-.9999)
      
        #set the price to the new price
        self.price = self.price * (1 + p_return)
        return
    
    def calculate_value(self):
        return self.price * self.quantity

    def calculate_exposure(self):
        return self.price * self.quantity
    
    def liq_ability(self,time_period):
        #Controls the ability to liquidate from this asset
        #if time_period<self.liquidity:
        #    self.liq_flag = 0
        #else:
        if time_period % self.liquidity == 0:
            self.liq_flag = 1
        else:
            self.liq_flag = 0
        return self.liq_flag
    
    def sub_ability(self,time_period):
        #Controls the ability to subscribe into this asset
        #if time_period<self.sub_period:
        #    self.sub_flag = 0
        #else:
        if time_period % self.sub_period == 0:
            self.sub_flag = 1
        else:
            self.sub_flag = 0
        return self.sub_flag
    
    def return_clean(self):
        self.returnstream['Date'] = pd.to_datetime(self.returnstream['Date'])
        return

    def period_converter(self,periodicity):
        #if we need to convert an asset periodicity to a higher periodicity
        # for example, if we have a monthly asset and we need to convert it to daily
        period_dict = {"D":1,"M":20,"Q":60,"Y":240}         
        adjustment = period_dict[periodicity]/period_dict[self.periodicity]
        return
    
    def reset_asset(self):
        self.price = self.price0
        self.quantity = self.quantity0

class Fund:
    def __init__(self, name, cashflows, asset_class= "Generic", reinvestment_rate=0,liquidity=999, prorate=1,age=0,periodicity="Q"):
        """
        Initialize a Fund object.

        Args:
            name (str): Name of the fund.
            asset_class (str): Investment strategy of the fund (used to aggregate funds).
            cashflows (pd.DataFrame): DataFrame containing fund cashflows with columns:
                - 'Age': Period age.
                - 'quarter_distribution': Distributions for the period.
                - 'quarter_contribution': Contributions for the period.
                - 'nav_eoq': Net Asset Value at the end of the period.
                - 'unfunded_eoq': Unfunded commitments at the end of the period.
                - index should be date value
            reinvestment_rate (float): Percentage of income reinvested (default: 0).
            liquidity (int): Liquidity period (default: length of cashflows).
            prorate (float): Proration factor for liquidation (default: 1).
            age (int): Initial age of the fund (default: 0).
            periodicity (str): Time period frequency ('D', 'M', 'Q', 'Y') (default: 'Q').
        """
        self.type = 'Fund'
        self.name = name
        self.asset_class = asset_class #used to aggregate funds, can be used to group funds by strategy
        """Cashflows will need the following columns to function:
        'gross_nav','net_distribution_post_fee', (confirm w Chih Wei if we should use summary_lp_total_net_distribution)
       'fee_paid_period', 'lp_contribution_post_fee', 'net_NAV'
        """
        self.cashflows = cashflows #takes in fund level cashflows: a dataframe with (fund_id	
                                    #committed	# nav_eoq	unfunded_eoq	quarter_contribution	quarter_distribution	Age)
        self.reinvestment_rate = reinvestment_rate #how much of the income is reinvested, default is 0%
        self.liquidity = liquidity if liquidity != 999 else len(self.cashflows)
        #how many in between periods it takes to sell the asset, default is 1, can be liquidated any periods
        #usually for close end funds, we should not allow liquidation prior to the end of the fund
        self.liq_flag = 0
        self.sub_flag = 0
        self.prorate = prorate
        self.age = 0
        self.periodicity = periodicity #are cashflows monthly, quarterly, etc ("D","M","Q","Y" for monthly, quarterly, yearly)
        self.max_age = max(self.cashflows['Age']) #maximum age of the fund
        self.quantity = 1 #default quantity of the fund, since we are using cashflows, we don't need to track quantity

    def get_current_period_data(self):
        """
        Get cashflow data for the current period (based on age).
        Returns None if age exceeds the maximum age or cashflows are empty.
        """
        if self.cashflows.empty or self.age > self.max_age:
            return None
        return self.cashflows[self.cashflows['Age'] == self.age].iloc[0]

    def income(self):
        period_data = self.get_current_period_data()
        return period_data['quarter_distribution'] if period_data is not None else 0 #.values[0]

    def call_needed(self):
        period_data = self.get_current_period_data()
        #in the base data, contributions are negative, but here we want to change them to positive
        return -period_data['quarter_contribution'] if period_data is not None else 0
    
    def growth(self):
        #Fund doesn't grow outside of increasing NAV
        return

    def calculate_value(self):
        period_data = self.get_current_period_data()
        if period_data is None:
            return 0
        return period_data['nav_eoq']

    def calculate_exposure(self):
        period_data = self.get_current_period_data()
        return max(0,period_data['nav_eoq']+max(0,period_data['committed_not_called'])) if period_data is not None else 0

    def calculate_nav(self):
        period_data = self.get_current_period_data()
        return period_data['nav_eoq'] if period_data is not None else 0

    def calculate_unfunded(self):
        period_data = self.get_current_period_data()
        return period_data['unfunded_eoq'] if period_data is not None else 0
    
    def calculate_cnc(self):
        period_data = self.get_current_period_data()
        return period_data['committed_not_called'] if period_data is not None else 0
    
    def calculate_mgmt_fee(self):
        period_data = self.get_current_period_data()
        return period_data['Management Fees Paid'] if period_data is not None else 0

    def calculate_perf_fee(self):
        period_data = self.get_current_period_data()
        return period_data['GP Distributions this Period'] if period_data is not None else 0

    def liq_ability(self,time_period):
        #Controls the ability to liquidate from this asset
        #We don't want to liquidate from fund
        if time_period < self.liquidity:
            self.liq_flag = 0
        else:
            self.liq_flag = 1 if time_period % self.liquidity == 0 else 0
        return self.liq_flag

    def sub_ability(self,time_period):
        #Controls the ability to subscribe into this asset
        return self.sub_flag
    
    def period_converter(self,periodicity):
        #if we need to convert an asset periodicity to a higher periodicity
        # for example, if we have a monthly asset and we need to convert it to daily        
        return

class Line_of_Credit:
    def __init__(self, name, balance, interest_rate, max_balance, liquidity=1, periodicity="M"):
        #Unused for now, but will be used to simulate credit lines in the future
        self.type = 'Line_of_Credit'
        self.name = name
        self.balance = balance
        self.interest_rate = interest_rate
        self.max_balance = max_balance
        self.liquidity = liquidity
        self.periodicity = periodicity
        self.period_dict = {"D":365,"M":12,"Q":4,"Y":1}
        self.accrued_interest = 0

    def availability(self):
        return max(0,self.max_balance - self.balance)
    
    def draw(self, amount):
        if amount > self.availability():
            self.balance += self.availability()
        else:
            self.balance += amount
    
    def repay(self, amount):
        if amount > self.balance:
            self.balance = 0
        else:
            self.balance -= amount
    
    def calculate_interest(self):
        return self.balance * self.interest_rate / self.period_dict[self.periodicity]
    
    def accrue_interest(self):
        interest = self.calculate_interest()
        self.accrued_interest += interest
    
    def pay_interest(self, payment):
        if payment > 0:
            interest = self.accrued_interest
            self.accrued_interest -= min(payment, interest)
    
class Option:
    def __init__(self, name,underlyingprice, quantity,und_return,und_vol,und_returnstream,dte,strike,volatility,optiontype,liquidity=1, prorate=1,periodicity="M"):
        self.type = 'Option'
        self.name = name
        self.underlyingprice = underlyingprice
        self.quantity = quantity
        self.und_return = und_return
        self.und_vol = und_vol
        self.returnstream = und_returnstream #a dataframe of historical growth returns
        self.strike = strike
        self.volatility = volatility
        self.optiontype = optiontype #call or put
        self.dte = dte #initial days to expiration
        self.liquidity = liquidity
        self.periodicity = periodicity
        self.price = self.american_option_pricer()
        self.prorate = prorate
        self.liq_flag = 0
    
    def underlying_growth(self):
        # Simulate a random price return from the given mean and standard deviation, then increases the asset price
        mean = self.und_return
        std_dev = self.und_vol
        random_value = np.random.normal(mean, std_dev)
        p_return = max(random_value,-.9999)
        #set the price to the new price
        self.underlyingprice = self.underlyingprice * (1 + p_return)

    def american_option_pricer(self):
        #parameters are 
        # S, K, T, r, sigma, option_type, n_steps
        n_steps = 100
        # spot price, strike price, time to maturity, risk free rate, volatility, option type, number of steps
        #to handle sqrt issues
        dt = self.dte/(365*100)
        u = np.exp(self.volatility * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(0.04 * dt) - d) / (u - d)
        #initialize the stock prices
        stock_prices = np.zeros((n_steps + 1, n_steps + 1))
        stock_prices[0, 0] = self.underlyingprice
        for i in range(1, n_steps + 1):
            stock_prices[i, 0] = stock_prices[i - 1, 0] * u
            for j in range(1, i + 1):
                stock_prices[i, j] = stock_prices[i - 1, j - 1] * d
        #initialize the option values
        option_values = np.zeros((n_steps + 1, n_steps + 1))
        for j in range(n_steps + 1):
            option_values[n_steps, j] = max(0, self.strike - stock_prices[n_steps, j]) if self.optiontype == 'put' else max(0, stock_prices[n_steps, j] - self.strike)
        #calculate the option values at each node
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[i, j] = max(0, self.strike - stock_prices[i, j]) if self.optiontype == 'put' else max(0, stock_prices[i, j] - self.strike)
                option_values[i, j] = max(option_values[i, j], np.exp(-.02 * dt) * (p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1]))
        return option_values[0, 0]*100

    def calculate_value(self):
        return self.price*self.quantity
    
    def income(self):
        return 0
    
    def growth(self):
        self.price = self.american_option_pricer()
        return
    
    def return_clean(self):
        self.returnstream['Date'] = pd.to_datetime(self.returnstream['Date'])
        return
    
    def liq_ability(self,time_period):
        #Controls the ability to liquidate from this asset
        #if time_period<self.liquidity:
        #    self.liq_flag = 0
        #else:
        if time_period % self.liquidity == 0:
            self.liq_flag = 1
        else:
            self.liq_flag = 0
        return self.liq_flag

################################# Asset, Fund, Option Objects ###########################################################################################

################################## Helper Functions #####################################################################################################

def combine_cashflows(cashflows_list):
    """
    Combine a list of consolidated_cashflows DataFrames based on quarter_end.
    Args:
        cashflows_list (list of pd.DataFrame): List of consolidated_cashflows DataFrames.
    Returns:
        pd.DataFrame: Combined DataFrame with summed values for each quarter_end.
    Note: If you are using with different vintage years, then only the first will be recorded.
    """
    for i in cashflows_list:
        i['quarter_end'] = pd.to_datetime(i.index)
    # Concatenate all DataFrames in the list
    tempcombined = pd.concat(cashflows_list, ignore_index=True)

    # Group by quarter_end and sum all numeric columns
    combined = tempcombined.drop(columns=['Vintage']).groupby('quarter_end').sum()
    combined['Vintage'] = tempcombined['Vintage'].iat[0]

    # Ensure cumulative columns are non-decreasing
    combined['cum_contributions_eoq'] = combined['cum_contributions_eoq'].cummin()
    combined['cum_distributions_eoq'] = combined['cum_distributions_eoq'].cummax()

    # Recalculate quarterly contributions and distributions
    combined['quarter_contribution'] = combined['cum_contributions_eoq'].diff().fillna(combined['cum_contributions_eoq'].iloc[0])
    combined['quarter_distribution'] = combined['cum_distributions_eoq'].diff().fillna(combined['cum_distributions_eoq'].iloc[0])

    # Add Age column
    combined['Age'] = range(len(combined))
    combined['committed'] = combined['committed'].max()
    combined['committed_not_called'] = combined['committed'] + combined['cum_contributions_eoq']
    combined['committed_not_called'].clip(lower=0, inplace=True)

    return combined

def plotter(sims):
    #plot the results of the simulations generated by a portfolio timecycle
    #sims is a dictionary with the asset_values, av_prerebal, warnings, subs, reds, rebal
    Post_Rebalancing = sims[0]
    Pre_Rebalancing = sims[1]
    IRR = sims[6]

    items_to_remove = ['Total', 'Redemption','Subscription','Date']
    assets = [item for item in list(sims[0].keys()) if item not in items_to_remove]

    for i in [Post_Rebalancing,Pre_Rebalancing]:
        df = pd.DataFrame(i)

        for col in df.columns:
            if col == 'Date':
                df = df.drop(col,axis=1)

        if i == Post_Rebalancing:
            df.plot(title='Post Rebalancing Asset Values over time')            
        else:
            df.plot(title='Pre Rebalancing Asset Values over time')
        plt.show()
        
        dfp2 = df.div(df['Total'], axis=0)

        #loop through portfolio2 asset names and then drop the columns that are not in the list
        for col in dfp2.columns:
            if col not in assets:
                dfp2= dfp2.drop(col, axis=1)

        if i == Post_Rebalancing:
            dfp2.plot(title=("Post Rebalancing Asset Percent over time"))
        else:
            dfp2.plot(title=("Pre Rebalancing Asset Percent over time"))
        plt.show()
    
    #plot the IRR, if we have chosen to calculate it
    try:
        df = pd.DataFrame(IRR['IRR'])
        df.plot(title='IRR over time')
        plt.show()
    except:
        pass
    return

########################
if __name__ == '__main__':
    # irr_data = load_irr()
    # fund_data = load_cashflows()

    # #merging the Preqin IRR data with the Preqin Cashflow data
    # fund_data2 = pd.merge(fund_data, irr_data, on='fund_id', how='left')
    # fund_data2 = fund_data2.rename(columns={'Value': 'IRR'})

    # # Apply the function to each group (by vintage_year)
    # fund_data3 = fund_data2.groupby('vintage').apply(calculate_quartiles)

    # # Reset index (optional, to remove the multi-index created by groupby)
    # fund_data3 = fund_data3.reset_index(drop=True)
    pass