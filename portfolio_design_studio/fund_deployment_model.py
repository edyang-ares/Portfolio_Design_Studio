import pandas as pd
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import numpy as np
from pandas.tseries.offsets import MonthEnd
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from .data import*

DEFAULT_SOFR_PATH = "Data Files/LIBOR SOFR Forward.xlsx"
DEFAULT_SOFR_RATE = 0.05

def _load_default_sofrcurve():
    try:
        return read_sofr_curve(DEFAULT_SOFR_PATH)
    except FileNotFoundError:
        return pd.DataFrame()

class AresFund:
    """Represents a private equity fund with cashflow generation capabilities.
    
    This class encapsulates fund parameters and provides methods to generate
    cashflow schedules, calculate management fees, and compute carry.
    
    Attributes:
        fund_name: Name of the fund
        fund_data: Dictionary containing all fund parameters
        fee_inputs: Dictionary containing fee calculation parameters
        subscription_line: Optional subscription line facility
        cashflows: DataFrame containing generated cashflows
        events: List of facility-related events
    """
    
    def __init__(self, fund_data: dict, fee_inputs: dict = None):
        """Initialize a Fund object.
        
        Args:
            fund_data: Dictionary with fund parameters (commitment, IRR, dates, etc.)
            fee_inputs: Dictionary with fee calculation parameters (optional)
        """
        self.fund_name = fund_data["Fund Name"]
        self.fund_data = fund_data
        self.fee_inputs = fee_inputs
        self.cashflows = None
    
    def generate_gross_cashflows(self) -> pd.DataFrame:
        """Generate gross cashflow schedule using existing logic.
        
        Returns:
            DataFrame with gross cashflows
        """
        self.cashflows = create_cashflow_schedule(self.fund_data)
        return self.cashflows
    
    def get_simplified_cashflows(self) -> pd.DataFrame:
        """Get simplified cashflow schedule.
        
        Returns:
            DataFrame with simplified cashflows
            
        Raises:
            ValueError: If cashflows haven't been generated yet
        """
        if self.cashflows is None:
            raise ValueError("Must generate cashflows first using generate_gross_cashflows()")
        return simplified_gross_cashflows(self.cashflows)
    
    def calculate_management_fees(self) -> pd.DataFrame:
        """Calculate management fees on the cashflows.
        
        Returns:
            DataFrame with management fees added
            
        Raises:
            ValueError: If cashflows or fee_inputs not available
        """
        if self.cashflows is None:
            raise ValueError("Must generate cashflows first using generate_gross_cashflows()")
        if self.fee_inputs is None:
            raise ValueError("Fee inputs not provided. Set fund.fee_inputs or pass to __init__")
        
        self.cashflows = management_fee(self.cashflows, self.fee_inputs)
        return self.cashflows
    
    def calculate_carry(self) -> pd.DataFrame:
        """Calculate carry (performance fees) on the cashflows.
        
        Returns:
            DataFrame with carry calculations added
            
        Raises:
            ValueError: If management fees haven't been calculated yet
        """
        if self.cashflows is None:
            raise ValueError("Must generate cashflows first")
        if 'Management Fees Charged' not in self.cashflows.columns:
            raise ValueError("Must calculate management fees first using calculate_management_fees()")
        
        self.cashflows = carry_calculations(self.cashflows, self.fee_inputs)
        return self.cashflows

mgmt_fee_mapping = {
    'Undrawn Commitment': 0, #Unfunded

    'Invested Assets':1, #Cumulative Equity - Cum_return_of_equity + debt balance outstanding (if no leverage, this is same as invested equity)
    'Invested Capital':1, #cumulative equity - cum_return_of_equity + Debt Balance Outstanding (if no leverage, this is same as invested equity)
    'Total Capital':1,#Treat as Invested Capital, shows up in AIREIT and AREIT
    'Invested Capital / Cashflow Available to LP / Purchase Price':1,#JRIF I, treat as Invested Capital
    
    'N/A':2, #zero
    '0':2, #zero
    'TBD':2, #zero
    np.nan:2, #zero
    
    'Invested Equity':3, #cumulative equity - cum_return_of_equity 
    'Unrealized Cost':3, #cumulative equity - cum_return_of_equity
    'Contributed Capital':3, #Contributed Capital = capital called from the LP (i.e. does not include funding through sub-lines)
                             #we should model this the same as Invested Equity since we cannot separate sub-lines in our model
    'Cost of Equity':3, #cumulatieve equity - cum_return_of_equity
    
    'NAV':4, #Gross Equity NAV
    'Managed Assets': 4, #APMF, Gross NAV
    'Outstanding Borrowings':4, #Applicable to the Revolver funds, we can model same as NAV
    
    'Committed Capital':5, #Committed
    'Invested Capital / Committed Capital':5 #AREMEX IV, Treat as Committed Capital to be conservative
    }

def get_quarterly_growth_rates(
    fund_data,
    dates,
    sofrcurve=None,
    hilo="High",
    growth_df=None,
    growth_start_date=None,
    growth_date_col="Date",
    growth_rate_col="Quarterly Growth",
    require_full_growth_history=True,
    default_sofr_rate=DEFAULT_SOFR_RATE
):
    """
    Produce quarterly growth rates for the fund model.

    Priority:
    1. If growth_df and growth_start_date are provided, try to use the
       supplied quarterly growth vector.
    2. If the vector is missing or insufficient, fall back to existing
       Fixed / SOFR logic.

    growth_df should contain:
        - growth_date_col, default "Date"
        - growth_rate_col, default "Quarterly Growth"

    growth_rate_col should already be quarterly, e.g. 0.02 for 2%.
    """

    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)

    # ------------------------------------------------------------
    # 1. Optional historical/vector growth path
    # ------------------------------------------------------------
    if growth_df is not None and growth_start_date is not None and len(growth_df) > 0:
        g = growth_df.copy()

        if growth_date_col not in g.columns:
            raise ValueError(f"growth_df is missing required date column: {growth_date_col}")

        if growth_rate_col not in g.columns:
            raise ValueError(f"growth_df is missing required growth column: {growth_rate_col}")

        g[growth_date_col] = pd.to_datetime(g[growth_date_col])
        g = g.sort_values(growth_date_col).reset_index(drop=True)

        start = pd.to_datetime(growth_start_date) + pd.offsets.QuarterEnd(0)

        # We need exactly as many quarterly growth observations as model dates.
        required_periods = len(dates)

        # Pull observations on or after the requested start quarter.
        candidate = g[g[growth_date_col] >= start].copy()

        if len(candidate) >= required_periods:
            selected = candidate.iloc[:required_periods].copy()
            selected_rates = selected[growth_rate_col].astype(float).reset_index(drop=True)

            # Use the model's dates as the index so downstream code lines up cleanly.
            return pd.Series(selected_rates.values, index=dates)

        if require_full_growth_history:
            # Fall through to fixed/SOFR logic instead of partially filling.
            pass
        else:
            # Optional partial mode: use what exists, then fill the rest later.
            growth_series = pd.Series(
                candidate[growth_rate_col].astype(float).values,
                index=pd.to_datetime(candidate[growth_date_col])
            ).sort_index()

            aligned = growth_series.reindex(dates, method="ffill").bfill()
            if aligned.notna().all():
                return aligned

    # ------------------------------------------------------------
    # 2. Existing Fixed / SOFR fallback path
    # ------------------------------------------------------------
    base = fund_data["Gross IRR Base"]
    spread = fund_data["Assumed Gross IRR High"] if hilo == "High" else fund_data["Assumed Gross IRR Low"]

    if base == "Fixed" or base == "N/A":
        annual_rates = pd.Series([spread] * len(dates), index=dates)

    elif base == "SOFR":
        if sofrcurve is None or len(sofrcurve) == 0:
            annual_rates = pd.Series([default_sofr_rate + spread] * len(dates), index=dates)
        else:
            sofr_series = pd.Series(
                sofrcurve["Growth"].values,
                index=pd.to_datetime(sofrcurve["Date"])
            ).sort_index()

            sofr_series = sofr_series.reindex(dates, method="ffill")
            annual_rates = sofr_series + spread
    else:
        annual_rates = pd.Series([spread] * len(dates), index=dates)

    quarterly_rates = (1 + annual_rates) ** (1 / 4) - 1
    quarterly_rates = quarterly_rates.ffill().bfill()

    return quarterly_rates

def create_cashflow_schedule(
    fund_data,
    growth_df=None,
    growth_start_date=None,
    growth_date_col="Date",
    growth_rate_col="Quarterly Growth",
    require_full_growth_history=True,
    sofrcurve=None
):
    """
    Creates a gross cashflow schedule based on fund parameters
    """
    if sofrcurve is None:
        sofrcurve = _load_default_sofrcurve()

    # Extract fund parameters
    fund_name = fund_data["Fund Name"]
    equity_commitment = fund_data["Original Equity Commitment"]*fund_data["Capital Called Estimate"]
    #assumed_gross_irr = fund_data["Assumed Gross IRR High"]/4  # Quarterly IRR
    cash_yield = fund_data["Cash Yield"]
    investment_period = fund_data["Investment Period"]
    post_investment_start = fund_data["Post Investment Start"] # When post-investment period starts (years)
    fund_term = fund_data["Fund Term"]
    try:
        investment_start_date = datetime.strptime(fund_data["Investment Start Date"], "%Y-%m-%d")
        #make investment_start_date quarter end
    except:
        investment_start_date = datetime.now()  # Default to current date if parsing fails
    investment_start_date = pd.to_datetime(investment_start_date) + pd.offsets.QuarterEnd(0)

    leverage = fund_data["Leverage"]
    ltv_target = fund_data["LTV Target"]
    #Controls if debt is paid down according to LTV or according to some schedule
    debt_paydown_logic = fund_data['Debt Paydown Logic']
    debt_paydown_quarters = fund_data['Debt Paydown Quarters']
    debt_paydown_start_year = fund_data['Debt Paydown Start']
    debt_paydown_end_year = fund_data['Debt Paydown End']
    debt_interest_rate = fund_data['Debt Interest Rate']/4  # Quarterly interest rate
    deployment_quarters = fund_data['Deployment Quarters']

    # Create date range (quarter-end periods)
    dates = pd.Series(
        pd.date_range(
            start=investment_start_date,
            periods=fund_term * 4 + 1,
            freq="QE-DEC",
        )
    )

    # Get quarterly growth rates based on fund parameters
    quarterly_growth_rates = get_quarterly_growth_rates(
        fund_data=fund_data,
        dates=pd.to_datetime(dates),
        sofrcurve=sofrcurve,
        growth_df=growth_df,
        growth_start_date=growth_start_date,
        growth_date_col=growth_date_col,
        growth_rate_col=growth_rate_col,
        require_full_growth_history=require_full_growth_history,
    )

    # Initialize dataframe
    cashflow_df = pd.DataFrame({
        'Date': dates,
        'fund_name': fund_name,
        'Period (Inv/Liq)': ['Inv'] * len(dates),  # Initialize all as investment
        'deployment_pace': 0.0,
        'cumulative_deployment_pace': 0.0,
        'cumulative_post_investment_deployment_pace': 0.0,
        'Inv. Equity': 0.0,
        'New Borrowing': 0.0,
        'remaining_cost': 0.0,
        'Debt Balance Outstanding': 0.0,
        'return_of_equity': 0.0,
        'gross_irr': quarterly_growth_rates.values,
        'bop_nav': 0.0,
        'eop_nav': 0.0
    })
    cashflow_df['Quarter'] = range(0, len(dates))
    cashflow_df['Year'] = cashflow_df['Quarter'] // 4
    # Calculate deployment pace during investment period
    investment_periods = investment_period * 4  # Convert to quarters
    post_investment_periods = post_investment_start * 4 # Convert to quarters
    
    # Simple linear deployment during investment period
    if deployment_quarters > 0:
        quarterly_deployment = 1.0 / deployment_quarters
        for i in range(min(deployment_quarters, len(dates))):
            cashflow_df.loc[i, 'deployment_pace'] = quarterly_deployment
    
    # Calculate cumulative deployment pace
    cashflow_df['cumulative_deployment_pace'] = cashflow_df['deployment_pace'].cumsum()
    
    # Set period labels and post-investment deployment pace
    post_inv_start_period = post_investment_periods
    harvest_periods = max(0, (fund_term * 4) - post_inv_start_period)
    
    # Update period labels for post-investment periods
    for i in range(post_inv_start_period, len(dates)):
        cashflow_df.loc[i, 'Period (Inv/Liq)'] = 'Liq'
    
    if harvest_periods > 0:
        # Distribution pace during post-investment period
        for i in range(post_inv_start_period, min(len(dates), fund_term * 4)):
            period_in_post_inv = i - post_inv_start_period
            # Exponential decay for distributions
            distribution_pace = 1 /harvest_periods
            cashflow_df.loc[i, 'cumulative_post_investment_deployment_pace'] = distribution_pace
    
    # Calculate equity deployment
    cashflow_df['Inv. Equity'] = cashflow_df['deployment_pace'] * equity_commitment
    
    # Calculate debt (based on leverage ratio)
    cashflow_df['New Borrowing'] = cashflow_df['Inv. Equity'] * leverage
    
    # Calculate cumulative equity and debt
    cashflow_df['cumulative_equity'] = cashflow_df['Inv. Equity'].cumsum()
    cashflow_df['cumulative_debt'] = cashflow_df['New Borrowing'].cumsum()
    
    # Calculate return of equity (distributions) - straight line over post investment period
    total_equity_deployed = equity_commitment  # Total equity to be returned
    post_investment_quarters = harvest_periods  # Number of quarters in post investment period
    
    if post_investment_quarters > 0:
        # Calculate total return based on IRR
        quarterly_return = total_equity_deployed / post_investment_quarters
        
        # Distribute returns evenly over post investment period
        for i in range(post_inv_start_period, min(len(dates), fund_term * 4)):
            cashflow_df.loc[i, 'return_of_equity'] = quarterly_return
    
    #debt paydown logic
    cashflow_df['LTV'] = 0.0
    cashflow_df['LTV Paydown'] = 0.0
    cashflow_df['Debt Paydown'] = 0.0
    cashflow_df['debt_paydown_schedule'] = 0.0
    paydown_start = debt_paydown_start_year *4
    paydown_end = debt_paydown_end_year *4
    cashflow_df['cum_debt_paydown'] = 0.0
    cashflow_df['Interest'] = 0.0
    cashflow_df['Debt Balance Outstanding'] = 0.0
    # Calculate NAV (Net Asset Value)
    cashflow_df['ret_cash_yield'] = 0.0
    cashflow_df['target_dist'] = 0.0
    cashflow_df['target_rate_of_dist'] = 0.0
    cashflow_df['distributions'] = 0.0
    cashflow_df['remaining_cost'] = 0.0
    cashflow_df['invested_assets'] = 0.0
    #Assume appreciation is linearly paid off during the post-investment period
    cashflow_df['appreciation_schedule'] = 0.0
    cashflow_df['appreciation_basis'] = 0.0
    cashflow_df['appreciation'] = 0.0
    cashflow_df['gross dists post debt-servicing'] = 0.0

    for i in range(len(dates)):
        period_growth = cashflow_df.loc[i, "gross_irr"]
        prev_growth = cashflow_df.loc[i - 1, "gross_irr"] if i > 0 else cashflow_df.loc[i, "gross_irr"]

        #BOP_NAV (beginning of period NAV)
        if i == 0:
            cashflow_df.loc[i, 'bop_nav'] = cashflow_df.loc[i, 'Inv. Equity'] + cashflow_df.loc[i, 'New Borrowing']
            cashflow_df.loc[i, 'remaining_cost'] = cashflow_df.loc[i, 'Inv. Equity']

        #Create the debt paydown schedule
        if i >= paydown_start and i < paydown_end+1:
            cashflow_df.loc[i, 'debt_paydown_schedule'] = (1/(paydown_end-paydown_start+1))/(1 - (1/(paydown_end-paydown_start+1))*(i-paydown_start))
        
        #Logic to pay down debt if LTV > target, or according to schedule
        if i>0:
            ltv = cashflow_df.loc[i-1, 'Debt Balance Outstanding'] / cashflow_df.loc[i-1, 'eop_nav'] if cashflow_df.loc[i, 'bop_nav'] != 0 else 0
            cashflow_df.loc[i,'LTV'] = ltv
            if debt_paydown_logic == 'Maintain LTV Target':
                if ltv > ltv_target and cashflow_df.loc[i, 'bop_nav'] != 0:
                    excess_debt = cashflow_df.loc[i-1, 'Debt Balance Outstanding'] - (ltv_target * cashflow_df.loc[i, 'bop_nav'])
                    cashflow_df.loc[i, 'LTV Paydown'] = excess_debt
                    cashflow_df.loc[i, 'Debt Paydown'] = cashflow_df.loc[i, 'Debt Paydown'] + excess_debt
            elif debt_paydown_logic == 'Follow Deployment Ratio':
                cashflow_df.loc[i, 'Debt Paydown'] = cashflow_df.loc[i-1, 'Debt Balance Outstanding'] * cashflow_df.loc[i, 'debt_paydown_schedule']
        if i == 0:
            cashflow_df.loc[i, 'cum_debt_paydown'] = cashflow_df.loc[i, 'Debt Paydown']
            cashflow_df.loc[i, 'Debt Balance Outstanding'] = cashflow_df.loc[i, 'New Borrowing']
        else:
            cashflow_df.loc[i, 'cum_debt_paydown'] = cashflow_df.loc[i-1, 'cum_debt_paydown'] + cashflow_df.loc[i, 'Debt Paydown']
            cashflow_df.loc[i, 'Debt Balance Outstanding'] = np.round(cashflow_df.loc[i-1, 'Debt Balance Outstanding'] + cashflow_df.loc[i, 'New Borrowing'] - cashflow_df.loc[i, 'Debt Paydown'],4)
        
        #Logic to return appreciation, if in the post-investment period
        if i >0:
            cashflow_df.loc[i, 'Interest'] = cashflow_df.loc[i-1, 'Debt Balance Outstanding'] * debt_interest_rate
            cashflow_df.loc[i, 'ret_cash_yield'] = cashflow_df.loc[i-1, 'eop_nav'] * (1+period_growth) * (cash_yield / 4) #cashflow_df.loc[i, 'bop_nav'] * (cash_yield / 4)  # Quarterly cash yield
            cashflow_df.loc[i,'appreciation_basis'] = cashflow_df.loc[i-1,'eop_nav']*(1+period_growth) - cashflow_df.loc[i-1,'remaining_cost'] - cashflow_df.loc[i-1,'Debt Balance Outstanding'] - cashflow_df.loc[i,'Interest'] - cashflow_df.loc[i,'ret_cash_yield']
        if cashflow_df.loc[i, 'Period (Inv/Liq)'] == 'Liq' and harvest_periods >0:
            denom = (1 - (1/(harvest_periods))*(i-post_investment_periods))
            if denom != 0:
                cashflow_df.loc[i,'appreciation_schedule'] = (1/(harvest_periods))/denom
            else:
                cashflow_df.loc[i,'appreciation_schedule'] = 1
        cashflow_df.loc[i,'appreciation'] = cashflow_df.loc[i,'appreciation_schedule'] * cashflow_df.loc[i,'appreciation_basis']
        if i != 0:
            cashflow_df.loc[i, 'bop_nav'] = cashflow_df.loc[i-1, 'eop_nav'] + cashflow_df.loc[i, 'Inv. Equity'] + cashflow_df.loc[i, 'New Borrowing']
            cashflow_df.loc[i, 'remaining_cost'] = cashflow_df.loc[i-1, 'remaining_cost'] + cashflow_df.loc[i, 'Inv. Equity'] - cashflow_df.loc[i, 'return_of_equity']        
            cashflow_df.loc[i, 'target_dist'] = cashflow_df.loc[i,'Debt Paydown'] + cashflow_df.loc[i,'return_of_equity'] + cashflow_df.loc[i,'ret_cash_yield'] + cashflow_df.loc[i,'Interest'] + cashflow_df.loc[i,'appreciation']
            cashflow_df.loc[i, 'target_rate_of_dist'] = min(1,(cashflow_df.loc[i,'target_dist'] / (cashflow_df.loc[i-1, 'eop_nav']*(1+period_growth)) if cashflow_df.loc[i-1, 'eop_nav'] !=0 else 0))
            cashflow_df.loc[i, 'distributions'] = cashflow_df.loc[i,'target_rate_of_dist'] * cashflow_df.loc[i-1, 'eop_nav']*(1+period_growth)
        cashflow_df.loc[i, 'invested_assets'] = cashflow_df.loc[i, 'remaining_cost'] + cashflow_df.loc[i, 'Debt Balance Outstanding']
        
        cashflow_df.loc[i, 'eop_nav'] = cashflow_df.loc[i, 'bop_nav']*(1+period_growth) - cashflow_df.loc[i, 'distributions']
        cashflow_df.loc[i, 'gross dists post debt-servicing'] = cashflow_df.loc[i,'distributions'] - cashflow_df.loc[i,'Interest'] - cashflow_df.loc[i,'Debt Paydown']
    # Calculate invested equity, total debt, and invested assets
    cashflow_df['cum_return_of_equity'] = cashflow_df['return_of_equity'].cumsum()
    cashflow_df['Gross Fund NAV'] = cashflow_df['eop_nav']
    cashflow_df['Gross Equity NAV'] = cashflow_df['Gross Fund NAV'] - cashflow_df['Debt Balance Outstanding']
    cashflow_df['Committed'] = fund_data["Original Equity Commitment"]
    cashflow_df['Unfunded'] = fund_data["Original Equity Commitment"] - cashflow_df['cumulative_equity']
    return cashflow_df

def simplified_gross_cashflows(fund_cashflow):
    """
    Wrapper function to create gross cashflow schedule
    """
    simplified = fund_cashflow[['Date','Quarter','Year','Inv. Equity','Period (Inv/Liq)','Debt Paydown','New Borrowing','Debt Balance Outstanding',
                            'Interest','distributions','Gross Fund NAV','gross dists post debt-servicing','Gross Equity NAV','cumulative_equity','remaining_cost','Committed','Unfunded']]
    #Gross Cont in fee model includes capital calls for interest and debt paydown
    return simplified

def management_fee(gross_cash_flows,fee_inputs):
    #First check if we have the required columns
    #columns we need: 'Unfunded','remaining_cost','Debt Balance Outstanding','Gross Equity NAV','Committed'
    
    gross_cash_flows['Fee Base'] = 0.0
    gross_cash_flows['Fee Base 2'] = 0.0
    
    inv_fee_base_type = mgmt_fee_mapping[fee_inputs["Primary Investment Period Mgmt Fee Basis"]]
    inv_fee_base_type2 = mgmt_fee_mapping[fee_inputs["Secondary Investment Period Mgmt Fee Basis"]]
    Liq_fee_base_type = mgmt_fee_mapping[fee_inputs["Liquidation Period Mgmt Fee Basis"]]

    gross_cash_flows['Fee Base Type'] = ''
    gross_cash_flows['Fee Base Type 2'] = ''

    for col in ['Fee Base Type', 'Fee Base Type 2']:
        for index, row in gross_cash_flows.iterrows():
            if row['Period (Inv/Liq)'] == 'Inv':
                if col == 'Fee Base Type':
                    fee_base_type = inv_fee_base_type
                    feebase = 'Fee Base'
                else:
                    fee_base_type = inv_fee_base_type2
                    feebase = 'Fee Base 2'
                gross_cash_flows.at[index, col] = fee_base_type
            else:
                fee_base_type = Liq_fee_base_type
                gross_cash_flows.at[index, col] = Liq_fee_base_type

            if fee_base_type == 0:
                gross_cash_flows.at[index, feebase] = row['Unfunded']
            elif fee_base_type == 1:
                gross_cash_flows.at[index, feebase] = row['remaining_cost'] + row['Debt Balance Outstanding']
            elif fee_base_type == 2:
                gross_cash_flows.at[index, feebase] = 0
            elif fee_base_type == 3:
                gross_cash_flows.at[index, feebase] = row['remaining_cost']
            elif fee_base_type == 4:
                gross_cash_flows.at[index, feebase] = row['Gross Equity NAV']
            elif fee_base_type == 5:
                gross_cash_flows.at[index, feebase] = row['Committed']
            
    total_discount = 0
    if fee_inputs["First Close Discount Applied?"]:
        total_discount += fee_inputs["First Close Discount"]
    if fee_inputs['Size Discount Applied?']:
        if row['Committed'] >= fee_inputs["3rd Tier Size Threshold"]:
            total_discount += fee_inputs["3rd Tier Size Discount"]
        elif row['Committed'] >= fee_inputs["2nd Tier Size Threshold"]:
            total_discount += fee_inputs["2nd Tier Size Discount"]
        elif row['Committed'] >= fee_inputs["1st Tier Size Threshold"]:
            total_discount += fee_inputs["1st Tier Size Discount"]

    if fee_inputs['Primary Standard Mgmt Fee'] is None:
        fee_inputs['Primary Standard Mgmt Fee'] = 0
    if fee_inputs['Secondary Standard Mgmt Fee'] is None:
        fee_inputs['Secondary Standard Mgmt Fee'] = 0
    if fee_inputs['Primary Post-Inv Period Mgmt Fee'] is None:
        fee_inputs['Primary Post-Inv Period Mgmt Fee'] = 0

    inv_fee_rate = max(fee_inputs["Primary Standard Mgmt Fee"] - total_discount, 0)
    inv_fee_rate = (1-fee_inputs['Partnership Discount']) * inv_fee_rate
    inv_fee_rate2 = max(fee_inputs["Secondary Standard Mgmt Fee"] - total_discount, 0)
    inv_fee_rate2 = (1-fee_inputs['Partnership Discount']) * inv_fee_rate2

    if fee_inputs['Apply Discounts to Liq. Period?']:
        liq_fee_rate = max(fee_inputs["Primary Post-Inv Period Mgmt Fee"] - total_discount, 0)
        liq_fee_rate = (1-fee_inputs['Partnership Discount']) * liq_fee_rate
    else:
        liq_fee_rate = fee_inputs["Primary Post-Inv Period Mgmt Fee"]
    
    gross_cash_flows['Management Fee Rate'] = 0.0
    gross_cash_flows['Management Fees Charged'] = 0.0
    for index, row in gross_cash_flows.iterrows():
        if row['Period (Inv/Liq)'] == 'Inv':
            fee_rate = inv_fee_rate
            fee_rate2 = inv_fee_rate2
        else:
            fee_rate = liq_fee_rate
            fee_rate2 = 0.0
        gross_cash_flows.at[index, 'Management Fee Rate'] = np.round(fee_rate,4)
        gross_cash_flows.at[index, 'Management Fees Charged'] = row['Fee Base'] * fee_rate / 4  + row['Fee Base 2'] * fee_rate2 / 4  # Quarterly fee
    
    #Add in Net Contributions / Distributions Columns
    if fee_inputs['Fees swept or called'] == 'Called':
        gross_cash_flows['Net Contributions'] = gross_cash_flows['Inv. Equity'] + gross_cash_flows['Management Fees Charged']
        gross_cash_flows['Net Distributions post debt-servicing'] = gross_cash_flows['gross dists post debt-servicing']
        gross_cash_flows['Accumulated Mgmt Fees'] = 0
        gross_cash_flows['Management Fees Paid'] = gross_cash_flows['Management Fees Charged']
    elif fee_inputs['Fees swept or called'] == 'Swept':
        gross_cash_flows['Net Contributions'] = gross_cash_flows['Inv. Equity'] 
        gross_cash_flows['Accumulated Mgmt Fees'] = 0
        gross_cash_flows['Management Fees Paid'] = 0
        for i in gross_cash_flows.index :
            period_fee_charged = gross_cash_flows['Management Fees Charged'][i]
            if i > 0:
                period_fee_paid = min(gross_cash_flows['gross dists post debt-servicing'][i],period_fee_charged+
                                      gross_cash_flows['Accumulated Mgmt Fees'][i-1])
            else:
                period_fee_paid = min(gross_cash_flows['gross dists post debt-servicing'][i],period_fee_charged)
            gross_cash_flows.loc[i, "Management Fees Paid"] = period_fee_paid 
            if i > 0:
                gross_cash_flows.loc[i, "Accumulated Mgmt Fees"] = gross_cash_flows['Accumulated Mgmt Fees'][i-1] + period_fee_charged - period_fee_paid
            else:
                gross_cash_flows.loc[i, "Accumulated Mgmt Fees"] = period_fee_charged - period_fee_paid
        gross_cash_flows['Net Distributions post debt-servicing'] = gross_cash_flows['gross dists post debt-servicing'] - gross_cash_flows['Management Fees Paid']
    gross_cash_flows['Cumulative Management Fees'] = gross_cash_flows['Management Fees Charged'].cumsum()
    cashflows_mgmt_fees = gross_cash_flows
    return cashflows_mgmt_fees

def carry_calculations(cashflows_mgmt_fees, fee_inputs):
    """
    Calculate carry (performance fees) based on:
    - Hurdle rate: minimum return threshold
    - Performance fee: % of gains above hurdle
    - GP catchup: GP's share of returns to catch up after hurdle
    
    Waterfall:
    1. LP receives distributions until LP MOIC = 1.0 (return of capital)
    2. LP receives distributions until hurdle is achieved
    3. GP receives distributions until it catches up to performance_fee share of profits
    4. Thereafter, distributions split: (1-performance_fee) to LP, performance_fee to GP
    """
    
    cashflows = cashflows_mgmt_fees.copy()
    
    # Extract fee inputs
    performance_fee = 0 if pd.isna(fee_inputs['Performance Fee']) else fee_inputs['Performance Fee']
    hurdle_rate = 0 if pd.isna(fee_inputs['Hurdle Rate']) else fee_inputs['Hurdle Rate']
    gp_catchup = 0 if pd.isna(fee_inputs['GP Catchup']) else fee_inputs['GP Catchup']
    
    # Initialize carry columns
    cashflows['Cumulative Profit Distributions'] = 0.0
    cashflows['Cumulative Contributions'] = 0.0
    cashflows['Return of Cost to LP'] = 0.0
    cashflows['Cumulative Hurdle'] = 0.0
    cashflows['LP Dist Hurdle'] = 0.0
    cashflows['GP Catchup'] = 0.0
    cashflows['LP Catchup Period Dist'] = 0.0
    cashflows['LP Terminal Split'] = 0.0
    cashflows['GP Terminal Split'] = 0.0
    cashflows['LP Distributions this Period'] = 0.0
    cashflows['GP Distributions this Period'] = 0.0
    cashflows['Cumulative GP Received'] = 0.0
    cashflows['Cumulative LP Received'] = 0.0

    # Explicitly convert to float64 to avoid dtype warnings
    for col in ['Cumulative Profit Distributions', 'Cumulative Contributions', 'Return of Cost to LP',
                'Cumulative Hurdle', 'LP Dist Hurdle', 'GP Catchup', 'LP Catchup Period Dist',
                'LP Terminal Split', 'GP Terminal Split', 'LP Distributions this Period',
                'GP Distributions this Period', 'Cumulative GP Received', 'Cumulative LP Received']:
        cashflows[col] = cashflows[col].astype('float64')
    
    cumulative_contributions = 0.0
    cumulative_distributions = 0.0
    cumulative_lp_received = 0.0
    cumulative_lp_cost_received = 0.0
    cumulative_lp_hurdle_received = 0.0
    cumulative_lp_catchup_received = 0.0
    cumulative_gp_received = 0.0
    
    quarterly_hurdle = (1 + hurdle_rate) ** (1/4) - 1
    
    for i in cashflows.index:
        # Track cumulative contributions and distributions
        if i == 0:
            cumulative_contributions = cashflows.loc[i, 'Net Contributions']
            cumulative_distributions = cashflows.loc[i, 'Net Distributions post debt-servicing']
        else:
            cumulative_contributions += cashflows.loc[i, 'Net Contributions']
            cumulative_distributions += cashflows.loc[i, 'Net Distributions post debt-servicing']
        
        cashflows.loc[i, 'Cumulative Profit Distributions'] = cumulative_distributions
        cashflows.loc[i, 'Cumulative Contributions'] = cumulative_contributions
        
        # Calculate hurdle return for LP (compound quarterly)
        quarters_elapsed = cashflows.loc[i, 'Quarter']
        cumulative_hurdle = 0
        if i > 0:
            for j in range(i):
                cumulative_hurdle += (cashflows.loc[j,'Net Contributions']-cashflows.loc[j, 'Return of Cost to LP']
                                      -cashflows.loc[j, 'LP Dist Hurdle'])*((1+quarterly_hurdle)**(quarters_elapsed-j)-1)
            cashflows.loc[i,"Cumulative Hurdle"] = cumulative_hurdle
        
        # Current period distribution
        period_distribution = cashflows.loc[i, 'Net Distributions post debt-servicing']
        
        # Waterfall logic
        lp_this_period = 0.0
        gp_this_period = 0.0
        
        # Step 1: LP receives until it gets back all capital (Return of Cost)
        capital_shortfall = max(0, cumulative_contributions - cumulative_lp_cost_received)
        lp_from_capital = min(period_distribution, capital_shortfall)
        lp_this_period += lp_from_capital
        cumulative_lp_received += lp_from_capital
        cumulative_lp_cost_received += lp_from_capital
        remaining_distribution = period_distribution - lp_from_capital
        
        cashflows.loc[i, 'Return of Cost to LP'] = lp_from_capital
        
        # Step 2: LP receives until hurdle is met
        lp_from_hurdle = min(remaining_distribution, max(0,cumulative_hurdle-cumulative_lp_hurdle_received))
        lp_this_period += lp_from_hurdle
        cumulative_lp_received += lp_from_hurdle
        cumulative_lp_hurdle_received += lp_from_hurdle
        remaining_distribution -= lp_from_hurdle
        
        cashflows.loc[i, 'LP Dist Hurdle'] = lp_from_hurdle
        
        # Step 3: GP catches up (only if LP has got capital back and hurdle is met)
        gp_catchup_target = max(0, (cumulative_lp_hurdle_received + cumulative_lp_catchup_received)*(performance_fee/(1-performance_fee)) - cumulative_gp_received)
        gp_from_catchup = min(remaining_distribution* gp_catchup, gp_catchup_target)
        if gp_catchup == 0:
            lp_from_catchup = 0
        else:
            lp_from_catchup = gp_from_catchup*(1-gp_catchup)/gp_catchup
        cumulative_lp_catchup_received += lp_from_catchup
        cumulative_lp_received += lp_from_catchup
        cumulative_gp_received += gp_from_catchup
        remaining_distribution -= (gp_from_catchup+lp_from_catchup)

        cashflows.loc[i, 'GP Catchup'] = gp_from_catchup
        cashflows.loc[i, 'LP Catchup Period Dist'] = lp_from_catchup
        
        # Step 4: Split remaining distributions
        lp_terminal = remaining_distribution * (1 - performance_fee)
        gp_terminal = remaining_distribution * performance_fee
        lp_this_period += lp_terminal
        gp_this_period += gp_terminal
        cumulative_lp_received += lp_terminal
        cumulative_gp_received += gp_terminal
    
        cashflows.loc[i, 'LP Terminal Split'] = lp_terminal
        cashflows.loc[i, 'GP Terminal Split'] = gp_terminal
        cashflows.loc[i, 'LP Distributions this Period'] = lp_this_period
        cashflows.loc[i, 'GP Distributions this Period'] = gp_this_period
        cashflows.loc[i, 'Cumulative GP Received'] = cumulative_gp_received
        cashflows.loc[i, 'Cumulative LP Received'] = cumulative_lp_received
    
    return cashflows