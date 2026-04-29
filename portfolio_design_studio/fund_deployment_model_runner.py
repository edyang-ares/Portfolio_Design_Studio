from .fund_deployment_model import*
from pyxirr import xirr

def parse_values(value):
        """Helper function to parse percentage strings"""
        if pd.isna(value):
            return None
        if isinstance(value, str):
            if "-" in value:
                #return low end of range if a range is provided
                value = value.split("-")[0].strip()
            
            if "%" in value:
                return float(value.rstrip("%").strip()) / 100
            elif ('x' in value) or ('X' in value):
                value = (value.strip().rstrip('x'))
                value = float(value.strip().rstrip('X'))
                return value  
            elif ('S' in value) or ('SOFR' in value) or ('s' in value):
                return value
            elif value == 'N/A' or value == 'TBD':
                return 0
        return float(value)

def period_handler(value):
     if pd.isna(value):
         return 10 #Default
     if isinstance(value, (int,float)):
          return int(value)
     elif isinstance(value, str):
        if value == 'N/A' or value == 'Open-ended':
            return 10 #assume 10 years for open ended funds
        else:
            return int(value)

def load_fund_data_from_mfr(mfr_data, fund_name):
    """
    Load fund data from MFR CSV file and populate fund_data_input and fund_fee_input dictionaries
    
    Parameters:
    -----------
    csv_file_path : file loaded from pandas read_csv
        Path to the MFR_Python Readable CSV file
    fund_name : str
        Strategy Short Name of the fund to extract (e.g., "SDL IV (Levered)")
    
    Returns:
    --------
    tuple : (fund_data_input, fund_fee_input) dictionaries populated with MFR data
    """

    
    # Filter for the specific fund
    fund_row = mfr_data[mfr_data['Strategy Short Name'] == fund_name]
    
    if fund_row.empty:
        raise ValueError(f"Fund '{fund_name}' not found in MFR data")
    
    fund_row = fund_row.iloc[0]  # Get first match
    
    # Parse dates
    try:
        investment_start = pd.to_datetime(fund_row['Investment Start Date']).strftime("%Y-%m-%d")
    except:
        investment_start = "2027-03-31"  # Default
    
    # Extract fund data

    fund_data_input = {}
    fund_fee_input = {}
    fund_data_input = {
            "Fund Name": fund_row['Strategy Short Name'],
            "Vintage": parse_values(fund_row['Vintage']) if pd.notna(fund_row['Vintage']) else 2026, #Default to 2026 vintage
            "Geography": 'Unknown', #Awaiting further refinement
            "Currency": fund_row['Currency'] if pd.notna(fund_row['Currency']) else 'USD',
            "Strategy": fund_row['Strategy'] if pd.notna(fund_row['Strategy']) else 'Ares', #Awaiting further refinement
            'Burgiss Map': fund_row['Burgiss Strategy Level 1'] if pd.notna(fund_row['Burgiss Strategy Level 1']) else '(All)', #Default to broadest category if not provided
            "Original Equity Commitment": 1000000000,  # Default $1B (adjust as needed)
            "Capital Called Estimate": 0.9,  # Default 90%
            
            "Gross IRR Base": fund_row['Target Net IRR High Base'] if pd.notna(fund_row['Target Net IRR High Base']) else "Fixed",  # Default to "Fixed" if not provided
            "Assumed Gross IRR Low": parse_values(fund_row['Gross IRR Low Assumption (% - Fund Level)']) if pd.notna(fund_row['Gross IRR Low Assumption (% - Fund Level)']) else 0.10,
            "Assumed Gross IRR High": parse_values(fund_row['Gross IRR High Assumption (% - Fund Level)']) if pd.notna(fund_row['Gross IRR High Assumption (% - Fund Level)']) else 0.15,
            
            "Cash Yield": parse_values(fund_row['Estimated Annual Distribution Yield (% of Invested Capital)']) if pd.notna(fund_row['Estimated Annual Distribution Yield (% of Invested Capital)']) else 0.05,
            "Deployment Quarters": max(1,int(parse_values(fund_row['Deployment Quarters']))) if pd.notna(fund_row['Deployment Quarters']) else 12,
            "Investment Period": max(1,period_handler(fund_row['Investment Period'])) if pd.notna(fund_row['Investment Period']) else 4,
            "Post Investment Start": period_handler(fund_row['Investment Period']) if pd.notna(fund_row['Investment Period']) else 4,  # Assume post-inv starts at end of inv period
            "Fund Term": max(1,period_handler(fund_row['Fund Term'])) if pd.notna(fund_row['Fund Term']) else 10,
            "Investment Start Date": investment_start,
            "Leverage": parse_values(fund_row['Leverage']) if pd.notna(fund_row['Leverage']) else 0.0,
            "LTV Target": 0.5,  # Default LTV target
            "Debt Paydown Quarters": 4,  # Default
            "Debt Paydown Start": (period_handler(fund_row['Fund Term']) - 1) if pd.notna(fund_row['Fund Term']) else 5,
            "Debt Paydown End": period_handler(fund_row['Fund Term']) if pd.notna(fund_row['Fund Term']) else 7,
            "Debt Interest Base": fund_row['Cost of Leverage Base'] if pd.notna(fund_row['Cost of Leverage Base']) else "Fixed",
            "Debt Interest Rate": parse_values(fund_row['Cost of Leverage']) if pd.notna(fund_row['Cost of Leverage']) else 0.05,
            "Debt Paydown Logic": 'Follow Deployment Ratio'
        }
    
    # Extract fee data
    fund_fee_input = {
            "Primary Standard Mgmt Fee": parse_values(fund_row['Primary Standard Mgmt Fee (%)']) if pd.notna(fund_row['Primary Standard Mgmt Fee (%)']) else 0,
            "Primary Investment Period Mgmt Fee Basis": fund_row['Primary Investment Period Mgmt Fee Basis'] if pd.notna(fund_row['Primary Investment Period Mgmt Fee Basis']) else 'Invested Assets',
            "Secondary Standard Mgmt Fee": parse_values(fund_row['Secondary Standard Mgmt Fee (%)']) if pd.notna(fund_row['Secondary Standard Mgmt Fee (%)']) else 0,
            "Secondary Investment Period Mgmt Fee Basis": fund_row['Secondary Investment Period Mgmt Fee Basis'] if pd.notna(fund_row['Secondary Investment Period Mgmt Fee Basis']) else np.nan,
            "Primary Post-Inv Period Mgmt Fee": parse_values(fund_row['Post-Inv. Period Mgmt Fee (%)']) if pd.notna(fund_row['Post-Inv. Period Mgmt Fee (%)']) else 0,
            "Liquidation Period Mgmt Fee Basis": fund_row['Liquidation Period Mgmt Fee Basis'] if pd.notna(fund_row['Liquidation Period Mgmt Fee Basis']) else 'Invested Assets',
            "First Close Discount": parse_values(fund_row['First Close Discount (%)']) if pd.notna(fund_row['First Close Discount (%)']) else 0.0,
            "First Close Discount Applied?": True,
            "Size Discount Applied?": True,
            "1st Tier Size Threshold": 100000000,
            "1st Tier Size Discount": 0.0005,
            "2nd Tier Size Threshold": 250000000,
            "2nd Tier Size Discount": 0.00075,
            "3rd Tier Size Threshold": 500000000,
            "3rd Tier Size Discount": 0.001,
            "Apply Discounts to Liq. Period?": fund_row['Apply Close+Size Discounts to Liq. Period?'] == 'Yes' if pd.notna(fund_row['Apply Close+Size Discounts to Liq. Period?']) else False,
            "Partnership Discount": 0.0,
            "Performance Fee": parse_values(fund_row['Performance Fee (%)']) if pd.notna(fund_row['Performance Fee (%)']) else np.nan,
            "Hurdle Rate": parse_values(fund_row['Hurdle Rate (%)']) if pd.notna(fund_row['Hurdle Rate (%)']) else np.nan,
            "GP Catchup": parse_values(fund_row['GP Catchup (%)']) if pd.notna(fund_row['GP Catchup (%)']) else 1.0,
            "Fees swept or called": 'Called'
        }
    
    return fund_data_input, fund_fee_input

def data_setup(mfr_source, verbose=False):
    #mfr_source = """Data Files/MFR_PYTHON_2026.04.csv"""
    if isinstance(mfr_source, pd.DataFrame):
        mfr_file = mfr_source.copy()
    else:
        mfr_file = pd.read_csv(mfr_source)

    # Load multiple funds at once
    fund_names = mfr_file['Strategy Short Name'].unique()

    fund_data_input = {}
    fund_fee_input = {}
    for fund_name in fund_names:
        if verbose:
            print(f"Loading data for fund: {fund_name}")
        fund_data_input[fund_name], fund_fee_input[fund_name] = load_fund_data_from_mfr(mfr_file, fund_name)
    
    return fund_data_input, fund_fee_input

def run_one_fund(fund_name, fund_data_input, fund_fee_input):
    fund_cashflow = create_cashflow_schedule(fund_data_input[fund_name])
    fund_cashflow = management_fee(fund_cashflow,fund_fee_input[fund_name])
    fund_cashflow_with_carry = carry_calculations(fund_cashflow, fund_fee_input[fund_name])
    #Add identifying columns
    fund_cashflow_with_carry['Vintage'] = fund_data_input[fund_name]['Vintage']
    fund_cashflow_with_carry['Geography'] = fund_data_input[fund_name]['Geography']
    fund_cashflow_with_carry['Strategy'] = fund_data_input[fund_name]['Strategy']
    fund_cashflow_with_carry['Currency'] = fund_data_input[fund_name]['Currency']

    return fund_cashflow_with_carry

def run_fund_flex(fund_name, fund_data_input, fund_fee_input, size_discount_options=[True, False], close_discount_options=[True, False], partnership_discount_options=[0.0, 0.075]):
    # Create a list to store results from each permutation
    results_list = []

    # Cycle between size / close on / off and partnership discount = 0 or 7.5%
    for i_size in size_discount_options:
        for i_close in close_discount_options:
            for i_p in partnership_discount_options:
                fund_fee_input[fund_name]['Size Discount Applied?'] = i_size
                fund_fee_input[fund_name]['First Close Discount Applied?'] = i_close
                fund_fee_input[fund_name]['Partnership Discount'] = i_p
                
                # Generate the cashflow schedule
                fund_cashflow = create_cashflow_schedule(fund_data_input[fund_name])
                fund_cashflow = management_fee(fund_cashflow, fund_fee_input[fund_name])
                fund_cashflow_with_carry = carry_calculations(fund_cashflow, fund_fee_input[fund_name])
                
                # Extract relevant data for this permutation
                permutation_data = fund_cashflow_with_carry[['Date', 'Quarter', 'Cumulative Management Fees']].copy()
                
                # Add permutation identifiers
                permutation_data['Size Discount'] = i_size
                permutation_data['First Close Discount'] = i_close
                permutation_data['Partnership Discount'] = i_p
                
                # Add a permutation ID for easier filtering
                permutation_id = f"Size:{i_size}_Close:{i_close}_Partner:{i_p}"
                permutation_data['Permutation ID'] = permutation_id
                
                results_list.append(permutation_data)

    # Combine all permutations into one dataframe
    all_permutations_df = pd.concat(results_list, ignore_index=True)
    return all_permutations_df

def run_ptf_funds(fund_names, fund_data_input, fund_fee_input):
    ptf_data = {}
    ptf_fees = {}
    for fund in fund_names:
        if fund in fund_data_input.keys():
            ptf_data[fund] = fund_data_input[fund]
        if fund in fund_fee_input.keys():
            ptf_fees[fund] = fund_fee_input[fund]
    #model the cashflows for the entire portfolio of funds
    ptf_results = {}
    for fund in fund_names:
        if fund in ptf_data.keys() and fund in ptf_fees.keys():
            cashflow = create_cashflow_schedule(ptf_data[fund])
            cashflow = management_fee(cashflow, ptf_fees[fund])
            cashflow_with_carry = carry_calculations(cashflow, ptf_fees[fund])
            ptf_results[fund] = cashflow_with_carry

    #combine into one dataframe for analysis
    ptf_combined = pd.DataFrame()
    for fund, df in ptf_results.items():
        df = df.copy()
        df['Fund'] = fund
        df['Vintage'] = ptf_data[fund]['Vintage']
        df['Geography'] = ptf_data[fund]['Geography']
        df['Strategy'] = ptf_data[fund]['Strategy']
        df['Currency'] = ptf_data[fund]['Currency']
        ptf_combined = pd.concat([ptf_combined, df], ignore_index=True)

    return ptf_combined

def ptf_combined_results(ptf_combined):
    # Create combined table with each fund filled forward to the last date
    # if ptf_combined.empty:
    #     return ptf_combined

    max_date = ptf_combined['Date'].max()
    min_date = ptf_combined['Date'].min()
    min_date = (pd.Timestamp(min_date) + pd.offsets.QuarterEnd(0))
    max_date = (pd.Timestamp(max_date) + pd.offsets.QuarterEnd(0))
    date_range = pd.date_range(start=min_date, end=max_date, freq='QE-DEC')

    # Fill each fund separately to max date
    filled_funds = []
    for fund_name, fund_df in ptf_combined.groupby('Fund'):
        fund_df = fund_df.copy().set_index('Date').sort_index()
        fund_df = fund_df.reindex(date_range)
        fund_df['Fund'] = fund_name
        for col in ['Cumulative Contributions', 'Cumulative LP Received', 'Cumulative GP Received', 'Cumulative Management Fees', 'Gross Fund NAV','Gross Equity NAV']:
            fund_df[col] = fund_df[col].ffill()
        filled_funds.append(fund_df)

    ptf_filled = pd.concat(filled_funds, axis=0).reset_index().rename(columns={'index': 'Date'})

    
    # Aggregate to portfolio totals after filling each fund to max date
    ptf_by_date = ptf_filled.groupby('Date').agg({
        'Cumulative Contributions': 'sum',
        'Cumulative LP Received': 'sum',
        'Cumulative GP Received': 'sum',
        'Cumulative Management Fees': 'sum',
        'Gross Fund NAV': 'sum',
        'Gross Equity NAV': 'sum'
    }).reset_index()

    return ptf_by_date


def calc_portfolio_irr(ptf_by_date):
    if ptf_by_date.empty:
        return np.nan

    contrib = ptf_by_date['Cumulative Contributions'].diff().fillna(
        ptf_by_date['Cumulative Contributions']
    )
    dists = ptf_by_date['Cumulative LP Received'].diff().fillna(
        ptf_by_date['Cumulative LP Received']
    )
    cashflows = (-contrib + dists).to_list()
    cashflows[-1] = cashflows[-1] + float(ptf_by_date['Gross Equity NAV'].iloc[-1])
    dates = [pd.Timestamp(d).to_pydatetime() for d in ptf_by_date['Date'].to_list()]

    if not any(abs(cf) > 0 for cf in cashflows):
        return np.nan


    return xirr(dates, cashflows)
  

def calc_portfolio_twr(ptf_by_date):
    if ptf_by_date.empty:
        return np.nan

    contrib = ptf_by_date['Cumulative Contributions'].diff().fillna(
        ptf_by_date['Cumulative Contributions']
    )
    dists = ptf_by_date['Cumulative LP Received'].diff().fillna(
        ptf_by_date['Cumulative LP Received']
    )
    # External cashflows into the fund (capital calls positive, distributions negative)
    net_cf = dists - contrib

    nav = ptf_by_date['Gross Equity NAV']
    growth = []
    try:
        growth.append(nav.iloc[0]/contrib.iloc[0])
    except ZeroDivisionError:
        growth.append(1.0)
    for i in range(1, len(ptf_by_date)):
        beginning_nav = np.round(nav.iloc[i - 1], 2)
        if beginning_nav == 0:
            growth.append(1.0)
        else:
            ending_nav = nav.iloc[i]
            period_cf = net_cf.iloc[i]
            period_return = (ending_nav - beginning_nav + period_cf) / beginning_nav
            growth.append(1.0 + period_return)
    
    if not growth:
        return np.nan

    twr_total = np.prod(growth) - 1.0
    years = len(growth) / 4.0
    if years <= 0:
        return np.nan

    twr_annualized = (1.0 + twr_total) ** (1.0 / years) - 1.0
    return twr_annualized

def calc_portfolio_tvpi(ptf_by_date):
    if ptf_by_date.empty:
        return np.nan

    paid_in = float(ptf_by_date['Cumulative Contributions'].iloc[-1])
    if paid_in == 0:
        return np.nan

    distributed = float(ptf_by_date['Cumulative LP Received'].iloc[-1])
    nav = float(ptf_by_date['Gross Equity NAV'].iloc[-1])
    return (distributed + nav) / paid_in

