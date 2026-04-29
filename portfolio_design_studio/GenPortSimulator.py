"""GenPortSimulator.py: A simple simulator for the GenPort protocol."""

import sys
import pandas as pd
import numpy as np
from .Lmk_Irr import BG_IRR_array
from datetime import datetime
from dateutil.relativedelta import relativedelta

# data access
from .data import get_engine, load_cashflows, load_irr, load_burgiss, calculate_quartiles
from .adapters import ares_cashflow_convert

# simulation / domain
from .GenPortvN import Portfolio, Asset, Fund, Line_of_Credit, fund_selector, combine_cashflows, mini_irr, mini_twr

# Default connection string (engine is NOT created at import time)
DEFAULT_CONN_STR = 'mssql+pyodbc://njd-p-lmksql01/FundMetrics?trusted_connection=yes&driver=SQL+Server'

def load_data(strategy, conn_str=None):
    """Load and prepare fund data with IRR and quartile information.
    
    Parameters
    ----------
    strategy : str
        Strategy filter for cashflows (e.g., 'Buyout', 'Private Debt').
    conn_str : str, optional
        Database connection string. If None, uses DEFAULT_CONN_STR.
    
    Returns
    -------
    pd.DataFrame
        Fund data merged with IRR and quartile information.
    """
    if conn_str is None:
        conn_str = DEFAULT_CONN_STR
    
    # Create engine only when load_data is called
    engine = get_engine(conn_str)
    
    irr_data = load_irr(engine)
    fund_data = load_cashflows(engine, vintage=None, strategy=strategy)
    
    # Merging the Preqin IRR data with the Preqin Cashflow data
    usd_buyout2 = pd.merge(fund_data, irr_data, on='fund_id', how='left').reset_index(drop=True)
    usd_buyout2 = usd_buyout2.rename(columns={'Value': 'IRR'})
    
    # Apply the function to each group (by vintage_year)
    # calculate quartiles for usd_buyout3 based on vintage
    usd_buyout3 = usd_buyout2.groupby('vintage', group_keys=False).apply(calculate_quartiles)
    usd_buyout3 = usd_buyout3.reset_index(drop=True)
    return usd_buyout3

def private_combine(simresults):
    """
    Combine the private cashflows in simresults from the run_single_simulation function.
    """
    initial = pd.DataFrame(simresults[0][0])
    privates = pd.DataFrame(simresults[0][4])
    failflag = 0
    for i in simresults.keys():
        if i != 0:
            try:
                temp = pd.DataFrame(simresults[i][0]) #this dataframe holds all cashflows
                initial = pd.concat([initial,temp],axis=0)

                temp1 = pd.DataFrame(simresults[i][4])
                privates = pd.concat([privates,temp1],axis=0)
            except:
                failflag = 1
                break    
    #fillna with 0
    initial.fillna(0,inplace=True)
    nav_columns = [col for col in initial.columns if 'NAV' in col]
    cnc_columns = [col for col in initial.columns if 'Com. Not Called' in col]
    initial['Total NAV'] = initial[nav_columns].sum(axis=1)
    initial['Total Com. Not Called'] = initial[cnc_columns].sum(axis=1)

    initial.reset_index(drop=True, inplace =True)
    privates.reset_index(drop=True, inplace =True)

    initial = pd.merge(initial,privates,how='left',left_index=True,right_index=True)
        
    return initial, failflag

def target_base_range(df,ptf_life):
    #Takes a dataframe that has two columns, target base and private range
    #If the length of data given is less than ptf_life, add more rows that are 0
    df = df[['Target Base','Target Private']]
    if len(df) < ptf_life+1:
        #create a new dataframe with the same columns
        newdf = pd.DataFrame(columns=df.columns)
        #add the rows from df to newdf
        newdf = pd.concat([newdf,df],axis=0)
        #add the rows that are 0
        for i in range(ptf_life+1-len(df)):
            newdf = pd.concat([newdf,pd.DataFrame([["Dollar",0.0]],columns=df.columns)],axis=0)
    else:
        newdf = df.reset_index(drop=True)[:ptf_life+1]
    return newdf.reset_index(drop=True)

def run_single_simulation(iteration,init_funds,ptf_size,ptf_life,init_age,new_commits,
                          funddata,target_base,private_range,target_cash,redrate,red_base,red_years,overcommit_amt,public_assets,select_year = False,d_year=None, 
                          commit_max=False,q1funds=None,q2funds=None,q3funds=None,q4funds=None,replacement=True):
    """
    Run a single simulation of the portfolio time cycle.

    Args:
        iteration (int): Simulation iteration number.
        init_funds (int): Number of initial funds to select.
        ptf_size (float): Initial portfolio size.
        start_year (int): Starting year for the funds in the simulation.
        ptf_life (int): Number of years to run the simulation.
        init_age (int): Initial age of the funds.
        new_commits (int): Number of new commitments to add each year.
        funddata (pd.DataFrame): DataFrame containing fund data. 
        private_range (array): Target private allocation range. ex. np.arange(0.10, 0.12, 0.01)
        target_cash (float): Target cash allocation.
        redrate (array): Redemption rate range. ex. np.arange(0.10, 0.12, 0.01)
        overcommit_amt (float): Overcommitment amount. (e.g.,  total exposure= target (1+x%)*Portfolio)
        public_assets (dict): list of public asset objects.
        select_year (bool): If True, select funds based on the d_year.
        d_year (int): start year.
        commit_max (bool): If True, commit the maxmimum of what can be committed
        q1 to q4 funds (int): Number of funds to select from each quartile, 4 being the top.
    Returns:
        dict: Simulation results for each target private allocation and redemption rate.
    """    
    overall_sims = {}
    overall_funds = {}
    overall_fail = {}

    # Iterate over target private allocations and redemption rates
    for target_private in private_range:
        for redemption_rate in redrate:
            if d_year == None:
                r_start_year = 2000
                t_start_year = r_start_year
            else:
                r_start_year = d_year
                t_start_year = d_year
            
            target_cash = np.round(target_cash, 2)

            if target_base == "Percentage" or target_base == "Target":
                target_private = np.round(target_private, 2)
                target_public = 1 - target_private - target_cash
            elif target_base == "Dollar":
                target_public = (ptf_size - target_private)*(1-target_cash)

            print("Simulation %s, Target Private %s, Redemption Rate %s" % (iteration,target_private,redemption_rate))
            #get the time0 initial funds
            scale_by_contrib = commit_max

            if init_funds != 0:
                if target_base == "Dollar":
                    funds0 = fund_selector(init_funds, ptf_size, target_base, min(target_private,ptf_size), init_age, funddata,
                                        select_year,r_start_year, final_liquidation=True,first_quarter=True,scale_by_contrib=scale_by_contrib,year_limit=min(ptf_life,10),replacement=replacement)
                elif target_base == "Percentage":
                    funds0 = fund_selector(init_funds, ptf_size, target_base, min(1,target_private*(1+overcommit_amt)),  init_age, funddata,
                                        select_year,r_start_year, final_liquidation=True,first_quarter=True,scale_by_contrib=scale_by_contrib,year_limit=min(ptf_life,10),replacement=replacement)
                elif target_base == "Target":
                    funds0 = fund_selector(init_funds, ptf_size, "Percentage", min(1,target_private*(1+overcommit_amt)),  init_age, funddata,
                                            select_year,r_start_year, final_liquidation=True,first_quarter=True,scale_by_contrib=scale_by_contrib,year_limit=min(ptf_life,10),replacement=replacement)
                
                consolidated_cashflows = funds0[0]

                fundlist = {r_start_year:funds0[1]}
                cashflowslist = {r_start_year:funds0[0]}

            """Run a portfolio and allocate to it when the % of Privates falls below target"""
            portfolio = Portfolio("My Portfolio")

            #creating assets for the initial portfolio
            #Asset parameters are: name, price0, quantity, return, volatility, income, income_volatility,
            # return_stream reinvestment_rate, liquidity, sub_period, prorate, periodicity
            #name, cashflows, reinvestment_rate=0,liquidity=999, prorate=1,age=0,periodicity="Q"
            assets = []
            Cash = Asset('Cash', 1, ptf_size*target_cash, 0.0, 0.0, 0.00, 0.0, {}, "Liquid",1, 1, 1,1,"Q") #not used in this case
            assets.append(Cash)
            target_weights = [target_cash]
            for asset in public_assets:
                assets.append(asset)
                if target_base == "Dollar":
                    target_weights.append(target_public/ptf_size)
                else:
                    target_weights.append(target_public)
            
            if init_funds != 0:
                Privates = Fund('Private Sleeve', cashflows=consolidated_cashflows) 
                assets.append(Privates)
                if target_base == "Dollar":
                    target_weights.append(target_private/ptf_size)
                else:
                    target_weights.append(target_private)
            
            asset_deviations = target_weights #assume any deviation is acceptable
            liquidity_ranks = list(range(1, len(assets)+1)) #assume all assets are equally liquid

            portfolio.set_assets(assets, target_weights,asset_deviations,liquidity_ranks)

            simulation_results  = {}
            startdate = '3/31/'+str(r_start_year)
            portdetails = {'private_percent':[]}
            growfirst = False
            failflag = 0
            for year in range(ptf_life):
                print("Year %s" % (year+1))
                time_periods = 4 #quarterly
                redemptions = {}
                if year in red_years:             
                    redemptions = {0:redemption_rate,
                                    1:redemption_rate,
                                    2:redemption_rate,
                                    3:redemption_rate} #redeem one time per year in the last quarter
                if year > 0:
                    growfirst = True

                if select_year == False:
                    historical = False
                else:
                    historical = True

                simulation_results[year] = portfolio.timecycle_drawdown(time_periods, redemptions=redemptions, red_base =red_base,subscriptions={}, red_max=True, max_pct=redemption_rate,
                                                            sub_max=False, smax_pct=0, verbose=False, historical=historical, startdate=startdate, periodicity="Q",earlybreak=True,growfirst=growfirst)
                
                #if earlybreak, then break this loop as well
                if simulation_results[year][5] == 1:
                    failflag = 1
                    break

                r_start_year += 1
                
                if year == ptf_life-1:
                    break
                    
                if init_funds == 0:
                    portdetails['private_percent'].append(0)
                else:
                    #Caluculate the private allocation percentage #Use calculate_private_exposure if we want to be more conservative
                    private_percent = portfolio.calculate_private_exposure()
                    #Use the below for allocation deciscions based on NAV exposure
                    #private_percent = 1 - portfolio.calculate_portfolio_weights()['Cash'] - portfolio.calculate_portfolio_weights()['Public Sleeve']
                    portdetails['private_percent'].append(private_percent)
                    
                    if commit_max:
                        #commit only the lesser of cash or the target
                        liquid_available = (1-private_percent)*portfolio.calculate_portfolio_value() #liquid available for new commitments
                        if target_base == "Dollar":
                            target_private_new = min(liquid_available,target_private)
                        elif target_base == "Percentage":
                            target_private_new = min(target_private*(1+overcommit_amt),(1-private_percent))
                        elif target_base == "Target":
                            target_private_new = min(max(target_private*(1+overcommit_amt)-private_percent,0),(1-private_percent))
                    else:
                        target_private_new = target_private 

                    #if no quartiles are specified, then select funds randomly from entire pool
                    if (q1funds is None) and (q2funds is None) and (q3funds is None) and (q4funds is None):
                        if (target_base == "Dollar") or (commit_max):
                            consolidated_cashflows, fundlist[r_start_year] = fund_selector(new_commits, portfolio.calculate_portfolio_value(), target_base,
                                                            target_private_new, init_age, funddata,select_year,r_start_year,True,True,scale_by_contrib,year_limit=min(ptf_life-year,10),replacement=replacement)
                        elif (target_base == "Percentage"):
                            consolidated_cashflows, fundlist[r_start_year] = fund_selector(new_commits, portfolio.calculate_portfolio_value(), target_base,
                                                        target_private_new*(1+overcommit_amt), init_age, funddata,select_year,r_start_year, True,True,scale_by_contrib,year_limit=min(ptf_life-year,10),replacement=replacement)
                        elif (target_base == "Target"):
                            consolidated_cashflows, fundlist[r_start_year] = fund_selector(new_commits, portfolio.calculate_portfolio_value(), target_base,
                                                        max(target_private_new*(1+overcommit_amt)-private_percent,0), init_age, funddata,select_year,r_start_year, True,True,scale_by_contrib,year_limit=min(ptf_life-year,10),replacement=replacement)
                    else:
                        q_match = {q1funds:'Q1',q2funds:'Q2',q3funds:'Q3',q4funds:'Q4'}
                        defined = {}
                        undefined = {}
                        for quartile in [q1funds,q2funds,q3funds,q4funds]:
                            if quartile is not None:
                                #Add the quartiles to those defined % and those left blank
                                defined[q_match[quartile]] = quartile
                            else:
                                undefined[q_match[quartile]] = quartile
                        
                        c_funds = []
                        c_list = []
                        pct_undef = 1

                        if (target_base == "Dollar") or (commit_max):
                            for quartile in defined:
                                funds0 = fund_selector(int(new_commits*defined[quartile]), portfolio.calculate_portfolio_value(), target_base,
                                                        target_private_new*defined[quartile],init_age,funddata[funddata['quartile']==quartile],
                                                        select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10),replacement=replacement)
                                pct_undef -= defined[quartile]
                                c_funds.append(funds0[0])
                                c_list.append(funds0[1])
                            if pct_undef > 0:
                                selectgroup = []
                                for quartile in undefined:
                                    selectgroup.append(quartile)
                                funds1 = fund_selector(int(new_commits*pct_undef), portfolio.calculate_portfolio_value(), target_base,
                                                        target_private_new*pct_undef,init_age,funddata[funddata['quartile'].isin(selectgroup)],
                                                        select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10),replacement=replacement)
                                c_funds.append(funds1[0])
                                c_list.append(funds1[1])
                        elif (target_base == "Percentage"):
                            for quartile in defined:
                                funds0 = fund_selector(int(new_commits*defined[quartile]), portfolio.calculate_portfolio_value(),target_base,
                                                        target_private*(1+overcommit_amt)*defined[quartile],init_age,funddata[funddata['quartile']==quartile],
                                                        select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10),replacement=replacement)
                                pct_undef -= defined[quartile]
                                c_funds.append(funds0[0])
                                c_list.append(funds0[1])
                            if pct_undef > 0:
                                selectgroup = []
                                for quartile in undefined:
                                    selectgroup.append(quartile)
                                funds1 = fund_selector(int(new_commits*pct_undef), portfolio.calculate_portfolio_value(), target_base,
                                                        target_private*(1+overcommit_amt)*pct_undef,init_age,funddata[funddata['quartile'].isin(selectgroup)],
                                                        select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10),replacement=replacement)
                                c_funds.append(funds1[0])
                                c_list.append(funds1[1])
                        elif (target_base == "Target"):
                            for quartile in defined:
                                funds0 = fund_selector(int(new_commits*defined[quartile]), portfolio.calculate_portfolio_value(), target_base,
                                                        max(target_private*(1+overcommit_amt)-private_percent,0)*defined[quartile],init_age,funddata[funddata['quartile']==quartile],
                                                        select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10),replacement=replacement)
                                pct_undef -= defined[quartile]
                                c_funds.append(funds0[0])
                                c_list.append(funds0[1])
                            if pct_undef > 0:
                                selectgroup = []
                                for quartile in undefined:
                                    selectgroup.append(quartile)
                                funds1 = fund_selector(int(new_commits*pct_undef), portfolio.calculate_portfolio_value(), target_base,
                                                        max(target_private*(1+overcommit_amt)-private_percent,0)*pct_undef,init_age,funddata[funddata['quartile'].isin(selectgroup)],
                                                        select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10),replacement=replacement)
                                c_funds.append(funds1[0])
                                c_list.append(funds1[1])

                        consolidated_cashflows = combine_cashflows(c_funds)

                        fundlist[r_start_year] = np.concatenate(c_list, axis=0)

                    cashflowslist[r_start_year] = consolidated_cashflows
                
                    portfolio.add_asset(Fund(str("Private " +str(year+1)), cashflows=consolidated_cashflows),.1,.1,3)
                
                startdate = (datetime.strptime(startdate, '%m/%d/%Y') + relativedelta(months=12)).strftime('%m/%d/%Y')                
                # Add 12 months to the start date
            
            #reset start year
            r_start_year = t_start_year
            #reset the public assets
            for asset in public_assets:
                asset.reset_asset()

            overall_sims[(target_private,redemption_rate)] = simulation_results
            if init_funds == 0:
                fundlist = ['N/A']
                cashflowslist = ['N/A']
            overall_funds[(target_private,redemption_rate)] = (fundlist,cashflowslist)
            overall_fail[(target_private,redemption_rate)] = failflag
    return overall_sims, overall_funds, overall_fail

def single_simulation_custompacing(iteration,init_funds,ptf_size,ptf_life,init_age,new_commits,
                          funddata,target_base,target_private,target_cash,redrate,red_base,red_years=[],
                          overcommit_amt=0,public_assets=[],select_year = False,d_year=None, 
                          commit_max=False,q1funds=None,q2funds=None,q3funds=None,q4funds=None,replacement=True):
    """
    Run a single simulation of the portfolio time cycle.

    Args:
        iteration (int): Simulation iteration number.
        init_funds (int): Number of initial funds to select.
        ptf_size (float): Initial portfolio size.
        start_year (int): Starting year for the funds in the simulation.
        ptf_life (int): Number of years to run the simulation.
        init_age (int): Initial age of the funds.
        new_commits (int): Number of new commitments to add each year.
        funddata (pd.DataFrame): DataFrame containing fund data. 
        private_range (dict): Private Range over time 
        target_cash (float): Target cash allocation.
        redrate (array): Redemption rate range. ex. np.arange(0.10, 0.12, 0.01)
        red_base (str): Redemption base. ex. "Fixed","NAV","Dist"
        red_years (list): Redemption years. ex. [1,2,3,4]
        overcommit_amt (float): Overcommitment amount. (e.g.,  total exposure= target (1+x%)*Portfolio)
        public_assets (dict): list of public asset objects.
        select_year (bool): If True, select funds based on the d_year.
        d_year (int): start year.
        commit_max (bool): If True, commit the maxmimum of what can be committed
        q1 to q4 funds (int): Number of funds to select from each quartile, 4 being the top.
    Returns:
        dict: Simulation results for each target private allocation and redemption rate.
    """    
    overall_sims = {}
    overall_funds = {}
    overall_fail = {}

    # Iterate over target private allocations and redemption rates
    
    if d_year == None:
        r_start_year = 2000
        t_start_year = r_start_year
    else:
        r_start_year = d_year
        t_start_year = d_year
    
    target_cash = np.round(target_cash, 2)

    if target_base[0] == "Percentage" or target_base[0] == "Target":
        target_private[0] = np.round(target_private[0], 2)
        target_public = 1 - target_cash #- target_private[0] 
    elif target_base[0] == "Dollar":
        target_public = ptf_size*(1-target_cash) #(ptf_size - target_private[0])*

    #print("Simulation %s" % (iteration))
    #get the time0 initial funds
    #fund_selector(init_funds, ptf_size, target_base, target_private, init_age, data,start_year=None, final_liquidation=True,first_quarter=True,scale_by_contrib=False)
    scale_by_contrib = commit_max

    if init_funds != 0:
        if target_base[0] == "Dollar":
            funds0 = fund_selector(init_funds, ptf_size, target_base[0], min(target_private[0],ptf_size), init_age, funddata,
                                    select_year,r_start_year, final_liquidation=True,first_quarter=True,scale_by_contrib=scale_by_contrib,year_limit=min(ptf_life-1,10-1),replacement=replacement)
        elif target_base[0] == "Percentage":
            funds0 = fund_selector(init_funds, ptf_size, target_base[0], min(1,target_private[0]*(1+overcommit_amt)),  init_age, funddata,
                                    select_year,r_start_year, final_liquidation=True,first_quarter=True,scale_by_contrib=scale_by_contrib,year_limit=min(ptf_life-1,10-1),replacement=replacement)
        elif target_base[0] == "Target":
            funds0 = fund_selector(init_funds, ptf_size, "Percentage", min(1,target_private[0]*(1+overcommit_amt)),  init_age, funddata,
                                    select_year,r_start_year, final_liquidation=True,first_quarter=True,scale_by_contrib=scale_by_contrib,year_limit=min(ptf_life-1,10-1),replacement=replacement)
        
        consolidated_cashflows = funds0[0]

        fundlist = {r_start_year:funds0[1]}
        cashflowslist = {r_start_year:funds0[0]}

    """Run a portfolio and allocate to it when the % of Privates falls below target"""
    portfolio = Portfolio("My Portfolio")

    #creating assets for the initial portfolio
    #Asset parameters are: name, price0, quantity, return, volatility, income, income_volatility,
    # return_stream reinvestment_rate, liquidity, sub_period, prorate, periodicity
    #name, cashflows, reinvestment_rate=0,liquidity=999, prorate=1,age=0,periodicity="Q"
    assets = []
    Cash = Asset('Cash', 1, ptf_size*target_cash, 0.0, 0.0, 0.00, 0.0, {}, "Liquid",1, 1, 1,1,"Q") #not used in this case
    assets.append(Cash)
    target_weights = [target_cash]
    for asset in public_assets:
        assets.append(asset)
        if target_base[0] == "Dollar":
            target_weights.append(target_public/ptf_size)
        else:
            target_weights.append(target_public)
    
    if init_funds != 0:
        Privates = Fund('Private Sleeve', cashflows=consolidated_cashflows) 
        assets.append(Privates)
        if target_base[0] == "Dollar":
            target_weights.append(0) #target_private[0]/ptf_size)
        else:
            target_weights.append(target_private[0])
    
    asset_deviations = target_weights #assume any deviation is acceptable
    liquidity_ranks = list(range(1, len(assets)+1)) #assume all assets are equally liquid

    portfolio.set_assets(assets, target_weights,asset_deviations,liquidity_ranks)

    simulation_results  = {}
    startdate = '3/31/'+str(r_start_year)
    portdetails = {'private_percent':[]}
    growfirst = False
    failflag = 0
    for year in range(1,ptf_life+1):
        #print("Year %s" % (year))
        time_periods = 4 #quarterly
        redemptions = {}
        if year in red_years:             
            redemptions = {0:redrate[0],
                            1:redrate[0],
                            2:redrate[0],
                            3:redrate[0]} 
            
        if year > 1:
            growfirst = True

        if select_year == False:
            historical = False
        else:
            historical = True

        simulation_results[year-1] = portfolio.timecycle_drawdown(time_periods, redemptions=redemptions, red_base =red_base, subscriptions={}, red_max=True, max_pct=redrate[0],
                                                    sub_max=False, smax_pct=0, verbose=False, historical=historical, startdate=startdate, periodicity="Q",earlybreak=True,growfirst=growfirst)
        
        #if earlybreak, then break this loop as well
        if simulation_results[year-1][5] == 1:
            failflag = 1
            break

        r_start_year += 1
        
        if year == ptf_life:
            break
    
        if init_funds == 0:
            portdetails['private_percent'].append(0)
        else:
            #Caluculate the private allocation percentage
            #Use calculate_private_exposure if we want to be more conservative
            private_percent = portfolio.calculate_private_exposure()
            #Use the below for allocation deciscions based on NAV exposure
            #private_percent = 1 - portfolio.calculate_portfolio_weights()['Cash'] - portfolio.calculate_portfolio_weights()['Public Sleeve']
            portdetails['private_percent'].append(private_percent)
            
            if target_private[year] != 0:
                if commit_max:
                    #commit only the lesser of cash or the target
                    liquid_available = (1-private_percent)*portfolio.calculate_portfolio_value() #liquid available for new commitments
                    if target_base[year] == "Dollar":
                        target_private_new = min(liquid_available,target_private[year])
                    elif target_base[year] == "Percentage":
                        target_private_new = min(target_private[year]*(1+overcommit_amt),(1-private_percent))
                    elif target_base[year] == "Target":
                        target_private_new = min(max(target_private[year]*(1+overcommit_amt)-private_percent,0),(1-private_percent))
                else:
                    target_private_new = target_private[year]

                #if no quartiles are specified, then select funds randomly from entire pool
                if (q1funds is None) and (q2funds is None) and (q3funds is None) and (q4funds is None):
                    if (target_base[year] == "Dollar") or (commit_max):
                        consolidated_cashflows, fundlist[r_start_year] = fund_selector(new_commits, portfolio.calculate_portfolio_value(), target_base[year],
                                                        target_private_new, init_age, funddata,select_year,r_start_year,True,True,scale_by_contrib,year_limit=min(ptf_life-year,10)-1,replacement=replacement)
                    elif (target_base[year] == "Percentage"):
                        consolidated_cashflows, fundlist[r_start_year] = fund_selector(new_commits, portfolio.calculate_portfolio_value(), target_base[year],
                                                        target_private_new*(1+overcommit_amt), init_age, funddata,select_year,r_start_year, True,True,scale_by_contrib,year_limit=min(ptf_life-year,10)-1,replacement=replacement)
                    elif (target_base[year] == "Target"):
                        consolidated_cashflows, fundlist[r_start_year] = fund_selector(new_commits, portfolio.calculate_portfolio_value(), target_base[year],
                                                        max(target_private_new*(1+overcommit_amt)-private_percent,0), init_age, funddata,select_year,r_start_year, True,True,scale_by_contrib,year_limit=min(ptf_life-year,10)-1,replacement=replacement)
                else:
                    q_match = {q1funds:'Q1',q2funds:'Q2',q3funds:'Q3',q4funds:'Q4'}
                    defined = {}
                    undefined = {}
                    for quartile in [q1funds,q2funds,q3funds,q4funds]:
                        if quartile is not None:
                            #Add the quartiles to those defined % and those left blank
                            defined[q_match[quartile]] = quartile
                        else:
                            undefined[q_match[quartile]] = quartile
                    
                    c_funds = []
                    c_list = []
                    pct_undef = 1

                    if (target_base[year] == "Dollar") or (commit_max):
                        for quartile in defined:
                            funds0 = fund_selector(int(new_commits*defined[quartile]), portfolio.calculate_portfolio_value(), target_base[year],
                                                    target_private_new*defined[quartile],init_age,funddata[funddata['quartile']==quartile],
                                                    select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10)-1,replacement=replacement)
                            pct_undef -= defined[quartile]
                            c_funds.append(funds0[0])
                            c_list.append(funds0[1])
                        if pct_undef > 0:
                            selectgroup = []
                            for quartile in undefined:
                                selectgroup.append(quartile)
                            funds1 = fund_selector(int(new_commits*pct_undef), portfolio.calculate_portfolio_value(), target_base[year],
                                                    target_private_new*pct_undef,init_age,funddata[funddata['quartile'].isin(selectgroup)],
                                                    select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10)-1,replacement=replacement)
                            c_funds.append(funds1[0])
                            c_list.append(funds1[1])
                    elif (target_base[year] == "Percentage"):
                        for quartile in defined:
                            funds0 = fund_selector(int(new_commits*defined[quartile]), portfolio.calculate_portfolio_value(),target_base[year],
                                                    target_private[year]*(1+overcommit_amt)*defined[quartile],init_age,funddata[funddata['quartile']==quartile],
                                                    select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10)-1,replacement=replacement)
                            pct_undef -= defined[quartile]
                            c_funds.append(funds0[0])
                            c_list.append(funds0[1])
                        if pct_undef > 0:
                            selectgroup = []
                            for quartile in undefined:
                                selectgroup.append(quartile)
                            funds1 = fund_selector(int(new_commits*pct_undef), portfolio.calculate_portfolio_value(), target_base[year],
                                                    target_private[year]*(1+overcommit_amt)*pct_undef,init_age,funddata[funddata['quartile'].isin(selectgroup)],
                                                    select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10)-1,replacement=replacement)
                            c_funds.append(funds1[0])
                            c_list.append(funds1[1])
                    elif (target_base[year] == "Target"):
                        for quartile in defined:
                            funds0 = fund_selector(int(new_commits*defined[quartile]), portfolio.calculate_portfolio_value(), target_base[year],
                                                    max(target_private[year]*(1+overcommit_amt)-private_percent,0)*defined[quartile],init_age,funddata[funddata['quartile']==quartile],
                                                    select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10)-1,replacement=replacement)
                            pct_undef -= defined[quartile]
                            c_funds.append(funds0[0])
                            c_list.append(funds0[1])
                        if pct_undef > 0:
                            selectgroup = []
                            for quartile in undefined:
                                selectgroup.append(quartile)
                            funds1 = fund_selector(int(new_commits*pct_undef), portfolio.calculate_portfolio_value(), target_base[year],
                                                    max(target_private[year]*(1+overcommit_amt)-private_percent,0)*pct_undef,init_age,funddata[funddata['quartile'].isin(selectgroup)],
                                                    select_year,r_start_year,True,True,True,year_limit=min(ptf_life-year,10)-1,replacement=replacement)
                            c_funds.append(funds1[0])
                            c_list.append(funds1[1])

                    consolidated_cashflows = combine_cashflows(c_funds)

                    fundlist[r_start_year] = np.concatenate(c_list, axis=0)

                cashflowslist[r_start_year] = consolidated_cashflows
        
                portfolio.add_asset(Fund(str("Private " +str(year+1)), cashflows=consolidated_cashflows),.1,.1,3)
        
        startdate = (datetime.strptime(startdate, '%m/%d/%Y') + relativedelta(months=12)).strftime('%m/%d/%Y')                
        # Add 12 months to the start date
    
    #reset start year
    r_start_year = t_start_year
    #reset the public assets
    for asset in public_assets:
        asset.reset_asset()

    overall_sims[(target_private[0],redrate[0])] = simulation_results
    if init_funds == 0:
        fundlist = ['N/A']
        cashflowslist = ['N/A']
    overall_funds[(target_private[0],redrate[0])] = (fundlist,cashflowslist)
    overall_fail[(target_private[0],redrate[0])] = failflag
    return overall_sims, overall_funds, overall_fail


def process_simulations(num_sims,init_funds,ptf_size,ptf_life,init_age,new_commits,
                          funddata,target_base,private_range,target_cash,redrate,red_base,red_years,overcommit_amt,public_assets,select_year = False,d_year=None, 
                          commit_max=False,q1funds=None,q2funds=None,q3funds=None,q4funds=None,public_returns=None,replacement=True):
    """Case 1: Single Time Period, Single Asset Class, New Private Markets Program:
    Start with 1bn of Total Assets, Create 100 random ptfs
    All ptfs start in 2009. 
    Each ptf selects 10 random funds each year
    Assume that 60% of funds are chosen from ex-post top quartile IRR, and 40% are random across entire universe of funds in that vintage
    red_base (str): Redemption base; either "Fixed", "NAV", "Dist"
    """
    mc_sims = {}
    mc_funds = {}
    mc_fail = {}

    if type(private_range) != list and type(private_range) != pd.DataFrame:
        private_range = [private_range]
    else:
        private_range = target_base_range(private_range,ptf_life)
        #print(private_range.columns)
    if type(redrate) != list and type(redrate) != dict:
        redrate = [redrate]

    for i in range(num_sims):
        #print("Simulation " + str(i))
        
        if type(private_range) == list:
            simresults, fundrecord,failflag = run_single_simulation(i, init_funds, ptf_size, ptf_life, init_age,new_commits, funddata, target_base, private_range,0,
                            redrate,red_base,red_years,overcommit_amt,public_assets,select_year,d_year, commit_max,replacement=replacement)
        elif type(private_range) != list:
            target_base = private_range['Target Base']
            target_private = private_range['Target Private']
            simresults, fundrecord,failflag = single_simulation_custompacing(i, init_funds, ptf_size, ptf_life, init_age,new_commits, funddata, target_base, target_private, 0,
                            redrate,red_base,red_years,overcommit_amt,public_assets,select_year,d_year, commit_max,q1funds=q1funds,q2funds=q2funds,q3funds=q3funds,q4funds=q4funds,replacement=replacement)
        
        finalcashflows = private_combine(simresults[list(simresults.keys())[0]])[0]
        finalcashflows['Subscription'][0] = ptf_size #set the initial subscription to the portfolio size
        irrs = pd.DataFrame(mini_irr(list(finalcashflows['Redemption']-finalcashflows['Subscription']),list(finalcashflows['Total']),list(finalcashflows['Date']),"LMK").items(),columns=["Date","Total Ptf IRR"])
        pvtirr = pd.DataFrame(mini_irr(list(finalcashflows['Distributions']-finalcashflows['Contributions']),list(finalcashflows['NAV']),list(finalcashflows['Date']),"LMK").items(),columns=["Date","Private IRR"])
        finalcashflows = finalcashflows.merge(irrs,how='left',on='Date')
        finalcashflows = finalcashflows.merge(pvtirr, how='left',on='Date')

        twr = mini_twr(list(finalcashflows['Redemption']-finalcashflows['Subscription']),list(finalcashflows['Total'])).rename(columns={'TWRR':'Ptf TWRR','hp_return':'Ptf HP Ret.'})
        pvttwr = mini_twr(list(finalcashflows['Distributions']-finalcashflows['Contributions']),list(finalcashflows['NAV'])).rename(columns={'TWRR':'Private TWRR','hp_return':'Private HP Ret.'})
        pubtwr = mini_twr(list(finalcashflows['Contributions']-finalcashflows['Distributions']),list(finalcashflows['Public Sleeve'])).rename(columns={'TWRR':'Public TWRR','hp_return':'Public HP Ret.'})
        finalcashflows = pd.concat([finalcashflows, twr],axis = 1)
        finalcashflows = pd.concat([finalcashflows, pvttwr],axis = 1)
        finalcashflows = pd.concat([finalcashflows, pubtwr],axis = 1)

        # if public_returns is not None:
        #     finalcashflows = finalcashflows.merge(public_returns,how='left',on='Date')

        #     #For evm, the cashflow file must include Date, Contribution, Distribution and NAV columns.
        #     #Calculate EVM of the total portfolio
        #     evm_subset = finalcashflows[['Date','Subscription','Redemption','Total','Benchmark']]
        #     evm = EVM(evm_subset, cont_col="Subscription", dist_col="Redemption", nav_col="Total", scaling_factor="proportional")
        #     result = evm.calculate()[['Cumulative_Excess_Value','TVPI']].rename(columns={'Cumulative_Excess_Value':'Ptf Cum. EVM','TVPI':'Ptf TVPI'})
        #     finalcashflows = pd.concat([finalcashflows, result], axis=1)

        #     #drop benchmark_return

        #     evm_sub2 = finalcashflows[['Date','Contributions','Distributions','NAV','Benchmark']]
        #     evm2 = EVM(evm_sub2, cont_col="Contributions", dist_col="Distributions", nav_col="NAV", scaling_factor="proportional")
        #     result2 = evm2.calculate()[['Cumulative_Excess_Value','TVPI']].rename(columns={'Cumulative_Excess_Value':'Private Cum. EVM','TVPI':'Private TVPI'})
        #     finalcashflows = pd.concat([finalcashflows, result2], axis=1)

        finalcashflows['Cum. Cont'] = finalcashflows['Contributions'].cumsum()
        finalcashflows['Cum. Dist'] = finalcashflows['Distributions'].cumsum()

        mc_sims[i] = finalcashflows
        mc_funds[i] = fundrecord
        mc_fail[i] = failflag

    return mc_sims, mc_funds, mc_fail

def sim_average(mc_sims):
    avg_total_df = {'Total NAV':[],
                'Total Com. Not Called':[],
                'Total':[],
                'Contributions':[],
                'Distributions':[],
                'Ptf TWRR':[],
                'Private IRR':[]}

    for j in avg_total_df.keys():
        temp = {}
        for i in mc_sims.keys():
            temp[i] = mc_sims[i][j]
        avg_total_df[j] = pd.DataFrame(temp).mean(axis=1)
    avg_total_df = pd.DataFrame(avg_total_df).reset_index(drop=True)
    avg_total_df['Private %'] = avg_total_df['Total NAV']/avg_total_df['Total']
    avg_total_df['Cum. Cont'] = avg_total_df['Contributions'].cumsum()
    avg_total_df['Cum. Dist'] = avg_total_df['Distributions'].cumsum()
    avg_total_df['Private TVPI'] = (avg_total_df['Total NAV'] + avg_total_df['Cum. Dist'])/avg_total_df['Cum. Cont']
    avg_total_df['Ptf TVPI'] = (avg_total_df['Total'])/avg_total_df['Total'][0]
    return avg_total_df

def sim_combine(mc_sims):
    total_dfs = {'Total NAV':pd.DataFrame(),
        'Total Com. Not Called':pd.DataFrame(),
        'Total':pd.DataFrame(),
        'Contributions':pd.DataFrame(),
        'Distributions':pd.DataFrame(),
        'Ptf TWRR':pd.DataFrame(),
        'Private IRR':pd.DataFrame(),
        'Date':pd.DataFrame()}
    for j in total_dfs.keys():
        temp = {}
        for i in mc_sims.keys():
            temp["sim "+str(i)] = mc_sims[i][j]
        total_dfs[j] = pd.DataFrame(temp)
    return total_dfs

def multiasset_sim(ptf_size,target_private,init_funds,target_base,new_commits,overcommit_amt,init_age,dfdata,target_public,Publics,select_year,r_start_year,ptf_life,
                   red_base, redemption_rate,commitment_period,scale_by_contrib=False,replacement=False):
    """
    ptf_size = portfolio size
    target_private = target private allocation. keys are strings (dictionary with each ASSET CLASS - specified in Asset objects and its target)
    init_funds = initial number of funds to select
    target_base = "Dollar", "Percentage" or "Target"
    new_commits = number of new commitments to add each year
    overcommit_amt = overcommitment amount (e.g.,  total exposure= target (1+x%)*Portfolio)
    init_age = initial fund age (used in fund selection)
    dfdata = data of cashflows
    target_public = dictionary with target public allocations relative to each other. Keys are asset objects 
                    (e.g., even if portfolio is 50% public, and we still want public targets to add to 100%) 
    Publics = list of public asset objects, e.g. [Cash, Publics]
    select_year = if True, select funds based on the d_year.
    r_start_year = start year
    ptf_life = number of years to run the simulation
    red_base = redemption base, either "Fixed", "NAV", "Dist"
    redemption_rate = redemption rate to use in the simulation
    commitment_period = How many times to commit to new funds each year (can only take 1 = each quarter, 2 = semi-annually, 4 = annually)
    """
    #Calculate the total private percent from the target_private dictionary
    total_private = sum(target_private.values())
    total_public = 1 - total_private
    asset_class_list = list(target_private.keys())

    #Initialize the private funds from the specified target allocations and the data provided
    privateassets = []
    funddata = {}
    for asset in asset_class_list:
        funddata[asset] = dfdata[dfdata['strategy']==asset].reset_index(drop=True)
        fundcfs = fund_selector(1, ptf_size, "Percentage", min(init_funds,target_private[asset]*(1+overcommit_amt)),  init_age, funddata[asset],
                                            select_year,r_start_year, final_liquidation=True,first_quarter=True,scale_by_contrib=scale_by_contrib,year_limit=min(ptf_life,10),replacement=replacement)
        privateassets.append(Fund(asset, cashflows = fundcfs[0],asset_class = asset))

    #Scale the public assets
    for asset in Publics:
        #overwrite what is originally specified to be the size to the target_public * portfolio size * amount of publics
        asset.quantity = ptf_size*target_public[asset]
    
    #Initialize a dummy cash
    Cash = Asset('Cash',1, 0,0.01,0, .000, 0, {}, "Liquid", 1,1, 1,1,"Q")
        
    overall_sims = {}
    overall_funds = {}
    overall_fail = {}
    
    #Record each year of cashflows for each asset class
    cashflowslist = {r_start_year:{}}
    for i in privateassets:
        cashflowslist[r_start_year][i.asset_class] = i.cashflows
    
    """Initialize and run a portfolio and allocate to it when the % of Privates falls below target"""
    portfolio = Portfolio("My Portfolio")

    assets = []
    assets.append(Cash)
    target_weights = [0] #if we initialize cash within this function
    for asset in Publics:
        assets.append(asset)
        target_weights.append(target_public[asset]) #Publics will serve as all the liquidity

    for fund in privateassets:
        assets.append(fund)
        target_weights.append(0) 
    print(sum(target_weights))

    asset_deviations = target_weights #assume any deviation is acceptable
    liquidity_ranks = list(range(1, len(assets)+1)) #assume all assets are equally liquid
    portfolio.set_assets(assets, target_weights,asset_deviations,liquidity_ranks)

    year_range = range(r_start_year, r_start_year + ptf_life) 

    simulation_results  = {}
    startdate = '3/31/'+str(r_start_year)

    growfirst = False
    simulation_results  = {} #Hold output from timecycle_drawdown for each year

    #Run the simulations, run for each quarter and year in the portfolio life
    startdate = '3/31/'+str(r_start_year) #When we start the portfolio
    portdetails = {} #Contains details around balances per period
    for asset_class in portfolio.asset_classes:
        portdetails[asset_class] = []

    growfirst = True #Is there growth in the first period?
    for year in year_range:
        for quarter in range(4): #Commit to new funds / run calculations/ 1 = each quarter, 2 = semi-annually, 4 = annually
            print(str(year)+" Q"+str(quarter+1))
            time_periods = 1 #If quarterly, then time_periods = 1, run the simulation/commitment pacing 1 quarter at a time. If annually, time periods = 4, run the commitment pacing once per 4 quarters
            redemptions = {0:redemption_rate}
            historical = select_year #True/False use Simulated vs Historical return data. Selecting a year will use historical data, not selecting a particular year will use simulated data

            simulation_results[str(year)+" Q"+str(quarter+1)] = portfolio.timecycle_drawdown(time_periods, redemptions=redemptions, red_base =red_base,subscriptions={}, red_max=True, max_pct=redemption_rate,
                                                                    sub_max=False, smax_pct=0, verbose=False, historical=historical, startdate=startdate, periodicity="Q",earlybreak=True,growfirst=growfirst)

            if quarter == 3: #end of year
                r_start_year += 1
                if year == ptf_life-1: #In the final year, do not need to select any more private funds
                    break

            #Caluculate the private allocation percentage for each asset class
            asset_class_exposures = portfolio.calculate_asset_classes_exposure()
            asset_class_values = portfolio.calculate_asset_classes_value()
            for asset_class in portfolio.asset_classes:
                private_percent = asset_class_exposures[asset_class]
                #Use the below for allocation deciscions based on NAV exposure
                #private_percent = 1 - portfolio.calculate_portfolio_weights()['Cash'] - portfolio.calculate_portfolio_weights()['Public Sleeve']
                private_values = asset_class_values[asset_class]
                portdetails[asset_class].append(private_values)

                #Check to see if we should be committing to new funds
                if quarter % commitment_period == 0: 
                    #Get new cashflows for the next period
                    cashflowslist[r_start_year] = {}
                    if asset_class in target_private:                    
                        #since we are not using commit_max for this example
                        target_private_new = target_private[asset_class]
                        
                        print(f"Private Percent for {asset_class} in year {year}: {private_percent:.2%}, Target: {target_private_new:.2%}")
                        if (target_base == "Target"):
                            consolidated_cashflows = fund_selector(new_commits, portfolio.calculate_portfolio_value(), target_base,
                                                        max(target_private_new*(1+overcommit_amt)-private_percent,0), init_age, 
                                                        funddata[asset_class],select_year,r_start_year, True,True,scale_by_contrib,year_limit=min(2023-year,10),replacement=replacement)[0] #ignore fundlist for now
                        cashflowslist[r_start_year][asset_class] = consolidated_cashflows
                        #Add the new cashflows to the portfolio
                        portfolio.add_asset(Fund(asset_class+str(r_start_year)+str(quarter), cashflows=consolidated_cashflows, asset_class=asset_class),0,0,3)

            startdate = (datetime.strptime(startdate, '%m/%d/%Y') + relativedelta(months=int(3))).strftime('%m/%d/%Y')
        
    return simulation_results, portdetails

