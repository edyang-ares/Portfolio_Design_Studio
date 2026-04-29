from typing import List
from datetime import date
import numpy as np
import math

DaysPerYear = 365
SearchStepMult = 2
ExcelMax = 1.79769313486231 * 10 ** 308
ExcelMIn = 4.94065645841246 * 10 ** -324
PrecComp = 6
MAXFLOAT = 1.79769313486231 * 10 ** 308

""" def PosAdd(Values):
    # Returns the sum of all of the positive elements of values
    count = len(Values)
    sum = 0
    for i in range(count):
        number = Values[i]
        if isinstance(number, (int, float)):
            if number >= 0:
                sum += number
    return sum

def NegAdd(Values):
    # Returns the sum of all of the negative elements of values
    count = len(Values)
    sum = 0
    for i in range(count):
        number = Values[i]
        if isinstance(number, (int, float)):
            if number <= 0:
                sum += number
    return sum


def XNPV_GS(Discount: float, cashflows: List[float], CFDates: List[date], ValDate: date = None, VDOption: int = 1, flag: bool = False, useFV: bool = False) -> float:
    # VDOption : ValDate Option
    # -1 : only use cashflows before given valuation date
    # 0  : use all cashflows regardless of occurence relative to valuation date
    # +1 : only use cashflows occuring after valuation date (default)
    n = len(cashflows)
    m = len(CFDates)
    if n != m:
        if flag:
            print("The Number of cfs and cfdates must be equal")
        raise ValueError("The Number of cfs and cfdates must be equal")
    if ValDate is None:
        ValDate = CFDates[0]
    if VDOption is None:
        VDOption = 1
    if flag is None:
        flag = False
    FvDate = CFDates[-1]
    lengths = [0] * n
    MinLength = ExcelMax
    MaxLength = 0
    for i in range(n):
        lengths[i] = (CFDates[i] - ValDate).days
        if lengths[i] < MinLength:
            MinLength = lengths[i]
        if lengths[i] > MaxLength:
            MaxLength = lengths[i]
    if MaxLength >= 365:
        epsilon = 1 / (1.7976913486231 * 10 ** 308) ** (365 / (MaxLength))
    else:
        epsilon = 0
    if Discount <= -1 + epsilon and not useFV:
        if flag:
            print("The Discount Rate is too close to -1 for the given time period")
        raise ValueError("The Discount Rate is too close to -1 for the given time period")
    if MinLength <= -365:
        epsilon = 1 / ((1.7976913486231 * 10 ** 308) ** (365 / (MinLength))) - 1
        if Discount > epsilon:
            if flag:
                print("The Discount Rate is too large for the given negative time period")
            raise ValueError("The Discount Rate is too large for the given negative time period")
    if useFV:
        alpha = (1 + Discount)
        output = sum([cashflows[i] * alpha ** ((FvDate - CFDates[i]).days / DaysPerYear) for i in range(n) if lengths[i] * VDOption >= 0])
    else:
        alpha = 1 / (1 + Discount)
        output = sum([cashflows[i] * alpha ** ((CFDates[i] - ValDate).days / DaysPerYear) for i in range(n) if lengths[i] * VDOption >= 0])
    return output
 """
def CalcNPV(CFValues: List[float], CFDates:List[date], discount:int, ValDate: date) -> float:

    MAXFLOAT = 1.79769313486231 * 10 ** 308
    max_length = max([(CFDates[i] - ValDate).days for i in range(len(CFDates))])
    min_length = min([(CFDates[i] - ValDate).days for i in range(len(CFDates))])

    epsilon = 1 / MAXFLOAT ** (365 / max_length) if max_length >= 365 else 0

    if discount <= -1 + epsilon:
        return None
    
    if min_length <= -365:
        epsilon = 1 / MAXFLOAT ** (365 / min_length) - 1
        if discount > epsilon:
            return None
        
    alpha = 1 / (1 + discount)
    output = 0

    for i in range(len(CFValues)):
        if (CFDates[i] - ValDate).days >= 0:
            output += CFValues[i] * alpha ** ((CFDates[i] - ValDate).days / 365)
    
    return output


def SearchLoop(CFValues: List[float], CFDates: List[date], tstep: float, iterations: int, precision: int, Mult: float, limit: float,  ValDate: date = None):
    
    bracketed = False
    dGuess1 = 0
    smallEnough = 0.1 ** precision

    dNPV1 = CalcNPV(CFValues, CFDates, dGuess1, ValDate)

    if abs(dNPV1) < smallEnough:
        return dGuess1
    
    i = 1
    done = False

    while i <= iterations:

        dGuess2 = dGuess1 + tstep
        
        if dGuess2 < limit:
            dGuess2 = limit
        
        dNPV2 = CalcNPV(CFValues, CFDates, dGuess2, ValDate)
        
        if abs(dNPV2) <= smallEnough:
            return dGuess2
        
        if dNPV1 * dNPV2 > 0:
            dGuess1 = dGuess2
            dNPV1 = dNPV2
        elif dNPV1 * dNPV2 < 0:
            tstep = tstep / 2
            bracketed = True
          
        if dGuess2 <= -0.95 and dNPV1 * dNPV2 > 1 and done == False:
            return None
        elif dGuess2 <= -0.95:
            done = True

        if bracketed == False:
            tstep = tstep * Mult
        
        i += 1

    return None

    
    
""" def Derivative(x1val: float, pgSummary: object = None, rBidPrice: object = None, rDR: object = None, rNPV: object = None, CFValues: List[float] = None, CFDates: List[date] = None, ValDate: date = None, j: int = None, switch: bool = False) -> float:
    x2val = x1val + 0.001
    y1val = CalcNPV(pgSummary, rBidPrice, rDR, rNPV, x1val, CFValues, CFDates, ValDate,  j, switch)
    y2val = CalcNPV(pgSummary, rBidPrice, rDR, rNPV, x2val, CFValues, CFDates, ValDate,  j, switch)
    if y1val - y2val > 0:
        return -1
    elif y1val - y2val < 0:
        return 1
    else:
        return 0 """

""" 
def FindMinMax(ax: float, cx: float, min: bool, pgSummary: object = None, rBidPrice: object = None, rDR: object = None, rNPV: object = None, CFValues: List[float] = None, CFDates: List[date] = None, ValDate: date = None, j: int = None, switch: bool = False) -> float:
    bx = (ax + cx) / 2
    fax = CalcNPV(pgSummary, rBidPrice, rDR, rNPV, ax, CFValues, CFDates, ValDate,  j, switch)
    fbx = CalcNPV(pgSummary, rBidPrice, rDR, rNPV, bx, CFValues, CFDates, ValDate,  j, switch)
    fcx = CalcNPV(pgSummary, rBidPrice, rDR, rNPV, cx, CFValues, CFDates, ValDate,  j, switch)
    if (fbx > fax or fbx > fcx) and min:
        return ax
    elif (fbx < fax or fbx < fcx) and not min:
        return ax
    else:
        x0 = ax
        x3 = cx
        if abs(cx - bx) > abs(bx - ax):
            x1 = bx
            x2 = bx + 0.61803399 * (cx - bx)
        else:
            x2 = bx
            x1 = bx - 0.61803399 * (bx - ax)
        f1 = CalcNPV(pgSummary, rBidPrice, rDR, rNPV, x1, CFValues, CFDates, ValDate,  j, switch)
        f2 = CalcNPV(pgSummary, rBidPrice, rDR, rNPV, x2, CFValues, CFDates, ValDate,  j, switch)
        done = False
        while not done:
            if abs(x3 - x0) > 1e-15 * (abs(x1) + abs(x2)):
                if check(f1, f2, min):
                    x0 = x1
                    x1 = x2
                    x2 = 0.61803399 * x1 + 0.38196601 * x3
                    f1 = f2
                    f2 = CalcNPV(pgSummary, rBidPrice, rDR, rNPV, x2, CFValues, CFDates, ValDate,  j, switch)
                else:
                    x3 = x2
                    x2 = x1
                    x1 = 0.61803399 * x2 + 0.38196601 * x0
                    f2 = f1
                    f1 = CalcNPV(pgSummary, rBidPrice, rDR, rNPV, x1, CFValues, CFDates, ValDate,  j, switch)
            else:
                done = True
        if check(f1, f2, min):
            return x2
        else:
            return x1 """


""" def check(f1: float, f2: float, min: bool) -> bool:
    if min:
        return f2 < f1
    else:
        return f2 > f1
     """

    
def PosCase(CFValues: List[float], CFDates: List[date], precision: int, iterations: int = 100) -> float:
    
    step = 0.1
    
    Position_first_non_zero = None
    
    #Set value date to first date of non-zero cashflow
    for i, item in enumerate(CFValues):
        if item != 0:
            Position_first_non_zero = i
            break

    if Position_first_non_zero is None:
        return 0

    ValDate = CFDates[Position_first_non_zero]
        
    
    return SearchLoop(CFValues, CFDates, step, iterations, precision, 2, -1, ValDate)

    
def NegCase(CFValues: List[float], CFDates: List[date], precision: int, iterations: int = 100 ) -> float:
    MAXFLOAT = 1.79769313486231 * 10 ** 308

    step = -0.1

    Position_first_non_zero = None

    #Set value date to first date of non-zero cashflow
    for i, item in enumerate(CFValues):
        if item != 0:
            Position_first_non_zero = i
            break

    if Position_first_non_zero is None:
        return 0

    ValDate = CFDates[Position_first_non_zero]
    MaxLength = max([(CFDates[i] - ValDate).days for i in range(len(CFDates))])
    
    limit = (1 / MAXFLOAT) ** (365 / MaxLength) - 1
    
    limit = max(-0.99, limit)
    
       
    return SearchLoop(CFValues, CFDates, step, iterations, precision, 1, limit, ValDate)
    
"""     
def SimpleCase(Values: List[float], dates: List[date], guess=None, step=None, iterations=None, precision=None) -> float:
    NegValue = 0.0
    NegDate = None
    PosValue = 0.0
    PosDate = None
    doneN = False
    doneP = False
    icount = len(Values)
    answer = 0.0
    
    for i in range(icount):
        number = Values[i]
        if isinstance(number, (int, float)) and not (doneP and doneN):
            if number > 0:
                PosValue = number
                PosDate = dates[i]
                doneP = True
            elif number < 0:
                NegValue = number
                NegDate = dates[i]
                doneN = True
    
    time = (PosDate - NegDate).days / 365.0
    
    answer = (1 + (PosValue + NegValue) / (-NegValue)) ** (1 / time) - 1
    
    return answer """
        
def BG_IRR_array(Values: List[float], dates: List[date], nav = 0, iterations:int = 100) -> float:
    
    if len(Values) > 0:  
        Values_copy = Values.copy()  #create a copy of the list so the original list is not modified
        Values_copy.iloc[-1] += int(nav)

    SumPos = sum(item for item in Values_copy if item > 0)
    SumNeg = sum(item for item in Values_copy if item < 0)
    NumPos = len([item for item in Values_copy if item > 0])
    NumNeg = len([item for item in Values_copy if item < 0])
    MaxAbsCf = max([abs(item) for item in Values_copy])
           
    precision = 6 - (0 if MaxAbsCf <= 0 else math.log10(MaxAbsCf))
    
    if NumNeg + NumPos == 0: # no non-zero cashflows
        return 0.0
    
    elif NumNeg == 0: # there are no negative cashflows
        return None # ("Unable to find IRR for these cashflows")
    
    elif NumPos == 0: # there are no positive cashflows:
        return -1.0
    
    elif SumPos + SumNeg > 0:
        return PosCase(Values_copy, dates, precision = precision, iterations = iterations)
    
    elif SumPos + SumNeg < 0:
        return NegCase(Values_copy, dates, precision = precision, iterations = iterations)
    
    else:
        return 0.0
    

######################################################################################################    
#Test the BG_IRR_array function

#Dates = [date(2020, 7, 8), date(2020, 12, 28), date(2021, 4, 1), date(2021, 10, 15), date(2022, 5, 24), date(2022, 11, 7), date(2023, 6, 2), date(2023, 6, 30)]
#Values = [-11000000, -35857, -30479, -30055, -32777, -13750, -66075, 35000]
#IRR1 = BG_IRR_array(Values, Dates)

#Case 1
# Dates1 = [date(2020, 7, 8), date(2020, 12, 28), date(2021, 4, 1), date(2021, 10, 15), date(2022, 5, 24), date(2022, 11, 7), date(2023, 6, 2), date(2023, 6, 30)]
# Values1 = [-11000000, 35857, 30479, 30055, 32777, 13750, 66075, 13500000]
# IRR2 = BG_IRR_array(Values1, Dates1)

# #Case 2
# Dates2 = [date(2020, 7, 8), date(2023, 6, 30)]
# Values2 = [-11000000, 13500000]
# IRR3 = BG_IRR_array(Values2, Dates2)




#print("The internal rate of return (IRR) is:", IRR3)