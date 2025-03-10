
"""
@author: LUKAS_ZIMMERMANN
"""

##Financial analysis functions


##Needed packages:

import pandas as pd 
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
#from scipy import stats
#from regpyhdfe import Regpyhdfe
#from fixedeffect.fe import fixedeffect
from stargazer.stargazer import Stargazer
from itertools import product

#import time
#import datetime as dt


##Portfolio sorts

def portfolio_sort(data, ids, time_id, sort_vars, quantiles,zeros=False):
    
    """
    Sort portfolios into quantiles.
    
    Parameters:
        
    data: DataFrame
    ids: list ['Date','Firm_ID']
    time_id: string ['Date']    
    sort_vars: list [variables to be sorted into portfolios]
    quantiles: numeric value [number of quantiles]
    zeros: If True all zero values of a variable are set to nan before sorting
    
    Examples:
    port = portfolio_sort(data = firmdata, ids = ['CSF_LCID', 'month_id'],
                          time_id = 'CAM_YEAR', sort_vars = ['Environmental Dimension',
                          'Governance & Economic Dimension', 'Social Dimension'],
                          quantiles = 5)
    
    port = portfolio_sort(data = firmdata, ids = ['CSF_LCID', 'month_id'],
                          time_id = 'month_id', sort_vars = list(firmdata.iloc[:,2:5].columns),
                          quantiles = 5)

    Output:
    data: DataFrame with new variables containing portfolio assignments
    vcount: Number of observations per portfolio
    """
    
    n = list(np.arange(1,quantiles+1,1))
    
    q = 1/quantiles
    quantiles = np.arange(q,1,q)
    
    data.sort_values(ids, inplace=True)
    
    #df = data.copy(deep=True)
    #df['month_id'] = df['month_id'].astype(str)
    #vcount = pd.DataFrame(list(product(df['month_id'].unique(), n)), columns=['month_id', 'group']) 
    #vcount.sort_values(['month_id', 'group'], inplace=True)
    #vcount.set_index(['month_id', 'group'], inplace=True)
    vcount = pd.DataFrame() 

    #j = 1

    for s in sort_vars:
        
        #Find portfolio breakpoints
        
        #data.loc[data['Environmental Dimension'] = 0, 'Environmental Dimension'] = np.nan
        
        if zeros == False:
            bp = data.groupby([time_id])[s].quantile(quantiles).unstack()
            bp.reset_index(inplace=True)
        elif zeros == True:
            data_wz = data[data[s]!=0]
            
            bp = data_wz.groupby([time_id])[s].quantile(quantiles).unstack()
            bp.reset_index(inplace=True)
        
        data = pd.merge(data, bp, on=[time_id], how='left')

        #Assign portfolio buckets based on breakpoints
        
        data['pt_'+s] = np.nan
        
        i = 1
        for q in quantiles:
            if q != max(quantiles):
                data.loc[((data[s] <= data[q]) & data['pt_'+s].isnull()), 'pt_'+s] = i
                i += 1
            elif q == max(quantiles):
                data.loc[((data[s] < data[q]) & data['pt_'+s].isnull()), 'pt_'+s] = i
                i += 1
                data.loc[(data[s] >= data[q]), 'pt_'+s] = i
                
        #Count number of firms in portfolios:
        
        #if j = 1:
        #    vcount = pd.DataFrame(list(product(df['month_id'].unique(), list(sdata['pt_'+s].unique()))), columns=['month_id', 'group'])
        #    vcount.sort_values(['month_id', 'group'], inplace=True)
        #    vcount.set_index(['month_id', 'group'], inplace=True)
            #vcount = pd.DataFrame() 
        #else:
        #    continue
        
        #j=+1        
                
        vc = pd.DataFrame(data.groupby([time_id])['pt_'+s].value_counts())
        vc.index.names = ['month_id', 'group']
        #vcount = pd.merge(vcount, vc, left_index=True, right_index=True)
        vcount = pd.concat([vcount, vc], axis=1)
                
        data.drop(quantiles, axis=1, inplace=True)
        
    return data, vcount


def portfolio_sort_short(data, ids, time_id, sort_vars, quantiles,zeros=False):
    
    """
    Sort portfolios into quantiles.
    
    Parameters:
        
    data: DataFrame
    ids: list ['Date','Firm_ID']
    time_id: string ['Date']    
    sort_vars: list [variables to be sorted into portfolios]
    quantiles: numeric value [number of quantiles]
    
    Examples:
    port = portfolio_sort(data = firmdata, ids = ['CSF_LCID', 'month_id'],
                          time_id = 'CAM_YEAR', sort_vars = ['Environmental Dimension',
                          'Governance & Economic Dimension', 'Social Dimension'],
                          quantiles = 5)
    
    port = portfolio_sort(data = firmdata, ids = ['CSF_LCID', 'month_id'],
                          time_id = 'month_id', sort_vars = list(firmdata.iloc[:,2:5].columns),
                          quantiles = 5)
    """
    
    q = 1/quantiles
    quantiles = np.arange(q,1,q)
    
    data.sort_values(ids, inplace=True)

    for s in sort_vars:
        
        #Find portfolio breakpoints
        
        #data.loc[data['Environmental Dimension'] = 0, 'Environmental Dimension'] = np.nan
        
        if zeros == False:
            bp = data.groupby([time_id])[s].quantile(quantiles).unstack()
            bp.reset_index(inplace=True)
        elif zeros == True:
            data_wz = data[data[s]!=0]
            
            bp = data_wz.groupby([time_id])[s].quantile(quantiles).unstack()
            bp.reset_index(inplace=True)
        
        data = pd.merge(data, bp, on=[time_id], how='left')

        #Assign portfolio buckets based on breakpoints
        
        data['pt_'+s] = np.nan
        
        i = 1
        for q in quantiles:
            if q != max(quantiles):
                data.loc[((data[s] <= data[q]) & data['pt_'+s].isnull()), 'pt_'+s] = i
                i += 1
            elif q == max(quantiles):
                data.loc[((data[s] < data[q]) & data['pt_'+s].isnull()), 'pt_'+s] = i
                i += 1
                data.loc[(data[s] >= data[q]), 'pt_'+s] = i
                
        data.drop(quantiles, axis=1, inplace=True)
        
    return data


#Computing portfolio returns

def portfolio_return(data, time_id, var_name, quantiles, ret, weight=None):   
    
    """
    Compute returns of sorted portfolios.
    
    Parameters:
        
    data: DataFrame
    time_id: string ['Date']    
    var_name: list [variables sorted into portfolios]
    quantiles: numeric value [number of quantiles]
    ret: string [return variable]
    weight: string [weight variable], if None and no value set then EW
    
    Examples:
    port_ret = portfolio_return(data = firmdata, time_id = 'CAM_YEAR', 
                                var_name = ['Environmental Dimension',
                                            'Governance & Economic Dimension', 
                                            'Social Dimension'],
                                ret='ret', weight=None)
    
    port_ret = portfolio_return(data = firmdata, time_id = 'month_id', 
                                sort_vars = list(firmdata.iloc[:,2:5].columns),
                                ret='ret', weight='market_cap')
    """
    
    prdata = pd.DataFrame()
    
    for v in var_name:
        #print(v)
        pt_names = []
        for i in range(1, quantiles+1):
            pt_names = pt_names + [v + '_ret_' + str(i)]
            
        #Equal-weighted return if no weight
    
        if weight == None:
            #rdata = data.groupby(['month_id', 'pt_'+v])['ret'].transform(np.mean, min_count=1)
            rdata = data.groupby([time_id, 'pt_'+v])[ret].mean().reset_index()
            rdata = pd.pivot(rdata, index=time_id, columns='pt_'+v, values=ret)              
             
            try:
                rdata.columns = pt_names
            except:
                pt_names = map(str, rdata.columns.astype(int))
                pt_names = [v+'_ret_'+p for p in list(pt_names)]
                rdata.columns = pt_names
                    
            #Long/Short portfolio return spread
            rdata[v+'_ls_'+str(quantiles)] = rdata[v+'_ret_'+str(quantiles)] - rdata[v+'_ret_1']  
        
        #Weighted return if weight set, set weight equal to market cap for value-weighting
        
        else:
            data[weight+'_tot'] = data.groupby([time_id, 'pt_'+v])[weight].transform(np.sum, min_count=1)
            data['weight'] = data[weight] / data[weight+'_tot']
            data['wret'] = data[ret] * data['weight'] 
                
            #rdata = data.groupby(['month_id', 'pt_'+v])['wret'].transform(np.sum, min_count=1)
            rdata = data.groupby([time_id, 'pt_'+v])['wret'].sum().reset_index()
            rdata = pd.pivot(rdata, index=time_id, columns='pt_'+v, values='wret')              
                 
            try:
                rdata.columns = pt_names
            except:
                pt_names = map(str, rdata.columns.astype(int))
                pt_names = [v+'_ret_'+p for p in list(pt_names)] 
                rdata.columns = pt_names
        
            #Long/Short portfolio return spread
            rdata[v+'_ls_'+str(quantiles)] = rdata[v+'_ret_'+str(quantiles)] - rdata[v+'_ret_1']    
        
        prdata = pd.concat([prdata, rdata], axis=1)

    return prdata


#Run time-series regression
    
def ts_reg(rdata, reg_vars, top, controls=None, nw=None):
    
    """
    Time-series regression to determine alpha or risk-adjusted return.
    
    Parameters:
        
    rdata: DataFrame
    time_id: string ['Date']    
    reg_vars: list [variables used to construct investment strategies]
    top: numeric value [number of quantiles/max quantile]
    controls: list [regression controls/risk factors to control for]
    nw: numeric value [time lags to compute Newey-West adjusted SEs]

    Example:
    port_ret = portfolio_return(data = firmdata, 
                                reg_vars = ['Environmental Dimension', 
                                            'Governance & Economic Dimension',
                                            'Social Dimension'], 
                                top = 5,
                                controls = ['MKT-rf', 'SMB', 'HML'],
                                nw = 6)
    """
    
    #Detail for robust and Newey-West standard errors
    
    if nw==None:
        cvt = 'HC1'
        cvk={}
    else:
        cvt = 'HAC'
        cvk={'maxlags': nw}
    
    regress_ls = dict()
    regs_ls = []
   
    j=0            
    
    for v in reg_vars:

        print(v)
        
        regress = dict()
        regs = []
        
        if controls==None:
            port = rdata.loc[:,(rdata.columns.str.contains(v))] 
        else:
            port = rdata.loc[:,(rdata.columns.str.contains(v))|(rdata.columns.isin(controls))] 
            
        if len(port.columns) == 0:
            continue    
        
        port = port.loc[:,~(port.columns.str.contains(v+':'))]
        port = port.loc[:,~(port.columns.str.contains(' '+v))]
        #port = port.loc[:,~(port.columns.str.contains(v+' '))]
        port = port.loc[:,~(port.columns.str.contains(v+' \('))]        
        
        #First regression of long/short portfolio
        
        Y = port.loc[:,port.columns.str.contains('_ls_')] 
        #Y2 = port.loc[:,port.columns.str.endswith('ret_'+str(top))] 
        #Y3 = port.loc[:,port.columns.str.endswith('ret_1')] 
        
        if controls==None:
            port = sm.add_constant(port)
            res = sm.OLS(Y,port['const']).fit(cov_type=cvt, cov_kwds=cvk)
        else:
            res = sm.OLS(Y,sm.add_constant(port[controls])).fit(cov_type=cvt, cov_kwds=cvk)
        
        regress[0] = res
        regs.append(res)
        
        regress_ls[j] = res
        regs_ls.append(res)
        
        j+=1
        
        #Then regression of each portfolio return
        
        for i in range(1, top+1):
            Y = port.loc[:,port.columns.str.endswith('ret_'+str(i))] 
            
            if len(Y.columns) == 0:
                continue
            
            if controls==None:
                res = sm.OLS(Y,port['const']).fit(cov_type=cvt, cov_kwds=cvk)
            else:
                res = sm.OLS(Y,sm.add_constant(port[controls])).fit(cov_type=cvt, cov_kwds=cvk)
            
            regress[i] = res
            regs.append(res)
        
        #Output regression results
        
        dfoutput = summary_col(regs,stars=True)
        print(dfoutput)
     
    #Store long/short strategy results/return spread regression results    
        
    return regress_ls, regs_ls, regress

#Time-series regression refering to variables names return and long/short

def ts_reg_det(rdata, reg_vars, top, controls=None, nw=None):
    
    """
    Time-series regression to determine alpha or risk-adjusted return.
    
    Parameters:
        
    rdata: DataFrame
    time_id: string ['Date']    
    reg_vars: list [variables used to construct investment strategies]
    top: numeric value [number of quantiles/max quantile]
    controls: list [regression controls/risk factors to control for]
    nw: numeric value [time lags to compute Newey-West adjusted SEs]

    Example:
    port_ret = portfolio_return(data = firmdata, 
                                reg_vars = ['Environmental Dimension', 
                                            'Governance & Economic Dimension',
                                            'Social Dimension'], 
                                top = 5,
                                controls = ['MKT-rf', 'SMB', 'HML'],
                                nw = 6)
    """
    
    #Detail for robust and Newey-West standard errors
    
    if nw==None:
        cvt = 'HC1'
        cvk={}
    else:
        cvt = 'HAC'
        cvk={'maxlags': nw}
    
    regress_ls = dict()
    regs_ls = []
   
    j=0            
    
    for v in reg_vars:

        print(v)
        
        regress = dict()
        regs = []

        #port = rdata.loc[:,rdata.columns.str.contains(v)] 
        if controls==None:
            port = rdata.loc[:,(rdata.columns.str.contains(v+'_r'))|(rdata.columns.str.contains(v+'_l'))] 
        else:
            port = rdata.loc[:,(rdata.columns.str.contains(v+'_r'))|(rdata.columns.str.contains(v+'_l'))|(rdata.columns.isin(controls))] 
        
        print(port.columns)
        
        #First regression of long/short portfolio
        
        Y = port.loc[:,port.columns.str.contains('_ls_')] 
        #Y2 = port.loc[:,port.columns.str.endswith('ret_'+str(top))] 
        #Y3 = port.loc[:,port.columns.str.endswith('ret_1')] 
        
        if controls==None:
            port = sm.add_constant(port)
            res = sm.OLS(Y,port['const']).fit(cov_type=cvt, cov_kwds=cvk)
        else:
            res = sm.OLS(Y,sm.add_constant(port[controls])).fit(cov_type=cvt, cov_kwds=cvk)
        
        regress[0] = res
        regs.append(res)
        
        regress_ls[j] = res
        regs_ls.append(res)
        
        j+=1
        
        #Then regression of each portfolio return
        
        for i in range(1, top+1):
            Y = port.loc[:,port.columns.str.endswith('ret_'+str(i))] 

            if len(Y.columns) == 0:
                continue
            
            if controls==None:
                res = sm.OLS(Y,port['const']).fit(cov_type=cvt, cov_kwds=cvk)
            else:
                res = sm.OLS(Y,sm.add_constant(port[controls])).fit(cov_type=cvt, cov_kwds=cvk)
            
            regress[i] = res
            regs.append(res)
        
        #Output regression results
        
        dfoutput = summary_col(regs,stars=True)
        print(dfoutput)
     
    #Store long/short strategy results/return spread regression results    
        
    return regress_ls, regs_ls


##Fama-Macbeth cross-sectional regressions

def fmb_reg(df,time_id,nw=None,num=None,sam=None,regtype='OLS', wt=None):
        
    """
    Fama-MacBeth cross-sectional regressions 
    Can be used for estimation of cross-sectional/slope factors
    
    Assumption of columns order in df: time_id, dependent variable, independent variables
    
    Parameters:
        
    df: DataFrame
    time_id: string ['Date']    
    nw: numeric value [time lags to compute Newey-West adjusted SEs]
    num: numeric value [percentage of full sample to be available in interval [0,1]]
    sam: numeric value [percentage of monthly sample to be available in interval [0,1]]
    regtype: Regression type, either OLS or WLS, in case of WLS need to specify weights parameter
    wt: WLS weights variable

    Examples:
    fmb, tb, reg = fmb_reg(data,'month_id')
    fmb, tb, reg = fmb_reg(data,'month_id', nw=6, num=0.5, sam=0.5)
    fmb, tb, reg = fmb_reg(data,'month_id', regtype='WLS', wt=['market_cap_usd'])
    """
    
    if num:
        n = df.notnull().sum()
        df = df[list(n.index[n > len(df)*num])] 

    ts = sorted(df[time_id].unique())
    betas = list()
    rsquared = list()
    nobs = list()
    
    #One cross-sectional regression in each time period

    for x in ts:
        sample = df[df[time_id] == x]
        sample = sample.dropna(axis=1, how='all')
        if sam:
            s = sample.count()
            s = s[s>(sample.count().max())*sam].index.to_list()
            sample = sample[s]
            sample = sample.dropna(axis=0)
        else:
            sample = sample.dropna(axis=0)
        
        if regtype == 'OLS':
            #res = smf.ols(formula, sample).fit()
            res = sm.OLS(sample.iloc[:,1],sm.add_constant(sample.iloc[:,2:])).fit()
        elif regtype == 'WLS':
            res = sm.WLS(sample.iloc[:,1],sm.add_constant(sample.iloc[:,2:]),
                         weights=np.array(sample[wt])).fit()
                         #weights=1./np.array(sample[wt])).fit()

        betas.append(res.params)
        rsquared.append(res.rsquared)
        nobs.append(res.nobs)
        
    rsquared = sum(rsquared)/len(rsquared)
    nobs = round(sum(nobs)/len(nobs))

    params = dict()
    tvalues = dict()
    fnobs = dict()
    regress = dict()
    regs = []
    
    #Computing sample average of coefficients

    indep = []
    for b in betas:
        var = list(b.index.values)  
        indep.extend(var)
     
    i = 0
    #if smf.ols: for v in formula.split('~')[1].split('+'):
    for v in np.unique(indep).tolist():
        i += 1
        #v = v.strip()
        #if v == '1':  # No intercept
            #continue
        beta_df = pd.DataFrame([b[v] for b in betas if v in b],
                               columns=['var'])
        if nw:
            res = smf.ols('var ~ 1', beta_df).fit(cov_type='HAC',
                                                  cov_kwds={'maxlags': nw})
        else:
            res = smf.ols('var ~ 1', beta_df).fit()
            
        regress[i] = res
        regs.append(res)
        
        params[v] = res.params['Intercept']
        tvalues[v] = res.tvalues['Intercept']
        fnobs[v] = res.nobs

    tbl = Stargazer(regress.values())

    results = pd.DataFrame.from_dict(params, orient='index')
    results = pd.merge(results, pd.DataFrame.from_dict(tvalues, orient='index'),
                       left_index=True, right_index=True)
    results = pd.merge(results, pd.DataFrame.from_dict(fnobs, orient='index'),
                       left_index=True, right_index=True)
    results.columns=['coef','t-stat','nobs']
    
    return results, tbl, regress


##Fama-Macbeth cross-sectional regressions with variables

def fmb_reg_var(df,time_id,depv,indepv,nw=None,num=None,sam=None,regtype='OLS', wt=None, intercept=True):
        
    """
    Fama-MacBeth cross-sectional regressions 
    Can be used for estimation of cross-sectional/slope factors
    
    Assumption of columns order in df: time_id, dependent variable, independent variables
    
    Parameters:
        
    df: DataFrame
    time_id: string ['Date'] 
    dep: dependent variable
    indep: independent variables
    nw: numeric value [time lags to compute Newey-West adjusted SEs]
    num: numeric value [percentage of full sample to be available in interval [0,1]]
    sam: numeric value [percentage of monthly sample to be available in interval [0,1]]
    regtype: Regression type, either OLS or WLS, in case of WLS need to specify weights parameter
    wt: WLS weights variable

    Examples:
    fmb, tb, reg = fmb_reg(data,'month_id')
    fmb, tb, reg = fmb_reg(data,'month_id', nw=6, num=0.5, sam=0.5)
    fmb, tb, reg = fmb_reg(data,'month_id', regtype='WLS', wt=['market_cap_usd'])
    """
    
    if num:
        n = df.notnull().sum()
        df = df[list(n.index[n > len(df)*num])] 

    ts = sorted(df[time_id].unique())
    betas = list()
    rsquared = list()
    nobs = list()
    
    #One cross-sectional regression in each time period

    for x in ts:
        sample = df[df[time_id] == x]
        sample = sample.dropna(axis=1, how='all')
        if sam:
            s = sample.count()
            s = s[s>(sample.count().max())*sam].index.to_list()
            sample = sample[s]
            sample = sample.dropna(axis=0)
        else:
            sample = sample.dropna(axis=0)

        if intercept == True:
            if regtype == 'OLS':
                #res = smf.ols(formula, sample).fit()
                res = sm.OLS(sample[depv],sm.add_constant(sample[indepv])).fit()
            elif regtype == 'WLS':
                res = sm.WLS(sample[depv],sm.add_constant[indepv],
                             weights=np.array(sample[wt])).fit()
                             #weights=1./np.array(sample[wt])).fit()
        elif intercept == False:
            if regtype == 'OLS':
                #res = smf.ols(formula, sample).fit()
                res = sm.OLS(sample[depv],sample[indepv]).fit()
            elif regtype == 'WLS':
                res = sm.WLS(sample[depv],indepv,
                             weights=np.array(sample[wt])).fit()
                             #weights=1./np.array(sample[wt])).fit()

        betas.append(res.params)
        rsquared.append(res.rsquared)
        nobs.append(res.nobs)
        
    rsquared = sum(rsquared)/len(rsquared)
    nobs = round(sum(nobs)/len(nobs))

    params = dict()
    tvalues = dict()
    fnobs = dict()
    regress = dict()
    regs = []
    
    #Computing sample average of coefficients

    indep = []
    for b in betas:
        var = list(b.index.values)  
        indep.extend(var)
    indep = indep[0:(len(indepv)+1)]
    
    i = 0
    #if smf.ols: for v in formula.split('~')[1].split('+'):
    #for v in np.unique(indep).tolist():
    for v in indep:
        i += 1
        #v = v.strip()
        #if v == '1':  # No intercept
            #continue
        beta_df = pd.DataFrame([b[v] for b in betas if v in b],
                               columns=['var'])
        if nw:
            res = smf.ols('var ~ 1', beta_df).fit(cov_type='HAC',
                                                  cov_kwds={'maxlags': nw})
        else:
            res = smf.ols('var ~ 1', beta_df).fit()
            
        regress[i] = res
        regs.append(res)
        
        params[v] = res.params['Intercept']
        tvalues[v] = res.tvalues['Intercept']
        fnobs[v] = res.nobs

    tbl = Stargazer(regress.values())

    results = pd.DataFrame.from_dict(params, orient='index')
    results = pd.merge(results, pd.DataFrame.from_dict(tvalues, orient='index'),
                       left_index=True, right_index=True)
    results = pd.merge(results, pd.DataFrame.from_dict(fnobs, orient='index'),
                       left_index=True, right_index=True)
    results.columns=['coef','t-stat','nobs']
    
    return results, tbl, regress, betas

