
"""
@author: LUKAS_ZIMMERMANN
"""

## Load packages

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import linearmodels as lm
from scipy import stats
from regpyhdfe import Regpyhdfe
from fixedeffect.fe import fixedeffect
from stargazer.stargazer import Stargazer
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
import gc
import glob
import time
import string 
import subprocess
import warnings
import dli
import itertools

import finfunc

pd.options.mode.chained_assignment = None  # default='warn'

## Set directories

input = "<Directory>\\Data"
input_econ = "<Directory>\\Data\\Economic Data\\"
zip_dir = "<Directory>\\Data\\Financials"

output = "<Directory>\\Data\\"

res_out = "<Directory>\\Projects\\Systematic Macro Shocks\\Results"


#Daterange to be used:
    
#mindate = '09/31/2013'
#maxdate = '05/01/2022'
#maxdate = '01/01/2022'

#mindate = '09-30-2013'
mindate = '08-31-2013'
#maxdate = '04-30-2022'
#maxdate = '05-01-2022'
#maxdate = '12-31-2021'
#maxdate = '09-30-2022'
maxdate = '12-31-2022'

#Set script directory to current working directory:
    
script = os.getcwd()
#os.chdir(script+'\\Data')
os.chdir('<Directory>\\Data')


#%% Load economic data:

#Economic policy uncertainty (Baker et al. 2016):
url="https://policyuncertainty.com/media/All_Country_Data.xlsx"
epu = pd.read_excel(url, skipfooter=29)

#Monetary policy uncertainty (Baker et al. 2016):
url="https://www.policyuncertainty.com/media/US_MPU_Monthly.xlsx"
mpu = pd.read_excel(url, skiprows=range(0), skipfooter=1)

#World uncertainty index (Ahir et al.):
url="https://worlduncertaintyindex.com/wp-content/uploads/2023/03/WUI_Data.xlsx"
wui = pd.read_excel(url, skiprows=range(0), sheet_name='T1')

#Climate policy uncertainty (Gavriilidis 2021):
#url="https://www.policyuncertainty.com/media/CPU%20index.csv"
#cpu = pd.read_csv(url, skiprows=range(2))
cpu = pd.read_csv(input_econ+"Climate_Policy_Uncertainty.csv", skiprows=range(2))  

#Geopolitical risk index:
url="https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
gr = pd.read_excel(url)

#Macroeconomic and financial uncertainty (Jurado et al. 2015 and Ludvigson et al. 2021):  
url="https://www.sydneyludvigson.com/s/MacroFinanceUncertainty_202302Update.zip"
resp = urlopen(url)
zipfile = ZipFile(BytesIO(resp.read()))
#fintu = pd.read_excel(zipfile.open(zipfile.namelist()[0]), sheet_name='Total Financial Uncertainty')
#fineu = pd.read_excel(zipfile.open(zipfile.namelist()[0]), sheet_name='Economic Financial Uncertainty')
#mactu = pd.read_excel(zipfile.open(zipfile.namelist()[1]), sheet_name='Total Macro Uncertainty')
#maceu = pd.read_excel(zipfile.open(zipfile.namelist()[1]), sheet_name='Economic Macro Uncertainty')
#realtu = pd.read_excel(zipfile.open(zipfile.namelist()[2]), sheet_name='Total Real Uncertainty')
#realeu = pd.read_excel(zipfile.open(zipfile.namelist()[2]), sheet_name='Economic Real Uncertainty')
fintu = pd.read_excel(zipfile.open(zipfile.namelist()[0]), sheet_name='Financial Uncertainty')
#fineu = pd.read_excel(zipfile.open(zipfile.namelist()[0]), sheet_name='Economic Financial Uncertainty')
mactu = pd.read_excel(zipfile.open(zipfile.namelist()[1]), sheet_name='Macro Uncertainty')
#maceu = pd.read_excel(zipfile.open(zipfile.namelist()[1]), sheet_name='Economic Macro Uncertainty')
realtu = pd.read_excel(zipfile.open(zipfile.namelist()[2]), sheet_name='Real Uncertainty')
#realeu = pd.read_excel(zipfile.open(zipfile.namelist()[2]), sheet_name='Economic Real Uncertainty')

#Twitter economic uncertainty (Baker et al. 2021): 
tweu = pd.read_excel(input_econ+"Twitter_Economic_Uncertainty.xlsx")    
   
#Monetary policy uncertainty (US):
mpu_us = pd.read_excel(input_econ+"HRS_MPU_monthly.xlsx") 

##Load other financial/economic data:

#Volatility index (VIX):
    
url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
vix = pd.read_csv(url)

#IHS purchasing manager index: 

#Logon and Authenticate to the Data Lake
dl = dli.connect()

#Loading Dataset
meta_ds = dl.datasets.get(short_code="PurchasingManagersIndexPMIMetadata")
ts_ds = dl.datasets.get(short_code="PurchasingManagersIndexPMITimeseries")

#Reading the Meta Data as a Dataframe
meta_df = meta_ds.dataframe(partitions=[f"as_of_date={meta_ds.last_datafile_at}"])
meta_df.sort_values('short_label', inplace=True)

#Check unique values of key columns and get the right meta information
regions=['Asia', 'Asia Excluding China, Japan', 'Asia Excluding China', 'Asia Excluding Japan',
         'China (mainland)', 'Developed Countries', 'Emerging Markets', 'Europe', 'European Union',
         'Eurozone', 'World', 'Japan', 'North America', 'United States']
meta_df = meta_df[(meta_df['economic_concept_name'].isin(['PMI', 'Headline'])) 
        & (meta_df['source_geographic_location_name'].isin(regions))
        & (meta_df['short_label'].str.contains('Composite')|meta_df['short_label'].str.contains('Manufacturing'))
        & (meta_df['series_type_name'] == 'Historical')]
 
#Reading the Time Series as a Dataframe
ts_df = ts_ds.dataframe(partitions=[f"as_of_date={ts_ds.last_datafile_at}"])

#Merge Timeseries with Metadata to get the required series 
df_PMI = pd.merge(ts_df, meta_df[['source_id', 'mnemonic', 'last_update', 
                                  'long_label', 'short_label', 
                                  'document_type', 'economic_concept_name',
                                  'source_geographic_location_name']], on="source_id")
del ts_df

    
## Merge data to one monthly dataset:

cpu['date'] = pd.to_datetime(cpu['date'], format='%b-%y')
cpu['month_id'] = pd.to_datetime(cpu['date']).dt.strftime('%Y-%m')

econ = cpu.copy(deep=True)
econ = econ[['month_id', 'cpu_index']]

epu['month_id'] = pd.to_datetime(epu[['Year', 'Month']].assign(DAY=1)).dt.strftime('%Y-%m') 
econ = pd.merge(econ, epu[['month_id', 'GEPU_current', 'GEPU_ppp']], on='month_id', how='outer')    
econ.sort_values(['month_id'], inplace=True)

mpu['month_id'] = pd.to_datetime(mpu[['Year', 'Month']].assign(DAY=1)).dt.strftime('%Y-%m') 
econ = pd.merge(econ, mpu.iloc[:,2:], on='month_id', how='outer')    
econ.sort_values(['month_id'], inplace=True)

mpu_us['Month'] = mpu_us['Month'].str.replace('m',' ')
mpu_us['month_id'] = pd.to_datetime(mpu_us['Month'], format='%Y %m').dt.strftime('%Y-%m')
econ = pd.merge(econ, mpu_us[['month_id', 'US MPU']], on='month_id', how='outer')    
econ.sort_values(['month_id'], inplace=True)

wui['month_id'] = pd.to_datetime(wui['Year']).dt.to_period('M')
wui = wui.set_index('month_id')
wui = wui.resample('M').mean()
wui = wui.ffill().reset_index()

econ = pd.merge(econ, wui[['month_id', 'Global (simple average)', 
                           'Global (GDP weighted average)']], on='month_id', how='outer')    
econ.sort_values(['month_id'], inplace=True)

gr['month_id'] = pd.to_datetime(gr['month']).dt.strftime('%Y-%m') 
econ = pd.merge(econ, gr[['month_id', 'GPR']], on='month_id', how='inner')    
econ.sort_values(['month_id'], inplace=True)

fintu['month_id'] = pd.to_datetime(fintu['Date']).dt.strftime('%Y-%m')
econ = pd.merge(econ, fintu[['month_id', 'h=1']], on='month_id', how='outer')   
econ.rename(columns={'h=1':'Financial Uncertainty'}, inplace=True)

mactu['month_id'] = pd.to_datetime(mactu['Date']).dt.strftime('%Y-%m')
econ = pd.merge(econ, mactu[['month_id', 'h=1']], on='month_id', how='outer')   
econ.rename(columns={'h=1':'Macro Uncertainty'}, inplace=True)

realtu['month_id'] = pd.to_datetime(realtu['Date']).dt.strftime('%Y-%m')
econ = pd.merge(econ, realtu[['month_id', 'h=1']], on='month_id', how='outer')   
econ.rename(columns={'h=1':'Real Uncertainty'}, inplace=True)
econ.sort_values(['month_id'], inplace=True)

tweu['month_id'] = pd.to_datetime(tweu['date']).dt.strftime('%Y-%m')
tweum = tweu.groupby(['month_id']).last().reset_index()
econ = pd.merge(econ, tweum[['month_id', 'TEU-ENG', 'TEU-SCA', 'TMU-ENG', 'TMU-SCA']], on='month_id', how='outer')    
econ.sort_values(['month_id'], inplace=True)

vix['month_id'] = pd.to_datetime(vix['DATE']).dt.strftime('%Y-%m')
vix['VIX_mean'] = vix.groupby(['month_id'])['CLOSE'].transform('mean')
vix = vix.groupby(['month_id']).last().reset_index()
econ = pd.merge(econ, vix[['month_id', 'CLOSE', 'VIX_mean']], on='month_id', how='outer')
econ.rename(columns={'CLOSE':'VIX'}, inplace=True)   

df_PMI["month_id"] = df_PMI["date"].dt.strftime("%Y-%m")
df_PMI = df_PMI.pivot(index="month_id", columns="source_id", values="value")
df_PMI = df_PMI.reset_index()
df_PMI = df_PMI[['month_id', '88505325', '93625215']]
df_PMI.rename(columns={'88505325':'Global Composite (M+S) PMI Headline Adjusted', 
                       '93625215':'Global Manufacturing PMI Adjusted'}, inplace=True)
econ = pd.merge(econ, df_PMI, on='month_id', how='outer')


## Load climate-related data:

#Media climate change concern index:
mccc = pd.read_excel(input_econ+"Sentometrics_US_Media_Climate_Change_Index.xlsx", 
                     sheet_name='SSRN 2022 version (monthly)', skiprows=range(5)) 

#Climate change news index:
ccn = mccc = pd.read_excel(input_econ+"EGLKS_data.xlsx") 


## Correlations between changes in economic shock variables   

econ.sort_values(['month_id'], inplace=True)
econd = econ.copy(deep=True)

for v in econd.columns[1:]:
    econd[v+'_L'] = econd[v].shift(1)
    econd[v+'_D'] = econd[v]-econd[v+'_L']
    econd.drop([v+'_L'], axis=1, inplace=True) 
    

#%% Load stock market data:    
 
ret = pd.read_csv(input+'\\Financials\\firm_returns_monthly.csv', low_memory=False)
#ret = pd.read_csv(input_dir+'\\Financials\\returns_monthly.csv', low_memory=False)

#ind = pd.read_csv(input+'\\Financials\\industries.csv', low_memory=False)
zipfile = ZipFile(input+'\\Financials\\industries.zip')
ind = pd.read_csv(zipfile.open(zipfile.namelist()[0]))  

#ret = pd.merge(ret, ind, on=['issue_id', 'month_id', 'Date', 'company_name',
#                            'capital_iq', 'snlinstitutionid'])
#ret = pd.merge(ret, ind, on=['issue_id', 'month_id', 'company_name',
#                             'capital_iq', 'snlinstitutionid'], suffixes=['', '_ind'], indicator=True)
ret = pd.merge(ret, ind, on=['capital_iq', 'month_id'], suffixes=['', '_ind'], 
               indicator=True, how='inner')

print(ret['_merge'].value_counts())
ret.drop(['_merge'], axis=1, inplace=True)

try:
    ret = ret.drop(['Unnamed: 0'], axis = 1)
except:
    pass

ret = ret.rename(columns={'capital_iq':'CIQ_ID'})
#ret['month_id'] = pd.to_datetime(ret['month_id']).dt.to_period("M")

for m in ['market_cap', 'market_cap_usd']:
    ret[m] = ret.groupby('CIQ_ID')[m].shift(1)
    ret[m] = ret[m]/1000

# Drop if CIQ ID not available:
ret = ret[~ret['CIQ_ID'].isnull()]

ind = ind.rename(columns={'capital_iq':'CIQ_ID'})
ind['month_id'] = pd.to_datetime(ind['month_id']).dt.to_period("M")

ind = ind[ind['month_id']>'01-01-2013']

gc.collect()

# Prepare data filtering:

countf = pd.read_csv(input+'\\Financials\\zero_return_filter.csv', low_memory=False)


#%% Apply filters:

# Winsorize returns by month at 1% and 99% percentiles
# Set max return to 300% if cumulative 2 month return less than 50%
# Set min excess return to -100%

print(len(ret))

def month_return_adjust(rd, ret_vars):
    for r in ret_vars:
        rd[r+'2'] = ((1+rd[r])*(1+rd[r].shift())-1)*100
        rd[r] = rd[r]*100
        #rd[r] = rd.groupby('month_id')[r].transform(lambda x: stats.mstats.winsorize(x, 
        #                                                                             limits=[0.01, 0.01]))
        #rd[r] = rd.groupby('month_id')[r].transform(lambda x: np.maximum(x.quantile(.01), 
        #                                                                 np.minimum(x, x.quantile(.99))))
        #rd.loc[(rd[r]>300)&(rd[r+'2']<50),r] = 300 
        rd.loc[(rd[r]>300)&(rd[r+'2']<50),r] = np.nan
        rd.loc[rd[r]<-100,r] = -100 
    
    return rd

ret.sort_values(['CIQ_ID', 'month_id'], inplace=True)

ret = month_return_adjust(ret, ['ret', 'ret_usd'])

# Drop small and illiquid firms:

#ret = ret[ret['cshtrd'] != 0]
ret = ret[ret['cshtrm'] != 0]
ret = ret[~((ret['prccd_usd']<1) & (ret['ret_usd']>300))]
ret = ret[~((ret['prccd_usd']<1) & (ret['ret_usd']<=-80))]
# data = data[data['prccd_usd']>1]
# data = data[data['market_cap_usd']>1]
# data = data[~((data['prccd_usd']<0.5) & (data['market_cap_usd']<1))]
print(len(ret))

# Drop if more than 30% zero return days per year

ret['year_id'] = pd.to_datetime(ret['Date']).dt.strftime('%Y').astype(int)
#ret = pd.merge(ret, countf, on=['issue_id', 'year_id'])
countf = countf.rename(columns={'capital_iq':'CIQ_ID'})
ret = pd.merge(ret, countf, on=['CIQ_ID', 'year_id'], how='left', indicator = True)
ret.loc[ret['perf'].isnull(), 'perf'] = 0.5
ret = ret[ret['perf'] < 0.3]
print(len(ret))

# Number of firms per country and month, require at least 10:

firms = ret.groupby(['country', 'month_id'])['ret'].count()
firms = firms.reset_index()
firms.rename(columns={'ret': 'firmno'}, inplace=True)

ret = pd.merge(ret, firms, on=['country', 'month_id'], how='outer')
ret = ret[ret['firmno'] >= 10]
print(len(ret))

# Drop time period where CSA data not available:

ret = ret[pd.to_datetime(ret['Date']) > '01-01-2013']
ret.drop('_merge', axis=1, inplace=True)
print(len(ret))

gc.collect()


#%% Merge accounting data and stock market data

acc = pd.read_csv(input+'\\Financials\\Controls_and_Predictors_post_2012_PIT.csv', low_memory=False)
acc.rename(columns={'companyId': 'CIQ_ID'}, inplace=True)

acc['md'] = pd.to_datetime(acc['filingDate']).dt.to_period('m').astype(int) - pd.to_datetime(
    acc['periodEndDate']).dt.to_period('m').astype(int)

ret['month_id'] = pd.to_datetime(ret['month_id']).dt.to_period("M")
acc['month_id'] = pd.to_datetime(acc['month_id']).dt.to_period("M")

ret = pd.merge(ret, acc[['CIQ_ID', 'month_id', 'periodEndDate', 
                         'As Reported Balance Sheet Date', 'Revenues', 'Total Assets', 
                         'Book Equity', 'Operating Profitability', 'Gross Profitability', 
                         'IAT', 'ICAPEX', 'md']], on=['CIQ_ID', 'month_id'], 
                how='left', indicator=True)

ret.sort_values(['CIQ_ID', 'month_id'], inplace=True)

#Number of months to forward accounting data:
fw=18

acc_vars = ['periodEndDate', 'As Reported Balance Sheet Date', 'Revenues', 'Total Assets', 
            'Book Equity', 'Operating Profitability', 'Gross Profitability', 'IAT', 'ICAPEX', 'md']
ret[acc_vars] = ret.groupby(['CIQ_ID'])[acc_vars].ffill(limit=fw)

ret['mcap_lag'] = np.nan
for i in range(0,18):
    ret['mcap_L'] = ret.groupby('CIQ_ID')['market_cap_usd'].shift(i)
    ret.loc[ret['md']==i, 'mcap_lag'] = ret['mcap_L']

#Book-to-market:
ret['BM'] = ret['Book Equity']/ret['mcap_lag']

#Size:
    
ret['size'] = np.log(ret['market_cap_usd'])

## Momentum:

mom = pd.read_csv(input+'\\Financials\\momentum_monthly.csv')
mom.rename(columns={'capital_iq': 'CIQ_ID'}, inplace=True)
mom['month_id'] = pd.to_datetime(mom['month_id']).dt.to_period("M")

ret = pd.merge(ret, mom[['CIQ_ID', 'month_id', 'mdiff', 'mom212', 'count_ret']], 
               on=['CIQ_ID', 'month_id'], how='left')


#%% Merge macroeconomic and stock market data

econ['month_id'] = pd.to_datetime(econ['month_id']).dt.to_period("M")
econd['month_id'] = pd.to_datetime(econd['month_id']).dt.to_period("M")
#ret = pd.merge(ret, econ, on='month_id', how='left')
ret = pd.merge(ret, econd, on='month_id', how='left')
#ret['month_id'] = pd.to_datetime(ret['month_id']).dt.to_period("M")
ret.sort_values(['CIQ_ID', 'month_id'], inplace=True)

print(ret.columns)

dvars = ['cusip', 'isin', 'issue_id', 'ticker', 'gvkey', 'cshoc', 'cshoc_unadj', 
         'Date_ind', 'iso_code', 'company_id_ind', 'gticker', 'gvkey_ind', 
         'ret2', 'ret_usd2', 'perf', 'firmno',
         'mcap_lag', 'mcap_L','md', '_merge']
ret.drop(dvars, axis=1, inplace=True)

del acc
del mom
del countf

gc.collect()


#%% Import ESG data

import csa_data_exec

csa = csa_data_exec.csa_import()

print('Executed')

csa = csa[csa['_merge']=='both']

#csa.drop(['CAM_ID', 'CAM_NAME', 'CAM_TYPE', 'EVR_PUBLISH_TIME', 'EVR_ID',
#          'aspect_type', 'Potential Score Contribution Combined', 'Data Availability Public',
#          'Data Availability Private', 'Data Availability Combined', 'SCORE_IMP_TEMP', '_merge'], 
#          axis=1, inplace=True)
csa.drop(['Potential Score Contribution Combined', 'Data Availability Public',
          'Data Availability Private', 'Data Availability Combined', 'SCORE_IMP_TEMP', '_merge'], 
          axis=1, inplace=True)
gc.collect()

#csa.drop(['Date', 'month_id', 'aspect_type', 'Disclosure Level Combined', 
#          'SCORE_IMP_NP', 'SCORE_ADJ', 'WEIGHT_ADJ', 'WEIGHT_IMP_NP', 'SCORE_IMP_ADJ', 
#          'WEIGHT_IMP_ADJ', 'SCORE_IMP_NP_ADJ', 'WEIGHT_IMP_NP_ADJ', 'SECTOR'], 
#          axis=1, inplace=True)
#csa.to_csv(output+'CSA\\csa_data.csv', index=False)


#Only keep the respective level of aggregated scores

#csa['CAM_TYPE'].value_counts()
#csa = csa[csa['CAM_TYPE']=='CA']

csa['CAM_YEAR'] = csa['CAM_YEAR_IMP']

question = csa[csa['QUESTION'].notnull()]
criterion = csa[(csa['QUESTION'].isnull())&(csa['CRITERION'].notnull())]
dimension = csa[(csa['QUESTION'].isnull())&(csa['CRITERION'].isnull())&(csa['DIMENSION'].notnull())]
esg = csa[(csa['QUESTION'].isnull())&(csa['CRITERION'].isnull())&(csa['DIMENSION'].isnull())]

#Set the score to be used in sorting:

#Potential choices are: 
#SCORE: classical scores, SCORE_IMP: imputed scores, SCORE_IMP_NP: imputed scores without disclosure penalization

#score = 'SCORE'
#score = 'SCORE_IMP'
#score = 'SCORE_IMP_NP'


#%% Prepare wide tables

#In wide tables the different score levels are next to each other instead of below each other, this enables sorting for multiple variables 

#ESG

esg.loc[esg['DIMENSION'].isnull(), 'DIMENSION'] = "ESG"

esg.sort_values(['CSF_LCID', 'month_id'], inplace=True)
firm_inf = esg.groupby(['CSF_LCID', 'month_id'])['INDUSTRY', 'INDUSTRYGROUP', 'CAM_TYPE', 
                                                 'CSF_LONGNAME', 'ISIN', 'GVKEY', 
                                                 'CIQ_ID', 'COUNTRYNAME', 'DJREGION',
                                                 'Date', 'CAM_YEAR', 'ASOF_DATE'].last()

#esg = esg.pivot_table(index=['CSF_LCID', 'month_id'], 
#                      columns='DIMENSION', values=score)

esgs = esg.pivot_table(index=['CSF_LCID', 'month_id'], 
                      columns='DIMENSION', values='SCORE')
esgi = esg.pivot_table(index=['CSF_LCID', 'month_id'], 
                      columns='DIMENSION', values='SCORE_IMP')
esgn = esg.pivot_table(index=['CSF_LCID', 'month_id'], 
                      columns='DIMENSION', values='SCORE_IMP_NP')
esg = pd.merge(esgs, esgi, left_index=True, right_index=True, how='outer', suffixes=('',' Imp'))
esg = pd.merge(esg, esgn, left_index=True, right_index=True, how='outer', suffixes=('',' NP'))

esg = pd.merge(esg, firm_inf, left_index=True, right_index=True)
esg = esg.reset_index()

del esgs
del esgi
del esgn

#Dimension:
 
dimension.replace({'DIMENSION':{'Economic Dimension':'Governance & Economic Dimension'}},
                 inplace=True)

#dimension = dimension.pivot_table(index=['CSF_LCID', 'month_id'], 
#                                  columns='DIMENSION', values=score)
   
dimensions = dimension.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='DIMENSION', values='SCORE')
dimensioni = dimension.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='DIMENSION', values='SCORE_IMP')
dimensionn = dimension.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='DIMENSION', values='SCORE_IMP_NP')
dimension = pd.merge(dimensions, dimensioni, left_index=True, right_index=True, how='outer', suffixes=('',' Imp'))
dimension = pd.merge(dimension, dimensionn, left_index=True, right_index=True, how='outer', suffixes=('',' NP'))

dimension = pd.merge(dimension, firm_inf, left_index=True, right_index=True)
dimension = dimension.reset_index()

del dimensions
del dimensioni
del dimensionn

#Criterion

# Only include criteria that have been used in the most recent CSA data vintage
# Historically, there were 104 aspects with 119 names, currently (2021) there are 68 aspects, 56 are available for the full period

csas = csa.groupby(['CAM_YEAR', 'ASP_LCID']).last()
csas.sort_values(['ASP_LCID', 'CAM_YEAR'], inplace=True)
csas = csas.reset_index()

csac = csas[(csas['QUESTION'].isnull()) & (csas['CRITERION'].notnull())]
csacl = csac[csac['CAM_YEAR'] == 2021]
csacl = csacl[['ASP_LCID', 'CRITERION']]

criterion.drop(['DIMENSION', 'CRITERION', 'QUESTION'], axis=1, inplace=True)
criterion = pd.merge(criterion, csacl, on=['ASP_LCID'], how='inner')

#Only use criteria that have been available in at least the last 6 years

#vc = csac['CRITERION'].value_counts()
vc = csac['ASP_LCID'].value_counts()

years = 6
#years = max(vc)
vc = vc[vc >= years].reset_index()
vc.rename(columns={'index': 'ASP_LCID',
                   'ASP_LCID': 'CRITERIONYEARS'}, inplace=True)

criterion = pd.merge(criterion, vc, on=['ASP_LCID'], how='inner')

firm_inf = criterion.groupby(['CSF_LCID', 'month_id'])['INDUSTRY', 'SECTOR', 'INDUSTRYGROUP',
                                                       'CSF_LONGNAME', 'ISIN', 'GVKEY',
                                                       'CIQ_ID', 'COUNTRYNAME', 'DJREGION',
                                                       'Date', 'CAM_YEAR', 'ASOF_DATE',
                                                       'CAM_TYPE', 'CRITERIONYEARS'].last()

#criterion = criterion.pivot_table(index=['CSF_LCID', 'month_id'], 
#                                  columns='CRITERION', values=score)

#crit = ['Biodiversity', 'Brand Management', 'Climate Strategy', 'Corporate Governance',
crit = ['Corporate Governance', 'Climate Strategy',
'Customer Relationship Management', 'Environmental Policy & Management Systems', 
'Environmental Reporting', 'Innovation Management', 'Supply Chain Management']

criterions = criterion.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='CRITERION', values='SCORE')
criterioni = criterion.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='CRITERION', values='SCORE_IMP') 
criterionn = criterion.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='CRITERION', values='SCORE_IMP_NP')                                  
#criterions = criterions[crit]
#criterioni = criterioni[crit]
#criterionn = criterionn[crit]
criterion = pd.merge(criterions, criterioni, left_index=True, right_index=True, how='outer', suffixes=('',' Imp'))
criterion = pd.merge(criterion, criterionn, left_index=True, right_index=True, how='outer', suffixes=('',' NP'))

criterion = pd.merge(criterion, firm_inf, left_index=True, right_index=True)
criterion = criterion.reset_index()

del criterions
del criterioni
del criterionn

cscores = csa.groupby('CRITERION').last().reset_index()
ccol = [c.replace(' NP','') for c in list(criterion.columns) if ' NP' in c]
cscores = cscores.loc[cscores['CRITERION'].isin(ccol)][['CRITERION', 'DIMENSION']]
cscores.to_excel(res_out+'\\Criteria+Dimensions.xlsx') 

#Question

#Only include questions that have been used in the most recent CSA data vintage
#Some question have been renamed, however, they have the same aspect identifier
# Historically, there were 642 aspects with 757 names, currently (2021) there are 325 aspects, 188 are available for the full period

#csaq = csas[csas['CAM_YEAR'] == 2021]
#csaq = csaq[csaq['QUESTION'].notnull()]
#csaq = csaq[['ASP_LCID', 'QUESTION']]

csaq = csas[csas['QUESTION'].notnull()]
csaql = csaq[csaq['CAM_YEAR'] == 2021]
csaql = csaql[['ASP_LCID', 'QUESTION']]

question.drop(['DIMENSION', 'CRITERION', 'QUESTION'], axis=1, inplace=True)
question = pd.merge(question, csaql, on=['ASP_LCID'], how='inner')

#Only use questions that have been available in at least the last 6 years

#vc = csaq['QUESTION'].value_counts()
vc = csaq['ASP_LCID'].value_counts()

years = 6
#years = max(vc)
vc = vc[vc >= years].reset_index()
vc.rename(columns={'index': 'ASP_LCID',
                   'ASP_LCID': 'QUESTIONYEARS'}, inplace=True)

question = pd.merge(question, vc, on=['ASP_LCID'], how='inner')

firm_inf = question.groupby(['CSF_LCID', 'month_id'])['INDUSTRY', 'SECTOR', 'INDUSTRYGROUP',
                                                      'CSF_LONGNAME', 'ISIN', 'GVKEY',
                                                      'CIQ_ID', 'COUNTRYNAME', 'DJREGION',
                                                      'Date', 'CAM_YEAR', 'ASOF_DATE',
                                                      'CAM_TYPE', 'QUESTIONYEARS'].last()

#question = question.pivot_table(index=['CSF_LCID', 'month_id'], 
#                                columns='QUESTION', values=score)

#qu = ['Climate Change Strategy', 'Climate Strategy Impacts', 'Corruption & Bribery',
qu = ['Corruption & Bribery',
'Direct Greenhouse Gas Emissions (Scope 1)', 'ESG Integration in SCM Strategy',
'Environmental Reporting - Assurance', 'Environmental Reporting - Coverage',
'Financial Opportunities Arising from Climate Change',
'Financial Risks of Climate Change', 'Indirect Greenhouse Gas Emissions (Scope 2)',
'Sensitivity Analysis & Stress Testing (including Water and Climate)',
'Supply Chain Risk Exposure', 'Waste Disposal', 'Water Use']
#'Internal Carbon Pricing', 'Process Innovations', 'Environmental Violations' 

questions = question.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='QUESTION', values='SCORE')
questioni = question.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='QUESTION', values='SCORE_IMP')  
questionn = question.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='QUESTION', values='SCORE_IMP_NP')                                  
questions = questions[qu]
questioni = questioni[qu]
questionn = questionn[qu]
question = pd.merge(questions, questioni, left_index=True, right_index=True, how='outer', suffixes=('',' Imp'))
question = pd.merge(question, questionn, left_index=True, right_index=True, how='outer', suffixes=('',' NP'))

question = pd.merge(question, firm_inf, left_index=True, right_index=True)
question = question.reset_index()

del questions
del questioni
del questionn

gc.collect()


#%% Industry-demeaned and standardized levels

ind.rename(columns={'capital_iq':'CIQ_ID'}, inplace=True)

def industry_demean(data, ind):
    var_name = list(data.loc[:,'INDUSTRY':'DJREGION'].columns)
    
    data[var_name] = data.groupby('CSF_LCID')[var_name].ffill()
    
    data = pd.merge(data, ind[['CIQ_ID', 'month_id', 'sector', 'sub_sector']], 
                    on=['CIQ_ID', 'month_id'], how='left')

    var_name = list(data.loc[:,'month_id':'INDUSTRY'].columns)
    var_name = var_name[1:-1]
    
    data.sort_values(['CSF_LCID', 'month_id'], inplace=True)

    #print(data.columns)
    
    for v in var_name:

        #Industry-demeaned:
        data[v+'_ind'] = data.groupby(['INDUSTRY', 'CAM_YEAR'])[v].transform('mean')
        #data[v+'_sd'] = data.groupby(['INDUSTRY', 'CAM_YEAR'])[v].transform('std')
        data[v+'_dm'] = data[v]-data[v+'_ind']
        
        #Industry-standardized:
        data[v+'_ind_sd'] = data.groupby(['INDUSTRY', 'CAM_YEAR'])[v].transform('std')
        data[v + '_sdm'] = data[v+'_dm']/data[v+'_ind_sd'] 

        #Standardized industry z-values:
        data[v + '_mean'] = data.groupby(['CAM_YEAR'])[v + '_sdm'].transform('mean')
        data[v + '_sd'] = data.groupby(['CAM_YEAR'])[v + '_sdm'].transform('std')
        data[v + '_stdm'] = (data[v + '_sdm'] - data[v + '_mean']) / data[v + '_sd']

        #Standardized:
        data[v+'_mean'] = data.groupby(['CAM_YEAR'])[v].transform('mean')
        data[v+'_sd'] = data.groupby(['CAM_YEAR'])[v].transform('std')
        data[v+'_std'] = (data[v]-data[v+'_mean'])/data[v+'_sd']

        #Sector-demeaned:
        #data[v+'_sec'] = data.groupby(['sector', 'CAM_YEAR'])[v].transform('mean')
        #data[v+'_dms'] = data[v]-data[v+'_sec']

        #Adjusted:
        #data[v+'_us'] = -(100-data[v])

        #data.drop([v+'_ind', v+'_sd', v+'_sec', v+'_mean'], axis=1, inplace=True)
        data.drop([v+'_ind', v+'_ind_sd', v+'_sd', v+'_mean'], axis=1, inplace=True)
        
    return data

question = industry_demean(question, ind)
criterion = industry_demean(criterion, ind)
dimension = industry_demean(dimension, ind)
esg = industry_demean(esg, ind)

del ind
del csa
gc.collect()


#%% Determine the type of CSA data to be analyzed:

#Possible choices are: esg, dimension, criterion, question
    
data = criterion.copy(deep=True)
#data = dimension.copy(deep=True)


#Levels:
var_name = list(data.loc[:,'month_id':'INDUSTRY'].columns)
var_name = var_name[1:-1]

#Standardized levels:
var_name_sd = list(data.loc[:,data.columns.str.endswith('_std')].columns)

#Industry-adjusted levels:
#var_name_dm = list(data.loc[:,data.columns.str.endswith('_dm')].columns)
var_name_dm = list(data.loc[:,data.columns.str.endswith('_sdm')].columns)

#Standardized industry z-values:
var_name_sdm = list(data.loc[:,data.columns.str.endswith('_stdm')].columns)

#Sector-demeaned levels:
#var_name_dms = list(data.loc[:,data.columns.str.endswith('_dms')].columns)

#Adjusted levels:
#var_name_us = list(data.loc[:,data.columns.str.endswith('_us')].columns)

try:
    data.drop(['_merge'], axis=1, inplace=True)
except:
    pass

data['month_id'] = pd.to_datetime(data['Date']) - pd.DateOffset(months=-1)
data['month_id'] = pd.to_datetime(data['month_id']).dt.to_period("M")

print('The following variables are analyzed: ' + str(var_name))


#%% Only keep one type of scores: classic, imputed, non-penalized

#Currently: Imputed scores 

print(len(data.columns))

#vars_all = list() 
#for v in var_name:
#    vars_all.extend(list(data.loc[:,data.columns.str.contains(v)].columns))
vars_all = list(data.columns)

vars_np =  [x for x in vars_all if 'NP' in x]
vars_imp = [x.replace(' NP', ' Imp') for x in vars_np]
#vars_imp =  [x for x in vars_all if 'Imp' in x]
#vars_imp =  [x for x in vars_imp if 'Impa' not in x and 'Impr' not in x]
#vars_cla = [x for x in vars_all if x not in vars_imp and x not in vars_np]
vars_cla = [x.replace(' NP', '') for x in vars_np]

#Imputed scores for SA and classic scores for CA: 
#i = 0
#for i in range(0,len(vars_cla)):
#    data.loc[data['CAM_TYPE']=='SA', vars_cla[i]] = data[vars_imp[i]]
#    i=+1
    
data.drop(vars_np, axis=1, inplace=True)
data.drop(vars_cla, axis=1, inplace=True)
#data.drop(vars_imp, axis=1, inplace=True)

#Delete the adjusted data that is not needed 
#Currently: Industry-adjusted

#vars_dm = [x for x in vars_imp if x.endswith('_dm')]
#vars_stdm = [x for x in vars_imp if x.endswith('_stdm')]
#vars_std = [x for x in vars_imp if x.endswith('_std')]

#data.drop(vars_dm, axis=1, inplace=True)
#data.drop(vars_stdm, axis=1, inplace=True)
#data.drop(vars_std, axis=1, inplace=True)

print(len(data.columns))
print(data.columns)

#var_name =  [x for x in list(data.columns) if x.endswith('Imp')]
var_name_sd =  [x for x in list(data.columns) if x.endswith('_std')]
var_name_dm =  [x for x in list(data.columns) if x.endswith('_sdm')]
var_name_sdm =  [x for x in list(data.columns) if x.endswith('_stdm')]
var_name =  [x.replace('_std', '') for x in var_name_sd]

data.drop(['ISIN', 'GVKEY', 'ASOF_DATE'], axis=1, inplace=True)

gc.collect()


#%% Merge financial data with csa data

ret.drop(['cshtrm', 'filingDate', 'fiscalYear', 'year_id', 'periodEndDate',
          'As Reported Balance Sheet Date', 'Revenues', 'Total Assets', 'Book Equity', 
          'mdiff', 'count_ret'], axis=1, inplace=True)
ret.drop(['currency', 'industry', 'region', 'sector', 'sub_industry', 'sub_sector'], 
         axis=1, inplace=True) 

ciq = pd.DataFrame(data['CIQ_ID'].unique()).rename(columns={0:'CIQ_ID'})
ret = pd.merge(ret, ciq, on='CIQ_ID')

#data1 = data.copy(deep=True)
#data1.drop(var_name, axis=1, inplace=True)
#data1.drop(var_name_dm, axis=1, inplace=True)

#del question
#del data
gc.collect()

data = pd.merge(ret, data, on=['CIQ_ID', 'month_id'], 
                how='outer', suffixes=['', '_csa'], indicator=True)

print(len(data))

#Keep specific dates

data = data[pd.to_datetime(data['Date']) > mindate]
data = data[pd.to_datetime(data['Date']) < maxdate]

data.sort_values(['CIQ_ID', 'month_id'], inplace=True)

print(data['_merge'].value_counts())
gc.collect()

#Forward fill, csa data to the next 24 months, industry data generally

info = [x for x in list(data.loc[:,'INDUSTRY':'CAM_YEAR'].columns) if x != 'CIQ_ID']

#Number of months to forward scores:
fw=24

data[['CSF_LCID', 
      'CAM_YEAR']] = data.groupby(['CIQ_ID'])[['CSF_LCID', 'CAM_YEAR']].ffill(limit=fw)
data[info] = data.groupby(['CIQ_ID'])[info].ffill()
data[var_name] = data.groupby(['CIQ_ID'])[var_name].ffill(limit=fw)
#data[var_name_d] = data.groupby(['CIQ_ID'])[var_name_d].ffill(limit=fw)
data[var_name_dm] = data.groupby(['CIQ_ID'])[var_name_dm].ffill(limit=fw)
data[var_name_sd] = data.groupby(['CIQ_ID'])[var_name_sd].ffill(limit=fw)
data[var_name_sdm] = data.groupby(['CIQ_ID'])[var_name_sdm].ffill(limit=fw)
#data[var_name_dms] = data.groupby(['CIQ_ID'])[var_name_dms].ffill(limit=fw)
#data[var_name_us] = data.groupby(['CIQ_ID'])[var_name_us].ffill(limit=fw)

#print(data.describe())
summary = data.describe()
len(data)

#Keep data with CSA values:
    
data = data[data['CSF_LCID'].notnull()]

print(len(data))

data.sort_values(['CIQ_ID', 'month_id'], inplace=True)
#data1 = data[data['month_id']<='2022-09']
#data1 = data.groupby('CIQ_ID').last().reset_index()
#data1 = data1[(data1['month_id']>='2021-11')]
#data1 = data1[(data1['month_id']>='2021-11')|(data1['company_id'].isnull())]
#data1 = data1[['CIQ_ID', 'company_id', 'CSF_LCID', 'company_name', 'market_cap_usd', 'market_cap']]
#data1 = data1[~data1['company_id'].isnull()]
#data1.to_excel(output+'CSA_market_cap.xlsx')

gc.collect()


#%% Add market adjusted returns

#zipfile = ZipFile(input_dir+'\\Financials\\betas_market.zip')
#beta = pd.read_csv(zipfile.open(zipfile.namelist()[1])) 
beta = pd.read_csv(input+'\\Financials\\betas_monthly_data.csv')  
beta = beta[~beta['beta'].isnull()] 
beta['month_id'] = pd.to_datetime(beta['month_id']).dt.to_period('M') 
beta.rename(columns={'capital_iq':'CIQ_ID'}, inplace=True)

market = pd.read_csv(input+'\\Financials\\returns_market_monthly.csv')  
market['month_id'] = pd.to_datetime(market['month_id']).dt.to_period('M') 
 
data = pd.merge(data, beta, on=['CIQ_ID', 'month_id'], how='left')
data = pd.merge(data, market, on=['country', 'month_id'], how='left')

data['ret_e'] = data['ret_usd']-data['beta']*data['mkt_usd']*100

del beta


#%% Prepare analysis

#os.chdir('<Directory>\\Projects')

# Define sets of variables

#reg_vars = list(data.loc[:,'CSF_LCID':'INDUSTRY'].columns[1:-1])
#reg_vars.extend([s + '_dm' for s in reg_vars]+[s + '_std' for s in reg_vars]+[s + '_stdm' for s in reg_vars]+[s + '_dms' for s in reg_vars])
vars = list(data.loc[:,'CSF_LCID':'INDUSTRY'].columns[1:-1])
reg_vars = [s + '_std' for s in vars]
#reg_vars_dm = [s + '_dm' for s in vars]
reg_vars_dm = [s + '_stdm' for s in vars]

#mac_vars = list(econ.columns)[1:]
mac_vars = list(econd.columns)[1:]
print(mac_vars)
#mac_vars = ['cpu_index', 'BBD MPU Index Based on Access World News', 'BBD MPU Index Based on 10 Major Papers', 'US MPU',
#            'Financial Uncertainty', 'Macro Uncertainty', 'Real Uncertainty', 'TEU-ENG', 'TEU-SCA', 'TMU-ENG', 'TMU-SCA', 'VIX', 'VIX_mean',
#            'Global Composite (M+S) PMI Headline Adjusted', 'Global Manufacturing PMI Adjusted']
#mac_vars = ['BBD MPU Index Based on Access World News', 'Financial Uncertainty', 'Macro Uncertainty', 'Real Uncertainty', 
#            'TEU-ENG', 'TMU-ENG', 'VIX_mean', 'Global Manufacturing PMI Adjusted']
#mac_vars = ['BBD MPU Index Based on Access World News', 'Financial Uncertainty', 'Macro Uncertainty', 'Real Uncertainty', 'TEU-ENG', 'TMU-ENG', 'VIX_mean', 
#            'Global Manufacturing PMI Adjusted', 'BBD MPU Index Based on Access World News_D', 'Financial Uncertainty_D', 'Macro Uncertainty_D', 'Real Uncertainty_D', 
#            'TEU-ENG_D', 'TMU-ENG_D', 'VIX_mean_D', 'Global Manufacturing PMI Adjusted_D']
mac_vars = ['Macro Uncertainty', 'Financial Uncertainty', 'GPR', 'TEU-ENG', 'VIX', 'Global Manufacturing PMI Adjusted']
mac_vars_d = ['Macro Uncertainty_D', 'Financial Uncertainty_D', 'GPR_D', 'TEU-ENG_D', 'VIX_D', 'Global Manufacturing PMI Adjusted_D']
#print(mac_vars)

controls = ['size', 'BM', 'IAT', 'Operating Profitability', 'mom212']

# Obtain final dataset

id_vars = ['month_id', 'ret_usd', 'ret_e']
#id_vars.extend(vars)
id_vars.extend(reg_vars)
#id_vars.extend(reg_vars_d)
id_vars.extend(reg_vars_dm)
id_vars.extend(mac_vars)
id_vars.extend(mac_vars_d)
id_vars.extend(controls)

regdata = data[id_vars]

#regdata = data[['month_id', 'ret_usd', 'Environmental Dimension', 'Environmental Dimension Imp', 'Environmental Dimension_dm', 'Environmental Dimension_std', 
#                'Environmental Dimension Imp_dm', 'Environmental Dimension Imp_std', 'cpu_index']]
#regdata['inter'] = (regdata['Environmental Dimension']*regdata['cpu_index'])/1000

#Prepare control variables:
    
for c in controls:
    regdata[c] = regdata.groupby('month_id')[c].transform(
        lambda x: np.maximum(x.quantile(.01), np.minimum(x, x.quantile(.99))))
    
    regdata[c+'_mean'] = regdata.groupby(['month_id'])[c].transform('mean')
    regdata[c+'_sd'] = regdata.groupby(['month_id'])[c].transform('std')
    regdata[c+'_std'] = (regdata[c]-regdata[c+'_mean'])/regdata[c+'_sd']
    regdata.drop([c+'_sd', c+'_mean'], axis=1, inplace=True)

controls = [c+'_std' for c in controls]

regdata.count()

regdata = regdata[regdata['month_id'] < '2022-12']

#print(econ.tail(10))

#Test
#regdt = regdata[~regdata['Water Related Risks Imp_stdm'].isnull()]
#b = regdt.groupby('month_id')['Water Related Risks Imp_stdm'].count()


#%% Cross-sectional regressions

def format_results(df, name):
    n = df.iloc[0,2]
    dfn = df.reset_index()
    dfn = dfn.iloc[0:,1:3]
    #dfn = df.iloc[0:,0:2].unstack()
    dfn = dfn.unstack()
    dfn = dfn.reset_index()
    dfn.sort_values(['level_1'],inplace=True)
    dfn.drop('level_1', axis=1, inplace=True)
    dfn.rename(columns={'level_0':'coefficient', 0:'value'}, inplace=True)

    dfn = dfn.set_index('coefficient')
    dfn = pd.DataFrame(pd.Series(dfn['value']).append(pd.Series(n)))
    dfn.rename(columns={0:name}, inplace=True)

    index_names = []
    for i in range(1,len(df['coef'])+1):
        index_names.append('coef_'+str(i))
        index_names.append('t_stats_'+str(i))
    index_names.append('n')
    dfn = dfn.set_index(pd.Series(index_names))

    return dfn

coef = {}
coefa = {}

# Regression results

results1 = {}
results2 = {}

results1a = {}
results2a = {}

coef1 = {}
coef1a = {}

#for v in vars:
for v in reg_vars:
#for v in reg_vars_dm:

    print(v)

    controls = ['size_std', 'BM_std', 'IAT_std', 'Operating Profitability_std', 'mom212_std']
    #controls = ['size', 'BM', 'IAT', 'Operating Profitability', 'mom212']
    #controls = ['ICAPEX', 'Gross Profitability', 'mom212']
    
    results1[v] = {}
    results2[v] = {}
    
    results1a[v] = {}
    results2a[v] = {}
    
    base_vars = ['month_id','ret_usd','ret_e',v]
    base_vars.extend(mac_vars)
    base_vars.extend(mac_vars_d)
    base_vars.extend(controls)
    regdt = regdata[base_vars]
    regdt = regdt[~regdt[v].isnull()]
    
    regdt.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    for c in controls:
        regdt = regdt[~regdt[c].isnull()]
        #regdt.dropna(subset=[c], inplace=True)
     
    #Minimum number of observations per month:
    regdt['month_obs'] = regdt.groupby(['month_id'])['ret_usd'].transform('count')  
    regdt = regdt[regdt['month_obs']>=20]
    
    #regdt = regdt[~regdt['ret_e'].isnull()]
    #regdt[v] = regdt[v]/100
    #regdt[v] = regdt[v]/10
    #regdt[v] = regdt[v]*10
    
    #fmb1, tb1, reg, beta1 = finfunc.fmb_reg_var(regdt,'month_id','ret_usd',v)
    fmb1, tb1, reg, beta1 = finfunc.fmb_reg_var(regdt,'month_id','ret_e',v)
    print(fmb1)

    #dfoutput = summary_col(reg,stars=True)
    #print(dfoutput)

    r1 = format_results(fmb1,v)
    #r1.rename(columns={v:m}, inplace=True)
    results1[v]=r1
    #results1[v][i]=r1
    beta1 = pd.concat(beta1, axis=1).sum(axis=1, level=0).transpose()
    coef1[v]=beta1
    
    controls.extend([v])
    
    #fmb1a, tb1a, rega, beta1a = finfunc.fmb_reg_var(regdt,'month_id','ret_usd',controls)
    fmb1a, tb1a, rega, beta1a = finfunc.fmb_reg_var(regdt,'month_id','ret_e',controls)
    print(fmb1a)

    r1 = format_results(fmb1a,v)
    results1a[v]=r1
    beta1a = pd.concat(beta1a, axis=1).sum(axis=1, level=0).transpose()
    coef1a[v]=beta1a
    
    i = 1
    #for m in mac_vars_d:
    for m in mac_vars:
        print(i)
        
        controls = ['size_std', 'BM_std', 'IAT_std', 'Operating Profitability_std', 'mom212_std']

        regdt['qu'] = pd.qcut(regdt[m], q=[0, .75, 1], labels=False)
        regdt['qun'] = (regdt['qu']==0).astype(int)
        regdt['inter'] = regdt[v]*regdt['qu']
        regdt['intern'] = regdt[v]*regdt['qun']
        
        rvars = ['inter','intern']

        #fmb2, t2, reg, beta2 = finfunc.fmb_reg_var(regdt,'month_id','ret_usd',[v,'qu','inter'])
        #fmb2, t2, reg, beta2 = finfunc.fmb_reg_var(regdt,'month_id','ret_e',[v,'qu','inter'])
        #fmb2, t2, reg, beta2 = finfunc.fmb_reg_var(regdt,'month_id','ret_usd',['inter','intern'])
        fmb2, t2, reg, beta2 = finfunc.fmb_reg_var(regdt,'month_id','ret_e',['inter','intern'])
        print(fmb2)

        r2 = format_results(fmb2,v)
        r2.rename(columns={v:m}, inplace=True)
        results2[v][i]=r2
        
        rvars.extend(controls)
        
        #fmb2a, t2a, rega, beta2a = finfunc.fmb_reg_var(regdt,'month_id','ret_usd',rvars)
        fmb2a, t2a, rega, beta2a = finfunc.fmb_reg_var(regdt,'month_id','ret_e',rvars)
        print(fmb2a)

        r2 = format_results(fmb2a,v)
        r2.rename(columns={v:m}, inplace=True)
        results2a[v][i]=r2

        i += 1

coef['reg_vars_exc'] = coef1
coefa['reg_vars_exc'] = coef1a
        
#Results 
# significant for Financial Uncertainty, TEU-ENG, 'VIX', 'Global Manufacturing PMI Adjusted'
# mixed for 'BBD MPU Index Based on 10 Major Papers', 'TEU-SCA', 'TMU-ENG'
# not significant for 'BBD MPU Index Based on Access World News', 'Macro Uncertainty', 'Real Uncertainty', 'TMU-SCA', 'VIX_mean', 'Global Composite (M+S) PMI Headline Adjusted'

#Overview over results by criterion/dimension:

cscoresd = cscores.set_index('CRITERION')
cscoresd.loc[cscoresd['DIMENSION']=='Economic Dimension', 'DIMENSION'] = 'Governance & Economic Dimension'

rest = pd.DataFrame()

res1_tab = pd.concat(results1, axis=1).sum(axis=1, level=0)
res1_tab = round(res1_tab, 3)
res1_tab.set_index(pd.Index(['cons', 't_cons', 'coef', 't_coef', 'n']), inplace=True)

for v in reg_vars:
#for v in reg_vars_dm:
    print(v)
        
    results2[v]=dict(zip(mac_vars,list(results2[v].values()))) 
    res2_tab = pd.concat(results2[v], axis=1).sum(axis=1, level=0)
    res2_tab = round(res2_tab, 3)
    res2_tab.set_index(pd.Index(['cons', 't_cons', 'coef_high', 't_coef_high', 
                                'coef_low', 't_coef_low', 'n']), inplace=True)
        
    res_tab = pd.DataFrame()
    res_tab.index = [v]
    #res_tab.columns = ['S'+str(i), 'Sl'+str(i), 'Sln'+str(i), 'Sh'+str(i), 'Shn'+str(i)]
    
    resn_tab = pd.DataFrame(np.empty((1,2)))
    resn_tab.index = [v]
    resn_tab.columns = ['Sl', 'Sln']
    
    res1_tab['S'] = res1_tab[v]>1.645
    res1_tab['Sl'] = res1_tab[v]>1.282
    res1_tab['Sln'] = (res1_tab[v]>0)
    
    resn_tab['Sl'][v] = res1_tab['Sl']['t_coef']
    resn_tab['Sln'][v] = res1_tab['Sln']['t_coef']
    
    res_tab = pd.merge(res_tab, resn_tab, left_index=True, right_index=True)
    
    i=0
    for m in mac_vars:
        
        resn_tab = pd.DataFrame(np.empty((1,12)))
        resn_tab.index = [v]
        resn_tab.columns = ['S'+str(i), 'Sl'+str(i), 'Sln'+str(i), 'Sh'+str(i), 'Shn'+str(i),
                            'B'+str(i), 'Bl'+str(i), 'Bln'+str(i), 'Bh'+str(i), 'Bhn'+str(i),
                            'SB'+str(i), 'SBn'+str(i)]
        
        res2_tab['S'+str(i)] = res2_tab[m]>1.645
        res2_tab['Sl'+str(i)] = res2_tab[m]>1.282
        res2_tab['Sln'+str(i)] = (res2_tab[m]>0)
        
        res2_tab['B'+str(i)] = res2_tab[m]<-1.645
        res2_tab['Bl'+str(i)] = res2_tab[m]<-1.282
        res2_tab['Bln'+str(i)] = (res2_tab[m]<0)
                
        #resn_tab['S'+str(i)][v] = res2_tab['Sl'+str(i)]['t_cons']
        resn_tab['Sl'+str(i)][v] = res2_tab['Sl'+str(i)]['t_coef_low']
        resn_tab['Sln'+str(i)][v] = res2_tab['Sln'+str(i)]['t_coef_low']
        resn_tab['Sh'+str(i)][v] = res2_tab['Sl'+str(i)]['t_coef_high']
        resn_tab['Shn'+str(i)][v] = res2_tab['Sln'+str(i)]['t_coef_high']
        
        resn_tab['Bl'+str(i)][v] = res2_tab['Bl'+str(i)]['t_coef_low']
        resn_tab['Bln'+str(i)][v] = res2_tab['Bln'+str(i)]['t_coef_low']
        resn_tab['Bh'+str(i)][v] = res2_tab['Bl'+str(i)]['t_coef_high']
        resn_tab['Bhn'+str(i)][v] = res2_tab['Bln'+str(i)]['t_coef_high']
        
        resn_tab['SB'+str(i)][v] = (resn_tab['Sl'+str(i)][v]==True)&(resn_tab['Bh'+str(i)]==True)
        resn_tab['SBn'+str(i)][v] = (resn_tab['Sln'+str(i)][v]==True)&(resn_tab['Bhn'+str(i)]==True)
        
        res_tab = pd.merge(res_tab, resn_tab, left_index=True, right_index=True)
        
        i+=1
        
    rest = rest.append(res_tab)
    
for c in list(rest.columns):
    rest[c] = rest[c].astype(bool)

rest['Sc'] = rest[['Sl']].sum(axis=1)
rest['Scl'] = rest[['Sl0', 'Sl1', 'Sl2', 'Sl3', 'Sl4', 'Sl5']].sum(axis=1)
rest['Scln'] = rest[['Sln0', 'Sln1', 'Sln2', 'Sln3', 'Sln4', 'Sln5']].sum(axis=1)
rest['Sch'] = rest[['Sh0', 'Sh1', 'Sh2', 'Sh3', 'Sh4', 'Sh5']].sum(axis=1)
rest['Schn'] = rest[['Shn0', 'Shn1', 'Shn2', 'Shn3', 'Shn4', 'Shn5']].sum(axis=1)
rest['Bcl'] = rest[['Bl0', 'Bl1', 'Bl2', 'Bl3', 'Bl4', 'Bl5']].sum(axis=1)
rest['Bcln'] = rest[['Bln0', 'Bln1', 'Bln2', 'Bln3', 'Bln4', 'Bln5']].sum(axis=1)
rest['Bch'] = rest[['Bh0', 'Bh1', 'Bh2', 'Bh3', 'Bh4', 'Bh5']].sum(axis=1)
rest['Bchn'] = rest[['Bhn0', 'Bhn1', 'Bhn2', 'Bhn3', 'Bhn4', 'Bhn5']].sum(axis=1)

rest.index = [x[:-8] for x in list(rest.index)]
rest = pd.merge(rest, cscoresd, left_index=True, right_index=True)    

rd_dim = rest.groupby('DIMENSION')['Sl', 'Sln', 'S0', 'Sl0', 'Sln0', 'Sh0', 'Shn0', 'B0', 'Bl0', 'Bln0',
       'Bh0', 'Bhn0', 'SB0', 'SBn0', 'S1', 'Sl1', 'Sln1', 'Sh1', 'Shn1', 'B1',
       'Bl1', 'Bln1', 'Bh1', 'Bhn1', 'SB1', 'SBn1', 'S2', 'Sl2', 'Sln2', 'Sh2',
       'Shn2', 'B2', 'Bl2', 'Bln2', 'Bh2', 'Bhn2', 'SB2', 'SBn2', 'S3', 'Sl3',
       'Sln3', 'Sh3', 'Shn3', 'B3', 'Bl3', 'Bln3', 'Bh3', 'Bhn3', 'SB3',
       'SBn3', 'S4', 'Sl4', 'Sln4', 'Sh4', 'Shn4', 'B4', 'Bl4', 'Bln4', 'Bh4',
       'Bhn4', 'SB4', 'SBn4', 'S5', 'Sl5', 'Sln5', 'Sh5', 'Shn5', 'B5', 'Bl5',
       'Bln5', 'Bh5', 'Bhn5', 'SB5', 'SBn5', 'Sc', 'Scl', 'Scln', 'Sch',
       'Schn', 'Bcl', 'Bcln', 'Bch', 'Bchn'].sum()
rd_dim['Sm'] = rd_dim[['Sh0', 'Sh1', 'Sh2', 'Sh3', 'Sh4', 'Sl5']].mean(axis=1)
rd_dim['Smn'] = rd_dim[['Shn0', 'Shn1', 'Shn2', 'Shn3', 'Shn4', 'Sln5']].mean(axis=1)
rd_dim['Bm'] = rd_dim[['Bh0', 'Bh1', 'Bh2', 'Bh3', 'Bh4', 'Bl5']].mean(axis=1)
rd_dim['Bmn'] = rd_dim[['Bhn0', 'Bhn1', 'Bhn2', 'Bhn3', 'Bhn4', 'Bln5']].mean(axis=1)
rd_dim['SBm'] = rd_dim[['SB0', 'SB1', 'SB2', 'SB3', 'SB4', 'SB5']].mean(axis=1)
rd_dim['SBmn'] = rd_dim[['SBn0', 'SBn1', 'SBn2', 'SBn3', 'SBn4', 'SBn5']].mean(axis=1)


#%% Results to table

#All characteristics in one table:

vnew = [x[:4] for x in reg_vars]

pd.set_option("display.max_rows", None, "display.max_columns", None)

res1_tab = pd.concat(results1, axis=1).sum(axis=1, level=0)
res1_tab = round(res1_tab, 3)
res1_tab.set_index(pd.Index(['cons', 't_cons', 'coef', 't_coef', 'n']), inplace=True)
res1_tab.columns = vnew

#sys.stdout = open(res_out+"\\dimension/_exc/_D/_DM.txt",'wt')
sys.stdout = open(res_out+"\\dimension_all.txt",'wt')
print('Dimension Level FMB Regressions')
print(res1_tab.to_markdown(), end=" ")
#print('Dimension Level FMB Regressions', res1_tab, end=" ")
#print(res1_tab, end=" ")
#np.savetxt(res_out+"\\dimension.txt", res1_tab,encoding='UTF-8')

#cont_var = list(itertools.chain(*[[c.replace('_std', ''), 't_'+c.replace('_std', '')] for c in controls]))
cont_var = ['cons', 't_cons']
for c in [[c.replace('_std', ''), 't_'+c.replace('_std', '')] for c in controls]:
    cont_var.extend(c)
cont_var.extend(['coef', 't_coef', 'n'])    

res1_tab = pd.concat(results1a, axis=1).sum(axis=1, level=0)
res1_tab = round(res1_tab, 3)
res1_tab.set_index(pd.Index(cont_var), inplace=True)
res1_tab.columns = vnew

sys.stdout = open(res_out+"\\dimension_all_ct.txt",'wt')
print('Dimension Level FMB Regressions with Controls')
print(res1_tab.to_markdown(), end=" ")

#A separate table for each characteristic:

for v in reg_vars:
#for v in reg_vars_dm:
    print(v)
    
    results2[v]=dict(zip(mac_vars,list(results2[v].values()))) 
    res2_tab = pd.concat(results2[v], axis=1).sum(axis=1, level=0)
    res2_tab = round(res2_tab, 3)
    res2_tab.set_index(pd.Index(['cons', 't_cons', 'coef_high', 't_coef_high', 
                                'coef_low', 't_coef_low', 'n']), inplace=True)

    #sys.stdout = open(res_out+"\\"+v+"_exc/_S/_D/_DM.txt",'wt')
    sys.stdout = open(res_out+"\\"+v+"_all_S.txt",'wt')
    print('Dimension Level FMB Regressions conditioned on Shocks')
    print(res2_tab.to_markdown(), end=" ")
    
    cont_var = ['cons', 't_cons', 'coef_high', 't_coef_high', 'coef_low', 't_coef_low']
    for c in [[c.replace('_std', ''), 't_'+c.replace('_std', '')] for c in controls]:
        cont_var.extend(c)
    cont_var.extend(['n'])  
    
    results2a[v]=dict(zip(mac_vars,list(results2a[v].values()))) 
    res2_tab = pd.concat(results2a[v], axis=1).sum(axis=1, level=0)
    res2_tab = round(res2_tab, 3)
    res2_tab.set_index(pd.Index(cont_var), inplace=True)

    sys.stdout = open(res_out+"\\"+v+"_all_S_ct.txt",'wt')
    print('Dimension Level FMB Regressions with Controls conditioned on Shocks')
    print(res2_tab.to_markdown(), end=" ")

        
#%% Results to graph   

#Baseline results

res1 = pd.concat(results1, axis=1).sum(axis=1, level=0).transpose()
res1 = round(res1, 3)

res1.sort_values('t_stats_2', inplace=True)
res1.index = [x.replace(' Imp_std', '') for x in res1.index.values.tolist()]
#res1.index = [x.replace(' Imp_stdm', '') for x in res1.index.values.tolist()]

res1a = pd.concat(results1a, axis=1).sum(axis=1, level=0).transpose()
res1a = round(res1a, 3)

res1a.sort_values('t_stats_7', inplace=True)
res1a.index = [x.replace(' Imp_std', '') for x in res1a.index.values.tolist()]
#res1a.index = [x.replace(' Imp_stdm', '') for x in res1a.index.values.tolist()]

#All characteristics ranked

res1['t_stats_2'].plot(kind='bar')

#Adjustements

cfsd1 = {}
cfsr1 = {}

cfsd1a = {}
cfsr1a = {}
    
for v in reg_vars:
#for v in reg_vars_dm:
    
    #Scale to volatility of 12% for comparability:    
    cfsd1[v] = coef1[v]*(12/(coef1[v].std()*np.sqrt(12)))
    cfsd1[v] = stats.ttest_1samp(cfsd1[v], popmean=0)[0]
    cfsd1[v] = pd.Series(cfsd1[v]).set_axis(['t_stats_1','t_stats_2'])
    cfsd1[v] = cfsd1[v].append((coef1[v]*(12/(coef1[v].std()*np.sqrt(12)))).mean().set_axis(['coef_1','coef_2']))
    
    cfsd1a[v] = coef1a[v]*(12/(coef1a[v].std()*np.sqrt(12)))
    cfsd1a[v] = stats.ttest_1samp(cfsd1a[v], popmean=0)[0]
    cfsd1a[v] = pd.Series(cfsd1a[v]).set_axis(['t_stats_1','t_stats_2','t_stats_3','t_stats_4','t_stats_5','t_stats_6','t_stats_7'])
    cfsd1a[v] = cfsd1a[v].append((coef1a[v]*(12/(coef1a[v].std()*np.sqrt(12)))).mean().set_axis(['coef_1','coef_2','coef_3','coef_4','coef_5','coef_6','coef_7']))

    #Annualized Sharpe ratio:
    cfsr1[v] = (coef1[v].mean()/coef1[v].std())*np.sqrt(12)   
    cfsr1[v] = cfsr1[v].set_axis(['const', 'coef'])
    
    cfsr1a[v] = (coef1a[v].mean()/coef1a[v].std())*np.sqrt(12)   
    cfsr1a[v] = cfsr1a[v].set_axis(['const','coef_1','coef_2','coef_3','coef_4','coef_5','coef_6'])

cfsd1 = pd.concat(cfsd1, axis=1).sum(axis=1, level=0).transpose()    
cfsd1.sort_values('t_stats_2', inplace=True)
cfsd1.index = [x.replace(' Imp_std', '') for x in cfsd1.index.values.tolist()]
#cfsd1.index = [x.replace(' Imp_stdm', '') for x in cfsd1.index.values.tolist()]

cfsd1a = pd.concat(cfsd1a, axis=1).sum(axis=1, level=0).transpose()    
cfsd1a.sort_values('t_stats_7', inplace=True)
cfsd1a.index = [x.replace(' Imp_std', '') for x in cfsd1a.index.values.tolist()]
#cfsd1a.index = [x.replace(' Imp_stdm', '') for x in cfsd1a.index.values.tolist()]

cfsr1 = pd.concat(cfsr1, axis=1).sum(axis=1, level=0).transpose()    
cfsr1.sort_values('coef', inplace=True)
cfsr1.index = [x.replace(' Imp_std', '') for x in cfsr1.index.values.tolist()]
#cfsr1.index = [x.replace(' Imp_stdm', '') for x in cfsr1.index.values.tolist()]

cfsr1a = pd.concat(cfsr1a, axis=1).sum(axis=1, level=0).transpose()    
cfsr1a.sort_values('coef_6', inplace=True)
cfsr1a.index = [x.replace(' Imp_std', '') for x in cfsr1a.index.values.tolist()]
#cfsr1a.index = [x.replace(' Imp_stdm', '') for x in cfsr1a.index.values.tolist()]
       
#Top and Bottom 5 characteristics

#res1['t_stats_2'].plot(kind='bar')
res1_top = pd.concat([res1.head(5), res1.tail(5)])

#Results depending on shocks indicators

results_m = {}  # Create a new empty dictionary
for i in range(len(mac_vars)):
    results = {}
    for key, value in results2.items():
        #print(value)
        #print(key)  
        results[key] = list(results2[key].values())[i]
    results_m[i] = results
    
    results_m[i] = pd.concat(results_m[i], axis=1).sum(axis=1, level=0).transpose()
    results_m[i] = round(results_m[i], 3)

## Graphs

#colors = [(6/235, 146/235, 126/235),(0, 94/235, 146/235)]
#colors = [(163/235, 147/235, 130/235),(0, 94/235, 146/235)]
colors = [(161/235, 195/235, 218/235),(0, 94/235, 146/235)]

def reg_graph(df, v1, v2, lab):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))

    ax2 = ax.twinx()
    df[v1].plot(kind='bar', color=colors[0], ax=ax2, width=0.25, 
                            position=0, align='center', label='Coefficient')
    df[v2].plot(kind='bar', color=colors[1], ax=ax, width=0.25,
                               position=1, align='center', label='T-statistics')
    plt.xticks()

    #ax.set_xlim(-0.8, 10)
    #ax.set_ylim(-2, 2)
    #ax.set_ylim(-2.5, 2.5)
    ax.set_ylim(-3.5, 3.5)
    ax2.set_ylim(-0.7, 0.7)
    #ax2.set_ylim(-0.8, 0.8)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc=0)

    #ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    #ax2.spines['left'].set_position(11)
    #ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['top'].set_visible(False)

    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    #ax.tick_params(axis='x', top=True, labeltop=True)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    ax.set_ylabel('T-statistics')
    ax2.set_ylabel('Coefficient')
    fig.text(0.5, 0.01, lab, ha='center')
    
def reg_graph_long(df, v1, v2):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 10))

    ax2 = ax.twinx()

    df[v1].plot(kind='bar', color=colors[0], ax=ax2, width=0.25, 
                            position=0, align='center', label='Coefficient')
    df[v2].plot(kind='bar', color=colors[1], ax=ax, width=0.25,
                               position=1, align='center', label='T-statistics')
    plt.xticks()

    #ax.set_xlim(-0.5, 10)
    ax.set_ylim(-2.5, 2.5)
    ax2.set_ylim(-1.25, 1.25)
    #ax.set_ylim(-3, 3)
    #ax2.set_ylim(-1.5, 1.5)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc=0)
    
    #ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    #ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['top'].set_visible(False)

    #If text is supposed to appear above graph:
    #ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    
    #If text is supposed to appear below graph:
    ax.tick_params(axis='x', pad=220)
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    ax.set_ylabel('T-statistics')
    ax2.set_ylabel('Coefficient')
    
def reg_graph_single(df, v1):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 10))

    df[v1].plot(kind='bar', color=colors[1], width=0.5, 
                            position=0, align='center', label='Sharpe Ratio')
    plt.xticks()

    #ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1, 1)

    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc=0)
    
    #ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)

    #If text is supposed to appear above graph:
    #ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    
    #If text is supposed to appear below graph:
    ax.tick_params(axis='x', pad=220)
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    ax.set_ylabel('Sharpe Ratio')
    
#All characteristics ranked

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

reg_graph_long(res1, 'coef_2', 't_stats_2')  
#plt.savefig(res_out+'\\_Criteria_Imp_Sd_Exc_DM.png')  
plt.savefig(res_out+'\\Criteria_Imp_Sd.pdf')

reg_graph_long(res1a, 'coef_7', 't_stats_7')  
#plt.savefig(res_out+'\\_Criteria_Imp_Sd_Exc_Con_DM.png')  
plt.savefig(res_out+'\\Criteria_Imp_Sd_Con.pdf')

#Top and Bottom 5 characteristics:    

reg_graph(res1_top, 'coef_2', 't_stats_2', lab=None)    
#plt.savefig(res_out+'\\_Criteria_Imp_Sd_Exc_DM.png')
plt.savefig(res_out+'\\Criteria_Imp_Sd.pdf')

#Scaled returns

reg_graph_long(cfsd1, 'coef_2', 't_stats_2')  
#plt.savefig(res_out+'\\_Criteria_Imp_Sd_Exc_DM.png')  
plt.savefig(res_out+'\\Criteria_Imp_Sd_Scaled.pdf')

reg_graph_long(cfsd1a, 'coef_7', 't_stats_7')  
#plt.savefig(res_out+'\\_Criteria_Imp_Sd_Exc_Con_DM.png')  
plt.savefig(res_out+'\\Criteria_Imp_Sd_Con_Scaled.pdf')

#Share ratio

reg_graph_single(cfsr1, 'coef')  
#plt.savefig(res_out+'\\_Criteria_Imp_Sd_Exc.png')  
plt.savefig(res_out+'\\Criteria_Imp_Sd_SR.pdf')

#For each shock

plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels

for i in range(len(mac_vars)):
    results_m[i].sort_values('t_stats_2', inplace=True)
    results_m[i].index = [x.replace(' Imp_std', '') for x in results_m[i].index.values.tolist()]
    #results_m[i].index = [x.replace(' Imp_stdm', '') for x in results_m[i].index.values.tolist()]
    res_top = pd.concat([results_m[i].head(5), results_m[i].tail(5)])
    
    reg_graph(res_top, 'coef_2', 't_stats_2', mac_vars[i]+' High')    
    plt.savefig(res_out+'\\Criteria_Imp_Sd_'+mac_vars[i]+'_H.pdf')
    #plt.savefig(res_out+'\\Criteria_Imp_Sd_Exc_Dm_'+mac_vars[i]+'_H.png')
    
    results_m[i].sort_values('t_stats_3', inplace=True)
    results_m[i].index = [x.replace(' Imp_std', '') for x in results_m[i].index.values.tolist()]
    #results_m[i].index = [x.replace(' Imp_stdm', '') for x in results_m[i].index.values.tolist()]
    res_top = pd.concat([results_m[i].head(5), results_m[i].tail(5)])
    
    reg_graph(res_top, 'coef_3', 't_stats_3', mac_vars[i]+' Low')    
    plt.savefig(res_out+'\\Criteria_Imp_Sd_'+mac_vars[i]+'_L.pdf')
    #plt.savefig(res_out+'\\Criteria_Imp_Sd_Exc_Dm_'+mac_vars[i]+'_L.png')
    

#%%1

fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))
#plt.xticks(rotation=90)

ax2 = ax.twinx()
#ax.bar(np.arange(len(res1_top['t_stats_2'])), res1_top['t_stats_2'], width=0.25, color='r')
#ax2.bar(np.arange(len(res1_top['t_stats_2']))+0.25, res1_top['coef_2'], width=0.25, color='b')
res1_top['coef_2'].plot(kind='bar', color='r', ax=ax2, width=0.25, 
                        position=0, align='center', label='Coefficient')
res1_top['t_stats_2'].plot(kind='bar', color='b', ax=ax, width=0.25,
                           position=1, align='center', label='T-statistics')
plt.xticks()

ax.set_xlim(-0.5, 10)
#ax.set_ylim(-2, 2)
ax.set_ylim(-2.5, 2.5)
#ax2.set_ylim(-0.5, 0.5)
ax2.set_ylim(-0.8, 0.8)

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc=0)
#ax.bar(range(len(res1_top['t_stats_2'])), res1_top['t_stats_2'], width=0.25, color='r')
#ax2.bar(range(len(res1_top['coef_2']))+0.25, res1_top['coef_2'], width=0.25, color='b')

#ax.spines['left'].set_position('zero')
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_visible(False)
#ax2.spines['left'].set_position(11)
#ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_position('zero')
ax2.spines['top'].set_visible(False)

ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#ax.tick_params(axis='x', top=True, labeltop=True)
plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

ax.set_ylabel('T-statistics')
ax2.set_ylabel('Coefficient')

##2

res1_top['t_stats_2'].plot.bar(
    xlabel='Family Member', ylabel='t-statistics').spines[
        'bottom'].set_position(('data', 0))
plt.twinx()   
plt.xticks(rotation=90)     
res1_top['coef_2'].plot.bar(left=plt.xticks()[0],
    xlabel='Family Member', ylabel='t-statistics', align='center', alpha=0.3).spines[
        'bottom'].set_position(('data', 0))

##3

res1_top[['coef_2', 't_stats_2']].plot.bar(
    xlabel='Family Member', ylabel='t-statistics').spines[
        'bottom'].set_position(('data', 0))
plt.xticks(rotation=90)
plt.spines['right'].set_visible(False)
plt.spines['top'].set_visible(False)

##4

res1_left = res1_top.copy(deep=True)
res1_left.iloc[0:5,:] = 0
res1_right = res1_top.copy(deep=True)
res1_right.iloc[5:10,:] = 0

fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))
#plt.xticks(rotation=90)

ax2 = ax.twinx()
ax3 = ax.twinx()
ax4 = ax.twinx()
#ax.bar(np.arange(len(res1_top['t_stats_2'])), res1_top['t_stats_2'], width=0.25, color='r')
#ax2.bar(np.arange(len(res1_top['t_stats_2']))+0.25, res1_top['coef_2'], width=0.25, color='b')
res1_right['coef_2'].plot(kind='bar', color='r', ax=ax2, width=0.25, 
                        position=0, align='center', label='Coefficient')
res1_right['t_stats_2'].plot(kind='bar', color='b', ax=ax, width=0.25,
                           position=1, align='center', label='T-statistics')

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc=0)

res1_left['coef_2'].plot(kind='bar', color='r', ax=ax2, width=0.25, 
                        position=0, align='center', label='Coefficient')
res1_left['t_stats_2'].plot(kind='bar', color='b', ax=ax, width=0.25,
                           position=1, align='center', label='T-statistics')
plt.xticks()

ax.set_xlim(-0.5, 10)
ax.set_ylim(-2, 2)
ax2.set_ylim(-0.5, 0.5)
ax3.set_ylim(-2, 2)
ax2.set_ylim(-0.5, 0.5)

#ax.spines['left'].set_position('zero')
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_visible(False)
#ax2.spines['left'].set_position(11)
#ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_position('zero')
ax2.spines['top'].set_visible(False)

ax3.spines['left'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['top'].set_visible(False)

#ax3.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax.tick_params(axis='x', top=True, labeltop=True)
ax3.tick_params(axis='x', top=True, labeltop=True)
plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
plt.setp(ax3.get_xticklabels(), rotation=90, ha='right')

ax.set_ylabel('T-statistics')
ax2.set_ylabel('Coefficient')


#%% Portfolio sorts:

reg_vars_new = list(data.loc[:,'CSF_LCID':'INDUSTRY'].columns[1:-1])
reg_vars_dm = [v+'_stdm' for v in reg_vars_new]
#reg_vars_dm = [v+'_dm' for v in reg_vars_new]
    
pdata = finfunc.portfolio_sort_short(data, ['CSF_LCID', 'month_id'], 'month_id', reg_vars_new, 5)
pdatad = finfunc.portfolio_sort_short(data, ['CSF_LCID', 'month_id'], 'month_id', reg_vars_dm, 5)

#Return computation:

rdata = finfunc.portfolio_return(pdata, 'month_id', reg_vars_new, 5, 'ret_usd', weight='market_cap_usd')
rdatad = finfunc.portfolio_return(pdatad, 'month_id', reg_vars_dm, 5, 'ret_usd', weight='market_cap_usd')

##Create macroeconomic control variables:

rdata.describe()

#rdata = pd.merge(rdata, econ, on='month_id', how='left')
rdata = pd.merge(rdata, econd, on='month_id', how='left')
#rdatad = pd.merge(rdatad, econ, on='month_id', how='left')
rdatad = pd.merge(rdatad, econd, on='month_id', how='left')

for i in range(0,len(mac_vars)):
    print(mac_vars[i])
    rdata['qu'+str(i)] = pd.qcut(rdata[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
    rdatad['qu'+str(i)] = pd.qcut(rdatad[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)

##Replace punctuation in column names

##column_names = list(rdata.columns)
#column_names = [''.join(c for c in s if c not in string.punctuation) for s in column_names]
##column_names = [''.join(c for c in s if c not in ['(',')']) for s in column_names]

##column_names_d = list(rdatad.columns)
##column_names_d = [''.join(c for c in s if c not in ['(',')']) for s in column_names_d]

##print(column_names)

##rdata.columns = column_names    
##rdatad.columns = column_names_d    

gc.collect()


#%% Time-series regressions preparation:

rdata.describe()

#reg_vars_new =  [v for v in reg_vars if not any(s in v for s in ['_std', '_stdm', '_dms'])] 
#reg_vars_new = list(data.loc[:,'CSF_LCID':'INDUSTRY'].columns[1:-1])
reg_vars_new = [''.join(c for c in s if c not in ['(',')']) for s in reg_vars_new]
reg_vars_dm = [''.join(c for c in s if c not in ['(',')']) for s in reg_vars_dm]

print(reg_vars_new)
print(rdata.columns)

##All portfolios and long/short portfolios:
    
#res = sm.OLS(reg_vars['Environmental Dimension_ls_5'],sm.add_constant(Y[c])).fit(cov_type=cvt, cov_kwds=cvk)

#regt,reg = finfunc.ts_reg_det(rdata, reg_vars, top=5, controls=None, nw=6)
#regt,reg = finfunc.ts_reg_det(rdata, reg_vars_new, top=5, controls=['qu5'], nw=6)
#regt,reg,res = finfunc.ts_reg(rdata, reg_vars_new, top=5, controls=['qu5'], nw=6)

#for i in range(0,len(mac_vars)):
#    regt,reg = finfunc.ts_reg_det(rdata, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)

##All macroeconomic variables and only long/short portfolios

ls = ['month_id']
lsd = ['month_id']
ls.extend(list(rdata.loc[:,(rdata.columns.str.contains('_ls_'))].columns))
lsd.extend(list(rdatad.loc[:,(rdatad.columns.str.contains('_ls_'))].columns))

rdatan = rdata[ls] 
rdatadn = rdatad[lsd] 

#rdatan = pd.merge(rdatan, econ, on='month_id', how='left')
rdatan = pd.merge(rdatan, econd, on='month_id', how='left')
rdatadn = pd.merge(rdatadn, econd, on='month_id', how='left')

rdatan = rdatan[rdatan['month_id'] < '2022-12']
rdatadn = rdatadn[rdatadn['month_id'] < '2022-12']


#%% Regressions and tables:

ff5 = pd.read_csv(input+'\\Financials\\Factors\\5_Factors.csv')
mom = pd.read_csv(input+'\\Financials\\Factors\\Mom_Factors.csv')

ff5 = ff5[ff5.region=='NAM']
mom = mom[mom.region=='NAM']

ff5['month_id'] = pd.to_datetime(ff5['month_id']).dt.to_period('M') 
mom['month_id'] = pd.to_datetime(mom['month_id']).dt.to_period('M') 

rdatan = pd.merge(rdatan, ff5, on='month_id', how='left')
rdatan = pd.merge(rdatan, mom, on='month_id', how='left')
rdatadn = pd.merge(rdatadn, ff5, on='month_id', how='left')
rdatadn = pd.merge(rdatadn, mom, on='month_id', how='left')

aqr = pd.read_csv(input+'\\Financials\\Factors\\AQR_Factors.csv')

aqr['month_id'] = pd.to_datetime(aqr['Date']).dt.to_period('M') 

rdatan = pd.merge(rdatan, aqr, on='month_id', how='left', suffixes=['_N',''])
rdatadn = pd.merge(rdatadn, aqr, on='month_id', how='left', suffixes=['_N',''])

factors = ['Mkt-RF', 'SMB_N', 'HML_N', 'RMW', 'CMA', 'WML']
factorsa = ['MKT', 'SMB', 'HML', 'QMJ', 'UMD']
mkt = ['Mkt-RF']

#Standard regression:

reg_rs = []
reg_rsd = []

regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, nw=6)
regtd_ls,tvld_ls,regtd,tvld,regsd = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, nw=6)
    
regt_ls,tvl_ls,regt,tvl,regsm = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, 
                                                   controls=['Mkt-RF'], nw=6)
regtd_ls,tvld_ls,regtd,tvld,regsdm = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, 
                                                        controls=['Mkt-RF'], nw=6)
    
regt_ls,tvl_ls,regt,tvl,regs5 = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, 
                                                   controls=factorsa, nw=6)
regtd_ls,tvld_ls,regtd,tvld,regsd5 = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, 
                                                        controls=factorsa, nw=6)
    
for j in range(0,len(reg_vars_new)):
    reg_rs.extend([regs[j]])
    reg_rs.extend([regsm[j]])
    reg_rs.extend([regs5[j]])
    reg_rsd.extend([regsd[j]])
    reg_rsd.extend([regsdm[j]])
    reg_rsd.extend([regsd5[j]])

#Dimension:
        
dfoutput = summary_col(reg_rs, model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3', 'Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['MKT', 'SMB', 'HML'])
                       #stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension.txt",'wt')
print(dfoutput)

dfoutput = summary_col(reg_rsd, model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3', 'Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f',regressor_order=['MKT', 'SMB', 'HML'])
                       #stars=True, float_format='%.3f',regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_DM.txt",'wt')
print(dfoutput)

#Criterion:

vnew = [x[:4] for x in reg_vars_new]
vnew = list(itertools.chain.from_iterable([[x+' 1', x+' 2', x+' 3'] for x in vnew]))

dfoutput = summary_col(reg_rs[:105], model_names=vnew[:105], stars=True, float_format='%.3f', 
                       regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_criterion_1.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rs[105:], model_names=vnew[105:], stars=True, float_format='%.3f', 
                       regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_criterion_2.txt",'wt')
print(dfoutput)

dfoutput = summary_col(reg_rsd[:105], model_names=vnew[:105], stars=True, float_format='%.3f', 
                       regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_criterion_DM_1.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rsd[105:], model_names=vnew[105:], stars=True, float_format='%.3f', 
                       regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_criterion_DM_2.txt",'wt')
print(dfoutput)

#Regressions controlling for macro factors:

reg_rsm = []
reg_rsdm = []

for i in range(0,len(mac_vars)):
    print(i)
    
    controlv=['qu'+str(i)]
    controlv.extend(factors)
        
    rdatan['qu'+str(i)] = pd.qcut(rdatan[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
    #regt,reg,regres = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)
    #regt_ls,reg_ls,regt,reg = ts_reg_det(rdatan, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)
    regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, 
                                                      controls=['qu'+str(i)], nw=6)
    
    rdatadn['qu'+str(i)] = pd.qcut(rdatadn[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
    #regtd,regd,regresd = finfunc.ts_reg_det(rdatadn, reg_vars_d, top=5, controls=['qu'+str(i)], nw=6)
    #regtd_ls,regd_ls,regtd,regd = ts_reg_det(rdatadn, reg_vars_d, top=5, controls=['qu'+str(i)], nw=6)
    regtd_ls,tvld_ls,regtd,tvld,regsd = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, 
                                                           controls=['qu'+str(i)], nw=6)
    
    regt_ls,tvl_ls,regt,tvl,regsm = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, 
                                                       controls=['qu'+str(i), 'Mkt-RF'], nw=6)
    regtd_ls,tvld_ls,regtd,tvld,regsdm = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, 
                                                           controls=['qu'+str(i), 'Mkt-RF'], nw=6)
    
    regt_ls,tvl_ls,regt,tvl,regs5 = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, 
                                                       controls=controlv, nw=6)
    regtd_ls,tvld_ls,regtd,tvld,regsd5 = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, 
                                                           controls=controlv, nw=6)
    
    for j in range(0,len(reg_vars_new)):
        reg_rsm.extend([regs[j]])
        reg_rsm.extend([regsm[j]])
        reg_rsm.extend([regs5[j]])
        reg_rsdm.extend([regsd[j]])
        reg_rsdm.extend([regsdm[j]])
        reg_rsdm.extend([regsd5[j]])
        
#Levels
    
dfoutput = summary_col(reg_rsm[:9],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_Macro_Uncertainty.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rsm[9:18],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_Financial_Uncertainty.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rsm[18:27],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_GPR.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rsm[27:36],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_TEU-ENG.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rsm[36:45],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_VIX.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rsm[45:54],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_PMI.txt",'wt')
print(dfoutput)

#Demeaned

dfoutput = summary_col(reg_rsdm[:9],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_Macro_Uncertainty_DM.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rsdm[9:18],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_Financial_Uncertainty_DM.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rsdm[18:27],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_TEU-ENG_DM.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rsdm[27:36],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_VIX_DM.txt",'wt')
print(dfoutput)
dfoutput = summary_col(reg_rsdm[36:45],model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
                                            'Gov 3','Soc 1','Soc 2','Soc 3'],
                       stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension_PMI_DM.txt",'wt')
print(dfoutput)


#%% Functions for later plotting

#colors = [(6/235, 146/235, 126/235),(0, 94/235, 146/235)]
#colors = [(163/235, 147/235, 130/235),(0, 94/235, 146/235)]
#colors = [(161/235, 195/235, 218/235),(0, 94/235, 146/235)]
colors = [(153/235, 228/235, 215/235),(10/235, 107/235, 91/235)]

def reg_graph(df, v1, v2, lab):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))

    ax2 = ax.twinx()
    df[v1].plot(kind='bar', color=colors[0], ax=ax2, width=0.25, 
                            position=0, align='center', label='Coefficient')
    df[v2].plot(kind='bar', color=colors[1], ax=ax, width=0.25,
                               position=1, align='center', label='T-statistics')
    plt.xticks()

    ax.set_xlim(-0.5, 10)
    #ax.set_ylim(-2, 2)
    #ax.set_ylim(-2.5, 2.5)
    ax.set_ylim(-4.5, 4.5)
    ax2.set_ylim(-4.5, 4.5)
    #ax2.set_ylim(-0.8, 0.8)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc=0)

    #ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    #ax2.spines['left'].set_position(11)
    #ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['top'].set_visible(False)

    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    #ax.tick_params(axis='x', top=True, labeltop=True)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    ax.set_ylabel('T-statistics')
    ax2.set_ylabel('Coefficient')
    fig.text(0.5, 0.01, lab, ha='center')
    
def reg_graph_long(df, v1, v2):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 10))

    ax2 = ax.twinx()

    df[v1].plot(kind='bar', color=colors[0], ax=ax2, width=0.25, 
                            position=0, align='center', label='Coefficient')
    df[v2].plot(kind='bar', color=colors[1], ax=ax, width=0.25,
                               position=1, align='center', label='T-statistics')
    plt.xticks()

    #ax.set_xlim(-0.5, 10)
    ax.set_ylim(-3.5, 3.5)
    ax2.set_ylim(-1.75, 1.75)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc=0)
    
    #ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    #ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['top'].set_visible(False)

    #If text is supposed to appear above graph:
    #ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    
    #If text is supposed to appear below graph:
    ax.tick_params(axis='x', pad=220)
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    ax.set_ylabel('T-statistics')
    ax2.set_ylabel('Coefficient')
    
def reg_graph_single(df, v1):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 10))

    df[v1].plot(kind='bar', color=colors[1], width=0.5, 
                            position=0, align='center', label='Sharpe Ratio')
    plt.xticks()

    #ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1, 1)

    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc=0)
    
    #ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)

    #If text is supposed to appear above graph:
    #ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    
    #If text is supposed to appear below graph:
    ax.tick_params(axis='x', pad=220)
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    ax.set_ylabel('Sharpe Ratio')


#%% All regression coefficients from same regression (Low state effects vs. high state difference effect)

rd = {}
rdn = {}
tv = {}
tvn = {}    

#for i in range(len(mac_vars_d)):
for i in range(0,len(mac_vars)):
    print(i)
    
    controlv=['qu'+str(i)]
    controlv.extend(factorsa)
    
    rdatan['qu'+str(i)] = pd.qcut(rdatan[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
    #regt,reg,regres = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)
    #regt_ls,reg_ls,regt,reg = ts_reg_det(rdatan, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)
    regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)
    #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, controls=controlv, nw=6)
    
    rdatadn['qu'+str(i)] = pd.qcut(rdatadn[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
    #regtd,regd,regresd = finfunc.ts_reg_det(rdatadn, reg_vars_d, top=5, controls=['qu'+str(i)], nw=6)
    #regtd_ls,regd_ls,regtd,regd = ts_reg_det(rdatadn, reg_vars_d, top=5, controls=['qu'+str(i)], nw=6)
    regtd_ls,tvld_ls,regtd,tvld,regsd = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, controls=['qu'+str(i)], nw=6)
    #regtd_ls,tvld_ls,regtd,tvld,regsd = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, controls=controlv, nw=6)
    
    rd[i] = pd.DataFrame(regt_ls)
    rdn[i] = pd.DataFrame(regtd_ls)
    
    rd[i].columns = reg_vars_new
    rdn[i].columns = reg_vars_new
    
    tv[i] = pd.DataFrame(tvl_ls)
    tvn[i] = pd.DataFrame(tvld_ls)
    
    tv[i].columns = reg_vars_new
    tvn[i].columns = reg_vars_new
    
    tv[i].index = ['t_const', 't_qu'+str(i)]
    tvn[i].index = ['t_const', 't_qu'+str(i)]
    #tv[i].index = ['t_const', 't_qu'+str(i), 't_1', 't_2', 't_3', 't_4', 't_5']
    #tvn[i].index = ['t_const', 't_qu'+str(i), 't_1', 't_2', 't_3', 't_4', 't_5']
    
    #rd[i].append(tv[i])
    #rdn[i].append(tvn[i])
    
    rd[i] = rd[i].transpose()
    rdn[i] = rdn[i].transpose()
    
    tv[i] = tv[i].transpose()
    tvn[i] = tvn[i].transpose()   
    
    rd[i] = pd.merge(rd[i], tv[i], left_index=True, right_index=True)
    rdn[i] = pd.merge(rdn[i], tvn[i], left_index=True, right_index=True)
   
## Plot results for each shock

plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels

#Generate plots:

#for i in range(len(mac_vars_d)):
for i in range(0,len(mac_vars)):
    
    rd[i].sort_values('t_const', inplace=True)
    rd[i].index = [x.replace(' Imp', '') for x in rd[i].index.values.tolist()]
    rd_top = pd.concat([rd[i].head(5), rd[i].tail(5)])
        
    reg_graph(rd_top, 'const', 't_const', mac_vars[i]+' Low')    
    plt.savefig(res_out+'\\LS_Criteria_Imp_'+mac_vars[i]+'_L.pdf')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_'+mac_vars[i]+'_alpha_L.png')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_D_'+mac_vars[i]+'_L.png')
    
    rd[i].sort_values('t_qu'+str(i), inplace=True)
    rd[i].index = [x.replace(' Imp', '') for x in rd[i].index.values.tolist()]
    rd_top = pd.concat([rd[i].head(5), rd[i].tail(5)])
    
    reg_graph(rd_top, 'qu'+str(i), 't_qu'+str(i), mac_vars[i]+' High')    
    plt.savefig(res_out+'\\LS_Criteria_Imp_'+mac_vars[i]+'_H.pdf')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_'+mac_vars[i]+'_alpha_H.png')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_D_'+mac_vars[i]+'_H.png')

#for i in range(len(mac_vars_d)):
for i in range(0,len(mac_vars)):
    
    rdn[i].sort_values('t_const', inplace=True)
    rdn[i].index = [x.replace(' Imp', '') for x in rdn[i].index.values.tolist()]
    rdn_top = pd.concat([rdn[i].head(5), rdn[i].tail(5)])
        
    reg_graph(rdn_top, 'const', 't_const', mac_vars[i]+' Low')    
    plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_'+mac_vars[i]+'_L.pdf')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_'+mac_vars[i]+'_alpha_L.png')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_D_'+mac_vars[i]+'_L.png')
    
    rdn[i].sort_values('t_qu'+str(i), inplace=True)
    rdn[i].index = [x.replace(' Imp', '') for x in rdn[i].index.values.tolist()]
    rdn_top = pd.concat([rdn[i].head(5), rdn[i].tail(5)])
    
    reg_graph(rdn_top, 'qu'+str(i), 't_qu'+str(i), mac_vars[i]+' High')    
    plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_'+mac_vars[i]+'_H.pdf')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_'+mac_vars[i]+'_alpha_H.png')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_D_'+mac_vars[i]+'_H.png')
    
    
#%% Full sample coefficients vs. adverse condition coefficients (Full sample vs. high state difference effect) 

rd = {}
rdn = {}
tv = {}
tvn = {}

#for i in range(0,len(mac_vars_d)):    
for i in range(0,len(mac_vars)):
    print(i)
        
    controlv=['qu'+str(i)]
    controlv.extend(factorsa)
    
    rdatan['qu'+str(i)] = pd.qcut(rdatan[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
    #rdatan['qu'+str(i)] = pd.qcut(rdatan[mac_vars_d[i]], q=[0, .75, 1], labels=False).astype(float)
    regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, nw=6)
    #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, controls=factorsa, nw=6)
    
    rdatadn['qu'+str(i)] = pd.qcut(rdatadn[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
    #rdatadn['qu'+str(i)] = pd.qcut(rdatadn[mac_vars_d[i]], q=[0, .75, 1], labels=False).astype(float)
    regtd_ls,tvld_ls,regtd,tvld,regsd = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, nw=6)
    #regtd_ls,tvld_ls,regtd,tvld,regsd = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, controls=factorsa, nw=6)
     
    rd[i] = pd.DataFrame(regt_ls)
    rdn[i] = pd.DataFrame(regtd_ls)
    
    rd[i].columns = reg_vars_new
    rdn[i].columns = reg_vars_new
    
    tv[i] = pd.DataFrame(tvl_ls)
    tvn[i] = pd.DataFrame(tvld_ls)
    
    tv[i].columns = reg_vars_new
    tvn[i].columns = reg_vars_new
    
    tv[i].index = ['t_const']
    tvn[i].index = ['t_const']
    #tv[i].index = ['t_const', 't_1', 't_2', 't_3', 't_4', 't_5']
    #tvn[i].index = ['t_const', 't_1', 't_2', 't_3', 't_4', 't_5']
    
    rd[i] = rd[i].transpose()
    rdn[i] = rdn[i].transpose()
    
    tv[i] = tv[i].transpose()
    tvn[i] = tvn[i].transpose()  
    
    rd[i] = pd.merge(rd[i], tv[i], left_index=True, right_index=True)
    rdn[i] = pd.merge(rdn[i], tvn[i], left_index=True, right_index=True)
    
    regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)
    regtd_ls,tvld_ls,regtd,tvld,regsd = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, controls=['qu'+str(i)], nw=6)
    #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, controls=controlv, nw=6)
    #regtd_ls,tvld_ls,regtd,tvld,regsd = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, controls=controlv, nw=6)
    
    regt_ls = pd.DataFrame(regt_ls)
    regtd_ls = pd.DataFrame(regtd_ls)
    
    regt_ls.columns = reg_vars_new
    regtd_ls.columns = reg_vars_new
    
    tvl_ls = pd.DataFrame(tvl_ls)
    tvld_ls = pd.DataFrame(tvld_ls)
    
    tvl_ls.columns = reg_vars_new
    tvld_ls.columns = reg_vars_new
    
    tvl_ls.index = ['t_const', 't_qu'+str(i)]
    tvld_ls.index = ['t_const', 't_qu'+str(i)]
    #tvl_ls.index = ['t_const', 't_qu'+str(i), 't_1', 't_2', 't_3', 't_4', 't_5']
    #tvld_ls.index = ['t_const', 't_qu'+str(i), 't_1', 't_2', 't_3', 't_4', 't_5']
    
    regt_ls = regt_ls.transpose()
    regtd_ls = regtd_ls.transpose()
    
    tvl_ls = tvl_ls.transpose()
    tvld_ls = tvld_ls.transpose()   
    
    rd[i] = pd.merge(rd[i], regt_ls['qu'+str(i)], left_index=True, right_index=True)
    rd[i] = pd.merge(rd[i], tvl_ls['t_qu'+str(i)], left_index=True, right_index=True)
    rdn[i] = pd.merge(rdn[i], regtd_ls['qu'+str(i)], left_index=True, right_index=True)
    rdn[i] = pd.merge(rdn[i], tvld_ls['t_qu'+str(i)], left_index=True, right_index=True)        
    
## Plot results for each shock

#Generate plots:
    
rd[0].sort_values('t_const', inplace=True)
rd[0].index = [x.replace(' Imp', '') for x in rd[0].index.values.tolist()]
rd_top = pd.concat([rd[0].head(5), rd[0].tail(5)])
    
reg_graph(rd_top, 'const', 't_const', lab=None)    
plt.savefig(res_out+'\\LS_Criteria_Imp.pdf')
#plt.savefig(res_out+'\\LS_Criteria_Imp_alpha.png')
#plt.savefig(res_out+'\\LS_Criteria_Imp_D.png')

#for i in range(len(mac_vars_d)):
for i in range(0,len(mac_vars)):

    rd[i].sort_values('t_qu'+str(i), inplace=True)
    rd[i].index = [x.replace(' Imp', '') for x in rd[i].index.values.tolist()]
    rd_top = pd.concat([rd[i].head(5), rd[i].tail(5)])
    
    reg_graph(rd_top, 'qu'+str(i), 't_qu'+str(i), mac_vars[i]+' High')    
    plt.savefig(res_out+'\\LS_Criteria_Imp_'+mac_vars[i]+'_H.pdf')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_'+mac_vars[i]+'_alpha_H.png')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_D_'+mac_vars[i]+'_H.png')

rdn[0].sort_values('t_const', inplace=True)
rdn[0].index = [x.replace(' Imp', '') for x in rdn[0].index.values.tolist()]
rdn_top = pd.concat([rdn[0].head(5), rdn[0].tail(5)])
    
reg_graph(rdn_top, 'const', 't_const', lab=None)    
plt.savefig(res_out+'\\LS_Criteria_Imp_Dm.pdf')
#plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_alpha.png')
#plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_D.png')

#for i in range(len(mac_vars_d)):
for i in range(0,len(mac_vars)):
    
    rdn[i].sort_values('t_qu'+str(i), inplace=True)
    rdn[i].index = [x.replace(' Imp', '') for x in rdn[i].index.values.tolist()]
    rdn_top = pd.concat([rdn[i].head(5), rdn[i].tail(5)])
    
    reg_graph(rdn_top, 'qu'+str(i), 't_qu'+str(i), mac_vars[i]+' High')    
    plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_'+mac_vars[i]+'_H.pdf')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_'+mac_vars[i]+'_alpha_H.png')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_D_'+mac_vars[i]+'_H.png')
    
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
    
## Regression coefficients and plotting full sample    
    
rd[0].sort_values('t_const', inplace=True)
rd[0].index = [x.replace(' Imp', '') for x in rd[0].index.values.tolist()]
reg_graph_long(rd[0], 'const', 't_const')   
plt.savefig(res_out+'\\LS_Criteria_Imp_All.pdf')
#plt.savefig(res_out+'\\LS_Criteria_Imp_alpha_All.png')

rdn[0].sort_values('t_const', inplace=True)
rdn[0].index = [x.replace(' Imp', '') for x in rdn[0].index.values.tolist()]
reg_graph_long(rdn[0], 'const', 't_const')   
plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_All.pdf') 
#plt.savefig(res_out+'\\LS_Criteria_Imp_Dm_alpha_All.png')   

#Overview over results by criterion/dimension:

cscoresd = cscores.set_index('CRITERION')

rd_all = rd[0].copy(deep=True)    
rd_all.index = [x[:-4] for x in list(rd_all.index)]
rd_all = pd.merge(rd_all, cscoresd, left_index=True, right_index=True)    

rd_all['S'] = rd_all['t_const']>1.645
rd_all['Sl'] = rd_all['t_const']>1.282
rd_all['Sln'] = rd_all['const']>0
#rd_all['S'] = abs(rd_all['t_const'])>1.645
#rd_all['Sl'] = abs(rd_all['t_const'])>1.282
rd_all['B'] = rd_all['t_const']<-1.645
rd_all['Bl'] = rd_all['t_const']<-1.282
rd_all['Bln'] = rd_all['const']<0

for i in range(0,len(mac_vars)):
    rd_st = rd[i].copy(deep=True) 
    rd_st.index = [x[:-4] for x in list(rd_st.index)]
    rd_st['S'+str(i)] = rd_st['t_qu'+str(i)]>1.645
    rd_st['Sl'+str(i)] = rd_st['t_qu'+str(i)]>1.282
    rd_st['Sln'+str(i)] = rd_st['qu'+str(i)]>0
    rd_all = pd.merge(rd_all, rd_st[['S'+str(i), 'Sl'+str(i), 'Sln'+str(i)]], left_index=True, right_index=True)

rd_all['St'] = rd_all[['S']].sum(axis=1)
rd_all['Stl'] = rd_all[['Sln']].sum(axis=1)
rd_all['Bt'] = rd_all[['B']].sum(axis=1)
rd_all['Btl'] = rd_all[['Bln']].sum(axis=1)
rd_all['Sc'] = rd_all[['S0', 'S1', 'S2', 'S3', 'S4', 'S5']].sum(axis=1)
rd_all['Scl'] = rd_all[['Sl0', 'Sl1', 'Sl2', 'Sl3', 'Sl4', 'Sl5']].sum(axis=1)
rd_all['Scln'] = rd_all[['Sln0', 'Sln1', 'Sln2', 'Sln3', 'Sln4', 'Sln5']].sum(axis=1)

rd_dim = rd_all.groupby('DIMENSION')['Sln', 'S0',
       'Sl0', 'Sln0', 'S1', 'Sl1', 'Sln1', 'S2', 'Sl2', 'Sln2', 'S3', 'Sl3',
       'Sln3', 'S4', 'Sl4', 'Sln4', 'S5', 'Sl5', 'Sln5'].sum()
rd_dim['Sm'] = rd_dim[['Sl0', 'Sl1', 'Sl2', 'Sl3', 'Sl4', 'Sl5']].mean(axis=1)
rd_dim['Smn'] = rd_dim[['Sln0', 'Sln1', 'Sln2', 'Sln3', 'Sln4', 'Sln5']].mean(axis=1)
#rd_dim['Bm'] = rd_dim[['Bl0', 'Bl1', 'Bl2', 'Bl3', 'Bl4', 'Bl5']].mean(axis=1)
#rd_dim['Bmn'] = rd_dim[['Bln0', 'Bln1', 'Bln2', 'Bln3', 'Bln4', 'Bln5']].mean(axis=1)
    

#%% Sharpe ratios and scaled cross-section factors

#Sharpe ratio

rd_sr = (rdatan.mean()/rdatan.std())*np.sqrt(12)
sr_var = [x for x in list(rd_sr.index) if '_ls_' in x]
rd_sr = pd.DataFrame(rd_sr.loc[sr_var].sort_values()).rename(columns={0:'value'})
rd_sr.index = [x.replace(' Imp_ls_5', '') for x in rd_sr.index.values.tolist()]

reg_graph_single(rd_sr, 'value')
plt.savefig(res_out+'\\SR_Criteria_Imp_All.pdf')

rdn_sr = (rdatadn.mean()/rdatadn.std())*np.sqrt(12)
sr_var = [x for x in list(rdn_sr.index) if '_ls_' in x]
rdn_sr = pd.DataFrame(rdn_sr.loc[sr_var].sort_values()).rename(columns={0:'value'})
rdn_sr.index = [x.replace(' Imp_stdm_ls_5', '') for x in rdn_sr.index.values.tolist()]

reg_graph_single(rdn_sr, 'value')
plt.savefig(res_out+'\\SR_Criteria_Imp_Dm_All.pdf')

#Scale cross-section factor to volatility of corresponding time-series factor

rd_std = rdatan.std()*np.sqrt(12)
rd_std.index = [x.replace(' Imp_ls_5', '') for x in rd_std.index.values.tolist()]

coef1 = coef['reg_vars'] ##'reg_vars_exc'
coef1a = coefa['reg_vars'] ##'reg_vars_exc'

sd1_cs = {}
sd1a_cs = {}
    
for v in reg_vars:
    
    x = v.replace(' Imp_std', '')
      
    sd1_cs[v] = coef1[v]*(rd_std[x]/(coef1[v].std()*np.sqrt(12)))
    sd1_cs[v] = stats.ttest_1samp(sd1_cs[v], popmean=0)[0]
    sd1_cs[v] = pd.Series(sd1_cs[v]).set_axis(['t_stats_1','t_stats_2'])
    sd1_cs[v] = sd1_cs[v].append((coef1[v]*(rd_std[x]/(coef1[v].std()*np.sqrt(12)))).mean().set_axis(['coef_1','coef_2']))
    
    sd1a_cs[v] = coef1a[v]*(rd_std[x]/(coef1a[v].std()*np.sqrt(12)))
    sd1a_cs[v] = stats.ttest_1samp(sd1a_cs[v], popmean=0)[0]
    sd1a_cs[v] = pd.Series(sd1a_cs[v]).set_axis(['t_stats_1','t_stats_2','t_stats_3','t_stats_4','t_stats_5','t_stats_6','t_stats_7'])
    sd1a_cs[v] = sd1a_cs[v].append((coef1a[v]*(rd_std[x]/(coef1a[v].std()*np.sqrt(12)))).mean().set_axis(['coef_1','coef_2','coef_3','coef_4','coef_5','coef_6','coef_7']))

rd_std = rdatan.std()*np.sqrt(12)
rd_std.index = [x.replace(' Imp_ls_5', '') for x in rd_std.index.values.tolist()]

coef1 = coef['reg_vars_dm'] ##'reg_vars_exc_dm'
coef1a = coefa['reg_vars_dm'] ##'reg_vars_exc_dm'

sd1_csn = {}
sd1a_csn = {}

for v in reg_vars_dm:
    
    x = v.replace(' Imp_stdm', '')
    
    sd1_csn[v] = coef1[v]*(rd_std[x]/(coef1[v].std()*np.sqrt(12)))
    sd1_csn[v] = stats.ttest_1samp(sd1_csn[v], popmean=0)[0]
    sd1_csn[v] = pd.Series(sd1_csn[v]).set_axis(['t_stats_1','t_stats_2'])
    sd1_csn[v] = sd1_csn[v].append((coef1[v]*(rd_std[x]/(coef1[v].std()*np.sqrt(12)))).mean().set_axis(['coef_1','coef_2']))
    
    sd1a_csn[v] = coef1a[v]*(rd_std[x]/(coef1a[v].std()*np.sqrt(12)))
    sd1a_csn[v] = stats.ttest_1samp(sd1a_csn[v], popmean=0)[0]
    sd1a_csn[v] = pd.Series(sd1a_csn[v]).set_axis(['t_stats_1','t_stats_2','t_stats_3','t_stats_4','t_stats_5','t_stats_6','t_stats_7'])
    sd1a_csn[v] = sd1a_csn[v].append((coef1a[v]*(rd_std[x]/(coef1a[v].std()*np.sqrt(12)))).mean().set_axis(['coef_1','coef_2','coef_3','coef_4','coef_5','coef_6','coef_7']))

sd1_cs = pd.concat(sd1_cs, axis=1).sum(axis=1, level=0).transpose()    
sd1_cs.sort_values('t_stats_2', inplace=True)
sd1_cs.index = [x.replace(' Imp_std', '') for x in sd1_cs.index.values.tolist()]

sd1a_cs = pd.concat(sd1a_cs, axis=1).sum(axis=1, level=0).transpose()    
sd1a_cs.sort_values('t_stats_7', inplace=True)
sd1a_cs.index = [x.replace(' Imp_std', '') for x in sd1a_cs.index.values.tolist()]

sd1_csn = pd.concat(sd1_csn, axis=1).sum(axis=1, level=0).transpose()    
sd1_csn.sort_values('t_stats_2', inplace=True)
sd1_csn.index = [x.replace(' Imp_stdm', '') for x in sd1_csn.index.values.tolist()]

sd1a_csn = pd.concat(sd1a_csn, axis=1).sum(axis=1, level=0).transpose()    
sd1a_csn.sort_values('t_stats_7', inplace=True)
sd1a_csn.index = [x.replace(' Imp_stdm', '') for x in sd1a_csn.index.values.tolist()]

reg_graph_long(sd1_cs, 'coef_2', 't_stats_2')  
plt.savefig(res_out+'\\Criteria_Imp_Sd_Scaled_TS.pdf')
#plt.savefig(res_out+'\\Criteria_Imp_Sd_Scaled_Exc_TS.png')

reg_graph_long(sd1_csn, 'coef_2', 't_stats_2')  
plt.savefig(res_out+'\\Criteria_Imp_Sd_Scaled_TS_DM.png')  
#plt.savefig(res_out+'\\_Criteria_Imp_Sd_Scaled_Exc_TS_DM.png')  

reg_graph_long(sd1a_cs, 'coef_7', 't_stats_7')   
plt.savefig(res_out+'\\Criteria_Imp_Sd_Con_Scaled_TS.pdf')
#plt.savefig(res_out+'\\Criteria_Imp_Sd_Con_Scaled_Exc_TS.png')

reg_graph_long(sd1a_csn, 'coef_7', 't_stats_7')  
plt.savefig(res_out+'\\Criteria_Imp_Sd_Con_Scaled_TS_DM.png')  
#plt.savefig(res_out+'\\_Criteria_Imp_Sd_Con_Scaled_Exc_TS_DM.png') 


