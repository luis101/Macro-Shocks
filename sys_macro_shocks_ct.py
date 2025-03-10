
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
#maxdate = '12-31-2022'
maxdate = '03-31-2023'

#Set script directory to current working directory:
    
script = os.getcwd()
#os.chdir(script+'\\Data')
os.chdir('<Directory>')


#%% Load economic data:

#Economic policy uncertainty (Baker et al. 2016) & (Davis 2016):
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
#ENG = total number, SCO = scaled by number of tweet with word "have", WGT - weighted by retweets
   
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

#df_PMI.to_csv(input_econ+"PMI_monthly.csv", index=False)
#df_PMI.to_excel(input_econ+"PMI_monthly.xlsx", index=False)
df_PMI = pd.read_csv(input_econ+"PMI_monthly.csv")

    
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
wui['month_id'] = wui['month_id'].dt.strftime('%Y-%m')

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
econ = pd.merge(econ, tweum[['month_id', 'TEU-ENG', 'TEU-USA', 'TEU-WGT', 'TEU-SCA',
                             'TMU-ENG', 'TMU-USA', 'TMU-WGT', 'TMU-SCA']], on='month_id', how='outer')    
econ.sort_values(['month_id'], inplace=True)

vix['month_id'] = pd.to_datetime(vix['DATE']).dt.strftime('%Y-%m')
vix['VIX_mean'] = vix.groupby(['month_id'])['CLOSE'].transform('mean')
vix = vix.groupby(['month_id']).last().reset_index()
econ = pd.merge(econ, vix[['month_id', 'CLOSE', 'VIX_mean']], on='month_id', how='outer')
econ.rename(columns={'CLOSE':'VIX'}, inplace=True)   

df_PMI["month_id"] = pd.to_datetime(df_PMI["date"]).dt.strftime("%Y-%m")
df_PMI = df_PMI.pivot(index="month_id", columns="source_id", values="value")
df_PMI = df_PMI.reset_index()
df_PMI = df_PMI[['month_id', 88505325, 93625215]]
df_PMI.rename(columns={88505325:'Global Composite (M+S) PMI Headline Adjusted', 
                       93625215:'Global Manufacturing PMI Adjusted'}, inplace=True)
#df_PMI = df_PMI[['month_id', '88505325', '93625215']]
#df_PMI.rename(columns={'88505325':'Global Composite (M+S) PMI Headline Adjusted', 
#                       '93625215':'Global Manufacturing PMI Adjusted'}, inplace=True)
econ = pd.merge(econ, df_PMI, on='month_id', how='outer')

for m in ['Global Composite (M+S) PMI Headline Adjusted', 'Global Manufacturing PMI Adjusted']:
    econ[m] = 100-econ[m]
    
    
## Load climate-related data:
    
#Media climate change concern index:
mccc = pd.read_excel(input_econ+"Sentometrics_US_Media_Climate_Change_Index.xlsx", 
                     sheet_name='SSRN 2022 version (monthly)', skiprows=range(5)) 

#Climate change news index:
ccn = mccc = pd.read_excel(input_econ+"EGLKS_data.xlsx")     

    
##First principle component of shock indicators:
    
epca = econ[['month_id', 'GEPU_current', 'BBD MPU Index Based on Access World News',
       'Global (GDP weighted average)', 'Financial Uncertainty', 'Macro Uncertainty', 
       'Real Uncertainty', 'TMU-ENG', 'TMU-SCA', 'VIX', 'Global Manufacturing PMI Adjusted']]   
epca = epca[(epca['month_id']>='2011-06')&(epca['month_id']<='2022-12')] 
#epca = epca[(epca['month_id']>='2013-06')&(epca['month_id']<='2022-12')] 
epca.drop('month_id', axis=1, inplace=True)

#Normalize data: 
#epca = epca.transform(lambda x: (x - x.mean())/x.std(ddof=0))  
epca = (epca - epca.mean())/epca.std(ddof=0)

#Covariance matrix
#E_cov = (1/len(epca)) * epca.T.dot(Xs)
E_cov = np.cov(epca.T, ddof=0)

#Eigenvalues and Eigenvectors
eigen_values, eigen_vectors = np.linalg.eig(E_cov)
assert(len(E_cov) > 1)
sort_index = np.argsort(eigen_values)[::-1]
sort_eigenval = eigen_values[sort_index]
sort_eigenvec = eigen_vectors[:,sort_index]

n_components = 10
projection_mat = sort_eigenvec[:,0:n_components]

expl_var = sort_eigenval[0:n_components]
total_var = sort_eigenval.sum()
expl_var_ratio_ = expl_var/total_var
cum_var_expl = np.cumsum(expl_var_ratio_)

#Principal component from eigenvector multiplication
pc1 = epca.dot(eigen_vectors[:, 0])
pc1 = pc1.rename('ShockPCA')

#econ = econ.merge(pd.DataFrame(pc1), left_index=True, right_index=True, how='left')
econ = econ.combine_first(pd.DataFrame(pc1))

#econ['ShockPCA'] = econ['ShockPCA']*(-1)
#econ['ShockPCAN'] = econ['ShockPCA']*(-1)
#econ['ShockPCA1'] = ((econ['ShockPCA'].max())/(econ['ShockPCA'].min()))*econ['ShockPCA']
#econ['ShockPCA2'] = ((econ['ShockPCA'].min())/(econ['ShockPCA'].max()))*econ['ShockPCA']
#econ['ShockPCA0'] = econ['ShockPCA'].copy(deep=True)
#econ['ShockPCA'] = 0
#econ.loc[econ['ShockPCA0']<0, 'ShockPCA'] = econ['ShockPCA1']
#econ.loc[econ['ShockPCA0']>0, 'ShockPCA'] = econ['ShockPCA2']


## Changes/First differences in economic shock variables   

econ.sort_values(['month_id'], inplace=True)
econd = econ.copy(deep=True)

econd.rename(columns={'GEPU_current':'Economic Policy Uncertainty', 'GPR':'Geopolitical Risk',
                     'BBD MPU Index Based on Access World News':'Monetary Policy Uncertainty',
                     'TMU-ENG':'Twitter Macro Uncertainty Absolute', 'TMU-SCA':'Twitter Macro Uncertainty',
                     'Global (GDP weighted average)':'Global Uncertainty'}, inplace=True)

#for v in econd.columns[1:]:
#for v in econd.columns[:-1]:
for v in [x for x in list(econd.columns) if 'month_id' not in x]:
    econd[v+'_L'] = econd[v].shift(1)
    econd[v+'_D'] = econd[v]-econd[v+'_L']
    econd.drop([v+'_L'], axis=1, inplace=True) 

#for g in ['Global (simple average)_D', 'Global (GDP weighted average)_D']:
for g in ['Global (simple average)_D', 'Global Uncertainty_D']:
    econd.loc[econd[g]==0, g] = np.nan
    econd[g].ffill(inplace=True)


## Correlations between level and changes of economic shock variables  

def corr_heat(data, s1, s2, b, l, t):
    fig, ax = plt.subplots(figsize=(s1,s2))
    plt.title(t,fontsize=9)
    plt.xlabel('Shock 1', fontsize=9)
    plt.ylabel('Shock 2', fontsize=9)
    plt.subplots_adjust(bottom=b, left=l)
    #sns.heatmap(data)
    #sns.heatmap(data, mask=np.triu(data)
    #sns.heatmap(data, mask=np.triu(data), annot=True, fmt='.2f', center = 0, cmap=sns.diverging_palette(20,120,n=256))
    sns.heatmap(data, mask=np.triu(data), annot=True, fmt='.2f', center = 0, cmap='RdYlGn') 

econdr = econd[(econd['month_id']>'2013-09')&(econd['month_id']<'2023-04')]

#mac_vars = ['Macro Uncertainty', 'Financial Uncertainty', 'Real Uncertainty', 'GEPU_current', 
#            'BBD MPU Index Based on Access World News', 'Global (GDP weighted average)', 
#            'GPR', 'TEU-ENG', 'TEU-SCA', 'TMU-ENG', 'TMU-SCA', 'VIX',
#            'Global Composite (M+S) PMI Headline Adjusted', 'Global Manufacturing PMI Adjusted']
#mac_vars_d = ['Macro Uncertainty_D', 'Financial Uncertainty_D', 'Real Uncertainty_D', 'GEPU_current_D', 
#              'BBD MPU Index Based on Access World News_D', 'Global (GDP weighted average)_D', 
#              'GPR_D', 'TEU-ENG_D', 'TEU-SCA_D', 'TMU-ENG_D', 'TMU-SCA_D', 'VIX_D',
#              'Global Composite (M+S) PMI Headline Adjusted_D', 'Global Manufacturing PMI Adjusted_D']

mac_vars = ['Macro Uncertainty', 'Financial Uncertainty', 'Real Uncertainty', 
            'Economic Policy Uncertainty', 'Monetary Policy Uncertainty', 'Global Uncertainty',
            'Twitter Macro Uncertainty Absolute', 'Twitter Macro Uncertainty',
            'VIX', 'Global Manufacturing PMI Adjusted', 'ShockPCA']
mac_vars_d = ['Macro Uncertainty_D', 'Financial Uncertainty_D', 'Real Uncertainty_D', 
              'Economic Policy Uncertainty_D', 'Monetary Policy Uncertainty_D', 'Global Uncertainty_D',
              'Twitter Macro Uncertainty Absolute_D', 'Twitter Macro Uncertainty_D',
              'VIX_D', 'Global Manufacturing PMI Adjusted_D', 'ShockPCA_D']

es = econdr[mac_vars].describe()
ec = econdr[mac_vars].corr()
corr_heat(ec, 12, 10, 0.325, 0.225, 'Correlation Macroeconomic Indicators')
plt.savefig(res_out+'\\Shocks Heat Map.png', bbox_inches='tight')
ec.to_csv(res_out+'\\Shocks Correlations.txt')

esd = econdr[mac_vars_d].describe()
ecd = econdr[mac_vars_d].corr()
corr_heat(ecd, 12, 10, 0.35, 0.25, 'Correlation Macroeconomic Indicator Differentials')
plt.savefig(res_out+'\\Shocks Diff Heat Map.png', bbox_inches='tight')
ec.to_csv(res_out+'\\Shocks Diff Correlations.txt')

#mac_vars = ['Macro Uncertainty', 'BBD MPU Index Based on Access World News', 'TEU-SCA', 'TMU-ENG', 
#            'Global (GDP weighted average)', 'Global Manufacturing PMI Adjusted']
#mac_vars_d = ['Macro Uncertainty_D', 'BBD MPU Index Based on Access World News_D', 'TEU-SCA_D', 'TMU-ENG_D', 
#              'Global (GDP weighted average)_D', 'Global Manufacturing PMI Adjusted_D']

mac_vars = ['Macro Uncertainty', 'Twitter Macro Uncertainty', 'Global Manufacturing PMI Adjusted', 'ShockPCA']
mac_vars_d = ['Macro Uncertainty_D', 'Twitter Macro Uncertainty_D', 'Global Manufacturing PMI Adjusted_D', 'ShockPCA_D']

es = econdr[mac_vars].describe()
ec = econdr[mac_vars].corr()
es.to_csv(res_out+'\\Shocks Summary Stats.txt')

esd = econdr[mac_vars_d].describe()
ecd = econdr[mac_vars_d].corr()
esd.to_csv(res_out+'\\Shocks Diff Summary Stats.txt')


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

ind = ind[ind['month_id']>'01-01-2011']

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
ret = ret[~((ret['prccd_unadj_usd']<1) & (ret['ret_usd']>300))]
ret = ret[~((ret['prccd_unadj_usd']<1) & (ret['ret_usd']<=-80))]

#ret = ret[ret['prccd_unadj_usd']>=5]
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

ret = ret[pd.to_datetime(ret['Date']) > '01-01-2011']
ret.drop('_merge', axis=1, inplace=True)
print(len(ret))

gc.collect()


#%% Merge accounting data and stock market data

#acc = pd.read_csv(input+'\\Financials\\Controls_and_Predictors_post_2012_PIT.csv', low_memory=False)
acc = pd.read_csv(input+'\\Financials\\Controls_and_Predictors_post_2010_PIT.csv', low_memory=False)
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
         #'ret2', 'ret_usd2', 'firmno', 'mcap_lag', 'mcap_L','md', '_merge']

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

#csa['CAM_YEAR'] = csa['CAM_YEAR_IMP']

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

#criterion.rename(columns={'Compliance with Applicable Export Control Regimes':'Compliance with Export Control Regimes',
#                          'Information Security/ Cybersecurity & System Availability':'Information Security/Cybersecurity',
#                          'Resource Conservation & Resource Efficiency':'Resource Conservation & Efficiency',
#                          'Partnerships Towards Sustainable Healthcare':'Partnerships For Sustainable Healthcare',
#                          'Strategy to Improve Access to Drugs or Products':'Strategy to Improve Drugs/Products Access'},
#                 inplace=True)

cscores = csa.groupby('CRITERION').last().reset_index()

ccol = [c.replace(' NP','') for c in list(criterion.columns) if ' NP' in c]

ccols = list(criterion.columns)
ccols = [x.replace('Codes of Business Conduct','Business Ethics') for x in ccols]
ccols = [x.replace('Compliance with Applicable Export Control Regimes','Compliance with Export Control Regimes') for x in ccols]
ccols = [x.replace('Information Security/ Cybersecurity & System Availability','Information Security/Cybersecurity') for x in ccols]
ccols = [x.replace('Resource Conservation & Resource Efficiency','Resource Conservation & Efficiency') for x in ccols]
ccols = [x.replace('Partnerships Towards Sustainable Healthcare','Partnerships For Sustainable Healthcare') for x in ccols]
ccols = [x.replace('Strategy to Improve Access to Drugs or Products','Strategy to Improve Drugs/Products Access') for x in ccols]

criterion.columns = ccols

cscores = cscores.loc[cscores['CRITERION'].isin(ccol)][['CRITERION', 'DIMENSION']]
cscores.to_excel(res_out+'\\Criteria+Dimensions.xlsx')

cmap = ['ACB', 'ACP', 'ACM', 'BIO', 'BRMN', 'BUI', 'CLS', 'COP', 'BE', 'CER', 'CCP', 
        'CGOV', 'CRM', 'ER', 'EG', 'EM', 'EPMS', 'ER', 'FI', 'FSSR', 'FM', 'FE', 'GMO', 
        'HN', 'HOC', 'HCD', 'HR', 'ICS', 'IM', 'LPI', 'LIBO', 'LCS', 'MO', 'MP', 'MAT', 
        'MWM', 'NR', 'OHS', 'OEE', 'PAC', 'PSH', 'PAS', 'PI', 'PP', 'PQRM', 'PRST', 
        'RS', 'RCE', 'RC', 'RCM', 'SIC', 'SIR', 'SR', 'SE', 'SEM', 'SADP', 'SCM', 'SAP', 
        'SC', 'SFP', 'TAR', 'TS', 'TD', 'WO', 'WRR']

cscores = pd.concat([cscores.reset_index(), pd.DataFrame(cmap, columns=['Abr'])], axis=1)
cscores.drop('index', axis=1, inplace=True)

cscores.loc[cscores['CRITERION']=='Codes of Business Conduct', 'CRITERION']='Business Ethics'
cscores.loc[cscores['CRITERION']=='Compliance with Applicable Export Control Regimes', 'CRITERION']='Compliance with Export Control Regimes'
cscores.loc[cscores['CRITERION']=='Information Security/ Cybersecurity & System Availability', 'CRITERION']='Information Security/Cybersecurity'
cscores.loc[cscores['CRITERION']=='Resource Conservation & Resource Efficiency', 'CRITERION']='Resource Conservation & Efficiency'
cscores.loc[cscores['CRITERION']=='Partnerships Towards Sustainable Healthcare', 'CRITERION']='Partnerships For Sustainable Healthcare'
cscores.loc[cscores['CRITERION']=='Strategy to Improve Access to Drugs or Products', 'CRITERION']='Strategy to Improve Drugs/Products Access'

cscores.to_excel(res_out+'\\Criteria+Dimensions_Names.xlsx') 

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
#qu = ['Brand Management Metrics', 'Brand Strategy & Sustainability Strategy', 
#      'Customer Privacy Information', 'Customer Satisfaction Measurement', 
#      'Direct-to-Consumer Marketing', 'Online Strategies & Customers Online', 
#      'Product Adaptation for Emerging Markets (B2B)', 'Product Adaptation for Emerging Markets (B2C)', 
#      'Product Design Criteria', 'Product Innovations', 'Product Recalls', 'Use of Customer Data']
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
        #data[v+'_ind_sd'] = data.groupby(['INDUSTRY', 'CAM_YEAR'])[v].transform('std')
        #data[v + '_sdm'] = data[v+'_dm']/data[v+'_ind_sd'] 

        #Standardized industry z-values:
        #data[v + '_mean'] = data.groupby(['CAM_YEAR'])[v + '_sdm'].transform('mean')
        #data[v + '_sd'] = data.groupby(['CAM_YEAR'])[v + '_sdm'].transform('std')
        #data[v + '_stdm'] = (data[v + '_sdm'] - data[v + '_mean']) / data[v + '_sd']

        #Standardized:
        #data[v+'_mean'] = data.groupby(['CAM_YEAR'])[v].transform('mean')
        #data[v+'_sd'] = data.groupby(['CAM_YEAR'])[v].transform('std')
        #data[v+'_std'] = (data[v]-data[v+'_mean'])/data[v+'_sd']

        #Sector-demeaned:
        #data[v+'_sec'] = data.groupby(['sector', 'CAM_YEAR'])[v].transform('mean')
        #data[v+'_dms'] = data[v]-data[v+'_sec']

        #Adjusted:
        #data[v+'_us'] = -(100-data[v])

        #data.drop([v+'_ind', v+'_sd', v+'_sec', v+'_mean'], axis=1, inplace=True)
        #data.drop([v+'_ind', v+'_ind_sd', v+'_sd', v+'_mean'], axis=1, inplace=True)
        data.drop([v+'_ind'], axis=1, inplace=True)
        
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

#Industry-adjusted levels:
var_name_dm = list(data.loc[:,data.columns.str.endswith('_dm')].columns)
#var_name_dm = list(data.loc[:,data.columns.str.endswith('_sdm')].columns)

#Standardized levels:
#var_name_sd = list(data.loc[:,data.columns.str.endswith('_std')].columns)

#Standardized industry z-values:
#var_name_sdm = list(data.loc[:,data.columns.str.endswith('_stdm')].columns)

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
var_name_dm =  [x for x in list(data.columns) if x.endswith('_dm')]
#var_name_sd =  [x for x in list(data.columns) if x.endswith('_std')]
#var_name_sdm =  [x for x in list(data.columns) if x.endswith('_stdm')]
#var_name =  [x.replace('_dm', '') for x in var_name_dm]
var_name =  [x for x in list(data.columns) if x.endswith(' Imp')]

data.drop(['ISIN', 'GVKEY', 'ASOF_DATE'], axis=1, inplace=True)

gc.collect()


#%% Merge financial data with csa data

ret.drop(['cshtrm', 'filingDate', 'fiscalYear', 'year_id', 'periodEndDate',
          'As Reported Balance Sheet Date', 'Revenues', 'Total Assets', 'Book Equity', 
          'mdiff', 'count_ret'], axis=1, inplace=True)
ret.drop(['currency', 'industry', 'region', 'sector', 'sub_industry', 'sub_sector'], 
         axis=1, inplace=True) 
#ret = ret[ret['prccd_unadj_usd']>=5]

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
data[var_name_dm] = data.groupby(['CIQ_ID'])[var_name_dm].ffill(limit=fw)
#data[var_name_sd] = data.groupby(['CIQ_ID'])[var_name_sd].ffill(limit=fw)
#data[var_name_sdm] = data.groupby(['CIQ_ID'])[var_name_sdm].ffill(limit=fw)
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


#%% Industry-demeaned and standardized levels

def industry_demean(data, var_name):
    
    data.sort_values(['CIQ_ID', 'month_id'], inplace=True)

    #Standardization with respect to relevant sample:
    
    for v in var_name:

        #Industry-demeaned:
        #data[v+'_ind'] = data.groupby(['INDUSTRY', 'CAM_YEAR'])[v].transform('mean')
        #data[v+'_sd'] = data.groupby(['INDUSTRY', 'CAM_YEAR'])[v].transform('std')
        #data[v+'_dm'] = data[v]-data[v+'_ind']
        
        #Standardized:
        #data[v+'_mean'] = data.groupby(['CAM_YEAR'])[v].transform('mean')
        #data[v+'_sd'] = data.groupby(['CAM_YEAR'])[v].transform('std')
        data[v+'_mean'] = data.groupby(['month_id'])[v].transform('mean')
        data[v+'_sd'] = data.groupby(['month_id'])[v].transform('std')
        data[v+'_std'] = (data[v] - data[v+'_mean']) / data[v+'_sd']
        
        #Standardized industry z-values:
        #data[v + '_mean'] = data.groupby(['CAM_YEAR'])[v + '_dm'].transform('mean')
        #data[v + '_sd'] = data.groupby(['CAM_YEAR'])[v + '_dm'].transform('std')
        data[v + '_mean'] = data.groupby(['month_id'])[v + '_dm'].transform('mean')
        data[v + '_sd'] = data.groupby(['month_id'])[v + '_dm'].transform('std')
        data[v + '_stdm'] = (data[v + '_dm'] - data[v + '_mean']) / data[v + '_sd']

        #Sector-demeaned:
        #data[v+'_sec'] = data.groupby(['sector', 'CAM_YEAR'])[v].transform('mean')
        #data[v+'_dms'] = data[v]-data[v+'_sec']

        #Adjusted:
        #data[v+'_us'] = -(100-data[v])

        #data.drop([v+'_ind', v+'_sd', v+'_sec', v+'_mean'], axis=1, inplace=True)
        data.drop([v+'_sd', v+'_mean'], axis=1, inplace=True)
        
    return data

data = industry_demean(data, var_name)


#%% Add market adjusted returns

#zipfile = ZipFile(zip_dir+'\\betas_market.zip')
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

#os.chdir('C:\\Users\\LUKAS_ZIMMERMANN\\OneDrive - S&P Global\\Projects')

# Define sets of variables

#Standardized levels:
var_name_sd = list(data.loc[:,data.columns.str.endswith('_std')].columns)
#Standardized industry-demeaned levels:
var_name_sdm = list(data.loc[:,data.columns.str.endswith('_stdm')].columns)

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

#mac_vars = ['Macro Uncertainty', 'Financial Uncertainty', 'GEPU_current', 'TEU-ENG', 'VIX', 
#            'Global Manufacturing PMI Adjusted']
#mac_vars_d = ['Macro Uncertainty_D', 'Financial Uncertainty_D', 'GEPU_current_D', 'TEU-ENG_D', 
#              'VIX_D', 'Global Manufacturing PMI Adjusted_D']
#mac_vars = ['Macro Uncertainty', 'BBD MPU Index Based on Access World News', 'TMU-ENG', 'TEU-ENG', 
#            'TMU-SCA', 'GPR', 'Global (GDP weighted average)']
#mac_vars_d = ['Macro Uncertainty_D', 'BBD MPU Index Based on Access World News_D', 'TMU-ENG_D', 
#              'TEU-ENG_D', 'TMU-SCA_D', 'GPR_D', 'Global (GDP weighted average)_D']
#mac_vars = ['Macro Uncertainty', 'BBD MPU Index Based on Access World News', 'TEU-SCA', 'TMU-ENG', 
#            'TMU-SCA', 'Global (GDP weighted average)', 'Global Manufacturing PMI Adjusted']
#mac_vars_d = ['Macro Uncertainty_D', 'BBD MPU Index Based on Access World News_D', 'TEU-SCA_D', 'TMU-ENG_D', 
#              'TMU-SCA_D', 'Global (GDP weighted average)_D', 'Global Manufacturing PMI Adjusted_D']
mac_vars = ['Macro Uncertainty', 'Twitter Macro Uncertainty', 'Global Manufacturing PMI Adjusted', 
            'ShockPCA', 'Global Manufacturing PMI Adjusted_D', 'ShockPCA_D']
#print(mac_vars)

controls = ['size', 'BM', 'IAT', 'Operating Profitability', 'mom212']

# Obtain final dataset

id_vars = ['month_id', 'ret_usd', 'ret_e']
#id_vars.extend(vars)
id_vars.extend(reg_vars)
#id_vars.extend(reg_vars_d)
id_vars.extend(reg_vars_dm)
id_vars.extend(mac_vars)
#id_vars.extend(mac_vars_d)
id_vars.extend(controls)

regdata = data[id_vars]

id_vars.extend(['COUNTRYNAME', 'DJREGION'])
regdata_ct = data[id_vars]

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
  
    regdata_ct[c] = regdata_ct.groupby('month_id')[c].transform(
        lambda x: np.maximum(x.quantile(.01), np.minimum(x, x.quantile(.99))))
    
    regdata_ct[c+'_mean'] = regdata_ct.groupby(['month_id'])[c].transform('mean')
    regdata_ct[c+'_sd'] = regdata_ct.groupby(['month_id'])[c].transform('std')
    regdata_ct[c+'_std'] = (regdata_ct[c]-regdata_ct[c+'_mean'])/regdata_ct[c+'_sd']
    regdata_ct.drop([c+'_sd', c+'_mean'], axis=1, inplace=True)

controls = [c+'_std' for c in controls]

regdata.count()

#regdata = regdata[regdata['month_id'] < '2023-04']
#regdata_ct = regdata_ct[regdata_ct['month_id'] < '2023-04']
regdata = regdata[regdata['month_id'] < '2023-01']
regdata_ct = regdata_ct[regdata_ct['month_id'] < '2023-01']

#print(econ.tail(10))

#Test
#regdt = regdata[~regdata['Water Related Risks Imp_stdm'].isnull()]
#b = regdt.groupby('month_id')['Water Related Risks Imp_stdm'].count()

reg_vars.remove('Mineral Waste Management Imp_std')


#%% Plotting functions

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
                               position=1, align='center', label='T-statistics', grid=True)
    plt.xticks()

    #ax.set_xlim(-0.8, 10)
    #ax.set_ylim(-2, 2)
    #ax.set_ylim(-2.5, 2.5)
    ax.set_ylim(-3.5, 3.5)
    ax2.set_ylim(-0.7, 0.7)
    #ax2.set_ylim(-0.8, 0.8)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax.legend(lines + lines2, labels + labels2, loc=0)
    ax.legend(lines + lines2, labels + labels2, loc=2)

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
                               position=1, align='center', label='T-statistics', grid=True)
    plt.xticks()

    #ax.set_xlim(-0.5, 10)
    ax.set_ylim(-2.5, 2.5)
    ax2.set_ylim(-1.25, 1.25)
    #ax.set_ylim(-3, 3)
    #ax2.set_ylim(-1.5, 1.5)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax.legend(lines + lines2, labels + labels2, loc=0)
    ax.legend(lines + lines2, labels + labels2, loc=2)
    
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
    
def reg_graph_single(df, v1, padding):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 10))

    df[v1].plot(kind='bar', color=colors[1], width=0.5, 
                            position=0, align='center', label='Sharpe Ratio', grid=True)
    plt.xticks()

    #ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1, 1)

    #lines, labels = ax.get_legend_handles_labels()
    #ax.legend(lines, labels, loc=0)
    
    #ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)

    #If text is supposed to appear above graph:
    #ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    
    #If text is supposed to appear below graph:
    ax.tick_params(axis='x', pad=padding)
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    ax.set_xlabel('') 
    #ax.set_ylabel('Sharpe Ratio') 
    
def reg_graph_single_s(df, v1, padding):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 9))

    df[v1].plot(kind='bar', color=colors[1], width=0.5, 
                            position=0, align='center', label='Sharpe Ratio', grid=True)
    plt.xticks()
    plt.ylabel('Sharpe Ratio', fontweight='bold')

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
    ax.tick_params(axis='x', pad=padding)
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    #ax.set_ylabel('Sharpe Ratio')       


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

# Regression results

def cs_reg(rdata, r_vars, m_vars):

    coef = {}
    coefa = {}
    
    results1 = {}
    results2 = {}

    results1a = {}
    results2a = {}

    coef1 = {}
    coef1a = {}
    
    #i=0
    #rlist=[]
    #for v in vars:
    for v in r_vars:

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
        #base_vars.extend(mac_vars_d)
        base_vars.extend(controls)
        regdt = regct[base_vars]
        regdt = regdt[~regdt[v].isnull()]
        
        regdt.replace([np.inf, -np.inf], np.nan, inplace=True)
                 
        #Minimum number of observations per month:
        regdt['month_obs'] = regdt.groupby(['month_id'])['ret_usd'].transform('count')  
        #regdt = regdt[regdt['month_obs']>=20]
        regdt = regdt[regdt['month_obs']>=50]
        
        for c in controls:
            regdt = regdt[~regdt[c].isnull()]
            #regdt.dropna(subset=[c], inplace=True)
        
        #print(len(regdt['month_id'].unique()))
        
        #Minimum number of months:
        if len(regdt)==0:
            continue
        #if len(regdt['month_id'].unique())<=48:
        if len(regdt['month_id'].unique())<=60:
            continue
        
        #No gaps:
        mn = pd.DataFrame(regdt['month_id'].unique())
        mn.sort_values(0, inplace=True)
        
        #mn['delta'] = mn[0].diff()[1:]
        mn['delta'] = mn[0].shift()[1:]
        mn['gap'] = mn[0].astype('int') - mn['delta'].astype('int')
        mn = mn[~mn['delta'].isnull()] 
        
        if mn['gap'].max()>1:
            continue

        #i=i+1
        #rlist.extend([v])
        
        #regdt = regdt[~regdt['ret_e'].isnull()]
        #regdt[v] = regdt[v]/100
        #regdt[v] = regdt[v]/10
        #regdt[v] = regdt[v]*10
        
        fmb1, tb1, reg, beta1 = finfunc.fmb_reg_var(regdt,'month_id','ret_usd',v)
        #fmb1, tb1, reg, beta1 = finfunc.fmb_reg_var(regdt,'month_id','ret_e',v)
        print(fmb1)

        #dfoutput = summary_col(reg,stars=True)
        #print(dfoutput)

        r1 = format_results(fmb1,v)
        #r1.rename(columns={v:m}, inplace=True)
        
        #results1[v][i]=r1
        beta1 = pd.concat(beta1, axis=1).sum(axis=1, level=0).transpose()
        fct_df = pd.concat([beta1, pd.DataFrame(regdt['month_id'].astype(str).unique())], axis=1)   
        
        #Check for gaps in time-series:
            
        fct_df.sort_values(0, inplace=True)
        
        fct_df['month'] = pd.to_datetime(fct_df[0]).dt.month
        fct_df['year'] = pd.to_datetime(fct_df[0]).dt.year
        fct_df['gap'] = (fct_df['year']-fct_df['year'].shift())*12+fct_df['month']-fct_df['month'].shift()
        
        #print(fct_df['gap'].max()>1)
        
        if fct_df['gap'].max()>1:
            continue
        
        fct_df.drop(['month', 'year', 'gap'], axis=1, inplace=True)
        
        results1[v]=r1
        coef1[v]=fct_df

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
            print(m)
            
            controls = ['size_std', 'BM_std', 'IAT_std', 'Operating Profitability_std', 'mom212_std']

            regdt['qu'] = pd.qcut(regdt[m], q=[0, .75, 1], labels=False)
            regdt['qun'] = (regdt['qu']==0).astype(int)
            #High
            regdt['inter'] = regdt[v]*regdt['qu']
            #Low
            regdt['intern'] = regdt[v]*regdt['qun']
            
            rvars = ['inter','intern']

            #fmb2, t2, reg, beta2 = finfunc.fmb_reg_var(regdt,'month_id','ret_usd',[v,'qu','inter'])
            #fmb2, t2, reg, beta2 = finfunc.fmb_reg_var(regdt,'month_id','ret_e',[v,'qu','inter'])
            fmb2, t2, reg, beta2 = finfunc.fmb_reg_var(regdt,'month_id','ret_usd',['inter','intern'])
            #fmb2, t2, reg, beta2 = finfunc.fmb_reg_var(regdt,'month_id','ret_e',['inter','intern'])
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
        
    coef['reg_vars'] = coef1
    coefa['reg_vars'] = coef1a  
            
    return results1, results1a, results2, results2a, coef, coefa

def cs_reg_raw(rdata, r_vars):

    coef = {}
    coefa = {}
    
    results1 = {}
    results1a = {}

    coef1 = {}
    coef1a = {}
    
    #for v in vars:
    for v in r_vars:
    #for v in reg_vars_dm:

        print(v)

        controls = ['size_std', 'BM_std', 'IAT_std', 'Operating Profitability_std', 'mom212_std']
        #controls = ['size', 'BM', 'IAT', 'Operating Profitability', 'mom212']
        #controls = ['ICAPEX', 'Gross Profitability', 'mom212']
        
        results1[v] = {}
        results1a[v] = {}
        
        base_vars = ['month_id','ret_usd','ret_e',v]
        base_vars.extend(controls)
        regdt = rdata[base_vars]
        regdt = regdt[~regdt[v].isnull()]
        
        regdt.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        for c in controls:
            regdt = regdt[~regdt[c].isnull()]
            #regdt.dropna(subset=[c], inplace=True)
         
        #Minimum number of observations per month:
        regdt['month_obs'] = regdt.groupby(['month_id'])['ret_usd'].transform('count')  
        #regdt = regdt[regdt['month_obs']>=20]
        regdt = regdt[regdt['month_obs']>=50]
        
        #Minimum number of months:
        if len(regdt)==0:
            continue
        #if len(regdt['month_id'].unique())<=48:
        if len(regdt['month_id'].unique())<=60:
            continue
        
        #regdt = regdt[~regdt['ret_e'].isnull()]
        #regdt[v] = regdt[v]/100
        #regdt[v] = regdt[v]/10
        #regdt[v] = regdt[v]*10
        
        fmb1, tb1, reg, beta1 = finfunc.fmb_reg_var(regdt,'month_id','ret_usd',v)
        #fmb1, tb1, reg, beta1 = finfunc.fmb_reg_var(regdt,'month_id','ret_e',v)
        print(fmb1)

        #dfoutput = summary_col(reg,stars=True)
        #print(dfoutput)

        r1 = format_results(fmb1,v)
        #r1.rename(columns={v:m}, inplace=True)
        results1[v]=r1
        #results1[v][i]=r1
        beta1 = pd.concat(beta1, axis=1).sum(axis=1, level=0).transpose()
        fct_df = pd.concat([beta1, pd.DataFrame(regdt['month_id'].astype(str).unique())], axis=1)   
        coef1[v]=fct_df
        
        controls.extend([v])
        
        #fmb1a, tb1a, rega, beta1a = finfunc.fmb_reg_var(regdt,'month_id','ret_usd',controls)
        fmb1a, tb1a, rega, beta1a = finfunc.fmb_reg_var(regdt,'month_id','ret_e',controls)
        print(fmb1a)

        r1 = format_results(fmb1a,v)
        results1a[v]=r1
        beta1a = pd.concat(beta1a, axis=1).sum(axis=1, level=0).transpose()
        coef1a[v]=beta1a
        
    coef['reg_vars'] = coef1
    coefa['reg_vars'] = coef1a  
            
    return results1, results1a, coef, coefa

#regdata_ct['COUNTRYNAME'].value_counts()
regdata_ct.loc[regdata_ct['COUNTRYNAME']=='Japan', 'DJREGION'] = 'JAP'
regdata_ct['DJREGION'].value_counts()

regions = ['All','NAM','EUR','APA','JAP','EM']

#gov = list(cscores[cscores['DIMENSION']=='Governance & Economic Dimension']['CRITERION'])
csc = cscores.set_index('CRITERION')

#mac_vars = ['ShockPCA', 'ShockPCA_D']


#%% Dimension Level

coef_all = {}
coefa_all = {}  

for reg in regions:
    
    sc = ''
    #sc = 'DM_'
    
    if reg == 'All':
        regct = regdata_ct.copy(deep=True)
    else:
        regct = regdata_ct[regdata_ct['DJREGION']==reg]
        
    r1, r1a, r2, r2a, cf1, cf1a = cs_reg(regct, reg_vars, mac_vars)
    #r1, r1a, r2, r2a, cf1, cf1a = cs_reg(regct, reg_vars_dm, mac_vars)
    #r1, r1a, r2, r2a = cs_reg(regct, reg_vars, mac_vars_d)
    
    coef_all[reg] = cf1 
    coefa_all[reg] = cf1a

    #All characteristics in one table:

    vnew = [x[:4] for x in reg_vars]

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    res1_tab = pd.concat(r1, axis=1).sum(axis=1, level=0)
    res1_tab = round(res1_tab, 3)
    res1_tab.set_index(pd.Index(['cons', 't_cons', 'coef', 't_coef', 'n']), inplace=True)
    res1_tab.columns = vnew

    #sys.stdout = open(res_out+'\\dimension_all_'+reg+'_D.txt','wt')
    sys.stdout = open(res_out+'\\dimension_all_'+sc+reg+'.txt','wt')
    print('Dimension Level FMB Regressions '+reg)
    print(res1_tab.to_markdown(), end=" ")

    cont_var = ['cons', 't_cons']
    for c in [[c.replace('_std', ''), 't_'+c.replace('_std', '')] for c in controls]:
        cont_var.extend(c)
    cont_var.extend(['coef', 't_coef', 'n'])    

    res1_tab = pd.concat(r1a, axis=1).sum(axis=1, level=0)
    res1_tab = round(res1_tab, 3)
    res1_tab.set_index(pd.Index(cont_var), inplace=True)
    res1_tab.columns = vnew

    #sys.stdout = open(res_out+'\\dimension_all_ct_'+reg+'_D.txt','wt')
    sys.stdout = open(res_out+'\\dimension_all_ct_'+sc+reg+'.txt','wt')
    print('Dimension Level FMB Regressions with Controls '+reg)
    print(res1_tab.to_markdown(), end=" ")

    #A separate table for each characteristic:
        
    for v in reg_vars:
    #for v in reg_vars_dm:
        print(v)
        
        r2[v]=dict(zip(mac_vars,list(r2[v].values()))) 
        res2_tab = pd.concat(r2[v], axis=1).sum(axis=1, level=0)
        res2_tab = round(res2_tab, 3)
        res2_tab.set_index(pd.Index(['cons', 't_cons', 'coef_high', 't_coef_high', 
                                    'coef_low', 't_coef_low', 'n']), inplace=True)
        
        #sys.stdout = open(res_out+"\\"+v+'_all_S_'+reg+'_D.txt','wt')
        sys.stdout = open(res_out+"\\"+v+'_all_S_'+sc+reg+'.txt','wt')
        print('Dimension Level FMB Regressions conditioned on Shocks '+reg)
        print(res2_tab.to_markdown(), end=" ")
        
        cont_var = ['cons', 't_cons', 'coef_high', 't_coef_high', 'coef_low', 't_coef_low']
        for c in [[c.replace('_std', ''), 't_'+c.replace('_std', '')] for c in controls]:
            cont_var.extend(c)
        cont_var.extend(['n'])  
        
        r2a[v]=dict(zip(mac_vars,list(r2a[v].values()))) 
        res2_tab = pd.concat(r2a[v], axis=1).sum(axis=1, level=0)
        res2_tab = round(res2_tab, 3)
        res2_tab.set_index(pd.Index(cont_var), inplace=True)
        
        #sys.stdout = open(res_out+"\\"+v+'_all_S_ct_'+reg+'_D.txt','wt')
        sys.stdout = open(res_out+"\\"+v+'_all_S_ct_'+sc+reg+'.txt','wt')
        print('Dimension Level FMB Regressions with Controls conditioned on Shocks')
        print(res2_tab.to_markdown(), end=" ")
        
    sys.stdout = open(res_out+'\\dimension_all_S_ct_'+sc+reg+'.txt','wt')


#%% Criterion Level

coef_all_c = {}
coefa_all_c = {}  

mac_vars = ['ShockPCA', 'ShockPCA_D']

for reg in regions:
    
    sc = ''
    #sc = 'DM_'
    
    if reg == 'All':
        regct = regdata_ct.copy(deep=True)
    else:
        regct = regdata_ct[regdata_ct['DJREGION']==reg]
        
    r1, r1a, r2, r2a, cf1, cf1a = cs_reg(regct, reg_vars, mac_vars)
    #r1, r1a, r2, r2a, cf1, cf1a = cs_reg(regct, reg_vars_dm, mac_vars)
    #r1, r1a, r2, r2a = cs_reg(regct, reg_vars, mac_vars_d)
    
    coef_all_c[reg] = cf1['reg_vars'] 
    coefa_all_c[reg] = cf1a['reg_vars'] 
    
    #Baseline results
    
    r1 = {k: v for k, v in r1.items() if len(v) != 0}
    r1a = {k: v for k, v in r1a.items() if len(v) != 0}
    r2 = {k: v for k, v in r2.items() if len(v) != 0}
    r2a = {k: v for k, v in r2a.items() if len(v) != 0}
    
    res1 = pd.concat(r1, axis=1).sum(axis=1, level=0).transpose()
    res1 = round(res1, 3)

    res1.sort_values('t_stats_2', inplace=True)
    res1.index = [x.replace(' Imp_std', '') for x in res1.index.values.tolist()]
    #res1.index = [x.replace(' Imp_stdm', '') for x in res1.index.values.tolist()]
    
    res1a = pd.concat(r1a, axis=1).sum(axis=1, level=0).transpose()
    res1a = round(res1a, 3)

    res1a.sort_values('t_stats_7', inplace=True)
    res1a.index = [x.replace(' Imp_std', '') for x in res1a.index.values.tolist()]
    #res1a.index = [x.replace(' Imp_stdm', '') for x in res1a.index.values.tolist()]

    #All characteristics ranked
    #res1['t_stats_2'].plot(kind='bar')
    
    #Adjustements
    
    cf1 = cf1['reg_vars'] 
    cf1a = cf1a['reg_vars'] 

    cfsr1 = {}
    cfsr1a = {}
        
    for v in reg_vars:
    #for v in reg_vars_dm:
        
        if v not in list(cf1.keys()):
            continue
        if v not in list(cf1a.keys()):
            continue

        #Annualized Sharpe ratio:
        cfsr1[v] = (cf1[v].mean()/cf1[v].std())*np.sqrt(12)   
        cfsr1[v] = cfsr1[v].set_axis(['const', 'coef'])
        
        cfsr1a[v] = (cf1a[v].mean()/cf1a[v].std())*np.sqrt(12)   
        cfsr1a[v] = cfsr1a[v].set_axis(['const','coef_1','coef_2','coef_3','coef_4','coef_5','coef_6'])
          
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
    results_ma = {}  # Create a new empty dictionary
    for i in range(len(mac_vars)):
        #print(i)
        results = {}
        for key, value in r2.items():
            results[key] = list(r2[key].values())[i]
        results_m[i] = results
    
        results_m[i] = pd.concat(results_m[i], axis=1).sum(axis=1, level=0).transpose()
        results_m[i] = round(results_m[i], 3)
        
        results = {}
        for key, value in r2a.items():
            results[key] = list(r2a[key].values())[i]
        results_ma[i] = results
    
        results_ma[i] = pd.concat(results_ma[i], axis=1).sum(axis=1, level=0).transpose()
        results_ma[i] = round(results_ma[i], 3)
        
    #All characteristics ranked

    #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    #plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
    plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
    plt.rc('font', weight='bold')    # bold font
    #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
    reg_graph_long(res1, 'coef_2', 't_stats_2')  
    plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_'+sc+reg+'.pdf',
                bbox_inches='tight')

    reg_graph_long(res1a, 'coef_7', 't_stats_7')  
    plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_Con_'+sc+reg+'.pdf',
                bbox_inches='tight')
        
    #Top and Bottom 5 characteristics:    

    reg_graph(res1_top, 'coef_2', 't_stats_2', lab=None)    
    plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_top5_'+sc+reg+'.pdf',
                bbox_inches='tight')
        
    #Sharpe ratio
    
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

    reg_graph_single(cfsr1, 'coef', 210)  
    plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_SR_'+sc+reg+'.pdf',
                bbox_inches='tight')
    #plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_SR_5_'+reg+'.pdf')
          
    reg_graph_single(cfsr1a, 'coef_6', 210)  
    plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_SR_Con_'+sc+reg+'.pdf',
                bbox_inches='tight')
    #plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_SR_Con_5_'+reg+'.pdf')

    #Sharpe ratio with Labels
    
    cfsr1i = pd.merge(cfsr1, csc, left_index=True, right_index=True, how='left')
    cfsr1i = cfsr1i.set_index('Abr').drop('DIMENSION', axis=1)
    
    cfsr1ai = pd.merge(cfsr1a, csc, left_index=True, right_index=True, how='left')
    cfsr1ai = cfsr1ai.set_index('Abr').drop('DIMENSION', axis=1)

    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    
    reg_graph_single(cfsr1i, 'coef', 325)  
    plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_SR_Lab_'+sc+reg+'.pdf',
                bbox_inches='tight')
        
    reg_graph_single(cfsr1ai, 'coef_6', 325)  
    plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_SR_Lab_Con_'+sc+reg+'.pdf',
                bbox_inches='tight')
        
    #Sharpe ratio Governance
    
    #cfsr1g = cfsr1[cfsr1.index.isin(gov)]
    #cfsr1ag = cfsr1a[cfsr1a.index.isin(gov)]
    
    #plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    
    #reg_graph_single_s(cfsr1g, 'coef', 200)  
    #plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Gov_Imp_Sd_SR_'+reg+'.pdf')
    #plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Gov_Imp_Sd_SR_5_'+reg+'.pdf')
         
    #reg_graph_single_s(cfsr1ag, 'coef_6', 200)  
    #plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Gov_Imp_Sd_SR_Con_'+reg+'.pdf')
    #plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Gov_Imp_Sd_SR_Con_5_'+reg+'.pdf')
    
    #For each shock

    plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=11)    # fontsize of the tick labels

    for i in range(len(mac_vars)):
        results_m[i].sort_values('t_stats_2', inplace=True)
        results_m[i].index = [x.replace(' Imp_std', '') for x in results_m[i].index.values.tolist()]
        #results_m[i].index = [x.replace(' Imp_stdm', '') for x in results_m[i].index.values.tolist()]
        res_top = pd.concat([results_m[i].head(5), results_m[i].tail(5)])
            
        #reg_graph(res_top, 'coef_2', 't_stats_2', mac_vars[i]+' High')  
        reg_graph(res_top, 'coef_2', 't_stats_2', lab=None)  
        plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_'+sc+mac_vars[i]+'_H_'+reg+'.pdf',
                    bbox_inches='tight')
            
        results_m[i].sort_values('t_stats_3', inplace=True)
        results_m[i].index = [x.replace(' Imp_std', '') for x in results_m[i].index.values.tolist()]
        #results_m[i].index = [x.replace(' Imp_stdm', '') for x in results_m[i].index.values.tolist()]
        res_top = pd.concat([results_m[i].head(5), results_m[i].tail(5)])
            
        #reg_graph(res_top, 'coef_3', 't_stats_3', mac_vars[i]+' Low')  
        reg_graph(res_top, 'coef_3', 't_stats_3', lab=None)  
        plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_'+sc+mac_vars[i]+'_L_'+reg+'.pdf',
                    bbox_inches='tight')
            
        results_ma[i].sort_values('t_stats_2', inplace=True)
        results_ma[i].index = [x.replace(' Imp_std', '') for x in results_ma[i].index.values.tolist()]
        #results_ma[i].index = [x.replace(' Imp_stdm', '') for x in results_ma[i].index.values.tolist()]
        res_top = pd.concat([results_ma[i].head(5), results_ma[i].tail(5)])
            
        #reg_graph(res_top, 'coef_2', 't_stats_2', mac_vars[i]+' High')   
        reg_graph(res_top, 'coef_2', 't_stats_2', lab=None)   
        plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_Con_'+sc+mac_vars[i]+'_H_'+reg+'.pdf',
                    bbox_inches='tight')
            
        results_ma[i].sort_values('t_stats_3', inplace=True)
        results_ma[i].index = [x.replace(' Imp_std', '') for x in results_ma[i].index.values.tolist()]
        #results_ma[i].index = [x.replace(' Imp_stdm', '') for x in results_ma[i].index.values.tolist()]
        res_top = pd.concat([results_ma[i].head(5), results_ma[i].tail(5)])
            
        #reg_graph(res_top, 'coef_3', 't_stats_3', mac_vars[i]+' Low')    
        reg_graph(res_top, 'coef_3', 't_stats_3', lab=None)    
        plt.savefig(res_out+'\\Criterion Country Level\\Criteria_Imp_Sd_Con_'+sc+mac_vars[i]+'_L_'+reg+'.pdf',
                    bbox_inches='tight')
   
    # Overview over results by criterion/dimension:

    cscoresd = cscores.set_index('CRITERION')
    cscoresd.loc[cscoresd['DIMENSION']=='Economic Dimension', 'DIMENSION'] = 'Governance & Economic Dimension'

    rest = pd.DataFrame()

    res1_tab = pd.concat(r1, axis=1).sum(axis=1, level=0)
    res1_tab = round(res1_tab, 3)
    res1_tab.set_index(pd.Index(['cons', 't_cons', 'coef', 't_coef', 'n']), inplace=True)

    reg_keys = {k for k in r2.keys()}

    for v in reg_keys:
    #for v in reg_vars:
        #print(v)
            
        r2[v]=dict(zip(mac_vars,list(r2[v].values()))) 
        res2_tab = pd.concat(r2[v], axis=1).sum(axis=1, level=0)
        res2_tab = round(res2_tab, 3)
        res2_tab.set_index(pd.Index(['cons', 't_cons', 'coef_high', 't_coef_high', 
                                    'coef_low', 't_coef_low', 'n']), inplace=True)
            
        res_tab = pd.DataFrame()
        res_tab.index = [v]
        #res_tab.columns = ['S'+str(i), 'Sl'+str(i), 'Sln'+str(i), 'Sh'+str(i), 'Shn'+str(i)]
        
        resn_tab = pd.DataFrame(np.empty((1,3)))
        resn_tab.index = [v]
        resn_tab.columns = ['S', 'Sl', 'Sln']
        
        res1_tab['S'] = res1_tab[v]>1.645
        res1_tab['Sl'] = res1_tab[v]>1.282
        res1_tab['Sln'] = (res1_tab[v]>0)
        
        resn_tab['S'][v] = res1_tab['S']['t_coef']
        resn_tab['Sl'][v] = res1_tab['Sl']['t_coef']
        resn_tab['Sln'][v] = res1_tab['Sln']['t_coef']
        
        res_tab = pd.merge(res_tab, resn_tab, left_index=True, right_index=True)
        
        i=0
        for m in mac_vars:
            
            #resn_tab = pd.DataFrame(np.empty((1,12)))
            #resn_tab.index = [v]
            #resn_tab.columns = ['S'+str(i), 'Sl'+str(i), 'Sln'+str(i), 'Sh'+str(i), 'Shn'+str(i),
            #                    'B'+str(i), 'Bl'+str(i), 'Bln'+str(i), 'Bh'+str(i), 'Bhn'+str(i),
            #                    'SB'+str(i), 'SBn'+str(i)]
            resn_tab = pd.DataFrame(np.empty((1,14)))
            resn_tab.index = [v]
            resn_tab.columns = ['SL'+str(i), 'Sl'+str(i), 'Sln'+str(i), 'SH'+str(i),
                                'Sh'+str(i), 'Shn'+str(i),'BL'+str(i), 'Bl'+str(i), 
                                'Bln'+str(i), 'BH'+str(i), 'Bh'+str(i), 'Bhn'+str(i),
                                'SB'+str(i), 'SBn'+str(i)]
            
            res2_tab['S'+str(i)] = res2_tab[m]>1.645
            res2_tab['Sl'+str(i)] = res2_tab[m]>1.282
            res2_tab['Sln'+str(i)] = (res2_tab[m]>0)
            
            res2_tab['B'+str(i)] = res2_tab[m]<-1.645
            res2_tab['Bl'+str(i)] = res2_tab[m]<-1.282
            res2_tab['Bln'+str(i)] = (res2_tab[m]<0)
                    
            #resn_tab['S'+str(i)][v] = res2_tab['Sl'+str(i)]['t_cons']
            resn_tab['SL'+str(i)][v] = res2_tab['S'+str(i)]['t_coef_low']
            resn_tab['Sl'+str(i)][v] = res2_tab['Sl'+str(i)]['t_coef_low']
            resn_tab['Sln'+str(i)][v] = res2_tab['Sln'+str(i)]['t_coef_low']
            resn_tab['SH'+str(i)][v] = res2_tab['S'+str(i)]['t_coef_high']
            resn_tab['Sh'+str(i)][v] = res2_tab['Sl'+str(i)]['t_coef_high']
            resn_tab['Shn'+str(i)][v] = res2_tab['Sln'+str(i)]['t_coef_high']
            
            resn_tab['BL'+str(i)][v] = res2_tab['B'+str(i)]['t_coef_low']
            resn_tab['Bl'+str(i)][v] = res2_tab['Bl'+str(i)]['t_coef_low']
            resn_tab['Bln'+str(i)][v] = res2_tab['Bln'+str(i)]['t_coef_low']
            resn_tab['BH'+str(i)][v] = res2_tab['B'+str(i)]['t_coef_high']
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
    #rest['Scl'] = rest[['Sl0', 'Sl1', 'Sl2', 'Sl3', 'Sl4', 'Sl5']].sum(axis=1)
    #rest['Scln'] = rest[['Sln0', 'Sln1', 'Sln2', 'Sln3', 'Sln4', 'Sln5']].sum(axis=1)
    #rest['Sch'] = rest[['Sh0', 'Sh1', 'Sh2', 'Sh3', 'Sh4', 'Sh5']].sum(axis=1)
    #rest['Schn'] = rest[['Shn0', 'Shn1', 'Shn2', 'Shn3', 'Shn4', 'Shn5']].sum(axis=1)
    #rest['Bcl'] = rest[['Bl0', 'Bl1', 'Bl2', 'Bl3', 'Bl4', 'Bl5']].sum(axis=1)
    #rest['Bcln'] = rest[['Bln0', 'Bln1', 'Bln2', 'Bln3', 'Bln4', 'Bln5']].sum(axis=1)
    #rest['Bch'] = rest[['Bh0', 'Bh1', 'Bh2', 'Bh3', 'Bh4', 'Bh5']].sum(axis=1)
    #rest['Bchn'] = rest[['Bhn0', 'Bhn1', 'Bhn2', 'Bhn3', 'Bhn4', 'Bhn5']].sum(axis=1)
    rest['ScL'] = rest[['SL0', 'SL1']].sum(axis=1)
    rest['Scl'] = rest[['Sl0', 'Sl1']].sum(axis=1)
    rest['Scln'] = rest[['Sln0', 'Sln1']].sum(axis=1)
    rest['ScH'] = rest[['SH0', 'SH1']].sum(axis=1)
    rest['Sch'] = rest[['Sh0', 'Sh1']].sum(axis=1)
    rest['Schn'] = rest[['Shn0', 'Shn1']].sum(axis=1)
    rest['BcL'] = rest[['BL0', 'BL1']].sum(axis=1)
    rest['Bcl'] = rest[['Bl0', 'Bl1']].sum(axis=1)
    rest['Bcln'] = rest[['Bln0', 'Bln1']].sum(axis=1)
    rest['BcH'] = rest[['BH0', 'BH1']].sum(axis=1)
    rest['Bch'] = rest[['Bh0', 'Bh1']].sum(axis=1)
    rest['Bchn'] = rest[['Bhn0', 'Bhn1']].sum(axis=1)

    rest.index = [x[:-8] for x in list(rest.index)]
    rest = pd.merge(rest, cscoresd, left_index=True, right_index=True) 
    
    rest.to_csv(res_out+'\\Criterion Country Level\\CS_Criteria_Imp_Comparison_'+sc+reg+'.csv')

    #rd_dim = rest.groupby('DIMENSION')['Sl', 'Sln', 'S0', 'Sl0', 'Sln0', 'Sh0', 'Shn0', 'B0', 'Bl0', 'Bln0',
           #'Bh0', 'Bhn0', 'SB0', 'SBn0', 'S1', 'Sl1', 'Sln1', 'Sh1', 'Shn1', 'B1',
           #'Bl1', 'Bln1', 'Bh1', 'Bhn1', 'SB1', 'SBn1', 'S2', 'Sl2', 'Sln2', 'Sh2',
           #'Shn2', 'B2', 'Bl2', 'Bln2', 'Bh2', 'Bhn2', 'SB2', 'SBn2', 'S3', 'Sl3',
           #'Sln3', 'Sh3', 'Shn3', 'B3', 'Bl3', 'Bln3', 'Bh3', 'Bhn3', 'SB3',
           #'SBn3', 'S4', 'Sl4', 'Sln4', 'Sh4', 'Shn4', 'B4', 'Bl4', 'Bln4', 'Bh4',
           #'Bhn4', 'SB4', 'SBn4', 'S5', 'Sl5', 'Sln5', 'Sh5', 'Shn5', 'B5', 'Bl5',
           #'Bln5', 'Bh5', 'Bhn5', 'SB5', 'SBn5', 'Sc', 'Scl', 'Scln', 'Sch',
           #'Schn', 'Bcl', 'Bcln', 'Bch', 'Bchn'].sum()
    #rd_dim['Sm'] = rd_dim[['Sh0', 'Sh1', 'Sh2', 'Sh3', 'Sh4', 'Sl5']].mean(axis=1)
    #rd_dim['Smn'] = rd_dim[['Shn0', 'Shn1', 'Shn2', 'Shn3', 'Shn4', 'Sln5']].mean(axis=1)
    #rd_dim['Bm'] = rd_dim[['Bh0', 'Bh1', 'Bh2', 'Bh3', 'Bh4', 'Bl5']].mean(axis=1)
    #rd_dim['Bmn'] = rd_dim[['Bhn0', 'Bhn1', 'Bhn2', 'Bhn3', 'Bhn4', 'Bln5']].mean(axis=1)
    #rd_dim['SBm'] = rd_dim[['SB0', 'SB1', 'SB2', 'SB3', 'SB4', 'SB5']].mean(axis=1)
    #rd_dim['SBmn'] = rd_dim[['SBn0', 'SBn1', 'SBn2', 'SBn3', 'SBn4', 'SBn5']].mean(axis=1)
    rd_dim = rest.groupby('DIMENSION')['S', 'Sl', 'Sln', 'SL0', 'Sl0', 'Sln0', 'SH0', 
            'Sh0', 'Shn0', 'BL0', 'Bl0', 'Bln0', 'BH0', 'Bh0', 'Bhn0', 'SB0', 'SBn0', 
            'SL1', 'Sl1', 'Sln1', 'SH1', 'Sh1', 'Shn1', 'BL1', 'Bl1', 'Bln1',
            'BH1', 'Bh1', 'Bhn1', 'SB1', 'SBn1', 'Sc', 'ScL', 'Scl', 'Scln', 'ScH', 'Sch', 
            'Schn', 'BcL', 'Bcl', 'Bcln', 'BcH', 'Bch', 'Bchn'].sum()
    rd_dim['SM'] = rd_dim[['SH0', 'SH1']].mean(axis=1)
    rd_dim['Sm'] = rd_dim[['Sh0', 'Sh1']].mean(axis=1)
    rd_dim['Smn'] = rd_dim[['Shn0', 'Shn1']].mean(axis=1)
    rd_dim['BM'] = rd_dim[['BH0', 'BH1']].mean(axis=1)
    rd_dim['Bm'] = rd_dim[['Bh0', 'Bh1']].mean(axis=1)
    rd_dim['Bmn'] = rd_dim[['Bhn0', 'Bhn1']].mean(axis=1)
    rd_dim['SBm'] = rd_dim[['SB0', 'SB1']].mean(axis=1)
    rd_dim['SBmn'] = rd_dim[['SBn0', 'SBn1']].mean(axis=1)
    
    rd_dim.to_csv(res_out+'\\Criterion Country Level\\CS_Criteria_DIM_Imp_Comparison_'+sc+reg+'.csv')


#%% Portfolio sorts:

#reg_vars_new = list(data.loc[:,'CSF_LCID':'INDUSTRY'].columns[1:-1])
#reg_vars_dm = [v+'_stdm' for v in reg_vars_new]
#reg_vars_dm = [v+'_dm' for v in reg_vars_new]
    
#pdata = finfunc.portfolio_sort_short(data, ['CSF_LCID', 'month_id'], 'month_id', reg_vars_new, 5)
#pdatad = finfunc.portfolio_sort_short(data, ['CSF_LCID', 'month_id'], 'month_id', reg_vars_dm, 5)

#Return computation:

#rdata = finfunc.portfolio_return(pdata, 'month_id', reg_vars_new, 5, 'ret_usd', weight='market_cap_usd')
#rdatad = finfunc.portfolio_return(pdatad, 'month_id', reg_vars_dm, 5, 'ret_usd', weight='market_cap_usd')

#By region:
    
data.loc[data['COUNTRYNAME']=='Japan', 'DJREGION'] = 'JAP'
data['DJREGION'].value_counts()

regions = ['All','NAM','EUR','APA','JAP','EM']
    
for reg in regions:
    
    print(reg)
    
    if reg == 'All':
        regct = data.copy(deep=True)
    else:
        regct = data[data['DJREGION']==reg]
        
    #regcount = regct.groupby('month_id')[reg_vars_new].count()
    #print(regcount)
            
    if reg == 'All':
        #Portfolio sorts:
        #pdata = finfunc.portfolio_sort_short(regct, ['CSF_LCID', 'month_id'], 'month_id', reg_vars_new, 5)
        pdata = finfunc.portfolio_sort_short(regct, ['CSF_LCID', 'month_id'], 'month_id', var_name, 5)
        #pdata = finfunc.portfolio_sort_short(regct, ['CSF_LCID', 'month_id'], 'month_id', var_name_dm, 5)
        
        for p in var_name:
        #for p in var_name_dm:
            pdata['month_obs'] = pdata.groupby(['month_id'])[p].transform('count')  
            #if pdatar['month_obs'].min()>=20:
            #    rdata_vars_n.extend([p.replace('pt_','')])
            pdata.loc[pdata['month_obs']<50, 'pt_'+p] = 5
        
        #Return computation:
        #rdata = finfunc.portfolio_return(pdata, 'month_id', reg_vars_new, 5, 'ret_usd', weight='market_cap_usd')
        rdata = finfunc.portfolio_return(pdata, 'month_id', var_name, 5, 'ret_usd', weight='market_cap_usd')
        #rdatadm = finfunc.portfolio_return(pdata, 'month_id', var_name_dm, 5, 'ret_usd', weight='market_cap_usd')

        rdata.columns = [x+'_All' for x in list(rdata.columns)]
        #rdatadm.columns = [x+'_All' for x in list(rdatadm.columns)]
    else:
        #Portfolio sorts:
        #pdatar = finfunc.portfolio_sort_short(regct, ['CSF_LCID', 'month_id'], 'month_id', reg_vars_new, 5)
        pdatar = finfunc.portfolio_sort_short(regct, ['CSF_LCID', 'month_id'], 'month_id', var_name, 5)
        #pdatar = finfunc.portfolio_sort_short(regct, ['CSF_LCID', 'month_id'], 'month_id', var_name_dm, 5)
        
        #pdatar1 = pdatar.copy(deep=True)
        
        #Minimum number of observations:
        #for p in [x for x in list(pdatar1.columns) if 'pt' in x]:
        #for p in reg_vars_new:
        for p in var_name:
        #for p in var_name_dm:
            pdatar[p+'_min'] = pdatar.groupby(['month_id'])['pt_'+p].transform('min') 
            pdatar[p+'_max'] = pdatar.groupby(['month_id'])['pt_'+p].transform('max')
            
            pdatar.loc[(pdatar[p+'_min'].isin(list(range(2,6))))&(
                pdatar[p]==0), 'pt_'+p] = 1
            pdatar.loc[(pdatar[p+'_max'].isin(list(range(1,5))))&(
                pdatar[p]>0)&(~pdatar[p].isnull()), 'pt_'+p] = 5
            
            pdatar['month_obs'] = pdatar.groupby(['month_id'])[p].transform('count')  
            #if pdatar['month_obs'].min()>=20:
            #    rdata_vars_n.extend([p.replace('pt_','')])
            pdatar.loc[pdatar['month_obs']<50, 'pt_'+p] = 5
            
        #Return computation:
        #rdatar = finfunc.portfolio_return(pdatar, 'month_id', reg_vars_new, 5, 'ret_usd', weight='market_cap_usd')
        rdatar = finfunc.portfolio_return(pdatar, 'month_id', var_name, 5, 'ret_usd', weight='market_cap_usd')
        #rdatar = finfunc.portfolio_return(pdatar, 'month_id', var_name_dm, 5, 'ret_usd', weight='market_cap_usd')
        
        pvar = ['month_id', 'CIQ_ID']
        pvar.extend([x for x in list(pdatar.columns) if 'pt_' in x])
        pdatar = pdatar[pvar] 
        
        pvar = ['month_id', 'CIQ_ID']
        pvar.extend([x+'_'+reg for x in list(pdatar.columns) if 'pt_' in x])
        pdatar.columns = pvar
        
        pdata = pd.merge(pdata, pdatar, on=['month_id', 'CIQ_ID'], how='left')
        
        rvar = [x for x in list(rdatar.columns) if 'ls_' in x]
        rdatar = rdatar[rvar] 
        rdatar.columns = [x+'_'+reg for x in list(rdatar.columns) if 'ls_' in x]
        
        rdata = pd.merge(rdata, rdatar, left_index=True, right_index=True, how='left')
        #rdatadm = pd.merge(rdatadm, rdatar, left_index=True, right_index=True, how='left')
        
##Create macroeconomic control variables:
    
mac_vars = ['Macro Uncertainty', 'Twitter Macro Uncertainty', 'Global Manufacturing PMI Adjusted', 
            'ShockPCA', 'Global Manufacturing PMI Adjusted_D', 'ShockPCA_D']

rdata.describe()
#rdatadm.describe()

#rdata = pd.merge(rdata, econ, on='month_id', how='left')
rdata = pd.merge(rdata, econd, on='month_id', how='left')
#rdatadm = pd.merge(rdatadm, econ, on='month_id', how='left')
#rdatadm = pd.merge(rdatadm, econd, on='month_id', how='left')

#for i in range(0,len(mac_vars_d)):
for i in range(0,len(mac_vars)):
    print(mac_vars[i])
    rdata['qu'+str(i)] = pd.qcut(rdata[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
    #rdata['qu'+str(i)] = pd.qcut(rdata[mac_vars_d[i]], q=[0, .75, 1], labels=False).astype(float)
    #rdatad['qu'+str(i)] = pd.qcut(rdatad[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
    rdata['qu_s'+str(i)] = (rdata[mac_vars[i]]-rdata[mac_vars[i]].mean())/rdata[mac_vars[i]].std()
    rdata['qu_n'+str(i)] = ((rdata[mac_vars[i]]-rdata[mac_vars[i]].min())/(
        rdata[mac_vars[i]].max()-rdata[mac_vars[i]].min()))

gc.collect()


#%% Time-series regressions preparation:

rdata.describe()

#reg_vars_new =  [v for v in reg_vars if not any(s in v for s in ['_std', '_stdm', '_dms'])] 
#reg_vars_new = list(data.loc[:,'CSF_LCID':'INDUSTRY'].columns[1:-1])
#reg_vars_new = [''.join(c for c in s if c not in ['(',')']) for s in reg_vars_new]
#reg_vars_dm = [''.join(c for c in s if c not in ['(',')']) for s in reg_vars_dm]

#print(reg_vars_new)
print(rdata.columns)

##All macroeconomic variables and only long/short portfolios

ls = ['month_id']
#lsd = ['month_id']
ls.extend(list(rdata.loc[:,(rdata.columns.str.contains('_ls_'))].columns))
#lsd.extend(list(rdatad.loc[:,(rdatad.columns.str.contains('_ls_'))].columns))
#ls.extend(list(rdatadm.loc[:,(rdatadm.columns.str.contains('_ls_'))].columns))

rdatan = rdata[ls].copy(deep=True) 
#rdatadn = rdatad[lsd] 
#rdatan = rdatadm[ls].copy(deep=True) 

#rdatan = pd.merge(rdatan, econ, on='month_id', how='left')
rdatan = pd.merge(rdatan, econd, on='month_id', how='left')
#rdatadn = pd.merge(rdatadn, econd, on='month_id', how='left')

rdatan = rdatan[rdatan['month_id'] < '2023-01']
#rdatan = rdatan[rdatan['month_id'] < '2023-04']
#rdatadn = rdatadn[rdatadn['month_id'] < '2023-04']


## Regression controls:

ff5 = pd.read_csv(input+'\\Financials\\Factors\\5_Factors.csv')
mom = pd.read_csv(input+'\\Financials\\Factors\\Mom_Factors.csv')

ff5n = ff5[ff5.region=='NAM']
momn = mom[mom.region=='NAM']
ff5e = ff5[ff5.region=='EUR']
mome = mom[mom.region=='EUR']
ff5a = ff5[ff5.region=='APA']
moma = mom[mom.region=='APA']
ff5j = ff5[ff5.region=='Japan']
momj = mom[mom.region=='Japan']
ff5m = ff5[ff5.region=='EM']
momm = mom[mom.region=='EM']
ff5d = ff5[ff5.region=='Developed']
momd = mom[mom.region=='Developed']

ff5n['month_id'] = pd.to_datetime(ff5n['month_id']).dt.to_period('M') 
momn['month_id'] = pd.to_datetime(momn['month_id']).dt.to_period('M') 
ff5e['month_id'] = pd.to_datetime(ff5e['month_id']).dt.to_period('M') 
mome['month_id'] = pd.to_datetime(mome['month_id']).dt.to_period('M') 
ff5a['month_id'] = pd.to_datetime(ff5a['month_id']).dt.to_period('M') 
moma['month_id'] = pd.to_datetime(moma['month_id']).dt.to_period('M') 
ff5j['month_id'] = pd.to_datetime(ff5j['month_id']).dt.to_period('M') 
momj['month_id'] = pd.to_datetime(momj['month_id']).dt.to_period('M') 
ff5m['month_id'] = pd.to_datetime(ff5m['month_id']).dt.to_period('M') 
momm['month_id'] = pd.to_datetime(momm['month_id']).dt.to_period('M') 
ff5d['month_id'] = pd.to_datetime(ff5d['month_id']).dt.to_period('M') 
momd['month_id'] = pd.to_datetime(momd['month_id']).dt.to_period('M') 

rdatan = pd.merge(rdatan, ff5n, on='month_id', how='left', suffixes=['', '_NAM'])
rdatan = pd.merge(rdatan, momn, on='month_id', how='left', suffixes=['', '_NAM'])
rdatan = pd.merge(rdatan, ff5e, on='month_id', how='left', suffixes=['', '_EUR'])
rdatan = pd.merge(rdatan, mome, on='month_id', how='left', suffixes=['', '_EUR'])
rdatan = pd.merge(rdatan, ff5a, on='month_id', how='left', suffixes=['', '_APA'])
rdatan = pd.merge(rdatan, moma, on='month_id', how='left', suffixes=['', '_APA'])
rdatan = pd.merge(rdatan, ff5j, on='month_id', how='left', suffixes=['', '_JAP'])
rdatan = pd.merge(rdatan, momj, on='month_id', how='left', suffixes=['', '_JAP'])
rdatan = pd.merge(rdatan, ff5m, on='month_id', how='left', suffixes=['', '_EM'])
rdatan = pd.merge(rdatan, momm, on='month_id', how='left', suffixes=['', '_EM'])
rdatan = pd.merge(rdatan, ff5d, on='month_id', how='left', suffixes=['', '_Dev'])
rdatan = pd.merge(rdatan, momd, on='month_id', how='left', suffixes=['', '_Dev'])

aqr = pd.read_csv(input+'\\Financials\\Factors\\AQR_Factors.csv')

aqr['month_id'] = pd.to_datetime(aqr['Date']).dt.to_period('M') 

rdatan = pd.merge(rdatan, aqr, on='month_id', how='left', suffixes=['_N',''])
#rdatadn = pd.merge(rdatadn, aqr, on='month_id', how='left', suffixes=['_N',''])

aqr_mkt = pd.read_csv(input+'\\Financials\\Factors\\MKT_Factor.csv')
aqr_mkt.set_index('Date', inplace=True)
aqr_mkt = aqr_mkt*100
aqr_mkt = aqr_mkt.reset_index()

aqr_mkt.columns = ['MKT_'+x for x in list(aqr_mkt.columns)]
aqr_mkt['month_id'] = pd.to_datetime(aqr_mkt['MKT_Date']).dt.to_period('M') 

rdatan = pd.merge(rdatan, aqr_mkt, on='month_id', how='left', suffixes=['','_A'])
#rdatadn = pd.merge(rdatadn, aqr, on='month_id', how='left', suffixes=['_N',''])

#factors = ['Mkt-RF', 'SMB_N', 'HML_N', 'RMW', 'CMA', 'WML']
#factorsa = ['MKT', 'SMB', 'HML', 'QMJ', 'UMD']
#mkt = ['Mkt-RF']

try:
    mwm = [x for x in list(rdatan.columns) if 'Mineral Waste Management' in x]
    rdatan.drop(mwm, axis=1, inplace=True)
    #reg_vars_new.remove('Mineral Waste Management Imp')
    var_name.remove('Mineral Waste Management Imp')
    var_name_dm.remove('Mineral Waste Management Imp_dm')
except:
    pass

#rdatadmn = rdatan.copy(deep=True)

reg_vars_new = var_name.copy()    
#reg_vars_new = var_name_dm.copy()    


#%% Standard regressions:

#regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, nw=6)
#regtd_ls,tvld_ls,regtd,tvld,regsd = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, nw=6)
    
#regt_ls,tvl_ls,regt,tvl,regsm = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, 
#                                                   controls=['Mkt-RF'], nw=6)
#regtd_ls,tvld_ls,regtd,tvld,regsdm = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, 
#                                                        controls=['Mkt-RF'], nw=6)
    
#regt_ls,tvl_ls,regt,tvl,regs5 = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, 
#                                                   controls=factors, nw=6)
#regtd_ls,tvld_ls,regtd,tvld,regsd5 = finfunc.ts_reg_det(rdatadn, reg_vars_dm, top=5, 
#                                                        controls=factorsa, nw=6)
    
#for j in range(0,len(reg_vars_new)):
    #reg_rs.extend([regs[j]])
    #reg_rs.extend([regsm[j]])
    #reg_rs.extend([regs5[j]])
    #reg_rsd.extend([regsd[j]])
    #reg_rsd.extend([regsdm[j]])
    #reg_rsd.extend([regsd5[j]])

#By country:

reg_rs = []
reg_rsd = []
    
regions = ['All','NAM','EUR','APA','JAP','EM']
    
for reg in regions:
    
    if reg == 'All':
        factors = ['MKT_Global', 'SMB_Dev', 'HML_Dev', 'RMW_Dev', 'CMA_Dev', 'WML_Dev']
    elif reg == 'NAM':
        factors = ['Mkt-RF', 'SMB_N', 'HML_N', 'RMW', 'CMA', 'WML']
    elif reg == 'EUR':
        factors = ['Mkt-RF_EUR', 'SMB_EUR', 'HML_EUR', 'RMW_EUR', 'CMA_EUR', 'WML_EUR']
    elif reg == 'APA':
        factors = ['Mkt-RF_APA', 'SMB_APA', 'HML_APA', 'RMW_APA', 'CMA_APA', 'WML_APA']
    elif reg == 'JAP':
        factors = ['Mkt-RF_JAP', 'SMB_JAP', 'HML_JAP', 'RMW_JAP', 'CMA_JAP', 'WML_JAP']
    elif reg == 'EM':
        factors = ['Mkt-RF_EM', 'SMB_EM', 'HML_EM', 'RMW_EM', 'CMA_EM', 'WML_EM']
  
    rdatar = rdatan.copy(deep=True)
    reg_vars_n = []
    
    #for v in reg_vars_new:
    for v in var_name:
    #for v in var_name_dm:
        rvar = list(rdatar.loc[:,(rdatar.columns.str.contains(v+'_r'))|(rdatar.columns.str.contains(v+'_l'))].columns) 
        rvar = [x for x in rvar if reg not in x]
        rdatar.drop(rvar, axis=1, inplace=True)
        
        #print(rdatar[v+'_ls_5_'+reg].count())
        if rdatar[v+'_ls_5_'+reg].count()>1:
            #rdatar.drop(v+'_ls_5_'+reg, axis=1, inplace=True)
            reg_vars_n.extend([v])
            
    regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatar, reg_vars_n, top=5, nw=6)

    regt_ls,tvl_ls,regt,tvl,regs5 = finfunc.ts_reg_det(rdatar, reg_vars_n, top=5, 
                                                       controls=factors, nw=6)
    
    #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatar, reg_vars_new, top=5, nw=6)
    #regt_ls,tvl_ls,regt,tvl,regs5 = finfunc.ts_reg_det(rdatar, reg_vars_new, top=5, 
    #                                                   controls=factors, nw=6)
    
    for j in range(0,len(reg_vars_n)):
        reg_rs.extend([regs[j]])
        reg_rs.extend([regs5[j]])

#Dimension:

region = ['All','NAM','EUR','APA','JAP','EM']    
model_name = ['Env1','Env2','Gov1','Gov2','Soc1','Soc2']
model_names = []
for r in region:
    mnames = [x+r for x in model_name]
    model_names.extend(mnames)
    
dfoutput = summary_col(reg_rs, stars=True, model_names=model_names,
                       float_format='%.3f', regressor_order=['MKT_Global', 'SMB_Dev', 'HML_Dev', 
                       'RMW_Dev', 'CMA_Dev', 'WML_Dev', 'Mkt-RF', 'SMB_N', 'HML_N', 'RMW', 'CMA', 
                       'WML', 'Mkt-RF_EUR', 'SMB_EUR', 'HML_EUR', 'RMW_EUR', 'CMA_EUR', 'WML_EUR',
                       'Mkt-RF_APA', 'SMB_APA', 'HML_APA', 'RMW_APA', 'CMA_APA', 'WML_APA',
                       'Mkt-RF_JAP', 'SMB_JAP', 'HML_JAP', 'RMW_JAP', 'CMA_JAP', 'WML_JAP',
                       'Mkt-RF_EM', 'SMB_EM', 'HML_EM', 'RMW_EM', 'CMA_EM', 'WML_EM'])
                       #stars=True, float_format='%.3f', regressor_order=['Mkt-RF', 'SMB', 'HML'])
sys.stdout = open(res_out+"\\ls_all_dimension.txt",'wt')
#sys.stdout = open(res_out+"\\ls_all_dimension_dm.txt",'wt')
print(dfoutput)

#dfoutput = summary_col(reg_rsd, model_names=['Env 1','Env 2','Env 3','Gov 1','Gov 2',
#                                            'Gov 3', 'Soc 1','Soc 2','Soc 3'],
#                       stars=True, float_format='%.3f',regressor_order=['MKT', 'SMB', 'HML'])
                       #stars=True, float_format='%.3f',regressor_order=['Mkt-RF', 'SMB', 'HML'])
#sys.stdout = open(res_out+"\\ls_all_dimension_DM.txt",'wt')
#print(dfoutput)


#%% Regressions controlling for macro factors dimension level:

for reg in regions:
    
    sc = ''
    #sc = 'DM_' 
    
    print(reg)
    
    reg_rsm = []
    reg_rsdm = []
    
    rd_col = [x for x in list(rdatan.columns) if 'ls_5' in x and reg not in x]  
    rdatanc = rdatan.copy(deep=True)
    rdatanc.drop(rd_col, axis=1, inplace=True)

    if reg == 'All':
        factors = ['MKT_Global', 'SMB_Dev', 'HML_Dev', 'RMW_Dev', 'CMA_Dev', 'WML_Dev']
    elif reg == 'NAM':
        factors = ['Mkt-RF', 'SMB_N', 'HML_N', 'RMW', 'CMA', 'WML']
    elif reg == 'EUR':
        factors = ['Mkt-RF_EUR', 'SMB_EUR', 'HML_EUR', 'RMW_EUR', 'CMA_EUR', 'WML_EUR']
    elif reg == 'APA':
        factors = ['Mkt-RF_APA', 'SMB_APA', 'HML_APA', 'RMW_APA', 'CMA_APA', 'WML_APA']
    elif reg == 'JAP':
        factors = ['Mkt-RF_JAP', 'SMB_JAP', 'HML_JAP', 'RMW_JAP', 'CMA_JAP', 'WML_JAP']
    elif reg == 'EM':
        factors = ['Mkt-RF_EM', 'SMB_EM', 'HML_EM', 'RMW_EM', 'CMA_EM', 'WML_EM']
        
    for i in range(0,len(mac_vars)):
        print(i)
        
        controlv=['qu'+str(i)]
        controlv.extend(factors)
        controln=['qu_n'+str(i)]
        controln.extend(factors)

        rdatanc['qu'+str(i)] = pd.qcut(rdatanc[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
        #rdatanc['qu'+str(i)] = abs(rdatanc['qu'+str(i)]-1)
        rdatanc['qu_s'+str(i)] = (rdatanc[mac_vars[i]]-rdatanc[mac_vars[i]].mean())/rdatanc[mac_vars[i]].std()
        rdatanc['qu_n'+str(i)] = ((rdatanc[mac_vars[i]]-rdatanc[mac_vars[i]].min())/(
            rdatanc[mac_vars[i]].max()-rdatanc[mac_vars[i]].min()))
        #regt,reg,regres = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)
        #regt_ls,reg_ls,regt,reg = ts_reg_det(rdatan, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)
        regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatanc, reg_vars_new, top=5, 
                                                          controls=['qu'+str(i)], nw=6)
        regtd_ls,tvld_ls,regtd,tvld,regsd = finfunc.ts_reg_det(rdatanc, reg_vars_new, top=5, 
                                                               controls=['qu_n'+str(i)], nw=6)
            
        regt_ls,tvl_ls,regt,tvl,regs5 = finfunc.ts_reg_det(rdatanc, reg_vars_new, top=5, 
                                                           controls=controlv, nw=6)
        regtd_ls,tvld_ls,regtd,tvld,regsd5 = finfunc.ts_reg_det(rdatanc, reg_vars_new, top=5, 
                                                               controls=controln, nw=6)
        
        for j in range(0,len(reg_vars_new)):
            reg_rsm.extend([regs[j]])
            reg_rsm.extend([regs5[j]])
            reg_rsdm.extend([regsd[j]])
            reg_rsdm.extend([regsd5[j]])
            
    #Splitting

    dfoutput = summary_col(reg_rsm[:18],model_names=['Env1MU','Env2MU','Gov1MU', 'Gov2MU',
                                                     'Soc1MU','Soc2MU','Env1TMU','Env2TMU',
                                                     'Gov1TMU','Gov2TMU','Soc1TMU','Soc2TMU',
                                                     'Env1PMI','Env2PMI','Gov1PMI','Gov2PMI',
                                                     'Soc1PMI','Soc2PMI'],
                           stars=True, float_format='%.3f', regressor_order=factors)
    sys.stdout = open(res_out+'\\ls_all_dimension_shocks_'+sc+reg+'_1.txt','wt')
    print(dfoutput)
    dfoutput = summary_col(reg_rsm[18:43],model_names=['Env1SPC','Env2SPC','Gov1SPC', 'Gov2SPC',
                                                       'Soc1SPC','Soc2SPC','Env1PID','Env2PID',
                                                       'Gov1PID','Gov2PID','Soc1PID','Soc2PID',
                                                       'Env1SCD','Env2SCD','Gov1SCD','Gov2SCD',
                                                       'Soc1SCD','Soc2SCD'],
                           stars=True, float_format='%.3f', regressor_order=factors)
    sys.stdout = open(res_out+'\\ls_all_dimension_shocks_'+sc+reg+'_2.txt','wt')
    print(dfoutput)
    
    #Interacting

    dfoutput = summary_col(reg_rsdm[:18],model_names=['Env1MU','Env2MU','Gov1MU', 'Gov2MU',
                                                      'Soc1MU','Soc2MU','Env1TMU','Env2TMU',
                                                      'Gov1TMU','Gov2TMU','Soc1TMU','Soc2TMU',
                                                      'Env1PMI','Env2PMI','Gov1PMI','Gov2PMI',
                                                      'Soc1PMI','Soc2PMI'],
                           stars=True, float_format='%.3f', regressor_order=factors)
    sys.stdout = open(res_out+'\\ls_all_dimension_shocks_int_'+sc+reg+'_1.txt','wt')
    print(dfoutput)
    dfoutput = summary_col(reg_rsdm[18:43],model_names=['Env1SPC','Env2SPC','Gov1SPC', 'Gov2SPC',
                                                       'Soc1SPC','Soc2SPC','Env1PID','Env2PID',
                                                       'Gov1PID','Gov2PID','Soc1PID','Soc2PID',
                                                       'Env1SCD','Env2SCD','Gov1SCD','Gov2SCD',
                                                       'Soc1SCD','Soc2SCD'],
                           stars=True, float_format='%.3f', regressor_order=factors)
    sys.stdout = open(res_out+'\\ls_all_dimension_shocks_int_'+sc+reg+'_2.txt','wt')
    print(dfoutput)
    
    
#%% Scale cross-section factor to volatility of corresponding time-series factor

#colors = [(0, 43/235, 95/235),(0, 126/235, 174/235),(161/235, 195/235, 218/235)]
#colors_pt = [(0, 43/235, 95/235),(6/235, 66/235, 61/235),
#             (0, 126/235, 174/235),(6/235, 146/235, 125/235),
#             (161/235, 195/235, 218/235),(153/235, 228/235, 215/235)]
#colors_ls = [(122/235, 104/235, 85/235), (6/235, 146/235, 124/235)]
#lines = [':',':','--','--','-','-']

#Specify line design:

colors_pt = [plt.cm.tab20c(8),plt.cm.tab20c(9),plt.cm.tab20c(0),
             plt.cm.tab20c(1),plt.cm.tab20c(4),plt.cm.tab20c(5)]
colors_ls = [(122/235, 104/235, 85/235), (6/235, 146/235, 124/235)]
#lines = ['--','-','--','-','--','-']
lines = ['solid','solid',(0,(5,3)),(0,(2,2)),(0,(4,1)),(0,(2,1))]
opacity = [1,0.75,1,0.75,1,0.75]
width = [1.5,4,1.5,4,1.5,4]

rd_std = rdatan.std()*np.sqrt(12)
#rd_std = rdatadmn.std()*np.sqrt(12)
rd_std.index = [x.replace(' Imp_ls_5', '') for x in rd_std.index.values.tolist()]

sd_cs = {}
sda_cs = {}

for reg in region:
    
    sc = ''
    #sc = 'DM_'
    
    print(reg)
    
    sd1_cs = {}
    sd1a_cs = {}
    
    coef1 = coef_all[reg]['reg_vars'] 
    coef1a = coefa_all[reg]['reg_vars']
    
    #Scale CS factors:
    
    for v in reg_vars:
    #for v in reg_vars_dm:

        x = v.replace(' Imp_std', '')
        #x = v.replace(' Imp_stdm', '')
      
        sd1_cs[v] = coef1[v]*(rd_std[x+'_'+reg]/(coef1[v].std()*np.sqrt(12)))
        sd1a_cs[v] = coef1a[v]*(rd_std[x+'_'+reg]/(coef1a[v].std()*np.sqrt(12)))
        
        sd1_cs[v] = sd1_cs[v][v]
        sd1a_cs[v] = sd1a_cs[v][v]
        
    sd_cs[reg] = sd1_cs
    sda_cs[reg] = sd1a_cs
    
    #Merge CS and TS factors:
   
    #rdata_u = rdata.iloc[7:,:]
    rdata_u = rdata.copy(deep=True)
    #rdata_u = rdatadm.copy(deep=True)
    
    fct_df = pd.concat(sda_cs[reg], axis=1) 
    fct_df = pd.concat([fct_df, pd.DataFrame(rdata_u['month_id'].astype(str).unique())], axis=1)    
    fct_df.rename(columns={0:'month_id'}, inplace=True)
    fct_df = fct_df.set_index('month_id')

    rd_col = [x for x in list(rdata_u.columns) if 'ls_5_'+reg in x]  
    rdata_u['month_id'] = rdata_u['month_id'].dt.strftime('%Y-%m')
    rdata_u = rdata_u.set_index('month_id')
    
    fct_sc = pd.merge(fct_df, rdata_u[rd_col], left_index=True, right_index=True)

    fct_sc.sort_index(inplace=True)
    fct_sc.index = pd.to_datetime(fct_sc.index).to_period('M')         

    #Plotting:
        
    plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
    plt.rc('font', weight='normal') 

    i=0
    fig, ax = plt.subplots(figsize=(7,5))
    for d in ['Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
        for s in [' Imp_std', ' Imp_ls_5_'+reg]:
            
            if s == ' Imp_std':
                ft = ' Score Factor'
            elif s == ' Imp_ls_5_'+reg:
                ft = ' Sort Factor'
                
            fct_sc.rename(columns={d+s:d+ft}, inplace=True) 
            fac = [d+ft]         
            
            for f in fac:
                #fig, ax = plt.subplots(layout='constrained')
                #fig = plt.subplot(layout='constrained')
                fig = plt.subplot()
                ((1+(fct_sc[f]/100)).cumprod()).plot(title='Cumulative Returns of Dimension Score Strategies '+reg,
                                                     ylabel='Value of 1$ Invested',
                                                     xlabel='Year',color=colors_pt[i],linestyle=lines[i],
                                                     grid=True, alpha=opacity[i], linewidth=width[i])
                                                     #xlabel='Year',color=plt.cm.tab20(i),linestyle=lines[i])
                fig.legend()
                #fig.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='upper left',
                #           mode="expand", borderaxespad=0.)
                i+=1

    plt.savefig(res_out+'\\Sustainability_Factors_'+sc+reg+'.pdf')
    plt.clf()
    
    i=0
    fig, ax = plt.subplots(figsize=(10,5))
    for d in ['Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
        for s in [' Imp_std', ' Imp_ls_5_'+reg]:
            
            if s == ' Imp_std':
                ft = ' Score Factor'
            elif s == ' Imp_ls_5_'+reg:
                ft = ' Sort Factor'
                
            fct_sc.rename(columns={d+s:d+ft}, inplace=True) 
            fac = [d+ft]   

            for f in fac:
                #fig, ax = plt.subplots(layout='constrained')
                #fig = plt.subplot(layout='constrained')
                #fig, ax = plt.subplots(figsize=(10,5))
                #plt.figure(1)
                #fig = plt.subplot()
                #ax = fig.add_subplot()
                ((1+(fct_sc[f]/100)).cumprod()).plot(title=reg, ylabel='Value of 1$ Invested',
                                                     xlabel='Year',color=colors_pt[i],linestyle=lines[i],
                                                     grid=True, alpha=opacity[i], linewidth=width[i])
                                                     #xlabel='Year',color=plt.cm.tab20(i),linestyle=lines[i])
                #fig.legend(bbox_to_anchor=(1.02, 0.34), loc='upper left',
                #           borderaxespad=0.)
                ax.set_position([0.1,0.1,0.5,0.8])
                i+=1
            fig.legend(bbox_to_anchor=(1.00, 0.375))

    plt.savefig(res_out+'\\Sustainability_Factors_Red_'+sc+reg+'.pdf')
    plt.clf()
    
    i=0
    fig, ax = plt.subplots(figsize=(7,5))
    for d in ['Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
        for s in [' Imp_std', ' Imp_ls_5_'+reg]:
            
            if s == ' Imp_std':
                ft = ' Score Factor'
            elif s == ' Imp_ls_5_'+reg:
                ft = ' Sort Factor'
                
            fct_sc.rename(columns={d+s:d+ft}, inplace=True) 
            fac = [d+ft]         
            
            for f in fac:
                #fig, ax = plt.subplots(layout='constrained')
                #fig = plt.subplot(layout='constrained')
                fig = plt.subplot()
                ((1+(fct_sc[f]/100)).cumprod()).plot(title=reg, ylabel='Value of 1$ Invested',
                                                     xlabel='Year',color=colors_pt[i],linestyle=lines[i],
                                                     grid=True, alpha=opacity[i], linewidth=width[i])
                                                     #xlabel='Year',color=plt.cm.tab20(i),linestyle=lines[i])
                i+=1

    plt.savefig(res_out+'\\Sustainability_Factors_NoLeg_'+sc+reg+'.pdf')
    plt.clf()


#%% Functions for later plotting

#colors = [(6/235, 146/235, 126/235),(0, 94/235, 146/235)]
#colors = [(163/235, 147/235, 130/235),(0, 94/235, 146/235)]
#colors = [(161/235, 195/235, 218/235),(0, 94/235, 146/235)]
colors = [(153/235, 228/235, 215/235),(10/235, 107/235, 91/235)]

plt.rc('font', weight='bold') 

def reg_graph(df, v1, v2, lab):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))

    ax2 = ax.twinx()
    df[v1].plot(kind='bar', color=colors[0], ax=ax2, width=0.25, 
                            position=0, align='center', label='Coefficient')
    df[v2].plot(kind='bar', color=colors[1], ax=ax, width=0.25,
                               position=1, align='center', label='T-statistics', grid=True)
    plt.xticks()

    ax.set_xlim(-0.5, 10)
    #ax.set_ylim(-2, 2)
    #ax.set_ylim(-2.5, 2.5)
    ax.set_ylim(-4.5, 4.5)
    ax2.set_ylim(-4.5, 4.5)
    #ax2.set_ylim(-0.8, 0.8)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax.legend(lines + lines2, labels + labels2, loc=0)
    ax.legend(lines + lines2, labels + labels2, loc=2, framealpha=1.0)

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
    
def reg_graph1(df, v1, v2, lab):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))

    ax2 = ax.twinx()
    df[v1].plot(kind='bar', color=colors[0], ax=ax2, width=0.25, 
                            position=0, align='center', label='Intercept')
    df[v2].plot(kind='bar', color=colors[1], ax=ax, width=0.25,
                               position=1, align='center', label='T-statistics', grid=True)
    plt.xticks()

    ax.set_xlim(-0.5, 10)
    #ax.set_ylim(-2, 2)
    #ax.set_ylim(-2.5, 2.5)
    ax.set_ylim(-4.5, 4.5)
    ax2.set_ylim(-4.5, 4.5)
    #ax2.set_ylim(-0.8, 0.8)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax.legend(lines + lines2, labels + labels2, loc=0)
    ax.legend(lines + lines2, labels + labels2, loc=2, framealpha=1.0)

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
    ax2.set_ylabel('Intercept')
    fig.text(0.5, 0.01, lab, ha='center')
 
def reg_graph_long(df, v1, v2):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 10))

    ax2 = ax.twinx()

    df[v1].plot(kind='bar', color=colors[0], ax=ax2, width=0.25, 
                            position=0, align='center', label='Coefficient')
    df[v2].plot(kind='bar', color=colors[1], ax=ax, width=0.25,
                               position=1, align='center', label='T-statistics', grid=True)
    plt.xticks()

    #ax.set_xlim(-0.5, 10)
    ax.set_ylim(-3.5, 3.5)
    ax2.set_ylim(-1.75, 1.75)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax.legend(lines + lines2, labels + labels2, loc=0)
    ax.legend(lines + lines2, labels + labels2, loc=2, framealpha=1.0)
    
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

def reg_graph_long1(df, v1, v2):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 10))

    ax2 = ax.twinx()

    df[v1].plot(kind='bar', color=colors[0], ax=ax2, width=0.25, 
                            position=0, align='center', label='Intercept')
    df[v2].plot(kind='bar', color=colors[1], ax=ax, width=0.25,
                               position=1, align='center', label='T-statistics', grid=True)
    plt.xticks()

    #ax.set_xlim(-0.5, 10)
    ax.set_ylim(-3.5, 3.5)
    ax2.set_ylim(-1.75, 1.75)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax.legend(lines + lines2, labels + labels2, loc=0)
    ax.legend(lines + lines2, labels + labels2, loc=2, framealpha=1.0)
    
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
    ax2.set_ylabel('Intercept')
    
    
def reg_graph_single(df, v1, lab, padding):    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 10))

    df[v1].plot(kind='bar', color=colors[1], width=0.5, 
                            position=0, align='center', label='Sharpe Ratio', grid=True)
    plt.xticks()

    #ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1, 1)

    #lines, labels = ax.get_legend_handles_labels()
    #ax.legend(lines, labels, loc=0)
    
    #ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)

    #If text is supposed to appear above graph:
    #ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    
    #If text is supposed to appear below graph:
    ax.tick_params(axis='x', pad=padding)
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    ax.set_xlabel('')
    #ax.set_ylabel('Sharpe Ratio')
    
    fig.text(0.5, 0.01, lab, ha='center')
    

#%% All regression coefficients from same regression (Low state effects vs. high state difference effect)

#rdatan = rdatadmn.copy(deep=True)

mac_vars = ['ShockPCA', 'ShockPCA_D']

for reg in regions:
    
    rd = {}
    tv = {}  
    
    if reg == 'All':
        factors = ['MKT_Global', 'SMB_Dev', 'HML_Dev', 'RMW_Dev', 'CMA_Dev', 'WML_Dev']
    elif reg == 'NAM':
        factors = ['Mkt-RF', 'SMB_N', 'HML_N', 'RMW', 'CMA', 'WML']
    elif reg == 'EUR':
        factors = ['Mkt-RF_EUR', 'SMB_EUR', 'HML_EUR', 'RMW_EUR', 'CMA_EUR', 'WML_EUR']
    elif reg == 'APA':
        factors = ['Mkt-RF_APA', 'SMB_APA', 'HML_APA', 'RMW_APA', 'CMA_APA', 'WML_APA']
    elif reg == 'JAP':
        factors = ['Mkt-RF_JAP', 'SMB_JAP', 'HML_JAP', 'RMW_JAP', 'CMA_JAP', 'WML_JAP']
    elif reg == 'EM':
        factors = ['Mkt-RF_EM', 'SMB_EM', 'HML_EM', 'RMW_EM', 'CMA_EM', 'WML_EM']

    #for i in range(len(mac_vars_d)):
    for i in range(0,len(mac_vars)):
        print(i)
        
        controlv=['qu'+str(i)]
        #controlv.extend(factors)
        
        #rdatan['qu'+str(i)] = pd.qcut(rdatan[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
        #regt,reg,regres = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)
        #regt_ls,reg_ls,regt,reg = ts_reg_det(rdatan, reg_vars_new, top=5, controls=['qu'+str(i)], nw=6)

        rdatan['qu'+str(i)] = pd.qcut(rdatan[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
        #rdatan['qu'+str(i)] = abs(rdatan['qu'+str(i)]-1)
        rdatan['qu_n'+str(i)] = ((rdatan[mac_vars[i]]-rdatan[mac_vars[i]].min())/(
            rdatan[mac_vars[i]].max()-rdatan[mac_vars[i]].min()))  

        rdatar = rdatan.copy(deep=True)
        reg_vars_n = []
        mac = mac_vars[i]
        
        for v in reg_vars_new:
            rvar = list(rdatar.loc[:,(rdatar.columns.str.contains(v+'_r'))|(rdatar.columns.str.contains(v+'_l'))].columns) 
            rvar = [x for x in rvar if reg not in x]
            rdatar.drop(rvar, axis=1, inplace=True)
            
            #rdatar['delta'] = rdatar[v+'_ls_5_'+reg].diff()[1:]
            rdatar['delta'] = rdatar[v+'_ls_5_'+reg].shift()[1:]
            rdatar['gap'] = rdatar[v+'_ls_5_'+reg].isnull()&~rdatar['delta'].isnull()
            
            #print(rdatar[v+'_ls_5_'+reg].count())
            #if rdatar[v+'_ls_5_'+reg].count()>1:
            #if rdatar[v+'_ls_5_'+reg].count()>72:
            #if rdatar[v+'_ls_5_'+reg].count()>48:
            if (rdatar[v+'_ls_5_'+reg].count()>60)&(rdatar[rdatar['gap']==True]['gap'].count()==0):
                #rdatar.drop(v+'_ls_5_'+reg, axis=1, inplace=True)
                reg_vars_n.extend([v])  
                
        #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatar, reg_vars_n, top=5, controls=['qu'+str(i)], nw=6)
        #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatar, reg_vars_n, top=5, controls=controlv, nw=6)
        #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_int(rdatar, reg_vars_n, top=5, cont_int=mac, reg=reg, nw=6, cut=True)
        regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_int(rdatar, reg_vars_n, top=5, cont_int=mac, reg=reg, nw=6)
        
        print('Done')
        
        rd[i] = pd.DataFrame(regt_ls)
        rd[i].columns = reg_vars_n
        
        tv[i] = pd.DataFrame(tvl_ls)
        tv[i].columns = reg_vars_n
        tv[i].index = ['t_const', 't_qu'+str(i)]

        #rd[i].append(tv[i])
        #rdn[i].append(tvn[i])
        
        rd[i] = rd[i].transpose()
        tv[i] = tv[i].transpose() 
        
        rd[i] = pd.merge(rd[i], tv[i], left_index=True, right_index=True)
   
    ## Plot results for each shock

    plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=11)    # fontsize of the tick labels

    #Generate plots:

    #for i in range(len(mac_vars_d)):
    for i in range(0,len(mac_vars)):
        
        rd[i].sort_values('t_const', inplace=True)
        rd[i].index = [x[:-4] for x in rd[i].index.values.tolist()]
        #rd[i].index = [x.replace(' Imp', '') for x in rd[i].index.values.tolist()]
        #rd[i].index = [x.replace(' Imp_dm', '') for x in rd[i].index.values.tolist()]
        rd_top = pd.concat([rd[i].head(5), rd[i].tail(5)])
            
        #reg_graph(rd_top, 'const', 't_const', mac_vars[i]+' Low '+reg)  
        reg_graph1(rd_top, 'const', 't_const', lab=None)  
        plt.savefig(res_out+'\\Criterion Country Level\\LS_Criteria_Imp_'+mac_vars[i]+reg+'_L.pdf',
                    bbox_inches='tight')
        #plt.savefig(res_out+'\\Criterion Country Level\\LS_Criteria_Imp_DM_'+mac_vars[i]+reg+'_L.pdf',
        #            bbox_inches='tight')
        #plt.savefig(res_out+'\\LS_Criteria_Imp_'+mac_vars[i]+'_alpha_L.png')
        #plt.savefig(res_out+'\\LS_Criteria_Imp_D_'+mac_vars[i]+'_L.png')
        
        rd[i].sort_values('t_qu'+str(i), inplace=True)
        #rd[i].index = [x.replace(' Imp', '') for x in rd[i].index.values.tolist()]
        #rd[i].index = [x.replace(' Imp_dm', '') for x in rd[i].index.values.tolist()]
        rd_top = pd.concat([rd[i].head(5), rd[i].tail(5)])
       
        #reg_graph(rd_top, 'qu'+str(i), 't_qu'+str(i), mac_vars[i]+' High '+reg)   
        #reg_graph(rd_top, 'qu', 't_qu'+str(i), mac_vars[i]+' High '+reg)    
        reg_graph(rd_top, 'qu', 't_qu'+str(i), lab=None)    
        plt.savefig(res_out+'\\Criterion Country Level\\LS_Criteria_Imp_'+mac_vars[i]+reg+'_H.pdf',
                    bbox_inches='tight')
        #plt.savefig(res_out+'\\Criterion Country Level\\LS_Criteria_Imp_DM_'+mac_vars[i]+reg+'_H.pdf',
        #            bbox_inches='tight')
        #plt.savefig(res_out+'\\LS_Criteria_Imp_'+mac_vars[i]+'_alpha_H.png')
        #plt.savefig(res_out+'\\LS_Criteria_Imp_D_'+mac_vars[i]+'_H.png')
        
    
#%% Full sample coefficients vs. adverse condition coefficients (Full sample vs. high state difference effect) 

for reg in regions:
    
    rd = {}
    tv = {}
    
    if reg == 'All':
        factors = ['MKT_Global', 'SMB_Dev', 'HML_Dev', 'RMW_Dev', 'CMA_Dev', 'WML_Dev']
    elif reg == 'NAM':
        factors = ['Mkt-RF', 'SMB_N', 'HML_N', 'RMW', 'CMA', 'WML']
    elif reg == 'EUR':
        factors = ['Mkt-RF_EUR', 'SMB_EUR', 'HML_EUR', 'RMW_EUR', 'CMA_EUR', 'WML_EUR']
    elif reg == 'APA':
        factors = ['Mkt-RF_APA', 'SMB_APA', 'HML_APA', 'RMW_APA', 'CMA_APA', 'WML_APA']
    elif reg == 'JAP':
        factors = ['Mkt-RF_JAP', 'SMB_JAP', 'HML_JAP', 'RMW_JAP', 'CMA_JAP', 'WML_JAP']
    elif reg == 'EM':
        factors = ['Mkt-RF_EM', 'SMB_EM', 'HML_EM', 'RMW_EM', 'CMA_EM', 'WML_EM']

    #for i in range(0,len(mac_vars_d)):    
    for i in range(0,len(mac_vars)):
        print(i)
            
        controlv=['qu'+str(i)]
        controlv.extend(factors)
        
        rdatan['qu'+str(i)] = pd.qcut(rdatan[mac_vars[i]], q=[0, .75, 1], labels=False).astype(float)
        #rdatan['qu'+str(i)] = abs(rdatan['qu'+str(i)]-1)
        #rdatan['qu'+str(i)] = pd.qcut(rdatan[mac_vars_d[i]], q=[0, .75, 1], labels=False).astype(float)
        
        rdatar = rdatan.copy(deep=True)
        reg_vars_n = []
        mac = mac_vars[i]
        
        for v in reg_vars_new:
            rvar = list(rdatar.loc[:,(rdatar.columns.str.contains(v+'_r'))|(rdatar.columns.str.contains(v+'_l'))].columns) 
            rvar = [x for x in rvar if reg not in x]
            rdatar.drop(rvar, axis=1, inplace=True)
            
            #rdatar['delta'] = rdatar[v+'_ls_5_'+reg].diff()[1:]
            rdatar['delta'] = rdatar[v+'_ls_5_'+reg].shift()[1:]
            rdatar['gap'] = rdatar[v+'_ls_5_'+reg].isnull()&~rdatar['delta'].isnull()
            
            #print(rdatar[v+'_ls_5_'+reg].count())
            #if rdatar[v+'_ls_5_'+reg].count()>1:
            #if rdatar[v+'_ls_5_'+reg].count()>72:
            #if rdatar[v+'_ls_5_'+reg].count()>48:
            #if (rdatar[v+'_ls_5_'+reg].count()>48)&(rdatar[rdatar['gap']==True]['gap'].count()==0):
            if (rdatar[v+'_ls_5_'+reg].count()>60)&(rdatar[rdatar['gap']==True]['gap'].count()==0):
                #rdatar.drop(v+'_ls_5_'+reg, axis=1, inplace=True)
                reg_vars_n.extend([v])
        
        regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatar, reg_vars_n, top=5, nw=6)
        #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatan, reg_vars_new, top=5, controls=factorsa, nw=6)
        
        rd[i] = pd.DataFrame(regt_ls)
        rd[i].columns = reg_vars_n
        
        tv[i] = pd.DataFrame(tvl_ls)
        tv[i].columns = reg_vars_n
        
        tv[i].index = ['t_const']
        #tv[i].index = ['t_const', 't_1', 't_2', 't_3', 't_4', 't_5']
        #tvn[i].index = ['t_const', 't_1', 't_2', 't_3', 't_4', 't_5']
        
        rd[i] = rd[i].transpose()
        tv[i] = tv[i].transpose()
        
        rd[i] = pd.merge(rd[i], tv[i], left_index=True, right_index=True)
        
        #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatar, reg_vars_n, top=5, controls=['qu'+str(i)], nw=6)
        #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatar, reg_vars_n, top=5, controls=controlv, nw=6)
        #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_int(rdatar, reg_vars_n, top=5, cont_int=mac, reg=reg, nw=6, cut=True)
        regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_int(rdatar, reg_vars_n, top=5, cont_int=mac, reg=reg, nw=6)
        
        regt_ls = pd.DataFrame(regt_ls)
        regt_ls.columns = reg_vars_n

        tvl_ls = pd.DataFrame(tvl_ls)
        tvl_ls.columns = reg_vars_n
        
        tvl_ls.index = ['t_const', 't_qu'+str(i)]
        #tvl_ls.index = ['t_const', 't_qu'+str(i), 't_1', 't_2', 't_3', 't_4', 't_5']
        #tvld_ls.index = ['t_const', 't_qu'+str(i), 't_1', 't_2', 't_3', 't_4', 't_5']
        
        regt_ls = regt_ls.transpose()
        tvl_ls = tvl_ls.transpose() 
        
        rd[i] = pd.merge(rd[i], regt_ls['qu'], left_index=True, right_index=True)
        #rd[i] = pd.merge(rd[i], regt_ls['qu'+str(i)], left_index=True, right_index=True)
        rd[i] = pd.merge(rd[i], tvl_ls['t_qu'+str(i)], left_index=True, right_index=True)

    ## Plot results for each shock

    #Generate plots:
        
    rd[0].sort_values('t_const', inplace=True)
    #rd[0].index = [x.replace(' Imp', '') for x in rd[0].index.values.tolist()]
    rd[0].index = [x[:-4] for x in rd[i].index.values.tolist()]
    #rd[0].index = [x.replace(' Imp_dm', '') for x in rd[0].index.values.tolist()]
    rd_top = pd.concat([rd[0].head(5), rd[0].tail(5)])
        
    reg_graph1(rd_top, 'const', 't_const', lab=None)    
    plt.savefig(res_out+'\\Criterion Country Level\\LS_Criteria_Imp '+reg+'.pdf',
                bbox_inches='tight')
    #plt.savefig(res_out+'\\Criterion Country Level\\LS_Criteria_Imp_DM '+reg+'.pdf',
    #            bbox_inches='tight')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_alpha.png')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_D.png')

    #for i in range(len(mac_vars_d)):
    for i in range(0,len(mac_vars)):

        rd[i].sort_values('t_qu'+str(i), inplace=True)
        if i > 0:
            rd[i].index = [x[:-4] for x in rd[i].index.values.tolist()]
        #rd[i].index = [x.replace(' Imp', '') for x in rd[i].index.values.tolist()]
        #rd[i].index = [x.replace(' Imp_dm', '') for x in rd[i].index.values.tolist()]
        rd_top = pd.concat([rd[i].head(5), rd[i].tail(5)])
        
        #reg_graph(rd_top, 'qu', 't_qu'+str(i), mac_vars[i]+' High '+reg)   
        reg_graph(rd_top, 'qu', 't_qu'+str(i), lab=None)   
        #reg_graph(rd_top, 'qu'+str(i), 't_qu'+str(i), mac_vars[i]+' High '+reg)    
        plt.savefig(res_out+'\\Criterion Country Level\\LS_Criteria_Imp_'+mac_vars[i]+reg+'_H.pdf',
                    bbox_inches='tight')
        #plt.savefig(res_out+'\\Criterion Country Level\\LS_Criteria_Imp_DM_'+mac_vars[i]+reg+'_H.pdf',
        #            bbox_inches='tight')
        #plt.savefig(res_out+'\\LS_Criteria_Imp_'+mac_vars[i]+'_alpha_H.png')
        #plt.savefig(res_out+'\\LS_Criteria_Imp_D_'+mac_vars[i]+'_H.png')
      
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
        
    ## Regression coefficients and plotting full sample    
        
    rd[0].sort_values('t_const', inplace=True)
    #rd[0].index = [x.replace(' Imp', '') for x in rd[0].index.values.tolist()]
    reg_graph_long(rd[0], 'const', 't_const')   
    plt.savefig(res_out+'\\LS_Criteria_Imp_Full '+reg+'.pdf', bbox_inches='tight')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_Full_DM '+reg+'.pdf')
    #plt.savefig(res_out+'\\LS_Criteria_Imp_alpha_All.png')
    
    ## Overview over results by criterion/dimension:
    
    cscoresd = cscores.set_index('CRITERION')
    
    rd_all = rd[0].copy(deep=True)    
    #rd_all.index = [x[:-4] for x in list(rd_all.index)]
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
        #rd_st.index = [x[:-4] for x in list(rd_st.index)]
        rd_st['S'+str(i)] = rd_st['t_qu'+str(i)]>1.645
        rd_st['Sl'+str(i)] = rd_st['t_qu'+str(i)]>1.282
        #rd_st['Sln'+str(i)] = rd_st['qu'+str(i)]>0
        rd_st['Sln'+str(i)] = rd_st['qu']>0
        rd_all = pd.merge(rd_all, rd_st[['S'+str(i), 'Sl'+str(i), 'Sln'+str(i)]], left_index=True, right_index=True)

    rd_all['St'] = rd_all[['S']].sum(axis=1)
    rd_all['Stl'] = rd_all[['Sln']].sum(axis=1)
    rd_all['Bt'] = rd_all[['B']].sum(axis=1)
    rd_all['Btl'] = rd_all[['Bln']].sum(axis=1)
    #rd_all['Sc'] = rd_all[['S0', 'S1', 'S2', 'S3', 'S4', 'S5']].sum(axis=1)
    #rd_all['Scl'] = rd_all[['Sl0', 'Sl1', 'Sl2', 'Sl3', 'Sl4', 'Sl5']].sum(axis=1)
    #rd_all['Scln'] = rd_all[['Sln0', 'Sln1', 'Sln2', 'Sln3', 'Sln4', 'Sln5']].sum(axis=1)
    rd_all['Sc'] = rd_all[['S0', 'S1']].sum(axis=1)
    rd_all['Scl'] = rd_all[['Sl0', 'Sl1']].sum(axis=1)
    rd_all['Scln'] = rd_all[['Sln0', 'Sln1']].sum(axis=1)
    
    rd_all.to_csv(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_Comparison_'+reg+'.csv')
    #rd_all.to_csv(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_DM_Comparison_'+reg+'.csv')

    #rd_dim = rd_all.groupby('DIMENSION')['Sln', 'S0',
    #       'Sl0', 'Sln0', 'S1', 'Sl1', 'Sln1', 'S2', 'Sl2', 'Sln2', 'S3', 'Sl3',
    #       'Sln3', 'S4', 'Sl4', 'Sln4', 'S5', 'Sl5', 'Sln5'].sum()
    #rd_dim['Sm'] = rd_dim[['Sl0', 'Sl1', 'Sl2', 'Sl3', 'Sl4', 'Sl5']].mean(axis=1)
    #rd_dim['Smn'] = rd_dim[['Sln0', 'Sln1', 'Sln2', 'Sln3', 'Sln4', 'Sln5']].mean(axis=1)
    #rd_dim['Bm'] = rd_dim[['Bl0', 'Bl1', 'Bl2', 'Bl3', 'Bl4', 'Bl5']].mean(axis=1)
    #rd_dim['Bmn'] = rd_dim[['Bln0', 'Bln1', 'Bln2', 'Bln3', 'Bln4', 'Bln5']].mean(axis=1)
    rd_dim = rd_all.groupby('DIMENSION')['Sln', 'S0','Sl0', 'Sln0', 'S1', 'Sl1', 'Sln1'].sum()
    rd_dim['Sm'] = rd_dim[['Sl0', 'Sl1']].mean(axis=1)
    rd_dim['Smn'] = rd_dim[['Sln0', 'Sln1']].mean(axis=1)
    
    rd_dim.to_csv(res_out+'\\Criterion Country Level\\SR_Criteria_DIM_Imp_Comparison_'+reg+'.csv')
    #rd_dim.to_csv(res_out+'\\Criterion Country Level\\SR_Criteria_DIM_Imp_DM_Comparison_'+reg+'.csv')
  

#%% Sharpe ratios and factor alphas

for reg in regions:
    
    rdatar = rdatan.copy(deep=True)
    reg_vars_n = []
    reg_vars_ex = reg_vars_new.copy()
    
    for v in reg_vars_new:
        rvar = list(rdatar.loc[:,(rdatar.columns.str.contains(v+'_r'))|(rdatar.columns.str.contains(v+'_l'))].columns) 
        rvar = [x for x in rvar if reg not in x]
        rdatar.drop(rvar, axis=1, inplace=True)
        
        #rdatar['delta'] = rdatar[v+'_ls_5_'+reg].diff()[1:]
        rdatar['delta'] = rdatar[v+'_ls_5_'+reg].shift()[1:]
        rdatar['gap'] = rdatar[v+'_ls_5_'+reg].isnull()&~rdatar['delta'].isnull()
        
        #print(rdatar[v+'_ls_5_'+reg].count())
        #if rdatar[v+'_ls_5_'+reg].count()>1:
        #if rdatar[v+'_ls_5_'+reg].count()>72:
        #if rdatar[v+'_ls_5_'+reg].count()>48:
        #if (rdatar[v+'_ls_5_'+reg].count()>48)&(rdatar[rdatar['gap']==True]['gap'].count()==0):
        if (rdatar[v+'_ls_5_'+reg].count()>60)&(rdatar[rdatar['gap']==True]['gap'].count()==0):
            #rdatar.drop(v+'_ls_5_'+reg, axis=1, inplace=True)
            reg_vars_n.extend([v])
            reg_vars_ex.remove(v)
            
    if reg == 'All':
        #reg_vars_list = pd.Series([x.replace(' Imp', '') for x in reg_vars_n], name='CRITERION')
        reg_vars_list = pd.Series([x[:-4] for x in reg_vars_n], name='CRITERION')
        cscoresn = cscores.merge(reg_vars_list, on='CRITERION')
        cscoresn.to_excel(res_out+'\\Criteria+Dimensions_ShortList.xlsx')
      
    reg_vars_ex = [x+'_ls_5_'+reg for x in reg_vars_ex]        
    rdatar.drop(reg_vars_ex, axis=1, inplace=True)
        
    #Sharpe ratio

    rd_sr = (rdatar.mean()/rdatar.std())*np.sqrt(12)
    sr_var = [x for x in list(rd_sr.index) if '_ls_' in x]
    rd_sr = pd.DataFrame(rd_sr.loc[sr_var].sort_values()).rename(columns={0:'value'})
    rd_sr.index = [x.replace(' Imp_ls_5_'+reg, '') for x in rd_sr.index.values.tolist()]
    #rd_sr.index = [x.replace(' Imp_dm_ls_5_'+reg, '') for x in rd_sr.index.values.tolist()]

    plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=11)    # fontsize of the tick labels

    reg_graph_single(rd_sr, 'value', lab=None, padding=220)
    plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_'+reg+'.pdf',
                bbox_inches='tight')
    #plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_DM_'+reg+'.pdf',
    #            bbox_inches='tight')
    #plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_5_'+reg+'.pdf')
    
    rd_sri = pd.merge(rd_sr, csc, left_index=True, right_index=True, how='left')
    rd_sri = rd_sri.set_index('Abr').drop('DIMENSION', axis=1)

    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    
    reg_graph_single(rd_sri, 'value', lab=None, padding=325)
    plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_Lab_'+reg+'.pdf',
                bbox_inches='tight')
    #plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_DM_Lab_'+reg+'.pdf',
    #            bbox_inches='tight')
    
    #Sharpe ratio Governance
    
    #rd_srg = rd_sr[rd_sr.index.isin(gov)]
    
    #plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    
    #reg_graph_single_s(rd_srg, 'value', padding=190)  
    #plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Gov_Imp_'+reg+'.pdf')
    #plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Gov_Imp_5_'+reg+'.pdf')

#Alphas of time-series factors
     
for reg in regions:
    
    rd = {}
    tv = {}
    
    if reg == 'All':
        factors = ['MKT_Global', 'SMB_Dev', 'HML_Dev', 'RMW_Dev', 'CMA_Dev', 'WML_Dev']
    elif reg == 'NAM':
        factors = ['Mkt-RF', 'SMB_N', 'HML_N', 'RMW', 'CMA', 'WML']
    elif reg == 'EUR':
        factors = ['Mkt-RF_EUR', 'SMB_EUR', 'HML_EUR', 'RMW_EUR', 'CMA_EUR', 'WML_EUR']
    elif reg == 'APA':
        factors = ['Mkt-RF_APA', 'SMB_APA', 'HML_APA', 'RMW_APA', 'CMA_APA', 'WML_APA']
    elif reg == 'JAP':
        factors = ['Mkt-RF_JAP', 'SMB_JAP', 'HML_JAP', 'RMW_JAP', 'CMA_JAP', 'WML_JAP']
    elif reg == 'EM':
        factors = ['Mkt-RF_EM', 'SMB_EM', 'HML_EM', 'RMW_EM', 'CMA_EM', 'WML_EM']

    rdatar = rdatan.copy(deep=True)
    reg_vars_n = []

    for v in reg_vars_new:
        rvar = list(rdatar.loc[:,(rdatar.columns.str.contains(v+'_r'))|(rdatar.columns.str.contains(v+'_l'))].columns) 
        rvar = [x for x in rvar if reg not in x]
        rdatar.drop(rvar, axis=1, inplace=True)
        
        rdatar['delta'] = rdatar[v+'_ls_5_'+reg].diff()[1:]
        rdatar['gap'] = rdatar[v+'_ls_5_'+reg].isnull()&~rdatar['delta'].isnull()
        
        #if rdatar[v+'_ls_5_'+reg].count()>72:
        #if rdatar[v+'_ls_5_'+reg].count()>48:
        #if (rdatar[v+'_ls_5_'+reg].count()>48)&(rdatar[rdatar['gap']==True]['gap'].count()==0):
        if (rdatar[v+'_ls_5_'+reg].count()>60)&(rdatar[rdatar['gap']==True]['gap'].count()==0):
            #rdatar.drop(v+'_ls_5_'+reg, axis=1, inplace=True)
            reg_vars_n.extend([v])
        
    #regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatar, reg_vars_n, top=5, nw=6)
    regt_ls,tvl_ls,regt,tvl,regs = finfunc.ts_reg_det(rdatar, reg_vars_n, top=5, controls=factors, nw=6)
        
    rd = pd.DataFrame(regt_ls)
    rd.columns = reg_vars_n

    for v in reg_vars_n:
        rdatar[v+'_ls_5_'+reg+'_alpha'] = rdatar[v+'_ls_5_'+reg]-rdatar[factors[0]]*rd[v][factors[0]]
        for i in range(1,6):
            rdatar[v+'_ls_5_'+reg+'_alpha'] = rdatar[v+'_ls_5_'+reg]-rdatar[factors[i]]*rd[v][factors[i]]
            
    #Sharpe ratio

    rd_sr = (rdatar.mean()/rdatar.std())*np.sqrt(12)
    sr_var = [x for x in list(rd_sr.index) if '_alpha' in x]
    rd_sr = pd.DataFrame(rd_sr.loc[sr_var].sort_values()).rename(columns={0:'value'})
    rd_sr.index = [x.replace(' Imp_ls_5_'+reg+'_alpha', '') for x in rd_sr.index.values.tolist()]
    #rd_sr.index = [x.replace(' Imp_dm_ls_5_'+reg+'_alpha', '') for x in rd_sr.index.values.tolist()]

    plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=11)    # fontsize of the tick labels

    reg_graph_single(rd_sr, 'value', lab=None, padding=220)
    plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_Alpha_'+reg+'.pdf',
                bbox_inches='tight')
    #plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_Alpha_5_'+reg+'.pdf')
    #plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_DM_Alpha_'+reg+'.pdf',
    #            bbox_inches='tight')
    
    rd_sri = pd.merge(rd_sr, csc, left_index=True, right_index=True, how='left')
    rd_sri = rd_sri.set_index('Abr').drop('DIMENSION', axis=1)

    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    
    reg_graph_single(rd_sri, 'value', lab=None, padding=325)
    plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_Lab_Alpha_'+reg+'.pdf',
                bbox_inches='tight')
    #plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Imp_DM_Lab_Alpha_'+reg+'.pdf',
    #            bbox_inches='tight')
    
    #Sharpe ratio Governance
    
    #rd_srg = rd_sr[rd_sr.index.isin(gov)]
    
    #plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    
    #reg_graph_single_s(rd_srg, 'value', padding=190)  
    #plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Gov_Imp_Alpha_'+reg+'.pdf')
    #plt.savefig(res_out+'\\Criterion Country Level\\SR_Criteria_Gov_Imp_Alpha_5_'+reg+'.pdf')
        
    #tv = pd.DataFrame(tvl_ls)
    #tv.columns = reg_vars_n
    
    
    
    
   