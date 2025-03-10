

##Packages

import pandas as pd
import numpy as np
import datetime as dt
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import dli

##Set directories:
    
input_econ = "<Directory>\\Data\\Economic Data\\"
input = "<Directory>\\Data"    

##Load economic data:

#Economic policy uncertainty (Baker et al. 2016):
url="https://policyuncertainty.com/media/All_Country_Data.xlsx"
epu = pd.read_excel(url, skipfooter=29)

#Monetary policy uncertainty (Baker et al. 2016):
url="https://www.policyuncertainty.com/media/US_MPU_Monthly.xlsx"
mpu = pd.read_excel(url, skiprows=range(0), skipfooter=1)

#Climate policy uncertainty (Gavriilidis 2021):
#url="https://www.policyuncertainty.com/media/CPU%20index.csv"
#cpu = pd.read_csv(url, skiprows=range(2))
cpu = pd.read_csv(input_econ+"Climate_Policy_Uncertainty.csv", skiprows=range(2))  

#Geopolitical risk index:
url="https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
gr = pd.read_excel(url)

#Macroeconomic and financial uncertainty (Jurado et al. 2015 and Ludvigson et al. 2021):  
#url="https://www.sydneyludvigson.com/s/MacroFinanceUncertainty_202202Update.zip"
url="https://www.sydneyludvigson.com/s/MacroFinanceUncertainty_202208Update.zip"
resp = urlopen(url)
zipfile = ZipFile(BytesIO(resp.read()))
fintu = pd.read_excel(zipfile.open(zipfile.namelist()[0]), sheet_name='Total Financial Uncertainty')
fineu = pd.read_excel(zipfile.open(zipfile.namelist()[0]), sheet_name='Economic Financial Uncertainty')
mactu = pd.read_excel(zipfile.open(zipfile.namelist()[1]), sheet_name='Total Macro Uncertainty')
maceu = pd.read_excel(zipfile.open(zipfile.namelist()[1]), sheet_name='Economic Macro Uncertainty')
realtu = pd.read_excel(zipfile.open(zipfile.namelist()[2]), sheet_name='Total Real Uncertainty')
realeu = pd.read_excel(zipfile.open(zipfile.namelist()[2]), sheet_name='Economic Real Uncertainty')

#Twitter economic uncertainty (Baker et al. 2021): 
tweu = pd.read_excel(input_econ+"Twitter_Economic_Uncertainty.xlsx")     
    
#Monetary policy uncertainty (US):
mpu_us = pd.read_excel(input_econ+"HRS_MPU_monthly.xlsx") 
    
#Merge data to one monthly dataset:

cpu['date'] = pd.to_datetime(cpu['date'], format='%b-%y')
cpu['month_id'] = pd.to_datetime(cpu['date']).dt.strftime('%Y-%m')

econ = cpu.copy(deep=True)
econ = econ[['month_id', 'cpu_index']]

mpu['month_id'] = pd.to_datetime(mpu[['Year', 'Month']].assign(DAY=1)).dt.strftime('%Y-%m') 
econ = pd.merge(econ, mpu.iloc[:,2:], on='month_id', how='outer')    
econ.sort_values(['month_id'], inplace=True)

mpu_us['Month'] = mpu_us['Month'].str.replace('m',' ')
mpu_us['month_id'] = pd.to_datetime(mpu_us['Month'], format='%Y %m').dt.strftime('%Y-%m')
econ = pd.merge(econ, mpu_us[['month_id', 'US MPU']], on='month_id', how='outer')    
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


##Load other financial/economic data:

#Volatility index (VIX):
    
url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
vix = pd.read_csv(url)

#IHS purchasing manager index: 

#Logon and Authenticate to the Data Lake
dl = dli.connect()

#Select a dataset by shortcode
dataset  = dl.datasets.get(organisation_short_code="IHSMarkit", short_code="PurchasingManagersIndexPMI")

# Extract to Pandas DataFrame
pmi = dataset.dataframe(nrows=50000)

pmi['st'] = pmi['short_label'].str.replace('PMI By S\&P Global, ','')
pmi['sts'] = pmi['st'].str[:12]
p = pmi['sts'].unique()

# Getting the SOURCE ID from the PMI METADATA

# Loading Dataset
meta_ds = dl.datasets.get(short_code="PurchasingManagersIndexPMIMetadata")
ts_ds = dl.datasets.get(short_code="PurchasingManagersIndexPMITimeseries")
print("**** INFORMATION ON THE DATA-SET ****")
print(meta_ds.description)
print("**** INFORMATION ON THE DATA-SET PARTITIONS ****")
print(meta_ds.partitions())

# Reading the Meta Data as a Dataframe
meta_df = meta_ds.dataframe(partitions=[f"as_of_date={meta_ds.last_datafile_at}"])
meta_df.sort_values('short_label', inplace=True)

# Check unique values of key columns and get the right meta information
regions=['Asia', 'Asia Excluding China, Japan', 'Asia Excluding China', 'Asia Excluding Japan',
         'China (mainland)', 'Developed Countries', 'Emerging Markets', 'Europe', 'European Union',
         'Eurozone', 'World', 'Japan', 'North America', 'United States']
meta_df = meta_df[(meta_df['economic_concept_name'].isin(['PMI', 'Headline'])) 
        & (meta_df['source_geographic_location_name'].isin(regions))
        & (meta_df['short_label'].str.contains('Composite')|meta_df['short_label'].str.contains('Manufacturing'))
        & (meta_df['series_type_name'] == 'Historical')]
        #].short_label.unique().tolist()
 
# Reading the Time Series as a Dataframe
ts_df = ts_ds.dataframe(partitions=[f"as_of_date={ts_ds.last_datafile_at}"])

# Merge Timeseries with Metadata to get the required series 

df_PMI = pd.merge(ts_df, meta_df[['source_id', 'mnemonic', 'last_update', 
                                  'long_label', 'short_label', 
                                  'document_type', 'economic_concept_name',
                                  'source_geographic_location_name']], on="source_id")
#df_PMI_ts_market["YearMonth"] = df_PMI_ts_market["date"].dt.strftime("%Y%m")

#Add data to monthly dataset:
    
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
  

##Load climate-related data:

#Media climate change concern index:
mccc = pd.read_excel(input_econ+"Sentometrics_US_Media_Climate_Change_Index.xlsx", 
                     sheet_name='SSRN 2022 version (monthly)', skiprows=range(5)) 

#Climate change news index:
ccn = mccc = pd.read_excel(input_econ+"EGLKS_data.xlsx") 

    
##Correlations between economic shock variables    
    
#ec = np.corrcoef(econ.loc[:,'cpu_index':'VIX_mean'])   
ec = econ.loc[:,'cpu_index':'VIX_mean'].corr()


###############################################################################


##Load stock market data 

ret = pd.read_csv(input+'\\Financials\\returns_monthly.csv', low_memory=False)
ind = pd.read_csv(input+'\\Financials\\industries_monthly.csv', low_memory=False)

#ret['month_id'] = pd.to_datetime(ret['Date']).dt.strftime('%Y-%m') 
#ind['month_id'] = pd.to_datetime(ind['Date']).dt.strftime('%Y-%m') 

ret = pd.merge(ret, ind, on=['issue_id', 'month_id', 'company_name',
                             'capital_iq', 'snlinstitutionid'], suffixes=['', '_ind'])
try:
    ret = ret.drop(['Unnamed: 0'], axis = 1)
except:
    pass

ret = ret.rename(columns={'capital_iq':'CIQ_ID'})

for m in ['market_cap', 'market_cap_usd']:
    ret.groupby('CIQ_ID')[m].shift()
    ret[m] = ret[m]/1000

del ind

#Add market level data

mret = pd.read_csv(input+'\\Financials\\returns_market_monthly.csv', low_memory=False)
mret = mret.rename(columns={'country':'icountry'})

#mret = mret[pd.to_datetime(mret['Date']) > mindate]
#mret = mret[pd.to_datetime(mret['Date']) < maxdate]

#Duplicates in financial data: Drop if no price available or if not issue #1

ret['dup'] = ret.duplicated(subset=['CIQ_ID', 'month_id'], keep=False)   
ret = ret.drop(ret[(ret['dup']==True) & (ret['prccd'].isnull())].index)
ret = ret.drop(['dup'], axis=1)

ret.sort_values(['CIQ_ID', 'issue_id', 'month_id'], inplace=True)
ret = ret.groupby(['CIQ_ID', 'month_id']).first().reset_index()


##Merge macroeconomic and stock market data

ret = pd.merge(ret, econ, on='month_id', how='left')
ret['month_id'] = pd.to_datetime(ret['month_id']).dt.to_period("M")
ret.sort_values(['CIQ_ID', 'month_id'], inplace=True)

