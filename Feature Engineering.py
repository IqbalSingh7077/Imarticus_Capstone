import os
os.chdir('/home/ryuzaki/E drive/Dekstop/projects/Imarticus')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# importing the dataset
df = pd.read_csv('df1.csv')

##************ Data Pre Processing****************##
##------------------------------------------------##

#1. Treating missing values

df.isnull().sum()

col = df.columns.to_list()
del col[11:44]
col
for i in col:
    print(df[i].unique())

'''
we have missing values in Employee Position and Supervisors name, we will replace missing values in Employee Position
by the mode category and replace missing values in Supervisor name as No Supervisor!
'''

df['Employee Position'].replace('-', df['Employee Position'].mode()[0], inplace = True)
df['Supervisor name'].replace('-', 'No Supervisor', inplace = True)
    

df['Employee Position'].unique()# No missing values
df['Supervisor name'].unique()# No missing values


# treating missing values in yearly totals
df['Utilization%'].isnull().sum()# 4 missing values
'''
all of these 4 missing values are of new joiners so we will replace these missing values by 0
'''
df['Utilization%'].replace(np.nan,0,inplace = True)
df.isnull().sum()# No missing values



'''
# Remove Outliers 
df1 = df.iloc[:,11:44]
col = list(df1).columns
for i in col:
    y = df[i]
    removed_outliers = y.between(y.quantile(.02), y.quantile(.98))
    index_names = df[~removed_outliers].index 
    df.drop(index_names, inplace=True)
'''



# Creating a new feature named Tenure 
'''
Tenure will tell for how many days a person was employed with the firm
'''

df['Join Date']=pd.to_datetime(df['Join Date'])

df['Join Date']
df['tenure'] =(pd.to_datetime('2018-03-31') - df['Join Date'])

df.dtypes
  
df['tenure']
'''
0      6391 days
1      6879 days
2      6268 days
3      6391 days
4      7214 days
  
1106    335 days
1107    243 days
1108    151 days
1109     90 days
1110      0 days
'''

df['tenure'] = df['tenure'].astype('timedelta64[D]')

df['tenure'] = df['tenure']/365

df['tenure']
'''
0       17.509589
1       18.846575
2       17.172603
3       17.509589
4       19.764384
   
1106     0.917808
1107     0.665753
1108     0.413699
1109     0.246575
1110     0.000000

'''                  
# T-Test
CS_A= df[df['Current Status'] == 'Active']
CS_R= df[df['Current Status']=='Resigned']
import scipy
scipy.stats.ttest_ind(CS_A['tenure'], CS_R['tenure'] )
'''
Ttest_indResult(statistic=-4.010405815616428, pvalue=6.467268187891957e-05)

p-val < 0.05, tenure can be a good predictor!
'''

## Dropping Unecessary Columns

df.columns

df.drop(columns = ['Employee No','Employee Name', 'People Group', 'Employee Category',
                   'Join Date','Termination Date','Jul-16 Utilization%',
                   'Aug-16 Utilization%','Sep-16 Utilization%','Oct-16 Utilization%','Nov-16 Utilization%',
                   'Dec-16 Utilization%','Jan-17 Utilization%','Feb-17 Utilization%','Employee Location', 'Total Hours'
                   ,'BD Hours','NC Hours','Utilization%'], inplace = True)
## Label Encoding

from sklearn import preprocessing
LE = preprocessing.LabelEncoder()

df['Supervisor name'] = LE.fit_transform(df['Supervisor name'])
df['Employee Position'] = LE.fit_transform(df['Employee Position'])
df['Current Status'] = df['Current Status'].replace("Active",0)
df['Current Status'] = df['Current Status'].replace("Resigned",1)


## Dummy variables

ML_df = pd.get_dummies(df, columns = ['Profit Center'])

ML_df.to_csv('ML_df.csv')

