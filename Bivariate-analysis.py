
import os
os.chdir('/home/ryuzaki/E drive/Dekstop/projects/Imarticus')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# importing the dataset

df = pd.read_excel('merged.xlsx')

# Structure of the dataset
df.shape #(1111, 43)
pd.set_option('display.max_columns',None)
df.head(5)
df.tail(5)
'''
rows with no values (missing) are given as '-'
'''
df.info()
'''
 #   Column                 Non-Null Count  Dtype         
---  ------                 --------------  -----         
 0   Employee No            1111 non-null   int64         
 1   Profit Center          1111 non-null   object        
 2   Employee Name          1111 non-null   object        
 3   Employee Position      1111 non-null   object        
 4   Employee Location      1111 non-null   object        
 5   People Group           1111 non-null   object        
 6   Employee Category      1111 non-null   object        
 7   Supervisor name        1111 non-null   object        
 8   Join Date              1111 non-null   datetime64[ns]
 9   Current Status         1111 non-null   object        
 10  Termination Date       283 non-null    object        
 11  Apr-16 Utilization%    1111 non-null   object        
 12  May-16 Utilization%    1111 non-null   object        
 13  June-16 Utilization%   1111 non-null   object        
 14  Jul-16 Utilization%    1111 non-null   object        
 15  Aug-16 Utilization%    1111 non-null   object        
 16  Sep-16 Utilization%    1111 non-null   object        
 17  Oct-16 Utilization%    1111 non-null   object        
 18  Nov-16 Utilization%    1111 non-null   object        
 19  Dec-16 Utilization%    1111 non-null   object        
 20  Jan-17 Utilization%    1111 non-null   object        
 21  Feb-17 Utilization%    1111 non-null   object        
 22  Mar-17 Utilization%    1111 non-null   object        
 23  Apr-17 Utilization%    1111 non-null   object        
 24  May-17 Utilization%    1111 non-null   object        
 25  June-17 Utilization%   1111 non-null   object        
 26  Jul-17 Utilization%    1111 non-null   object        
 27  Aug-17 Utilization%    1111 non-null   object        
 28  Sep-17 Utilization%    1111 non-null   object        
 29  Oct-17 Utilization%    1111 non-null   object        
 30  Nov-17 Utilization%    1111 non-null   object        
 31  Dec-17 Utilization%    1111 non-null   object        
 32  Jan-18 Utilization%    1111 non-null   object        
 33  Feb-18 Utilization%    1111 non-null   object        
 34  Mar-18 Utilization%    1111 non-null   object        
 35  Total Hours            1111 non-null   int64         
 36  Total Available Hours  1111 non-null   float64       
 37  Work Hours             1111 non-null   float64       
 38  Leave Hours            1111 non-null   float64       
 39  Training Hours         1111 non-null   float64       
 40  BD Hours               1111 non-null   float64       
 41  NC Hours               1111 non-null   float64       
 42  Utilization%           1107 non-null   float64       
dtypes: datetime64[ns](1), float64(7), int64(2), object(33)

datatype of month wise utilization needs to be converted to float
'''

df.isnull().sum()
'''
4 missing values in the Total Utilization%, will be taken care of during Feature Engineering
'''

df.describe()




#######****** EDA *******########
#-------------------------------#
####### Bi-variate Analysis ########

# Current Status (Dependent Variable)

df['Current Status'].value_counts()

plt.figure(figsize=(14,10))
sns.countplot(df['Current Status'])
plt.title('Countplot of Current Status')
plt.legend(loc = 'upper left')

'''
Current Status has total 4 categories, we will convert it to 2, Acitve & Resigned
'''

df['Current Status'].replace('Secondment','Active', inplace =True)
df['Current Status'].replace('New Joiner','Active', inplace =True)
df['Current Status'].replace('Sabbatical','Active', inplace =True)

df['Current Status'].value_counts()# replaced

# 1. Profit Center Vs Current Status

plt.figure(figsize=(12,8))
sns.countplot(df['Profit Center'], hue = df['Current Status'])
plt.title('Status of emplyess in different Proft Centers')
plt.legend(loc = 'upper right')
 
cross_tab = pd.crosstab(df['Profit Center'], df['Current Status'])
cross_tab
'''
Current Status  Active  Resigned
Profit Center                   
PC - 1             228        56
PC - 10              0         1
PC - 11              1         0
PC - 2             263        80
PC - 3             339        92
PC - 4               1         0
PC - 5               0        28
PC - 6               0        18
PC - 7               0         2
PC - 8               1         0
PC - 9               1         0

from the plot we can observe the following:  
1- proportion of people resigning in PC-5 & PC-6 is more.
2- PC-1, PC-2 & PC-3 has more employess compared to other centers combined
3- PC-10, PC-5, PC-6 & PC-7 are inactive as all the employees from these centers already resigned,
   we can club these as inactive profit centers to make our model less expensive.

Clubiing the category as PC - Inactive or few employees & PC -1, PC -2, PC - 3 will remain as it is
'''

df['Profit Center'].replace('PC - 10','PC - Inactive or few employees',inplace = True)
df['Profit Center'].replace('PC - 11','PC - Inactive or few employees',inplace = True)
df['Profit Center'].replace('PC - 4','PC - Inactive or few employees',inplace = True)
df['Profit Center'].replace('PC - 5','PC - Inactive or few employees',inplace = True)
df['Profit Center'].replace('PC - 6','PC - Inactive or few employees',inplace = True)
df['Profit Center'].replace('PC - 7','PC - Inactive or few employees',inplace = True)
df['Profit Center'].replace('PC - 8','PC - Inactive or few employees',inplace = True)
df['Profit Center'].replace('PC - 9','PC - Inactive or few employees',inplace = True)

## chisquare test
## Hypothesis - Does Profit Center has any relation to employees resignation ?
from scipy.stats import chi2_contingency  
cross_tab = pd.crosstab(df['Profit Center'], df['Current Status'])
cross_tab
'''
Current Status                  Active  Resigned
Profit Center                                   
PC - 1                             228        56
PC - 2                             263        80
PC - 3                             339        92
PC - Inactive or few employees       4        49


'''

c, p, dof, exp = chi2_contingency(cross_tab)
p
'''
P-value =  1.981377701870508e-29, Profit Center can be good predictor.
'''


# 2. Employee Position Vs Current Status

plt.figure(figsize=(14,10))
sns.countplot(df['Employee Position'],hue = df['Current Status'])
plt.title('Employement status at different Positions')
plt.legend(loc='upper left')
plt.xticks(rotation = 20)
'''
From the above plot we can see that most of the employees work at levels 6, 7 & 8.

Employee Position follow an order so, we will encode it as an ordinal variable!
'''
## chi-square test
# Hypothesis - Does Employee Position has any affect on the Employement Status ?

cross_tab = pd.crosstab(df['Employee Position'], df['Current Status'])
cross_tab
'''
Current Status     Active  Resigned
Employee Position                  
-                       7         2
Level 1                 2         0
Level 10                0         1
Level 2                10         1
Level 3                19         1
Level 4                41         8
Level 5                82        18
Level 6               177        56
Level 7               274        86
Level 8               214       104
Level A1                2         0
Level A2                3         0
Level A3                3         0

level 10 has no active employees.
'''
c, p, dof, exp = chi2_contingency(cross_tab)
p
'''
p-value = 0.007309333501753395
Null hypothesis Rejected. Accepting the variable
'''



# 3. Employee Location vs Current Status

plt.figure(figsize=(14,10))
sns.countplot(df['Employee Location'], hue=df['Current Status'])
plt.title('Employement Status at different Locations')
plt.legend(loc = 'upper right')

'''
Observations from the plot-
1- Location 3, 7 & 1 has the moost count of employees compared to other locations combined.
2- Propotion of employees resigning at location 3, 7 & 2 is more
3- most employees resigned at location 3

we will perform multivariate analysis on this variable to futher explore!!
'''

## Chisquare test
## Hypothesis - Does Different working locations affect employement status?

cross_tab = pd.crosstab(df['Employee Location'],df['Current Status'])
cross_tab
'''
Current Status     Active  Resigned
Employee Location                  
Location 1            268        78
Location 2             31        17
Location 3            180        67
Location 4             33        10
Location 5              7         1
Location 6              5         3
Location 7            218        81
Location 8             45         6
Location 9             47        14

1- 27% of the employees out of total 299 emplyees at location 7 resigned
2- 35% of the employees out of total 48 employees at location 2 resigned
'''

c, p, dof, exp = chi2_contingency(cross_tab)
p
'''
p-value = 0.17465165527812224
Null Hypothesis accepted, Rejecting the variable!
'''

# 4. People Group Vs Current Status
df['People Group'].value_counts()
1101/1111
plt.figure(figsize = (14,10))
sns.countplot(df['People Group'], hue = df['Current Status'])
plt.legend(loc ='upper right')
'''
Rejecting the variable as Data is more dominant towards 'Client Service Staff' by 99% 
'''


# 5. Employee Category Vs Current Status

df['Employee Category'].value_counts()

plt.figure(figsize = (14,10))
sns.countplot(df['Employee Category'])
plt.xticks(rotation = 30)
'''
Rejecting the variable because this variable is quite similar to the Dependent variable and if
used may create a biased model!
'''


# 6. Supervisor name vs Current Status
df['Supervisor name'].nunique()
'''
there are total 152 different supervisors, it will be hard to plot all of them so instead, will
create two filters, one of all active employess and one of all resigned employees and then check
the supervisor with percentage of resignation under each!
'''

act_filt = df['Current Status']!= 'Resigned'
res_filt = df['Current Status'] == 'Resigned'

df_act = df.loc[act_filt,'Supervisor name']
df_res = df.loc[res_filt,'Supervisor name']

act = df_act.value_counts() # storing the values in a variable
res = df_res.value_counts()

act=tuple(zip(act,act.index))
res=tuple(zip(res,res.index))
act
res
  
       
#creating a function to calculate the probability 
def perct(tup1, tup2):
    
    perc = []
    name = []
    for i,sup1 in tup1:
        for j,sup2 in tup2:
            if sup1 == sup2: #matching the supervisor name
                k = j/(i+j) #probability calculation
                perc.append(k) # appending it to a list
                name.append(sup1) 
            else:
                pass
    return (perc,name)

prob, name =perct(act,res)

prob=sorted(prob, reverse = True)
prob[:11]
#converting the results into a tuple
sorted_prob=()
for item in zip(prob, name):
    sorted_prob = sorted_prob+(item,)

sorted(sorted_prob, reverse=True)

# Pie Chart  (most attritions)
mylabels = ['Sile Lorrie','Katuscha Pru','Vonni Bethena','Lindy Marguerite','Blondy Tatiania','Trenna Mureil','Dottie Bidget',
            'Gert Editha','Tatiana Roxanna','Rosabella Arlina']

plt.figure(figsize=(14,10))
plt.barh(mylabels,prob[:10], align = 'edge', edgecolor = 'black')
plt.xlabel('Percent of attrition')
plt.ylabel('Supervisor Name')
plt.title('Top 10 supervisor with most resignations under them')
plt.show()

# Pie Chart  (least attritions)
prob, name =perct(act,res)

prob=sorted(prob)
prob[:11]
#converting the results into a tuple
sorted_prob=()
for item in zip(prob, name):
    sorted_prob = sorted_prob+(item,)

sorted(sorted_prob)

mylabels = ['Cacilia Aimee','Jillian Lorelei','Tiena Hatti','Jolyn Briney','Tallia Eyde','Dottie Bidget','Gert Editha',
            'Rosabella Arlina','Tatiana Roxanna','Trenna Mureil']

plt.figure(figsize=(14,10))
plt.barh(mylabels,prob[:10], align = 'edge', edgecolor = 'black')
plt.xlabel('Percent of attrition')
plt.ylabel('Supervisor Name')
plt.title('Top 10 supervisor with least resignations under them')
plt.show()



# 7. Total Hours Vs Current Status

plt.figure(figsize=(14,10))
sns.kdeplot(df.loc[df['Current Status'] == 'Resigned', 'Total Hours'], label='R', shade = True)
sns.kdeplot(df.loc[df['Current Status'] != 'Resigned', 'Total Hours'], label='A', shade = True)
plt.title('Total Hours Distribution as per Current Status', fontsize = 20)
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(df['Current Status'], df['Total Hours'])
plt.xlabel('Employement Status')
plt.ylabel('Total chargeble available hours')
'''
Distribution of Total chargeble hours available is right skewed whereas it is rleft skewed for
all the active employees
'''

## T-test
## Hypothesis - Does Total Chargable Hours affect the employement status ?
CS_A= df[df['Current Status'] == 'Active']
CS_R= df[df['Current Status']=='Resigned']
import scipy
scipy.stats.ttest_ind(CS_A['Total Hours'], CS_R['Total Hours'] )

'''
Ttest_indResult(statistic=12.127629257973059, pvalue=7.108462869599731e-32)

p-value < 0.05, hence Null Hypothesis is rejected
Total Hours can be a good predictor!
'''


# 8. Total Available Hours Vs Current Status

plt.figure(figsize=(14,10))
sns.kdeplot(df.loc[df['Current Status'] == 'Resigned', 'Total Available Hours'], label='R', shade = True)
sns.kdeplot(df.loc[df['Current Status'] != 'Resigned', 'Total Available Hours'], label='A', shade = True)
plt.title('Actual Available chargeble Hours Distribution as per Current Status', fontsize = 20)
plt.legend()
plt.show()

'''
Distribution for Resigned employees is right skewed whereas it is left skewed for active employees!
'''

## T-Test
## Hypothesis - Does actual available hours affect Employement Status?

CS_A= df[df['Current Status'] == 'Active']
CS_R= df[df['Current Status']=='Resigned']
import scipy
scipy.stats.ttest_ind(CS_A['Total Available Hours'], CS_R['Total Available Hours'] )
'''
Ttest_indResult(statistic=11.754566745354413, pvalue=3.783092774756236e-30)

p-value < 0.05, NUll Hypothesis rejected,
can be a good predictor!
'''


# 9. Work Hours Vs Current Status

plt.figure(figsize=(14,10))
sns.kdeplot(df.loc[df['Current Status'] == 'Resigned', 'Work Hours'], label='R', shade = True)
sns.kdeplot(df.loc[df['Current Status'] != 'Resigned', 'Work Hours'], label='A', shade = True)
plt.title('Work Hours Distribution as per Current Status', fontsize = 20)
plt.legend()
plt.show()

'''
Distribution of work hours for resigned employees is right skewed, indicating less hours of work.
whereas for active employees it in little bit uniform indicating probability of employees working
less or more hours is same!
'''

## T-Test
## Hypothesis - Does actual work hours affect employement status ?

CS_A= df[df['Current Status'] == 'Active']
CS_R= df[df['Current Status']=='Resigned']
import scipy
scipy.stats.ttest_ind(CS_A['Work Hours'], CS_R['Work Hours'] )
'''
Ttest_indResult(statistic=10.233550003809558, pvalue=1.5034673645844107e-23)

p-value < 0.05, Null Hypothesis rejeceted.

can be a good predictor!
'''



# 10. Leave Hours Vs Current status

plt.figure(figsize=(14,10))
sns.kdeplot(df.loc[df['Current Status'] == 'Resigned', 'Leave Hours'], label='R', shade = True)
sns.kdeplot(df.loc[df['Current Status'] != 'Resigned', 'Leave Hours'], label='A', shade = True)
plt.title('Leave Hours Distribution as per Current Status', fontsize = 20)
plt.legend()
plt.show()

'''
observation - 
1- Resigned employees have taken less number of leaves compared to active employees!
'''

## T-test
## Hypothesis - Does leave hours affect employement status ?

CS_A= df[df['Current Status'] == 'Active']
CS_R= df[df['Current Status']=='Resigned']
import scipy
scipy.stats.ttest_ind(CS_A['Leave Hours'], CS_R['Leave Hours'] )
'''
Ttest_indResult(statistic=7.478275812329296, pvalue=1.5264813658836308e-13)

p-value < 0.05, Null Hypothesis rejeceted.

can be a good predictor!
'''



# 11. Training Hours Vs Current status

plt.figure(figsize=(14,10))
sns.kdeplot(df.loc[df['Current Status'] == 'Resigned', 'Training Hours'], label='R', shade = True)
sns.kdeplot(df.loc[df['Current Status'] != 'Resigned', 'Training Hours'], label='A', shade = True)
plt.title('Training Hours Distribution as per Current Status', fontsize = 20)
plt.legend()
plt.show()

'''
observation - 
1- Distribution for both resigned and active employees in right skewed
2- active employees have been in training more than resigned employees
'''

## T-Test
## Hypothesis - Does training hours affect employement status ?

CS_A= df[df['Current Status'] == 'Active']
CS_R= df[df['Current Status']=='Resigned']
import scipy
scipy.stats.ttest_ind(CS_A['Training Hours'], CS_R['Training Hours'] )
''' 
Ttest_indResult(statistic=9.947483503742305, pvalue=2.169317831159363e-22)

p-value < 0.05, Null Hypothesis rejeceted.

can be a good predictor!
'''



# 12. BD Hours Vs Current status

plt.figure(figsize=(14,10))
sns.kdeplot(df.loc[df['Current Status'] == 'Resigned', 'BD Hours'], label='R', shade = True)
sns.kdeplot(df.loc[df['Current Status'] != 'Resigned', 'BD Hours'], label='A', shade = True)
plt.title('BD Hours Distribution as per Current Status', fontsize = 20)
plt.legend()
plt.show()

'''
observation - 
1- Distribution for both resigned and active employees is gaussian with outliers
2- both the active and resigned employees devoted simialer hours to business developtment
'''

## Anova
## Hypothesis - Does BD Hours affect employement status ?

CS_A= df[df['Current Status'] == 'Active']
CS_R= df[df['Current Status']=='Resigned']
import scipy
scipy.stats.ttest_ind(CS_A['BD Hours'], CS_R['BD Hours'] )
'''      
Ttest_indResult(statistic=2.6115845821634385, pvalue=0.00913406616632078

p-value < 0.05, Null Hypothesis Rejected.

BD hours can be a good predictor
'''



# 13. NC Hours Vs Current status

plt.figure(figsize=(14,10))
sns.kdeplot(df.loc[df['Current Status'] == 'Resigned', 'NC Hours'], label='R', shade = True)
sns.kdeplot(df.loc[df['Current Status'] != 'Resigned', 'NC Hours'], label='A', shade = True)
plt.title('NC Hours Distribution as per Current Status', fontsize = 20)
plt.legend()
plt.show()

'''
observation - 
1- Distribution for both resigned and active employees is gaussian (if outliers are removed)
2- both the active and resigned employees similar Non chargeble hours 
'''

## Anova
## Hypothesis - Does NC Hours affect employement status ?

CS_A= df[df['Current Status'] == 'Active']
CS_R= df[df['Current Status']=='Resigned']
import scipy
scipy.stats.ttest_ind(CS_A['NC Hours'], CS_R['NC Hours'] )
'''      
Ttest_indResult(statistic=3.732302263855521, pvalue=0.00019936223742571182)

p-value < 0.05, Null Hypothesis Rejected.

can be a good predictor!
'''




# 14. Utilization% Vs Current status

plt.figure(figsize=(14,10))
sns.kdeplot(df.loc[df['Current Status'] == 'Resigned', 'Utilization%'], label='R', shade = True)
sns.kdeplot(df.loc[df['Current Status'] != 'Resigned', 'Utilization%'], label='A', shade = True)
plt.title('Total Utilization as per Current Status', fontsize = 20)
plt.legend()
plt.show()

'''
observation - 
1- Distribution for both resigned and active employees is quite similar(extreme outliers are present)
2- both the active and resigned employees similar yearly Utilization 
'''

## T-Test
## Hypothesis - Does Utilization% affect employement status ?

CS_A= df[df['Current Status'] == 'Active']
CS_R= df[df['Current Status']=='Resigned']
import scipy
scipy.stats.ttest_ind(CS_A['Utilization%'], CS_R['Utilization%'] )
'''      
p-value < 0.05, Null Hypothesis Rejected.

can be a good predictor!
'''


# changing dataype of monthly wise utilization to float
df.dtypes

col = [x for x in df.columns]

del col[0:11]

del col[24:32]

col

for i in col:
    for i in col:
        df[i].replace('-',0, inplace = True)
    df[i].astype('float')  


df1 = df.copy()

df1.to_csv('/home/ryuzaki/E drive/Dekstop/projects/Imarticus/df1.csv')

