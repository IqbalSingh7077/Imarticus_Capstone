
import os
os.chdir('/home/ryuzaki/E drive/Dekstop/projects/Imarticus')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importing the dataset

df = pd.read_csv('df1.csv')


##************* Multivariate Analysis ****************##
##----------------------------------------------------##


# 1. Correlation table and Plot
df_ml = df.copy()
df_corr = df.drop(df_ml.iloc[:,0:35], axis = 1)
df_corr.columns

cor = df_corr.corr()
cor

plt.figure(figsize = (14,10))
sns.heatmap(cor, annot = True, cmap = 'YlGnBu')
plt.title('Correlation Plot')



plt.close();
sns.pairplot(df_ml, hue = 'Current Status')
plt.show()
'''
Observations-
1- Total Hours and To Avialable Hours are perfectly correlated with each other
2- Leave Hours and Total Hours has 0.65 correlation value
3- Work Hours has around correlation value of around 0.8 with Total Hours and Total Available hours
4- other variables has little to no-correlation
'''

## Aggregate analysis

#1. creating a group on Current status
Emp_st = df_ml.groupby(['Current Status'])

## 
Emp_st['Employee Position'].value_counts()
'''
Observation-
1- 7 missing values in Active group and 2 in Resigned group, will fill them with the mode level of 
   the group, i.e. Level 7.
2- Level 8 has the most employee resignations followed by level 7 and level 5
'''

#2. creating a group of Current status and Employee Position
Emp_st = df_ml.groupby(['Current Status','Employee Position'])
Emp_st['Utilization%'].mean()
'''
Current Status  Employee Position
Active          -                    0.804529
                Level 1              0.209650
                Level 2              0.318820
                Level 3              0.226284
                Level 4              0.388822
                Level 5              0.616396
                Level 6              0.837719
                Level 7              0.834971
                Level 8              0.761448
                Level A1             0.000000
                Level A2             0.000000
                Level A3             0.000000
Resigned        -                    0.971850
                Level 10             0.971400
                Level 2              0.156400
                Level 3              0.102300
                Level 4              0.299775
                Level 5              0.375228
                Level 6              0.703207
                Level 7              0.698033
                Level 8              0.721058
Name: Utilization%, dtype: float64

Observations-

1- Level A1, A2 & A3 has 0 average utilization in Active group
2- level 10 in resigned has the highest utilization
3- level 7 & 6 has the highest average utilization in active group
'''


# 3. Creaing a group on Current Status and Employee Location

Emp_st = df_ml.groupby(['Current Status','Employee Location'])

Emp_st[['Leave Hours','Work Hours']].agg(['mean','median'])

'''
                                 Leave Hours          Work Hours          
                                        mean median         mean    median
Current Status Employee Location                                          
Active         Location 1         289.768657  248.0  1744.556381  1383.500
               Location 2         348.000000  368.0  2114.370968  2474.000
               Location 3         369.861111  336.0  1996.944278  1909.750
               Location 4         427.666667  440.0  2248.393939  2514.000
               Location 5         359.428571  424.0  2014.214286  2496.000
               Location 6         484.200000  506.0  1634.800000  1060.000
               Location 7         330.254587  324.0  2046.261193  2114.000
               Location 8         247.688889  240.0  1768.950000  1696.000
               Location 9         302.914894  320.0  1945.459149  1950.500
Resigned       Location 1         213.153846  132.0  1103.619872   749.925
               Location 2         185.117647  174.0  1158.941176   804.000
               Location 3         154.910448  104.0  1005.477612   784.000
               Location 4         198.400000  136.0  1349.750000  1546.250
               Location 5         348.000000  348.0   172.000000   172.000
               Location 6         120.000000   32.0   319.000000     0.000
               Location 7         186.851852  152.0  1150.428519   742.500
               Location 8         282.666667  256.0  1650.000000  1370.000
               Location 9         157.357143   96.0  1051.142857   774.000

Observations - 
Leave Hours-
1- highest mean & median leaves are taken at location 5 & 8 for the group resigned
2- lowest mean & median leaves are taken at location 6, 3 & 9 for the group resigned
3- highest mean & median leaves are taken at location 2,3,4,5 & 6 for the group Active
4- lowest mean & median leaves are taken at location 1 & 8 for the group Active 

Work Hours-
1- highest mean & median work is done at location 4 & 2 for the group Active
2- lowest mean & median work is done at location 1,3 & 6 for the group Active
3- highest mean & median work is done at location 2 & 8 for the group Active
4- lowest mean & median work is done at location 5 & 6 for the group Active 
'''


Emp_st['Utilization%'].agg(['mean','median'])
'''
                                      mean   median
Current Status Employee Location                   
Active         Location 1         0.721009  0.86100
               Location 2         0.698752  0.74060
               Location 3         0.714269  0.82475
               Location 4         0.754142  0.85020
               Location 5         0.623100  0.73090
               Location 6         0.629780  0.72630
               Location 7         0.778324  0.86260
               Location 8         0.782544  0.87650
               Location 9         0.836760  0.89500
Resigned       Location 1         0.732910  0.77125
               Location 2         0.626094  0.73790
               Location 3         0.653139  0.72320
               Location 4         0.624910  0.77840
               Location 5         0.055400  0.05540
               Location 6         0.184933  0.00000
               Location 7         0.669019  0.75090
               Location 8         0.829950  0.90170
               Location 9         0.651864  0.80520
               
Observations - 
Active - 
1- Employees are highly utilized at location 9 & 8
2- Employees are not utilized properly at location 2, 5 & 6

Resigned -
1- Employees were higly utilized at location 8
2- Employees were not properly utilized at location 5 & 6 
'''



# creating a function to check for anova values for month wise utilization

def testt(col, dat):
    import scipy
    
    res = []
    for i in col:  
        CS_A= df_ml[df_ml['Current Status'] == 'Active']
        CS_R= df_ml[df_ml['Current Status']=='Resigned']
        
        r=scipy.stats.ttest_ind(CS_A[i], CS_R[i] )
        res.append(r)
        
    for k,j  in zip(col,res):
             print("column:%s and Test score: %s" % (k,j))    


col = df_ml.columns.tolist()

del col[0:11]
del col[24:32]

col

testt(col, df_ml)

'''

column:Apr-16 Utilization% and Test score: Ttest_indResult(statistic=-5.687918935227921, pvalue=1.6440612934565812e-08)

column:May-16 Utilization% and Test score: Ttest_indResult(statistic=-4.504898508536065, pvalue=7.343489997567106e-06)

column:June-16 Utilization% and Test score: Ttest_indResult(statistic=-4.556836992511695, pvalue=5.7685566353918625e-06)

column:Jul-16 Utilization% and Test score: Ttest_indResult(statistic=-1.29107297689781, pvalue=0.19694742881035884)

column:Aug-16 Utilization% and Test score: Ttest_indResult(statistic=-2.1480584703356485, pvalue=0.0319252206577402)

column:Sep-16 Utilization% and Test score: Ttest_indResult(statistic=-0.18431205183098784, pvalue=0.8538023563128296)

column:Oct-16 Utilization% and Test score: Ttest_indResult(statistic=0.5242014837873606, pvalue=0.6002431587876558)

column:Nov-16 Utilization% and Test score: Ttest_indResult(statistic=2.059744515938246, pvalue=0.03965593673824266)

column:Dec-16 Utilization% and Test score: Ttest_indResult(statistic=1.5318925274042825, pvalue=0.12583415406408063)

column:Jan-17 Utilization% and Test score: Ttest_indResult(statistic=2.387887716831124, pvalue=0.017112132758577488)

column:Feb-17 Utilization% and Test score: Ttest_indResult(statistic=2.7631474468230617, pvalue=0.00581939371343011)

column:Mar-17 Utilization% and Test score: Ttest_indResult(statistic=4.279872468314245, pvalue=2.0315317302046157e-05)

column:Apr-17 Utilization% and Test score: Ttest_indResult(statistic=5.118544675737463, pvalue=3.627187309486468e-07)

column:May-17 Utilization% and Test score: Ttest_indResult(statistic=5.897782475477677, pvalue=4.886551007595489e-09)

column:June-17 Utilization% and Test score: Ttest_indResult(statistic=7.103343645769039, pvalue=2.173080065272367e-12)

column:Jul-17 Utilization% and Test score: Ttest_indResult(statistic=4.360673352820452, pvalue=1.4172274695536553e-05)

column:Aug-17 Utilization% and Test score: Ttest_indResult(statistic=9.348761113657691, pvalue=4.7309063067826125e-20)

column:Sep-17 Utilization% and Test score: Ttest_indResult(statistic=12.720391181398895, pvalue=1.062736690585265e-34)

column:Oct-17 Utilization% and Test score: Ttest_indResult(statistic=12.817782319536024, pvalue=3.5704149193872823e-35)

column:Nov-17 Utilization% and Test score: Ttest_indResult(statistic=15.014687552642922, pvalue=1.5837034220259956e-46)

column:Dec-17 Utilization% and Test score: Ttest_indResult(statistic=16.701552323635976, pvalue=4.965205461894902e-56)

column:Jan-18 Utilization% and Test score: Ttest_indResult(statistic=20.115645707442763, pvalue=5.713845766388738e-77)

column:Feb-18 Utilization% and Test score: Ttest_indResult(statistic=21.10461099865866, pvalue=2.198245939098848e-83)

column:Mar-18 Utilization% and Test score: Ttest_indResult(statistic=22.007260148793506, pvalue=2.3695812512757182e-89)

so p-value for columns Jul-16,Aug-16, Sep-16, Oct-16, Nov-16, Dec-16, Jan-17, Feb-17 is > 0.05
Hence we will not consider these variables while building our model !
'''


plt.figure(figsize = (14,10))
sns.boxplot(x =df['Apr-16 Utilization%'], y= df['Current Status'])
plt.title('Boxplot of April 16 Utilization as per Employee Status')

plt.figure(figsize = (14,10))
sns.boxplot(df['Apr-16 Utilization%'])
plt.title('Boxplot of April 16 Utilization ')


