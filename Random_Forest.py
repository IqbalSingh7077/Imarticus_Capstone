
import os
os.chdir('/home/ryuzaki/E drive/Dekstop/projects/Imarticus')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# importing the dataset
df = pd.read_csv('ML_df.csv')

df.dtypes
df.drop(columns = ['Unnamed: 0'], inplace = True)



##**************** Modeling **************##
##----------------------------------------## 




##***************** Randon Forest Classifier *********************##
##----------------------------------------------------------------##

# spliting data into train & split
# train test split 
from sklearn.model_selection import train_test_split

X = df.drop('Current Status', axis=1)
y = df['Current Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
'''
Train set: (774, 27) (774,)
Test set: (334, 27) (334,)
'''



## Balacing the dataset for random forest
plt.figure(figsize=(12,10))
sns.countplot(y_train)
plt.title('Unbalanced Classes')

from imblearn.over_sampling import SMOTE

Y =y_train
X = X_train
print(X.shape,Y.shape)
'''
(777, 27) (777,)
'''

# OverSampling using SMOTE
smt = SMOTE(random_state=123)
smt_X, smt_Y = smt.fit_resample(X,Y)

print(smt_X.shape, smt_Y.shape)
'''
(1162, 27) (1162,)
'''
from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
'''
Original dataset shape Counter({0: 581, 1: 196})
'''

print('Resampled dataset shape {}'.format(Counter(smt_Y)))#
'''
Resampled dataset shape Counter({0: 581, 1: 581})
'''

#After re-sampling
plt.figure(figsize=(12,10))
sns.countplot(smt_Y)
plt.title('After Resmapling')


X_train = smt_X
y_train = smt_Y



# building the model
from sklearn.model_selection import RandomizedSearchCV

param_grid = {"n_estimators": [int(x) for x in np.linspace(start = 100, stop = 300, num = 50)],
              "max_depth" : [int(x) for x in np.linspace(1, 50, num = 11)],
              "max_features" : ['auto', 'sqrt'],
              "min_samples_split" : [2, 5, 10],
              "min_samples_leaf" : [1, 2, 4],
              "bootstrap" : [True, False],
              "criterion":["gini","entropy"]}

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_jobs=-1)    

randm_src = RandomizedSearchCV(estimator=rf_classifier, param_distributions = param_grid,
                               cv = 5, n_iter = 10, n_jobs=-1, random_state=42)
randm_src.fit(X_train, y_train)

print("\n The best estimator across ALL searched params:\n", randm_src.best_estimator_)
print("\n The best score across ALL searched params:\n", randm_src.best_score_)
print("\n The best parameters across ALL searched params:\n", randm_src.best_params_)

"""

 The best estimator across ALL searched params:
 RandomForestClassifier(criterion='entropy', max_depth=30, max_features='sqrt',
                       n_estimators=283, n_jobs=-1)

 The best score across ALL searched params:
 0.953844693373837

 The best parameters across ALL searched params:
 {'n_estimators': 283, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 
  'max_depth': 30, 'criterion': 'entropy', 'bootstrap': True}
"""
 
rf_final = RandomForestClassifier(n_estimators = 283,
                                 max_depth = 30,
                                 max_features = 'sqrt',
                                 min_samples_split = 2,
                                 min_samples_leaf = 1,
                                 criterion = 'entropy',
                                 bootstrap = True,
                                 random_state=42)


trained_model = rf_final.fit(X_train,y_train)


from sklearn import metrics

print( "Train Accuracy :: ", metrics.accuracy_score(y_train, trained_model.predict(X_train)))
#Train Accuracy ::  1.0

y_pred = trained_model.predict(X_test)

print( "Test Accuracy  :: ", metrics.accuracy_score(y_test, y_pred))

#Test Accuracy  ::  0.9579579579579579


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
'''
array([[248,   5],
       [ 10,  71]])
'''

#Evaluation


yhat = trained_model.predict(X_test)
yhat_train = trained_model.predict(X_train)

yhat_prob = trained_model.predict_proba(X_test)
yhat_prob_train = trained_model.predict_proba(X_train)

yhat[0:5]
yhat_prob[0:5]


from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat)
'''
0.8255813953488372
'''


#Confusion matrix plotting
from sklearn.metrics import confusion_matrix
labels = ['Active', 'Resigned']
cm=confusion_matrix(y_test, yhat)
plt.figure(figsize=(12,8))
axes=sns.heatmap(cm, square=True, annot=True,fmt='d',cbar=True,cmap=plt.cm.Blues)
ticks=np.arange(len(labels))+0.5
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
axes.set_xticks(ticks)
axes.set_xticklabels(labels,rotation=0)
axes.set_yticks(ticks)
axes.set_yticklabels(labels,rotation=0)

from sklearn.metrics import classification_report # test data
print (classification_report(y_test, yhat))
'''
              precision    recall  f1-score   support

           0       0.96      0.98      0.97       253
           1       0.93      0.88      0.90        81

    accuracy                           0.96       334
   macro avg       0.95      0.93      0.94       334
weighted avg       0.95      0.96      0.95       334
'''

print (classification_report(y_train, yhat_train))# train data
'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       581
           1       1.00      1.00      1.00       581

    accuracy                           1.00      1162
   macro avg       1.00      1.00      1.00      1162
weighted avg       1.00      1.00      1.00      1162
'''

#different accuracy scores (Test data)
from sklearn.metrics import log_loss
import sklearn.metrics as metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
print("Random Forest's Accuracy: ", metrics.accuracy_score(y_test, yhat))
print("Random Forest's LogLoss : ", log_loss(y_test, yhat_prob))
print("Random Forest's F1-Score: ", f1_score(y_test, yhat, average='weighted'))
-np.mean(cross_val_score(trained_model,X_test,y_test, scoring = 'neg_mean_absolute_error', cv= 4))
'''
Random Forest's Accuracy:  0.9579579579579579
Random Forest's LogLoss :  0.16952678480810052
Random Forest's F1-Score:  0.9575603510487231
Cross_val_score: 0.05389414802065404
'''

#Different accuracy scores (Train Data)
print("Random Forest's Accuracy: ", metrics.accuracy_score(y_train, yhat_train))
print("Random Forest's LogLoss : ", log_loss(y_train, yhat_prob_train))
print("Random Forest's F1-Score: ", f1_score(y_train, yhat_train, average='weighted'))
-np.mean(cross_val_score(trained_model,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 4))
'''
Random Forest's Accuracy:  1.0
Random Forest's LogLoss :  0.03949450728681169
Random Forest's F1-Score:  1.0
Cross_val_score: 0.03871015523166252
'''

#ROC curve
#!pip install scikit-plot
import scikitplot as skplt
y_true = y_test
y_probas = yhat_prob

plt.figure(figsize=(20,15))
skplt.metrics.plot_roc(y_true, y_probas)
plt.title("ROC Curves of Random Forest Classifier")
plt.show()

###---checking for multicolinearity
#manual calculation:vif 1/1-0.84=0.160
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data=pd.DataFrame()
vif_data["feature"]=df.columns
###calculating VIF for each feature---multicolinearity

vif_data["VIF"]=[variance_inflation_factor(df.values, i)
                 for i in range(len(df.columns))]
print(vif_data)
'''
Work Hours, Profit Center_PC - 1, Profit Center_PC - 2 & Profit Center_PC - 3 have a VIF value of > 5
'''






##************** Predicting Attrition *************##
##-------------------------------------------------##

df_act = df[df['Current Status'] == 0]

df_act_x = df_act.drop(columns=['Current Status'])
df_act_y = df_act['Current Status']

Prob_att = trained_model.predict_proba(df_act_x)
Prob_att

Prob_att = Prob_att[:,-1]
plt.hist(Prob_att)

np.mean(Prob_att)

poss_att = []

for i in Prob_att:
    if i > 0.08:
        poss_att.append('Yes')
    elif i <= 0.08:
        poss_att.append('No')        


unique, count = np.unique(poss_att, return_counts = True) 
count
'''
array([666, 168])

around 20% i.e. 168 employees may resign 
'''

# Importing Csv dataset with New feature Promotion

df_prob_att = pd.read_csv('Att_Prob_Ret.csv')

df_prob_att['Current Status'].value_counts()
'''
Converting Secondament, New Joiner & Sabbatical to Active Employees
'''
df_prob_att['Current Status'].replace('Secondment','Active', inplace =True)
df_prob_att['Current Status'].replace('New Joiner','Active', inplace =True)
df_prob_att['Current Status'].replace('Sabbatical','Active', inplace =True)



df_prob_att_act = df_prob_att[df_prob_att['Current Status'] == 'Active']
'''
Creating a dataframe with only Active Employees
'''

df_prob_att_act['Current Status'].value_counts()

# adding a new Feature as Probaility_Attrition
df_prob_att_act['Prob_Attrition'] = poss_att
'''
Prob_Attrition shows the predicted value wether the Active Employee will resign or not!
'''

# Exporting the file as Final Dataset
df_prob_att_act.to_csv('Final_dataset.csv')



# Importing the Final Dataset
final = pd.read_csv('Final_dataset.csv')

final['Promotion'].value_counts()
'''
False    671
True     163
Name: Promotion, dtype: int64
'''
grp = final.groupby(['Prob_Attrition'])
'''
Creating a group of Prob_Attrition
'''

grp['Promotion'].value_counts()
'''
Prob_Attrition  Promotion
No              False        532
                True         134
Yes             False        139
                True          29
'''