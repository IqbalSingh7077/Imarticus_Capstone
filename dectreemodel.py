# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 20:20:41 2021

@author: Lakshmi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dfdec = pd.read_csv('C:/Users/Lakshmi/Desktop/dfm1.csv')


#####SPLITTING DATA IN TO TRAIN AND TEST DATA SETS

from sklearn.model_selection import train_test_split

y = dfdec["Current Status"]

X = dfdec.drop(columns=['Current Status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1234)

print("Number of employyes information X_train dataset: ", X_train.shape)
print("Number of employees information y_train dataset: ", y_train.shape)
print("Number of employees information X_test dataset: ", X_test.shape)
print("Number of employees information y_test dataset: ", y_test.shape)
##774+333=1107


###---Applying SMOTE to balance  variable


pip install imblearn
from imblearn.over_sampling import SMOTE


#import collections
# summarize class distribution
#counter = collections.Counter(y_train)
#print(counter)

# transform the dataset
smote = SMOTE()

X_train_smote, y_train_smote = smote.fit_resample(X_train.astype('float'), y_train)

from collections import Counter
print("Before SMOTE:", Counter(y_train))
print("After SMOTE:", Counter(y_train_smote))


# Create Decision Tree classifer object

from sklearn.tree import DecisionTreeClassifier

clfentropy = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Train Decision Tree Classifer
clfentropy = clfentropy.fit(X_train_smote,y_train_smote)
print(clfentropy)

##FOR TRAINING DATASET
predxtrain = clfentropy.predict(X_train_smote)

predxtrain

# Model Accuracy, how often is the classifier correct?
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_train_smote, predxtrain))
'''
print("Accuracy:",metrics.accuracy_score(y_train_smote, predxtrain))
Accuracy: 0.9748263888888888
'''

##---confusion matrix---

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_smote, predxtrain)

'''
rray([[565,  11],
       [ 18, 558]], dtype=int64)
'''
###checking accuracy 

from sklearn.metrics import accuracy_score

accuracy_score(y_train_smote, predxtrain)
'''
accuracy_score(y_train_smote, predxtrain)
Out[311]: 0.9748263888888888
'''
from sklearn.metrics import classification_report
print(classification_report(y_train_smote, predxtrain))
'''
    print(classification_report(y_train_smote, predxtrain))
              precision    recall  f1-score   support

           0       0.97      0.98      0.97       576
           1       0.98      0.97      0.97       576

    accuracy                           0.97      1152
   macro avg       0.97      0.97      0.97      1152
weighted avg       0.97      0.97      0.97      1152
'''

# Create Decision Tree classifer object----GINI
clfgini = DecisionTreeClassifier()
# Train Decision Tree Classifer
clfgini = clfentropy.fit(X_train_smote,y_train_smote)
print(clfgini)

##FOR TRAINING DATASET
predxtrain = clfgini.predict(X_train_smote)

predxtrain

# Model Accuracy, how often is the classifier correct?
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_train_smote, predxtrain))






##Predict the response for TEST dataset

# Create Decision Tree classifer object----GINI
clfgini = DecisionTreeClassifier(criterion="gini", max_depth=3)

# Train Decision Tree Classifer
clfgini = clfgini.fit(X_train_smote,y_train_smote)
clfgini
#Predict the response for test dataset
y_predgini = clfgini.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_predgini))


'''
print("Accuracy:",metrics.accuracy_score(y_test, y_predgini))
Accuracy: 0.963963963963964
'''
##---confusion matrix---

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_predgini)
'''
a([[248,   6],
       [  6,  73]], dtype=int64)
'''
###checking accuracy 

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_predgini)
'''
accuracy_score(y_test,y_predgini)
Out[344]: 0.9639639639639645
'''
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predgini))

'''
print(classification_report(y_test,y_predgini))
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       254
           1       0.92      0.92      0.92        79

    accuracy                           0.96       333
   macro avg       0.95      0.95      0.95       333
weighted avg       0.96      0.96      0.96       333
'''


# Create Decision Tree classifer object---ENTROPY
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train_smote,y_train_smote)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


'''
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
Accuracy: 0.960960960960961
'''
##---confusion matrix---

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
from sklearn.metrics import confusion_matrix
labels = ['Active', 'Resigned']
cm=confusion_matrix(y_test, y_pred)
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
'''
array([[247,   7],
       [  6,  73]], dtype=int64)
'''
###checking accuracy 

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
'''
accuracy_score(y_test,y_pred)
Out[324]: 0.960960960960961
'''
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
'''
    precision    recall  f1-score   support

           0       0.98      0.97      0.97       254
           1       0.91      0.92      0.92        79

    accuracy                           0.96       333
   macro avg       0.94      0.95      0.95       333
weighted avg       0.96      0.96      0.96       333
'''
###tree building
from sklearn import tree
tree.plot_tree(clf)

#ROC curve
!pip install scikit-plot
yhat_prob=clfentropy.predict_proba(X_test)
import scikitplot as skplt
y_true = y_test
y_probas = yhat_prob


plt.figure(figsize=(20,15))
skplt.metrics.plot_roc(y_true, y_probas)
plt.title("ROC Curves of decision tree")
plt.show()
'''

feature_labels = np.array(["Profit Center","Employee Position","Supervisor name",'Apr-16 Utilization%', 'May-16 Utilization%',
                 'June-16 Utilization%','Mar-17 Utilization%',
       'Apr-17 Utilization%', 'May-17 Utilization%', 'June-17 Utilization%',
       'Jul-17 Utilization%', 'Aug-17 Utilization%', 'Sep-17 Utilization%',
       'Oct-17 Utilization%', 'Nov-17 Utilization%', 'Dec-17 Utilization%',
       'Jan-18 Utilization%', 'Feb-18 Utilization%', 'Mar-18 Utilization%',
           "Total Hours","Total Available Hours","Work Hours","Leave Hours",
          "Training Hours","NC Hours","Tenure","Current Status"])
importance = clf.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))
    analysis_result += ('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))
    '''
