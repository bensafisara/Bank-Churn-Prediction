# -*- coding: utf-8 -*-
"""Classifiers_Withpreprocessing_OHE_SCALER

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fM_V-8LJJJagcdU9qWjI8-3dSJP8VK86

Dans ce notebook on effectue le prétraitement avec One hot encoder, Scaler et on applique le Over/UnderSampling
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
from sklearn import *
import pickle
from pprint import pprint
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from matplotlib import pyplot
from sklearn.naive_bayes import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler #fixed import

"""#Metrics"""


def Metrics(y_pred , y) :
  #List of Metrics
  list=['micro',None,'weighted','macro']
  evaluate_result = dict()
  for i in list :
    precision, recall, f_score, _ = precision_recall_fscore_support(y, y_pred, average=i)
    evaluate_result[str(i)] = {
            'f_score': f_score,
            'precision': precision,
            'recall': recall,
        }
  return evaluate_result

"""#Classifiers"""

#Dans le bloc suivant on fait la trasformation en one hot

#Load Data and Target and # shuffle the dataset
donnees=pd.read_csv("/Donnees/donnees.csv")
donnees = donnees.sample(frac=1).reset_index(drop=True)
target_name = "DEM"
target = donnees[target_name]
donnees = donnees.drop(columns=[target_name])
#Creat pipline to transform the data to OHE and Scale 
numeric_features = ['CDSEXE','NBENF','CDTMT'	, 'CDCATCL',	'AGEAD',	'ADH' ,	'MADH'	 , 'ANNEEADH']
numeric_transformer = Pipeline(steps=[ ("scaler",  StandardScaler())])
categorical_features = ['CDSITFAM'	, 'RANGAGEAD' ,	'RANGADH']
# NB :I had to use A sparse=false because by default it s true, and this on is not matrix containg a lot of zeros, could skip it
#it was the problem of my errors for a long moment 
categorical_transformer = Pipeline(steps=[ ('ohe',OneHotEncoder(sparse=False,handle_unknown='ignore')) ])
preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),])
#Split data 
X_train, X_rem, y_train, y_rem = train_test_split(donnees,target, train_size=0.8) #on sépare les données en train et en test
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5) #test_size=0.5 signifie 50% des données restantes

"""#WEIGHTED NAIVE BAYES """

print("naive_bayes")
NB = ComplementNB(alpha = 1.0, class_prior = None, fit_prior = True, norm = False)
name_clf="ComplementNB"
model = Pipeline(
steps=[("preprocessor", preprocessor), ("classifier", NB)])
model.fit(X_train, y_train )
y_pred =  model.predict(X_valid)
print('Accuracy of '+name_clf+' classifier on training set: {:.2f}'
      .format(model.score(X_train, y_train)))
print('Accuracy of '+name_clf+' classifier on valid set: {:.2f}'
  .format(model.score(X_valid, y_valid)))
print(confusion_matrix(y_valid, y_pred))
print(f"naive_bayes Report : \n\n {classification_report(y_valid, y_pred)}")
pprint(Metrics(y_valid, y_pred))

"""#Weighted SVM"""

# create a SVM classifier
print("SVM")
name_clf="svm"
#demissioner => 1 pas dem => 0 nous allons doner un poid plus grand a la classe minoritaire 
weights = {1:0.1, 0:1.0}
SVM_ = svm.SVC(gamma='auto', class_weight=weights)
#SVM_ = svm.SVC(gamma='auto')
model = Pipeline(
steps=[("preprocessor", preprocessor), ("classifier", SVM_)])
model.fit(X_train, y_train )
y_pred =  model.predict(X_valid)
print('Accuracy of '+name_clf+' classifier on training set: {:.2f}'
      .format(model.score(X_train, y_train)))
print('Accuracy of '+name_clf+' classifier on valid set: {:.2f}'
  .format(model.score(X_valid, y_valid)))
print(confusion_matrix(y_valid, y_pred))
print(f"svm Report : \n\n {classification_report(y_valid, y_pred)}")
print(Metrics(y_valid, y_pred))

#dans ce bloc on fait la transformation avec le pipline puis on applique le OVERsamp et UNDERsamp

#load data and target and # shuffle the dataset
donnees=pd.read_csv("/Donnees/donnees.csv")
non_dem=donnees.loc[donnees['DEM'] == 0]
donnees_=pd.concat([donnees,non_dem])
donnees=donnees_
donnees = donnees.sample(frac=1).reset_index(drop=True)
###print("Data")
###print(donnees)
print(end="\n")
print(end="\n")
########
dt_features=donnees.copy()
dt_labels=dt_features.copy()
dt_features=dt_features.drop(['DEM'],axis=1)
dt_labels=dt_labels.pop('DEM')

#RandomOverSampler and UnderSampler
print(end="\n")
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

ros = RandomUnderSampler(sampling_strategy="majority")  #RandomOverSampler(sampling_strategy="not majority") 
X_res, y_res = ros.fit_resample(dt_features, dt_labels)
y_res.value_counts().plot.pie(autopct='%.2f')
plt.show()
print(dt_features)


X_train, X_rem, y_train, y_rem = train_test_split(X_res,y_res, train_size=0.8) #on sépare les données en train et en test
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5) #test_size=0.5 signifie 50% des données restantes

# implement k-nearest neighbors classifier
print("KNN")
knn = KNeighborsClassifier(n_neighbors=3)
name_clf="KNN"
model = Pipeline(
steps=[("preprocessor", preprocessor), ("classifier", knn)])
model.fit(X_train, y_train )
ytr =  model.predict(X_train)
y_pred =  model.predict(X_valid)
print('Accuracy of '+name_clf+' classifier on training set: {:.2f}'
      .format(model.score(X_train, y_train)))
print('Accuracy of '+name_clf+' classifier on valid set: {:.2f}'
  .format(model.score(X_valid, y_valid)))
print(confusion_matrix(y_valid, y_pred))
print(f"KNN Report : \n\n {classification_report(y_valid, y_pred)}")
#on validation
print(Metrics(y_valid, y_pred))
#on training
print(Metrics(y_train, ytr))

#Create a Random Forest Classifier
print("RFC")
RFC= sklearn.ensemble.RandomForestClassifier(n_estimators=10)
model = Pipeline(
steps=[("preprocessor", preprocessor),("classifier", RFC)])
model.fit(X_train, y_train )
y_pred =  model.predict(X_valid)
print('Accuracy of '+name_clf+' classifier on training set: {:.2f}'
      .format(model.score(X_train, y_train)))
print('Accuracy of '+name_clf+' classifier on valid set: {:.2f}'
  .format(model.score(X_valid, y_valid)))
print(confusion_matrix(y_valid, y_pred))
print(f"RFC Report : \n\n {classification_report(y_valid, y_pred)}")
print(Metrics(y_valid, y_pred))
with open('/Modeles/RandomForest_Undersamp_pipeline', 'wb') as files:
    pickle.dump(RFC, files)



# implement a naive bayes classifier
print("Naive Bayes")
name_clf="Nb"
clf = GaussianNB()
model = Pipeline(
steps=[("preprocessor", preprocessor), ("classifier", clf)])
model.fit(X_train, y_train )
y_pred =  model.predict(X_valid)
ytr =  model.predict(X_train)
print('Accuracy of '+name_clf+' classifier on training set: {:.2f}'
      .format(model.score(X_train, y_train)))
print('Accuracy of '+name_clf+' classifier on valid set: {:.2f}'
  .format(model.score(X_valid, y_valid)))
print(confusion_matrix(y_valid, y_pred))
print(f"nb Report : \n\n {classification_report(y_valid, y_pred)}")
print(Metrics(y_valid, y_pred))
#on training
print(Metrics(y_train, ytr))

with open('/Modeles/RandomForest_Undersamp_pipeline', 'rb') as f:
  RandomForest_Undersamp_pipeline=pickle.load(f)


model = Pipeline(
steps=[("preprocessor", preprocessor), ("classifier", RandomForest_Undersamp_pipeline)])
y_pred =  model.predict(X_test)
pprint(Metrics(y_test, y_pred))