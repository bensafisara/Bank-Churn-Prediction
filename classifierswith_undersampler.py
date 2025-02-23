# -*- coding: utf-8 -*-
"""ClassifiersWith_Undersampler.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fLNI5YkDqrpr4svUe_LM5rhazQPtwcn_

Dans ce notebook on applique le mécanisme de UnderSampler pour les algorithme knn, GaussianNB et RandomForest.
Le prétraitement sera avec la transformation oneHote seulemnt.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from matplotlib import pyplot
from sklearn.naive_bayes import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler #fixed import
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

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

"""#Data"""

#Load Data and Target and # shuffle the dataset
donnees=pd.read_csv("/Donnees/donnees.csv") #on charge les données
non_dem=donnees.loc[donnees['DEM'] == 0] # on récupère les lignes qui ont la valeur 0 dans la colonne 'DEM' -> non démissionnaire
donnees_=pd.concat([donnees,non_dem]) # on concatène les données avec les lignes qui ont la valeur 0 dans la colonne 'DEM' -> non démissionnaire
donnees=donnees_
donnees = donnees.sample(frac=1).reset_index(drop=True) # on mélange les données
##print("Data")
##print(donnees)
print(end="\n")
print(end="\n")
dt_features=donnees.copy() # on copie les données dans dt_features
dt_labels=dt_features.copy() # on copie les données dans dt_labels
dt_labels=dt_labels.pop('DEM') # on supprime la colonne 'DEM' de dt_labels
###

# one hot encoder
one_hot=OneHotEncoder() # on crée un objet one hot encoder
encoder = OneHotEncoder(handle_unknown='ignore') 
#perform one-hot encoding on 'team' column 
encoder_df = pd.DataFrame(encoder.fit_transform(dt_features[['CDSITFAM']]).toarray()) 
final_df = dt_features.join(encoder_df)
encoder_df = pd.DataFrame(encoder.fit_transform(dt_features[['RANGAGEAD']]).toarray())
final_df = dt_features.join(encoder_df)
encoder_df = pd.DataFrame(encoder.fit_transform(dt_features[['RANGADH']]).toarray())
final_df = dt_features.join(encoder_df)
dt_features=final_df.drop(['CDSITFAM','RANGAGEAD','RANGADH','DEM'],axis=1)

print(sorted(Counter(dt_labels).items())) # on affiche le nombre de démissionnaires et de non démissionnaires

rus = RandomUnderSampler(sampling_strategy="majority") # on crée un objet RandomUnderSampler
X_Uresampled, y_Uresampled = rus.fit_resample(dt_features, dt_labels) # on applique le mécanisme de RandomUnderSampler
print(sorted(Counter(y_Uresampled).items())) # on affiche le nombre de démissionnaires et de non démissionnaires après l'application du mécanisme de RandomUnderSampler

#Split data 
X_train, X_rem, y_train, y_rem = train_test_split(X_Uresampled,y_Uresampled, train_size=0.8) #on sépare les données en train et en test
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5) #test_size=0.5 signifie 50% des données restantes

# implement a naive bayes classifier
print("Naive Bayes")
clf = GaussianNB()
# clf = ComplementNB()
# train the classifier
y_train = np.array(y_train)
y_train=y_train.reshape(-1,1)
clf.fit(X_train, y_train)
# predict the response
pred = clf.predict(X_valid)
# evaluate accuracy
print(accuracy_score(y_valid, pred))
# print the confusion matrix
print(confusion_matrix(y_valid, pred))
pprint(Metrics(y_valid, pred))
print(f"Naive Bayes Report : \n\n {classification_report(y_valid, pred)}")
with open('/Modeles/GaussianNB_Undersamp', 'wb') as files: # on sauvegarde le modèle
    pickle.dump(clf, files) 



# implement k-nearest neighbors classifier
print("KNN")
knn = KNeighborsClassifier(n_neighbors=3)
# train the classifier
knn.fit(X_train, y_train)
# predict the response
pred = knn.predict(X_valid)
# evaluate accuracy
print(accuracy_score(y_valid, pred))
# print the confusion matrix
print(confusion_matrix(y_valid, pred))
pprint(Metrics(y_valid, pred))
print(f"KNN Report : \n\n {classification_report(y_valid, pred)}")
with open('/Modeles/KNN_Undersamp', 'wb') as files:
    pickle.dump(clf, files) 






#Create a Random Forest Classifier
print("RFC")
clf=RandomForestClassifier(n_estimators=10)
clf.fit(X_train,y_train)
y_tr=clf.predict(X_train)
pred=clf.predict(X_valid)
# evaluate accuracy
print(accuracy_score(y_valid, pred))
# print the confusion matrix
print(confusion_matrix(y_valid, pred))
pprint(Metrics(y_valid, pred))
pprint(Metrics(y_train, y_tr))
print(f"Random Forest Report : \n\n {classification_report(y_valid, pred)}")
with open('/Modeles/RandomForest_Undersamp', 'wb') as files:
    pickle.dump(clf, files)

# create a SVM classifier
print("SVM")
name_clf="svm"
clf = svm.SVC(gamma='auto')
y_train = np.array(y_train)
y_train.reshape(-1,1)
clf.fit(X_train, y_train)
pred=clf.predict(X_valid)
# evaluate accuracy
print(accuracy_score(y_valid, pred))
# print the confusion matrix
print(confusion_matrix(y_valid, pred))
print(f"Classifier Report : \n\n {classification_report(y_valid, pred)}")
print(f"SVM : \n\n {classification_report(y_valid, pred)}")

