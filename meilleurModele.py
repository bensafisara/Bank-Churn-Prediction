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

"""Test """


#Load Data and Target and # shuffle the dataset
donnees=pd.read_csv("/Donnees/donnees.csv") #on charge les données
non_dem=donnees.loc[donnees['DEM'] == 0] # on récupère les lignes qui ont la valeur 0 dans la colonne 'DEM' -> non démissionnaire
donnees_=pd.concat([donnees,non_dem]) # on concatène les données avec les lignes qui ont la valeur 0 dans la colonne 'DEM' -> non démissionnaire
donnees=donnees_
donnees = donnees.sample(frac=1).reset_index(drop=True) # on mélange les données

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


# On applique le meilleur modèle sur les données de test

with open('/Modeles/RandomForest_Undersamp', 'rb') as f:
  RFC_UnderSamp=pickle.load(f)

pred = RFC_UnderSamp.predict(X_test)
print(accuracy_score(y_test, pred))