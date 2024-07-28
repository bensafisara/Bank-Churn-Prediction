Pour traiter les données il faut tout d'abord lancer le fichier pretraitement.py
Une fois ce fichier exécuté, il faut fournir le chemin du fichier contenant les données à pré-traiter.

Dès que l'exécution du code dans pretraitement.py est terminé, il y a 2 possibilités:
    - Entrainer tous les classifieurs que j'ai choisis avec différents paramètres
		- classifierswith_undersampler.py -> c'est ici que j'ai créé le meilleur modèle
		- classifierswith_oversampl_smote_onehot.py
		- classifiers_withpreprocessing_ohe_scaler.py
		- classifiers_withpreprocessing_ohe_scaler_smote.py
    - Charger le meilleur modèle en exécutant le fichier meilleurModele.py
		- Si vous voulez charger un autre modèle il suffir de changer le nom du fichier appelé dans le code du fichier meilleurModele.py