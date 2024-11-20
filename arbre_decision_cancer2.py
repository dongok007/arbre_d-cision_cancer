import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Charger les données
data_cancer = pd.read_csv('Cancer_Data.csv')

# Aperçu des données
print(data_cancer.head())

# Diagramme à barre de la variable 'diagnosis'
sns.countplot(x='diagnosis', data=data_cancer).set_title("Distribution de la variable 'diagnosis'")
plt.show()

# Suppression des variables 'id' et 'Unnamed: 32'
data_cancer.drop(columns=['id', 'Unnamed: 32'], inplace=True)

# Conversion de la variable cible en binaire (M = 1, B = 0)
data_cancer['diagnosis'] = data_cancer['diagnosis'].map({'M': 1, 'B': 0})

# Variable cible
y = data_cancer['diagnosis']

# Variables indépendantes
X = data_cancer.drop('diagnosis', axis=1)

# Séparation des données en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

# Affichage des tailles des jeux de données
print("Shape des données d'entraînement X:", X_train.shape)
print("Shape des données d'entraînement y:", y_train.shape)
print("Shape des données de test X:", X_test.shape)
print("Shape des données de test y:", y_test.shape)

# Création du modèle d'arbre de décision
tree_model = DecisionTreeClassifier(random_state=42)

# Entraînement du modèle
tree_model.fit(X_train, y_train)

# Précision sur les données d'entraînement
train_score = tree_model.score(X_train, y_train)
print(f"Précision sur les données d'entraînement: {train_score:.2f}")

# Précision sur les données de test
test_score = tree_model.score(X_test, y_test)
print(f"Précision sur les données de test: {test_score:.2f}")

# Visualisation de l'arbre de décision avec plot_tree
plt.figure(figsize=(36, 24))  # Ajuster la taille de l'arbre pour le rendre lisible
plot_tree(tree_model, feature_names=X.columns, class_names=['B', 'M'], filled=True, fontsize=10)
plt.title("Arbre de décision")
plt.show()
