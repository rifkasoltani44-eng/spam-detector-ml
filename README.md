# Détecteur de Spam SMS — Machine Learning avec Naive Bayes

Mon premier projet de Machine Learning : un modèle qui apprend à distinguer un SMS normal ("ham") d'un SMS indésirable ("spam"), à partir de 5572 messages réels déjà classés.

## Pourquoi ce projet
L'objectif n'était pas seulement d'obtenir un modèle qui fonctionne, mais de comprendre en profondeur chaque étape d'un pipeline de Machine Learning supervisé appliqué au texte : de la donnée brute jusqu'à l'évaluation critique des résultats.

## Données utilisées
Le dataset "SMS Spam Collection" : 5572 SMS réels, étiquetés par des humains en amont.
Répartition : 87% de messages normaux (ham), 13% de spam — un déséquilibre à prendre en compte pour bien évaluer le modèle.

## Méthodologie
1. **Exploration** — vérification de la répartition des classes pour anticiper les biais d'évaluation
2. **Division des données** — 80% pour l'entraînement, 20% mis de côté pour un test honnête sur des messages jamais vus
3. **Vectorisation du texte** — transformation des messages en représentations numériques via l'approche Bag of Words (`CountVectorizer`)
4. **Entraînement** — modèle Naive Bayes Multinomial, adapté aux données de comptage de mots
5. **Évaluation** — mesure de la performance sur le jeu de test, avec une attention particulière portée au recall de la classe minoritaire (spam)

## Résultats obtenus
- **Précision globale : 99,2%**
- **Recall sur la classe spam : 94%** — ce chiffre confirme que le modèle détecte réellement les spams, et ne se contente pas d'un score flatteur dû au déséquilibre des classes

## Limites et axes d'amélioration
- Comparer les performances avec d'autres algorithmes (régression logistique, SVM)
- Remplacer le comptage brut de mots par une pondération TF-IDF
- Étendre le modèle à des messages multilingues

## Stack technique
Python · pandas · scikit-learn · Google Colab
