# Données

## Source
Credit Card Fraud Detection — Kaggle (ULB Machine Learning Group).
Licence : Open Database License (DbCL v1.0).

## Jeu complet (non versionné)
- Fichier : `creditcard.csv`, 284 807 transactions, 31 colonnes, 492 fraudes (0,1727 %)
- SHA-256 : '76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`
- Emplacement attendu : `data/creditcard.csv` (à télécharger depuis Kaggle), exclu par `.gitignore`

## Colonnes
- `Time` : secondes écoulées depuis la première transaction du jeu
- `V1` à `V28` : composantes PCA anonymisées (non interprétables métier)
- `Amount` : montant de la transaction
- `Class` : cible (0 = normale, 1 = fraude)

## Échantillon versionné : `sample/creditcard_sample.csv`
- 492 fraudes + 5 000 transactions normales tirées au hasard (graine 42), soit 5 492 lignes
- **Échantillon enrichi en fraudes (≈ 9 %)** : il sert uniquement aux tests automatisés
  (CI). Les métriques obtenues sur cet échantillon ne représentent pas la performance réelle.
