# Plan de test — AI-TestGuard

Qualification d'un modèle de détection de fraude bancaire (dataset Kaggle *Credit Card Fraud Detection*).

**Auteur :** Mohamed KABORE — **Version :** 1.0 — **Statut :** validé avant toute évaluation sur le jeu de test

---

## 1. Objet et périmètre

Le plan de test définit ce qui est testé, pourquoi, comment et selon quels critères le modèle est accepté ou rejeté. Il est rédigé **avant** l'entraînement et l'évaluation : les critères d'acceptation ne sont pas ajustés après observation du jeu de test.

| Composant | Rôle | Module |
|---|---|---|
| Données | Chargement et contrôle du schéma (31 colonnes, types, cible binaire) | `app/preprocessing.py` |
| Prétraitement | Contrôles qualité, séparation train / validation / test, normalisation | `app/preprocessing.py` |
| Modèle | Pipeline scikit-learn (StandardScaler + classifieur), sauvegarde et rechargement | `app/training.py` |
| Prédiction | Contrôle des entrées, probabilités, application du seuil de décision | `app/prediction.py` |
| Évaluation | Métriques, matrice de confusion, choix du seuil sur la validation | `app/evaluation.py` |
| Intégration | Enchaînement complet et automatisation (GitHub Actions) | `tests/`, `.github/workflows/tests.yml` |

**Hors périmètre :** API HTTP déployée, données client réelles, équité selon des attributs démographiques (absents du jeu de données, voir § 7).

## 2. Choix méthodologiques fixés à l'avance

| Point | Décision | Justification |
|---|---|---|
| Doublons | Les 1 081 lignes strictement dupliquées sont supprimées **avant** la séparation | Une même transaction ne doit pas se trouver à la fois dans l'entraînement et le test (fuite) |
| Séparation | 60 % entraînement / 20 % validation / 20 % test, **stratifiée** sur `Class`, graine 42 | Conserver le taux de fraude (≈ 0,17 %) dans chaque jeu |
| Transformations | `StandardScaler` ajusté **uniquement** sur l'entraînement, dans un `Pipeline` | Éviter toute fuite d'information du test |
| Déséquilibre | `class_weight="balanced"` | Éviter un modèle qui prédit toujours « normale » |
| Modèles comparés | Régression logistique, arbre de décision | Modèles légers demandés par le sujet |
| Choix du modèle et du seuil | Sur la **validation** uniquement (voir règle § 5) | Le test reste réservé à l'évaluation finale |
| Graines | `random_state=42` partout | Reproductibilité |
| Variables V1–V28 | Non interprétées comme des caractéristiques métier | Composantes PCA anonymisées |

## 3. Exigences, risques et cas de test

| ID | Exigence | Type de test | Risque maîtrisé | Résultat attendu | Fichier de test |
|---|---|---|---|---|---|
| T01 | Les colonnes nécessaires sont présentes | Données | Pipeline appliqué à un fichier incomplet | Aucune colonne indispensable manquante, sinon rejet explicite | `test_preprocessing.py` |
| T02 | Les données respectent les types attendus | Unitaire | Calculs faux sur des colonnes texte | Types numériques conformes au contrat, sinon rejet | `test_preprocessing.py` |
| T03 | Le prétraitement ne produit pas de valeurs non finies | Unitaire | Probabilités ou métriques invalides | Aucune valeur NaN ou infinie en sortie | `test_preprocessing.py` |
| T04 | Le modèle produit une probabilité valide | Fonctionnel ML | Décision prise sur une valeur incohérente | Probabilité finie, comprise entre 0 et 1 | `test_prediction.py` |
| T05 | Les performances sont évaluées sur des données indépendantes | ML | Performances surestimées | Jeu de test non utilisé pour le réglage | `test_integration.py` |
| T06 | Les résultats sont reproductibles | Non-régression | Résultats impossibles à vérifier | Deux exécutions identiques : écart de probabilité ≤ 1e-9 | `test_regression.py` |
| T07 | Les tests s'exécutent sur GitHub | CI | Régression non détectée | Workflow réussi, ou échec correctement signalé | `.github/workflows/tests.yml` |
| **T08** | La cible ne contient que les valeurs 0 et 1 | Données | Labels corrompus faussant l'entraînement et les métriques | Toute autre valeur est rejetée explicitement | `test_preprocessing.py` |
| **T09** | Les entrées invalides sont rejetées avec un message explicite (données vides, colonne dupliquée, valeur manquante ou infinie, montant négatif) | Unitaire | Prédiction silencieuse sur des données corrompues | Exception `ValueError` avec un message identifiant le problème | `test_preprocessing.py`, `test_prediction.py` |
| **T10** | Aucune fuite de données entre les jeux | Intégration | Performances surestimées en production | Scaler ajusté sur l'entraînement seul ; aucune transaction commune entre entraînement et test | `test_integration.py` |
| **T11** | La stratification conserve le taux de fraude | Données | Jeu de validation ou de test sans fraude, métriques instables | Taux de fraude de chaque jeu égal au taux global à ± 0,02 point près | `test_integration.py` |
| **T12** | Le seuil de décision est choisi sur la validation selon une règle documentée | ML | Seuil arbitraire ou réglé sur le test | Seuil reproductible, calculé sans accès au test | `test_integration.py` |
| **T13** | Le modèle atteint les critères d'acceptation sur le test (§ 5) | ML | Mise en pilote d'un modèle insuffisant | Tous les critères du § 5 respectés | rapport d'évaluation |
| **T14** | Le pipeline sauvegardé puis rechargé donne les mêmes prédictions | Intégration | Comportement différent entre développement et exploitation | Écart de probabilité ≤ 1e-9, classes identiques | `test_integration.py` |
| **T15** | Le système réagit de façon contrôlée aux entrées perturbées | Robustesse | Comportement imprévisible en exploitation | Rejet, traitement ou alerte conformément au § 6 ; sensibilité mesurée | `test_robustness.py` |
| **T16** | Une nouvelle version ne dégrade pas les performances de la référence | Non-régression | Régression silencieuse après une modification | Baisse de PR-AUC ≤ 0,02 et de rappel ≤ 0,03 par rapport à la référence figée | `test_regression.py` |
| **T17** | Le temps d'inférence et les ressources respectent les limites fixées | Performance | Délais incompatibles avec l'autorisation de paiement | Critères du § 5 (latence P95, débit, taille, mémoire) | `test_performance.py` |
| **T18** | Aucune donnée sensible ni secret n'est exposé | Sécurité | Fuite de données ou d'identifiants | Aucun secret dans le dépôt ; journaux sans valeurs brutes de transaction ; champs inconnus rejetés | `test_prediction.py`, CI |
| **T19** | Le code respecte les règles de qualité | CI | Code difficile à maintenir, défauts non testés | Ruff sans erreur ; couverture de `app/` ≥ 80 % | `.github/workflows/tests.yml` |

## 4. Contrat des données d'entrée

| Élément | Règle | Comportement si non respecté |
|---|---|---|
| Colonnes | `Time`, `V1`…`V28`, `Amount` (+ `Class` pour l'entraînement) | Rejet |
| Colonnes dupliquées ou inconnues | Interdites | Rejet |
| Types | Numériques | Rejet |
| Valeurs manquantes ou infinies | Interdites (le jeu d'entraînement n'en contient pas : aucune règle d'imputation validée) | Rejet |
| `Amount` | ≥ 0 | Rejet si négatif |
| `Class` | 0 ou 1 | Rejet |
| Données vides | Interdites | Rejet |
| `Amount` très élevé (au-delà du maximum observé à l'entraînement) | Accepté | Traitement + alerte (journalisée) |
| Distribution différente de l'entraînement | Acceptée | Traitement + alerte de dérive |

## 5. Critères d'acceptation et de rejet

Les critères sont fixés à partir des exigences métier, avant toute évaluation sur le jeu de test.

**Contexte métier :** un faux négatif (fraude non détectée) entraîne une perte financière directe ; un faux positif (transaction légitime bloquée) dégrade l'expérience client et occupe les analystes. Le rappel est donc prioritaire, mais le volume de fausses alertes doit rester supportable.

| Critère | Seuil | Justification |
|---|---|---|
| Rappel (fraudes détectées) | ≥ 0,75 | Détecter au moins trois fraudes sur quatre |
| Précision | ≥ 0,30 | Au plus environ deux fausses alertes par fraude réelle : charge d'analyse supportable |
| Taux de faux positifs (FPR) | ≤ 0,1 % | Au plus une transaction légitime bloquée sur 1 000 |
| PR-AUC (Average Precision) | ≥ 0,70 | Indicateur adapté à une classe rare (un classifieur aléatoire obtient ≈ 0,0017) |
| ROC-AUC | ≥ 0,95 | Bonne séparation des classes, à interpréter avec prudence vu le déséquilibre |
| Latence P95 d'une prédiction isolée | ≤ 50 ms | Fraction réduite du délai d'autorisation d'un paiement par carte |
| Débit par lot | ≥ 1 000 prédictions / s | Traitement de lots de rejeu ou de contrôle a posteriori |
| Taille du modèle sauvegardé | ≤ 10 Mo | Déploiement léger |
| Mémoire supplémentaire à l'inférence | ≤ 200 Mo | Exécution sur un serveur partagé |

**Règle de choix du seuil (sur la validation) :** le seuil retenu est celui qui **maximise la précision tout en garantissant un rappel ≥ 0,80** sur la validation. La marge de 0,05 au-dessus du critère d'acceptation (0,75) absorbe une partie de la variabilité entre validation et test, qui ne contiennent chacun qu'environ 95 fraudes.

**Règle de choix du modèle :** le modèle retenu est celui qui obtient la meilleure PR-AUC sur la validation.

**Décision :**
- **Accepté pour un pilote** : tous les critères respectés et aucune anomalie bloquante ouverte.
- **À investiguer** : un critère non respecté de peu ou une anomalie majeure.
- **Rejeté** : rappel < 0,75, ou anomalie bloquante (fuite de données, prédictions non reproductibles).

L'acceptation technique ne vaut pas autorisation de mise en production : celle-ci exige aussi des validations métier, opérationnelle, de sécurité et réglementaire.

## 6. Scénarios de robustesse

| Scénario | Comportement attendu | Mesure |
|---|---|---|
| Valeurs manquantes | Rejet | Exception explicite |
| Valeurs infinies ou non numériques | Rejet | Exception explicite |
| `Amount` extrêmement élevé | Traitement + alerte | Probabilités valides, alerte journalisée |
| Perturbation limitée de V1–V28 (bruit proportionnel à l'écart-type de chaque variable) | Traitement | Variation des probabilités, changements de classe, rappel, précision, FPR |
| Perturbation de `Amount` (± quelques %) | Traitement | Idem |
| Distribution décalée | Traitement + alerte de dérive | Part des valeurs hors de la plage d'entraînement |
| Schéma invalide | Rejet | Exception explicite |

## 7. Limites connues

- Pas d'attribut démographique (âge, sexe, localisation) : l'équité ne peut pas être évaluée sur ce jeu.
- Données de septembre 2013, sur deux jours : elles ne représentent pas les fraudes actuelles.
- Environ 95 fraudes dans le jeu de test : les métriques de la classe fraude ont une incertitude importante, mesurée par intervalle de confiance (bootstrap).
- Séparation aléatoire et non temporelle, conformément au sujet : une validation chronologique serait plus proche de l'exploitation réelle.

## 8. Outils et preuves conservées

| Outil | Usage | Preuve conservée |
|---|---|---|
| Pytest, pytest-cov | Tests unitaires, d'intégration, de robustesse, de non-régression | Rapports d'exécution et de couverture (`reports/`) |
| Hypothesis | Génération de cas de test | Tests de propriétés |
| scikit-learn, joblib | Pipeline, sauvegarde | Modèle et métadonnées |
| matplotlib | Courbes ROC, précision-rappel, matrices de confusion | Figures (`reports/`) |
| time.perf_counter, psutil | Latence, débit, mémoire | Rapport de performance |
| Ruff | Qualité du code | Journal de la CI |
| GitHub Actions | Automatisation | Exécutions réussie et en échec volontaire |
