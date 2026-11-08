# Note d'audit des biais — AI-TestGuard (Partie VII)

**Auteur :** Mohamed KABORE — **Modèle examiné :** v2 (régression logistique, `log(1 + Amount)`)

## 1. Objet

Vérifier si le système présente des disparités de performance entre groupes de clients : rappel, taux de faux positifs et précision par groupe, avec leurs incertitudes.

## 2. Constat : l'analyse d'équité n'est pas réalisable sur ce jeu de données

Le jeu *Credit Card Fraud Detection* ne contient que :

- `Time` : secondes écoulées depuis la première transaction du jeu ;
- `V1` à `V28` : composantes issues d'une analyse en composantes principales (PCA), anonymisées ;
- `Amount` : montant de la transaction ;
- `Class` : fraude ou non.

Il ne contient **aucune caractéristique des clients** (âge, sexe, lieu de résidence, nationalité, revenu, type de carte, canal de paiement).

**Conséquence : aucune conclusion sur l'équité du système n'est possible.** L'absence de résultat ne signifie pas l'absence de biais : un modèle entraîné sur des données historiques peut reproduire des disparités, par exemple au détriment de clients dont les habitudes de paiement sont peu représentées.

## 3. Ce qui n'a volontairement pas été fait

- **Aucun attribut démographique n'a été reconstruit** à partir des composantes V1–V28. Ces composantes sont des combinaisons anonymisées de variables inconnues ; leur attribuer une signification (âge, origine…) serait une invention, non vérifiable et potentiellement discriminante.
- **Aucun groupe n'a été construit à partir du montant** pour en tirer une conclusion d'équité. Une analyse par tranche de montant est une analyse de **performance par segment**, pas d'équité envers des personnes.

## 4. Analyse de segments réellement possible (indicative)

| Segment | Observation | Portée |
|---|---|---|
| Jour 1 / jour 2 (validation) | Rappel 0,789 / 0,865 ; précision 0,882 / 0,727 ; FPR 0,021 % / 0,043 % | Stabilité temporelle, pas équité |
| Montant (analyse des erreurs, Partie III) | Fraudes manquées surtout de petit montant (médiane 2,99 contre 33,59 pour les fraudes détectées) | Performance par segment, pas équité |
| Montant (robustesse, Partie V) | v1 très sensible à l'échelle des montants, corrigé en v2 | Performance par segment, pas équité |

Avec 26 fraudes manquées et 37 à 57 fraudes par jour, ces écarts sont **statistiquement fragiles**.

## 5. Données complémentaires nécessaires

Pour une analyse d'équité réelle, il faudrait, **sous réserve d'une base légale et d'une analyse d'impact sur la protection des données** :

- des attributs de groupe pertinents et légalement utilisables (tranche d'âge, zone géographique, type de client ou de carte, canal : en ligne, terminal, retrait) ;
- un volume suffisant de fraudes **par groupe** pour estimer rappel et FPR avec une incertitude acceptable ;
- des labels fiables et datés (fraudes confirmées, contestations).

## 6. Méthode recommandée lorsque ces données seront disponibles

1. Calculer par groupe le rappel, le taux de faux positifs et la précision, avec intervalles de confiance (bootstrap).
2. Comparer les écarts entre groupes : différence de rappel (*equal opportunity*) et de taux de faux positifs (*equalized odds*).
3. Fixer **avant l'analyse** un écart maximal acceptable, validé avec les métiers, la conformité et le délégué à la protection des données.
4. Outils : **Fairlearn** (`MetricFrame`) ou **AI Fairness 360**.
5. En cas d'écart : analyser les causes (représentation, qualité des labels), envisager un seuil par segment ou un réentraînement, et documenter la décision.

## 7. Recommandations

- Inscrire l'audit d'équité comme **condition préalable à une mise en production**, non à un pilote technique.
- Surveiller dès le pilote le **taux de faux positifs par segment disponible** (canal, type de carte) : un blocage injustifié touche directement le client.
- Conserver la traçabilité des décisions (version du modèle, seuil) pour permettre un audit ultérieur.
