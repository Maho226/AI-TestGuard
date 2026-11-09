# AI-TestGuard

Qualification d'un modèle de détection de fraude bancaire : stratégie de test, tests automatisés, évaluation sur données déséquilibrées, robustesse, non-régression, performance, sécurité opérationnelle, intégration continue et surveillance.

**Auteur :** Mohamed KABORE — **Module :** Test logiciel (ESMT) — **Enseignant :** Dr. Moustapha DER

[![Tests AI-TestGuard](https://github.com/Maho226/AI-TestGuard/actions/workflows/tests.yml/badge.svg)](https://github.com/Maho226/AI-TestGuard/actions/workflows/tests.yml)

## Résultat de la qualification

| | v1 (référence) | v2 (modèle actif) |
|---|---|---|
| Modèle | Régression logistique, `class_weight="balanced"` | Régression logistique, `log(1 + Amount)`, poids des fraudes = 10 |
| Rappel sur le test | 0,726 | 0,747 |
| Précision sur le test | 0,885 | 0,826 |
| PR-AUC sur le test | 0,674 | 0,703 |
| Décision | **Rejeté** | **Rejeté pour un pilote** (rappel < 0,75), retenu comme base technique |

Le système **réussit les tests techniques** (128 tests, contrats de données, robustesse, performance, sécurité, CI), mais le modèle **n'atteint pas le critère métier de rappel**. Il n'est pas autorisé à poursuivre vers un pilote. Détails : [`docs/rapport_qualification.md`](docs/rapport_qualification.md).

## Structure

```
AI-TestGuard/
├── app/
│   ├── preprocessing.py     contrat des données, chargement, doublons, séparation stratifiée
│   ├── training.py          pipeline, versions v1/v2, réglage sur la validation, sauvegarde, référence
│   ├── prediction.py        prédiction, alertes, mesures de performance, surveillance (PSI)
│   └── evaluation.py        métriques, seuil, intervalles de confiance, robustesse, non-régression
├── tests/                   128 tests Pytest (exigences T01–T19)
│   ├── test_preprocessing.py   qualité des données (36)
│   ├── test_prediction.py      prédiction, métriques, seuil, décision (25)
│   ├── test_integration.py     pipeline de bout en bout, fuites de données (16)
│   ├── test_robustness.py      données perturbées et invalides (17)
│   ├── test_regression.py      reproductibilité et non-régression (13)
│   ├── test_security.py        secrets, fichiers reçus, intégrité du modèle (8)
│   ├── test_monitoring.py      dérive et taux de signalement (7)
│   └── test_performance.py     latence, débit, taille, mémoire (6, marqués « slow »)
├── data/
│   ├── README.md               source, licence, empreinte du jeu complet
│   └── sample/                 échantillon versionné (492 fraudes + 5 000 normales)
├── docs/                    plan de test, rapport de qualification, audit des biais
├── notebooks/               évaluation, robustesse, non-régression, performance, surveillance
├── reports/                 métriques, figures, références figées, rapports d'exécution
├── .github/workflows/tests.yml
├── pyproject.toml           configuration Pytest et Ruff
└── requirements.txt         versions figées (Python 3.13)
```

## Installation

```bash
git clone https://github.com/Maho226/AI-TestGuard.git
cd AI-TestGuard
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Le jeu complet (`creditcard.csv`, Kaggle) n'est pas versionné : le télécharger dans `data/` pour les notebooks. Les tests n'en ont pas besoin, ils utilisent l'échantillon versionné.

## Utilisation

```bash
python -m pytest -m "not slow" --cov=app      # tests rapides (Pull Requests)
python -m pytest -m slow                      # tests de performance
ruff check . && ruff format --check .         # qualité du code

# création MANUELLE d'une référence de non-régression (jamais écrasée sans --force)
python -m app.training --data data/sample/creditcard_sample.csv --version v1 --output reports/baseline_ci.json
```

## Intégration continue

`.github/workflows/tests.yml` :

| Job | Déclenchement | Contenu |
|---|---|---|
| `qualite` | Pull Request, push sur `main` | Ruff (règles et formatage) |
| `tests` | Pull Request, push sur `main` | Pytest (tests rapides), couverture ≥ 80 %, rapports JUnit et couverture en artefacts |
| `performance` | manuel ou chaque lundi | tests marqués `slow` |

La branche `main` est protégée : Pull Request obligatoire et réussite des deux contrôles exigée.

## Méthodologie

- Doublons supprimés avant la séparation ; séparation 60 / 20 / 20 stratifiée, graine 42.
- Transformations ajustées sur l'entraînement uniquement (`Pipeline` scikit-learn).
- Hyperparamètres, seuil et choix du modèle réglés sur la **validation** ; le **test** est réservé à l'évaluation finale.
- Critères d'acceptation fixés avant l'évaluation : [`docs/plan_de_test.md`](docs/plan_de_test.md).
