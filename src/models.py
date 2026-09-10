"""Entraînement, sélection de variables et évaluation des modèles."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Grille explorée par la recherche aléatoire. Elle reprend celle du projet
# d'origine, dont les 50 tirages ont représenté environ douze heures de calcul.
XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5, 1],
    "tree_method": ["hist"],
}


def knn_baseline(
    features: pd.DataFrame,
    target: pd.Series,
    n_neighbors: int = 5,
    cv: int = 5,
) -> float:
    """Référence k plus proches voisins.

    Sert d'étalon bas. Les distances euclidiennes perdent leur pouvoir
    discriminant en grande dimension : toutes les parcelles finissent à peu
    près équidistantes, d'où une performance médiocre attendue.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return float(cross_val_score(model, features, target, cv=cv).mean())


def random_forest_baseline(
    features: pd.DataFrame,
    target: pd.Series,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    cv: int = 5,
    random_state: int = 0,
) -> float:
    """Référence forêt aléatoire, insensible à l'échelle des variables."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    return float(cross_val_score(model, features, target, cv=cv).mean())


def select_k_best_curve(
    features: pd.DataFrame,
    target: pd.Series,
    k_values: Sequence[int],
    cv: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Mesure l'exactitude en fonction du nombre de variables retenues.

    Le test du khi-deux exige des valeurs positives ; les variables issues
    des dates et des géométries le sont toutes ici.

    Returns:
        Une table ``k`` / ``accuracy``, pour tracer la courbe de sélection.
    """
    scores = []
    for k in k_values:
        pipeline = Pipeline(
            [
                ("selection", SelectKBest(score_func=chi2, k=k)),
                (
                    "classification",
                    RandomForestClassifier(random_state=random_state, n_jobs=-1),
                ),
            ]
        )
        accuracy = cross_val_score(pipeline, features, target, cv=cv).mean()
        scores.append({"k": k, "accuracy": float(accuracy)})

    return pd.DataFrame(scores)


def tune_xgboost(
    features: pd.DataFrame,
    target: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 0,
    verbose: int = 1,
):
    """Recherche aléatoire d'hyperparamètres pour XGBoost.

    La recherche aléatoire est préférée à une grille exhaustive : à budget
    égal, elle couvre bien mieux les dimensions qui comptent réellement,
    au lieu de dépenser tous les essais sur des paramètres sans effet.

    Returns:
        L'objet ``RandomizedSearchCV`` ajusté.
    """
    from xgboost import XGBClassifier  # import différé : dépendance lourde

    model = XGBClassifier(
        random_state=random_state,
        eval_metric="merror",
        tree_method="hist",
    )

    search = RandomizedSearchCV(
        model,
        param_distributions=XGB_PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=random_state,
        verbose=verbose,
        n_jobs=-1,
    )
    search.fit(features, target)
    return search


def cumulative_importance_features(
    model, feature_names: Sequence[str], threshold: float = 0.95
) -> List[str]:
    """Retient les variables couvrant une part donnée de l'importance totale.

    Sur ce jeu de données, 81 des 108 variables suffisaient à couvrir 95 %
    de l'importance : les 27 restantes allongeaient l'entraînement sans rien
    apporter.
    """
    importances = np.asarray(model.feature_importances_)
    order = importances.argsort()[::-1]

    cumulative = np.cumsum(importances[order])
    keep = int(np.argmax(cumulative >= threshold)) + 1

    return [feature_names[i] for i in order[:keep]]


def evaluate(model, features: pd.DataFrame, target: pd.Series, cv: int = 5) -> Dict:
    """Évalue un modèle par validation croisée."""
    scores = cross_val_score(model, features, target, cv=cv, scoring="accuracy")
    return {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "folds": scores.tolist(),
    }


def make_submission(
    model, features_test: pd.DataFrame, index: Sequence, path: str = "submission.csv"
) -> pd.DataFrame:
    """Écrit le fichier de soumission au format attendu par Kaggle."""
    predictions = model.predict(features_test)
    submission = pd.DataFrame({"Id": index, "Prediction": predictions})
    submission.to_csv(path, index=False)
    return submission
