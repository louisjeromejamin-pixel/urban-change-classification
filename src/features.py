"""Construction des variables explicatives à partir des polygones bruts.

Le jeu de données décrit des parcelles urbaines observées à cinq dates. Chaque
ligne porte une géométrie, un statut de chantier par date, et des descripteurs
urbains et géographiques. La cible est le type de changement observé.

Ce module rassemble le pipeline de préparation qui était dispersé dans le
notebook, afin de pouvoir le rejouer sur de nouvelles données sans réexécuter
toutes les cellules.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Les six classes du challenge, dans l'ordre attendu pour la soumission.
CLASS_NAMES = [
    "Demolition",
    "Road",
    "Residential",
    "Commercial",
    "Industrial",
    "Mega Projects",
]
CLASS_MAPPING = {name: index for index, name in enumerate(CLASS_NAMES)}

DATE_COLUMNS = [f"date{i}" for i in range(5)]
STATUS_COLUMNS = [f"change_status_date{i}" for i in range(5)]


def drop_missing_rows(
    features: pd.DataFrame, target: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Retire les lignes incomplètes.

    Les valeurs manquantes sont rares dans ce jeu de données ; les supprimer
    coûte moins qu'une imputation qui introduirait du bruit. La cible est
    filtrée en parallèle pour conserver l'alignement des index.
    """
    mask = features.notna().all(axis=1)
    if target is None:
        return features[mask], None
    return features[mask], target[mask]


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Décompose chaque date en variables numériques exploitables.

    Un modèle à base d'arbres ne sait rien faire d'un horodatage brut. On en
    extrait donc deux informations de nature différente :

    - la position absolue dans le temps (année, mois, jour, jour de semaine),
      qui distingue une construction ancienne d'une récente ;
    - le délai écoulé depuis la première observation, qui mesure la *vitesse*
      de transformation de la parcelle — un lotissement et un mégaprojet ne
      progressent pas au même rythme.
    """
    frame = frame.copy()

    for column in DATE_COLUMNS:
        if column not in frame:
            continue
        parsed = pd.to_datetime(frame[column], format="%d-%m-%Y", errors="coerce")
        frame[f"{column}_year"] = parsed.dt.year
        frame[f"{column}_month"] = parsed.dt.month
        frame[f"{column}_day"] = parsed.dt.day
        frame[f"{column}_weekday"] = parsed.dt.weekday
        frame[f"{column}_elapsed"] = (parsed - parsed.min()).dt.days
        frame = frame.drop(columns=[column])

    return frame


def add_geometry_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Remplace la géométrie par sa surface et son périmètre.

    L'échelle d'une parcelle sépare assez bien les classes : les emprises
    étendues correspondent aux zones industrielles et aux mégaprojets, les
    petites au résidentiel. On ajoute la compacité — le rapport entre surface
    et périmètre au carré — qui distingue une parcelle ramassée d'un tracé
    allongé, typique des routes.
    """
    if "geometry" not in frame:
        return frame

    frame = frame.copy()
    frame["area"] = frame["geometry"].apply(lambda geom: geom.area)
    frame["perimeter"] = frame["geometry"].apply(lambda geom: geom.length)

    perimeter_squared = frame["perimeter"] ** 2
    frame["compactness"] = np.where(
        perimeter_squared > 0,
        4 * np.pi * frame["area"] / perimeter_squared,
        0.0,
    )

    return frame.drop(columns=["geometry"])


def encode_categoricals(
    frame: pd.DataFrame, columns: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Encode les statuts de chantier en indicatrices.

    ``pd.get_dummies`` est préféré à ``OneHotEncoder`` ici : sur ce jeu de
    données, l'encodeur complet produisait un espace très creux dont la
    plupart des colonnes n'apportaient rien, alors que les statuts n'ont
    qu'un petit nombre de modalités.
    """
    columns = list(columns) if columns is not None else STATUS_COLUMNS
    present = [c for c in columns if c in frame]
    if not present:
        return frame
    return pd.get_dummies(frame, columns=present)


def reduce_memory_usage(frame: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes 64 bits en 32 bits.

    Le jeu de données est volumineux et la recherche d'hyperparamètres
    entraîne des centaines de modèles ; diviser l'empreinte mémoire par deux
    évite les échecs d'allocation sans perte de précision utile.
    """
    frame = frame.copy()
    for column in frame.columns:
        dtype = frame[column].dtype
        if dtype == "int64":
            frame[column] = frame[column].astype("int32")
        elif dtype == "float64":
            frame[column] = frame[column].astype("float32")
    return frame


def align_columns(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Donne aux deux tables les mêmes colonnes, dans le même ordre.

    L'encodage par indicatrices est calculé séparément sur chaque table :
    une modalité absente du test crée un décalage de colonnes qui fait
    échouer la prédiction. On complète donc les manquantes par des zéros.
    """
    missing_in_test = set(train.columns) - set(test.columns)
    for column in missing_in_test:
        test[column] = 0

    extra_in_test = set(test.columns) - set(train.columns)
    test = test.drop(columns=list(extra_in_test))

    return train, test[train.columns]


def build_features(
    frame: pd.DataFrame, drop_columns: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Applique la chaîne complète de préparation.

    Args:
        frame: Table brute lue depuis le GeoJSON.
        drop_columns: Colonnes à écarter. Par défaut ``geography_type`` et
            ``urban_type``, dont l'encodage produisait de nombreuses
            indicatrices d'importance négligeable.

    Returns:
        La table prête pour l'entraînement.
    """
    drop_columns = (
        list(drop_columns)
        if drop_columns is not None
        else ["geography_type", "urban_type"]
    )

    frame = add_geometry_features(frame)
    frame = add_date_features(frame)
    frame = encode_categoricals(frame)
    frame = frame.drop(columns=[c for c in drop_columns if c in frame])

    return reduce_memory_usage(frame)


def encode_target(target: pd.Series) -> pd.Series:
    """Convertit les libellés de classe en entiers, comme l'attend XGBoost."""
    return target.map(CLASS_MAPPING)


def decode_target(encoded: Iterable[int]) -> List[str]:
    """Opération inverse de :func:`encode_target`."""
    return [CLASS_NAMES[i] for i in encoded]
