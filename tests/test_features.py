"""Tests du pipeline de préparation des données."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    CLASS_MAPPING,
    CLASS_NAMES,
    add_date_features,
    add_geometry_features,
    align_columns,
    build_features,
    decode_target,
    drop_missing_rows,
    encode_categoricals,
    encode_target,
    reduce_memory_usage,
)


class FakeGeometry:
    """Substitut minimal d'un polygone shapely, pour tester sans geopandas."""

    def __init__(self, area: float, length: float):
        self.area = area
        self.length = length


def test_drop_missing_rows_aligne_la_cible():
    features = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    target = pd.Series(["x", "y", "z"])
    kept_features, kept_target = drop_missing_rows(features, target)
    assert len(kept_features) == 2
    assert list(kept_target) == ["x", "z"]


def test_drop_missing_rows_sans_cible():
    features = pd.DataFrame({"a": [1.0, np.nan]})
    kept, target = drop_missing_rows(features)
    assert len(kept) == 1 and target is None


def test_add_date_features_extrait_les_composantes():
    frame = pd.DataFrame({"date0": ["01-01-2020", "15-06-2021"]})
    result = add_date_features(frame)

    assert "date0" not in result, "la colonne brute doit être remplacée"
    for suffix in ("year", "month", "day", "weekday", "elapsed"):
        assert f"date0_{suffix}" in result

    assert result["date0_year"].tolist() == [2020, 2021]
    assert result["date0_month"].tolist() == [1, 6]


def test_elapsed_compte_depuis_la_premiere_date():
    frame = pd.DataFrame({"date0": ["01-01-2020", "11-01-2020"]})
    result = add_date_features(frame)
    assert result["date0_elapsed"].tolist() == [0, 10]


def test_add_geometry_features():
    frame = pd.DataFrame({"geometry": [FakeGeometry(100.0, 40.0)]})
    result = add_geometry_features(frame)

    assert "geometry" not in result
    assert result["area"].iloc[0] == 100.0
    assert result["perimeter"].iloc[0] == 40.0


def test_compactness_maximale_pour_un_disque():
    """La compacité vaut 1 pour un disque et moins pour toute autre forme."""
    radius = 3.0
    disc = FakeGeometry(np.pi * radius**2, 2 * np.pi * radius)
    elongated = FakeGeometry(10.0, 100.0)

    result = add_geometry_features(pd.DataFrame({"geometry": [disc, elongated]}))
    assert result["compactness"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert result["compactness"].iloc[1] < 0.1


def test_compactness_nulle_si_perimetre_nul():
    result = add_geometry_features(pd.DataFrame({"geometry": [FakeGeometry(0.0, 0.0)]}))
    assert result["compactness"].iloc[0] == 0.0


def test_encode_categoricals_cree_des_indicatrices():
    frame = pd.DataFrame({"change_status_date0": ["A", "B", "A"]})
    result = encode_categoricals(frame)
    assert "change_status_date0" not in result
    assert result.shape[1] == 2


def test_encode_categoricals_ignore_colonnes_absentes():
    frame = pd.DataFrame({"autre": [1, 2]})
    assert encode_categoricals(frame).equals(frame)


def test_reduce_memory_usage():
    frame = pd.DataFrame({"i": np.array([1, 2], dtype="int64"),
                          "f": np.array([1.0, 2.0], dtype="float64")})
    result = reduce_memory_usage(frame)
    assert result["i"].dtype == "int32"
    assert result["f"].dtype == "float32"


def test_align_columns_complete_les_manquantes():
    train = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    test = pd.DataFrame({"a": [1], "c": [3], "d": [9]})
    aligned_train, aligned_test = align_columns(train, test)

    assert list(aligned_test.columns) == list(aligned_train.columns)
    assert aligned_test["b"].iloc[0] == 0, "modalité absente du test -> zéro"
    assert "d" not in aligned_test, "colonne surnuméraire supprimée"


def test_encode_decode_target_sont_inverses():
    target = pd.Series(CLASS_NAMES)
    encoded = encode_target(target)
    assert encoded.tolist() == list(range(len(CLASS_NAMES)))
    assert decode_target(encoded) == CLASS_NAMES


def test_class_mapping_couvre_les_six_classes():
    assert len(CLASS_MAPPING) == 6
    assert set(CLASS_MAPPING.values()) == set(range(6))


def test_build_features_bout_en_bout():
    frame = pd.DataFrame(
        {
            "geometry": [FakeGeometry(100.0, 40.0), FakeGeometry(200.0, 60.0)],
            "date0": ["01-01-2020", "11-01-2020"],
            "change_status_date0": ["A", "B"],
            "geography_type": ["x", "y"],
            "urban_type": ["u", "v"],
        }
    )
    result = build_features(frame)

    for dropped in ("geometry", "date0", "geography_type", "urban_type"):
        assert dropped not in result
    assert "area" in result and "date0_year" in result
    assert len(result) == 2
    # Toutes les colonnes doivent être numériques pour les modèles d'arbres.
    assert all(np.issubdtype(dtype, np.number) or dtype == bool
               for dtype in result.dtypes)
