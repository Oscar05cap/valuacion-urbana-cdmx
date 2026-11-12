import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.extract.listings import (
    clean_and_validate_listings,
    detect_bank_foreclosure,
    generate_bootstrap_dataset,
)
from src.features.geospatial_inegi import (
    build_bootstrap_ageb_grid,
    clean_inegi_census_table,
    enrich_with_geospatial_features,
)
from src.features.vision_street import generate_bootstrap_vision_features
from src.schemas import ListingsCleanSchema, ListingsGeoSchema, ListingsMultimodalSchema


def test_regex_remates_bancarios():
    """Verifica que el filtro atrape litigios e ignore anuncios legítimos."""
    assert detect_bank_foreclosure("Gran oportunidad remate bancario en Del Valle") is True
    assert detect_bank_foreclosure("Departamento en venta, cesión de derechos, solo contado") is True
    assert detect_bank_foreclosure("No créditos, adjudicación directa en Narvarte") is True
    assert detect_bank_foreclosure("Hermoso departamento iluminado en Coyoacán, se aceptan créditos") is False


def test_limpieza_censo_inegi_asteriscos_y_ceros():
    """Verifica que CVEGEO conserve el '09' inicial y que los '*' se imputen sin romper tipos."""
    df_dirty = pd.DataFrame([
        {"CVEGEO": "90140001001A", "POBTOT": "1200", "TVIVPARHAB": "400", "GRAPROES": "14.5", "VPH_AUTOM": "300", "VPH_INTER": "380"},
        {"CVEGEO": "90140001002B", "POBTOT": "*", "TVIVPARHAB": "400", "GRAPROES": "*", "VPH_AUTOM": "N/D", "VPH_INTER": "350"},
    ])
    df_clean = clean_inegi_census_table(df_dirty)

    # 1. El cero inicial debe haberse restaurado a 13 caracteres
    assert df_clean.loc[0, "CVEGEO"] == "090140001001A"
    assert df_clean["CVEGEO"].str.len().eq(13).all()

    # 2. Los asteriscos deben haberse imputado con la mediana de la alcaldía (14.5), no con 0 ni NaN
    assert df_clean["escolaridad_prom"].isna().sum() == 0
    assert df_clean.loc[1, "escolaridad_prom"] == pytest.approx(14.5)


def test_contratos_pandera_end_to_end():
    """Ejecuta una muestra por los Módulos 1, 2 y 3 y valida los contratos de datos con Pandera."""
    df_raw = generate_bootstrap_dataset(n_samples=250, seed=123)
    df_clean = clean_and_validate_listings(df_raw)
    ListingsCleanSchema.validate(df_clean)

    gdf_ageb = build_bootstrap_ageb_grid(df_clean, seed=123)
    df_geo = enrich_with_geospatial_features(df_clean, gdf_ageb)
    ListingsGeoSchema.validate(df_geo)

    df_multi = generate_bootstrap_vision_features(df_geo, seed=123)
    ListingsMultimodalSchema.validate(df_multi)


def test_ausencia_de_fuga_espacial_en_groupkfold():
    """Garantiza que ninguna AGEB (CVEGEO) de entrenamiento aparezca en validación."""
    df_raw = generate_bootstrap_dataset(n_samples=200, seed=99)
    df_clean = clean_and_validate_listings(df_raw)
    gdf_ageb = build_bootstrap_ageb_grid(df_clean, seed=99)
    df_geo = enrich_with_geospatial_features(df_clean, gdf_ageb)

    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(df_geo, df_geo["precio_m2"], df_geo["CVEGEO"]):
        agebs_train = set(df_geo.iloc[train_idx]["CVEGEO"])
        agebs_val = set(df_geo.iloc[val_idx]["CVEGEO"])
        assert agebs_train.isdisjoint(agebs_val), "Fuga de datos espacial detectada: AGEB compartida entre train y val"