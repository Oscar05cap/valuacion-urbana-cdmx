from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Definición de bloques de variables para el estudio de ablación
FEATURES_BASE = [
    "m2_construidos",
    "recamaras",
    "banos",
    "estacionamientos",
    "es_departamento",
    "alcaldia_cat",
]

FEATURES_GEO = [
    "escolaridad_prom",
    "pct_viv_auto",
    "pct_viv_internet",
    "densidad_pob_km2",
    "densidad_comercial_denue",
    "dist_transporte_m",
]

FEATURES_VISION_INTERPRETABLE = [
    "pct_vegetacion",
    "pct_banqueta",
    "pct_edificio",
    "pct_cielo",
    "clip_urban_score",
]

CLIP_RAW_COLS = [f"clip_emb_{i}" for i in range(8)]


def prepare_base_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Codifica variables categóricas de forma determinista para LightGBM y SHAP"""
    df = df.copy()
    df["es_departamento"] = (df["tipo_propiedad"] == "departamento").astype(int)

    alcaldias_orden = ["Iztapalapa", "Coyoacán", "Benito Juárez", "Cuauhtémoc"]
    alcaldia_map = {name: idx for idx, name in enumerate(alcaldias_orden)}
    df["alcaldia_cat"] = df["alcaldia"].map(alcaldia_map).fillna(0).astype(int)

    return df, alcaldia_map


def evaluate_ablation_stage(
    df: pd.DataFrame,
    feature_cols: list[str],
    use_clip_pca: bool = False,
    n_splits: int = 5,
) -> tuple[dict, np.ndarray]:
    """
    Evalúa un conjunto de variables usando GroupKFold espacial por CVEGEO (AGEB).
    Si use_clip_pca=True, ajusta StandardScaler + PCA exclusivamente en X_train de cada fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["CVEGEO"].values
    y = df["precio_m2"].values
    oof_preds = np.zeros(len(df))

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 23,
        "max_depth": 5,
        "min_child_samples": 25,
        "subsample": 0.8,
        "colsample_bytree": 0.85,
        "n_estimators": 350,
        "random_state": 42,
        "verbose": -1,
    }

    for train_idx, val_idx in gkf.split(df, y, groups):
        X_train = df.iloc[train_idx][feature_cols].copy()
        X_val = df.iloc[val_idx][feature_cols].copy()
        y_train, y_val = y[train_idx], y[val_idx]

        # Reducción dimensional de embeddings 
        if use_clip_pca:
            scaler = StandardScaler()
            pca = PCA(n_components=3, random_state=42)

            emb_train = scaler.fit_transform(df.iloc[train_idx][CLIP_RAW_COLS])
            emb_val = scaler.transform(df.iloc[val_idx][CLIP_RAW_COLS])

            pca_train = pca.fit_transform(emb_train)
            pca_val = pca.transform(emb_val)

            for i in range(3):
                X_train[f"clip_pca_{i}"] = pca_train[:, i]
                X_val[f"clip_pca_{i}"] = pca_val[:, i]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        oof_preds[val_idx] = model.predict(X_val)

    metrics = {
        "RMSE_MXN_m2": round(float(root_mean_squared_error(y, oof_preds)), 1),
        "MAPE_pct": round(float(mean_absolute_percentage_error(y, oof_preds) * 100), 2),
        "R2_espacial": round(float(r2_score(y, oof_preds)), 4),
    }
    return metrics, oof_preds


def train_final_production_bundle(df: pd.DataFrame, alcaldia_map: dict) -> dict:
    """
    Entrena el modelo multimodal final; media más cuantiles P15 y P85 para intervalos de confianza
    y calcula el explicador SHAP para producción
    """
    feature_cols = FEATURES_BASE + FEATURES_GEO + FEATURES_VISION_INTERPRETABLE
    X = df[feature_cols].copy()
    y = df["precio_m2"].values

    scaler = StandardScaler()
    pca = PCA(n_components=3, random_state=42)
    pca_feats = pca.fit_transform(scaler.fit_transform(df[CLIP_RAW_COLS]))
    for i in range(3):
        X[f"clip_pca_{i}"] = pca_feats[:, i]

    all_features = list(X.columns)

    # Modelo central 
    model_mean = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.04,
        num_leaves=23,
        max_depth=5,
        min_child_samples=25,
        n_estimators=220,
        random_state=42,
        verbose=-1,
    )
    model_mean.fit(X, y)

    # Modelos cuantílicos 
    model_p15 = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.15,
        learning_rate=0.04,
        num_leaves=19,
        max_depth=5,
        n_estimators=180,
        random_state=42,
        verbose=-1,
    )
    model_p85 = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.85,
        learning_rate=0.04,
        num_leaves=19,
        max_depth=5,
        n_estimators=180,
        random_state=42,
        verbose=-1,
    )
    model_p15.fit(X, y)
    model_p85.fit(X, y)

    # Importancia global con SHAP
    explainer = shap.TreeExplainer(model_mean)
    sample_X = X.sample(min(500, len(X)), random_state=42)
    shap_vals = explainer.shap_values(sample_X)
    shap_importance = pd.Series(np.abs(shap_vals).mean(axis=0), index=all_features).sort_values(ascending=False)

    return {
        "model_mean": model_mean,
        "model_p15": model_p15,
        "model_p85": model_p85,
        "scaler_clip": scaler,
        "pca_clip": pca,
        "feature_names": all_features,
        "alcaldia_map": alcaldia_map,
        "shap_importance": shap_importance,
    }


def main():
    in_path = Path("data/processed/listings_multimodal.parquet")
    if not in_path.exists():
        raise FileNotFoundError("Primero ejecuta el Módulo 3: python src/features/vision_street.py --bootstrap")

    df_raw = pd.read_parquet(in_path)
    df, alcaldia_map = prepare_base_dataframe(df_raw)

    print("[Ablación] Evaluando 3 arquitecturas con GroupKFold Espacial por AGEB (CVEGEO)...")

    # EXP 1: Solo datos del anuncio
    m1, _ = evaluate_ablation_stage(df, FEATURES_BASE, use_clip_pca=False)

    # EXP 2: Anuncio + INEGI (Censo AGEB + DENUE + Metro)
    m2, _ = evaluate_ablation_stage(df, FEATURES_BASE + FEATURES_GEO, use_clip_pca=False)

    # EXP 3: Multimodal Completo (Anuncio + INEGI + SegFormer + CLIP)
    m3, oof_multimodal = evaluate_ablation_stage(
        df, FEATURES_BASE + FEATURES_GEO + FEATURES_VISION_INTERPRETABLE, use_clip_pca=True
    )

    df_ablation = pd.DataFrame([
        {"Experimento": "1. Baseline (Solo Inmueble)", **m1},
        {"Experimento": "2. + Geoespacial (INEGI AGEB + Metro)", **m2},
        {"Experimento": "3. + IA Multimodal (SegFormer + CLIP)", **m3},
    ])

    print("\n=== RESULTADOS DEL ESTUDIO DE ABLACIÓN (Validación Cruzada Espacial) ===")
    print(df_ablation.to_string(index=False))

    # Entrenar paquete de producción (Media + Cuantiles + SHAP)
    print("\n[Producción] Entrenando modelos finales (Media, P15, P85) y calculando valores SHAP...")
    bundle = train_final_production_bundle(df, alcaldia_map)

    print("\nTop 8 Variables con Mayor Impacto en el Precio/m2 (SHAP medio absoluto en MXN):")
    print(bundle["shap_importance"].head(8).round(1).to_string())

    # Guardar predicciones Out-of-Fold y bandas para el tablero interactivo
    X_full = df[FEATURES_BASE + FEATURES_GEO + FEATURES_VISION_INTERPRETABLE].copy()
    pca_full = bundle["pca_clip"].transform(bundle["scaler_clip"].transform(df[CLIP_RAW_COLS]))
    for i in range(3):
        X_full[f"clip_pca_{i}"] = pca_full[:, i]

    df["pred_precio_m2"] = oof_multimodal.round(0)
    df["pred_p15_m2"] = bundle["model_p15"].predict(X_full).round(0)
    df["pred_p85_m2"] = bundle["model_p85"].predict(X_full).round(0)
    df["descuento_pct"] = (((df["pred_precio_m2"] - df["precio_m2"]) / df["pred_precio_m2"]) * 100).round(1)
    df["clasificacion_mercado"] = np.select(
        [df["precio_m2"] < df["pred_p15_m2"], df["precio_m2"] > df["pred_p85_m2"]],
        ["Subvaluado (Oportunidad)", "Sobrevaluado"],
        default="Precio Justo",
    )

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(bundle, models_dir / "lgbm_multimodal.joblib")
    df_ablation.to_csv("data/processed/ablation_metrics.csv", index=False)
    df.to_parquet("data/processed/predictions_oof.parquet", index=False)

    print("\n[Éxito] Modelos guardados en 'models/lgbm_multimodal.joblib' y predicciones en 'data/processed/predictions_oof.parquet'")


if __name__ == "__main__":
    main()