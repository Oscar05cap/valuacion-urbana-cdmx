from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import shap
import streamlit as st

st.set_page_config(
    page_title="Valuador Inmobiliario Multimodal CDMX",
    layout="wide",
)


@st.cache_resource
def load_model_bundle():
    model_path = Path("models/lgbm_multimodal.joblib")
    if not model_path.exists():
        st.error("No se encontró 'models/lgbm_multimodal.joblib'. Ejecuta primero los Módulos 1 al 4.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def load_datasets():
    oof_path = Path("data/processed/predictions_oof.parquet")
    abl_path = Path("data/processed/ablation_metrics.csv")
    if not oof_path.exists() or not abl_path.exists():
        st.error("Faltan archivos en 'data/processed/'. Ejecuta primero el Módulo 4.")
        st.stop()
    return pd.read_parquet(oof_path), pd.read_csv(abl_path)


def lookup_nearest_spatial_context(df: pd.DataFrame, alcaldia: str, lat: float, lon: float) -> pd.Series:
    """
    Simula un Feature Store espacial: recupera las variables del INEGI y de Visión
    de la cuadra/AGEB más cercana a las coordenadas ingresadas.
    """
    sub = df[df["alcaldia"] == alcaldia]
    if sub.empty:
        sub = df
    dist2 = (sub["latitud"] - lat) ** 2 + (sub["longitud"] - lon) ** 2
    nearest_idx = dist2.idxmin()
    return sub.loc[nearest_idx]


def main():
    bundle = load_model_bundle()
    df_oof, df_ablation = load_datasets()

    st.title("Valuador Inmobiliario Multimodal CDMX")

    tab_sim, tab_map, tab_abl = st.tabs([
        "1. Valuador y Explicabilidad SHAP",
        "2. Radar de Inversión (Subvaluados)",
        "3. Evidencia de Ablación Espacial",
    ])

    # PRIMERA PESTAÑA: VALUADOR Y EXPLICABILIDAD 
    with tab_sim:
        col_in, col_out = st.columns([1, 1.35], gap="large")

        with col_in:
            st.subheader("Características del Inmueble")
            alcaldia = st.selectbox(
                "Alcaldía",
                options=["Benito Juárez", "Cuauhtémoc", "Coyoacán", "Iztapalapa"],
                index=0,
            )

            sub_alc = df_oof[df_oof["alcaldia"] == alcaldia]
            lat_default = float(sub_alc["latitud"].median())
            lon_default = float(sub_alc["longitud"].median())

            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitud", value=lat_default, format="%.5f", step=0.002)
            lon = c2.number_input("Longitud", value=lon_default, format="%.5f", step=0.002)

            c3, c4, c5 = st.columns(3)
            m2 = c3.number_input("m² Construidos", min_value=30, max_value=500, value=85, step=5)
            recamaras = c4.selectbox("Recámaras", options=[1, 2, 3, 4], index=1)
            banos = c5.selectbox("Baños", options=[1, 2, 3], index=1)

            c6, c7 = st.columns(2)
            estac = c6.selectbox("Estacionamientos", options=[0, 1, 2, 3], index=1)
            tipo_prop = c7.selectbox("Tipo", options=["departamento", "casa"], index=0)

            precio_oferta_millones = st.number_input(
                "Precio de lista publicado (Millones MXN) para evaluar trato",
                min_value=0.8,
                max_value=35.0,
                value=3.85,
                step=0.1,
            )

            # Recuperar contexto del Feature Store espacial
            ctx = lookup_nearest_spatial_context(df_oof, alcaldia, lat, lon)

            st.markdown("**Ajuste de Entorno Callejero (Extraído por SegFormer / CLIP en la AGEB)**")
            pct_veg = st.slider(
                "Proporción de vegetación / árboles en calle ",
                min_value=0.01,
                max_value=0.55,
                value=float(ctx["pct_vegetacion"]),
                step=0.01,
            )
            clip_score = st.slider(
                "Índice de mantenimiento urbano",
                min_value=-0.95,
                max_value=0.95,
                value=float(ctx["clip_urban_score"]),
                step=0.05,
            )

        with col_out:
            # Construir vector de inferencia respetando el orden exacto del modelo
            raw_clip_vec = np.array([[ctx[f"clip_emb_{i}"] for i in range(8)]])
            pca_clip_vec = bundle["pca_clip"].transform(bundle["scaler_clip"].transform(raw_clip_vec))[0]

            row_dict = {
                "m2_construidos": float(m2),
                "recamaras": int(recamaras),
                "banos": int(banos),
                "estacionamientos": int(estac),
                "es_departamento": 1 if tipo_prop == "departamento" else 0,
                "alcaldia_cat": bundle["alcaldia_map"].get(alcaldia, 0),
                "escolaridad_prom": float(ctx["escolaridad_prom"]),
                "pct_viv_auto": float(ctx["pct_viv_auto"]),
                "pct_viv_internet": float(ctx["pct_viv_internet"]),
                "densidad_pob_km2": float(ctx["densidad_pob_km2"]),
                "densidad_comercial_denue": float(ctx["densidad_comercial_denue"]),
                "dist_transporte_m": float(ctx["dist_transporte_m"]),
                "pct_vegetacion": float(pct_veg),
                "pct_banqueta": float(ctx["pct_banqueta"]),
                "pct_edificio": float(ctx["pct_edificio"]),
                "pct_cielo": float(ctx["pct_cielo"]),
                "clip_urban_score": float(clip_score),
                "clip_pca_0": float(pca_clip_vec[0]),
                "clip_pca_1": float(pca_clip_vec[1]),
                "clip_pca_2": float(pca_clip_vec[2]),
            }

            X_infer = pd.DataFrame([row_dict])[bundle["feature_names"]]

            pred_m2 = float(bundle["model_mean"].predict(X_infer)[0])
            p15_m2 = float(bundle["model_p15"].predict(X_infer)[0])
            p85_m2 = float(bundle["model_p85"].predict(X_infer)[0])

            valor_justo_total = pred_m2 * m2
            oferta_total = precio_oferta_millones * 1_000_000
            oferta_m2 = oferta_total / m2
            diferencia_pct = ((valor_justo_total - oferta_total) / valor_justo_total) * 100

            st.subheader(f"Resultado de Valuación (AGEB: {ctx['CVEGEO']})")
            k1, k2, k3 = st.columns(3)
            k1.metric("Precio Justo Estimado", f"${valor_justo_total:,.0f} MXN", f"${pred_m2:,.0f} / m²")
            k2.metric("Banda Cuantílica (P15 - P85)", f"${p15_m2:,.0f} - ${p85_m2:,.0f} / m²")
            k3.metric(
                "Diferencia vs Precio Lista",
                f"{diferencia_pct:+.1f}%",
                "Descuento a favor" if diferencia_pct > 0 else "Sobreprecio",
            )

            if oferta_m2 < p15_m2:
                st.success(
                    f"OPORTUNIDAD DE INVERSIÓN: El precio publicado (${oferta_m2:,.0f}/m²) está por debajo del piso cuantílico P15 (${p15_m2:,.0f}/m²)."
                )
            elif oferta_m2 > p85_m2:
                st.warning(
                    f"ALERTA DE SOBREPRECIO: El inmueble (${oferta_m2:,.0f}/m²) rebasa el techo cuantílico P85 (${p85_m2:,.0f}/m²)."
                )
            else:
                st.info("PRECIO EN RANGO DE MERCADO: El inmueble se encuentra dentro de la banda normal de su AGEB.")

            # Explicabilidad local con SHAP en pesos por m2
            explainer = shap.TreeExplainer(bundle["model_mean"])
            shap_local = explainer.shap_values(X_infer)[0]
            df_shap = pd.DataFrame({
                "Variable": bundle["feature_names"],
                "Impacto_MXN_m2": np.round(shap_local, 0),
            }).sort_values(by="Impacto_MXN_m2", key=abs, ascending=False).head(8)

            st.markdown("**Desglose Local SHAP: ¿Qué está empujando el precio por m² hacia arriba o abajo?**")
            st.bar_chart(df_shap.set_index("Variable")["Impacto_MXN_m2"], horizontal=True)

            st.caption(
                f"Contexto censal automático (INEGI): Escolaridad {ctx['escolaridad_prom']:.1f} años | "
                f"Viviendas con internet {ctx['pct_viv_internet']*100:.0f}% | "
                f"Estación más cercana: {ctx['estacion_cercana']} a {ctx['dist_transporte_m']:.0f} m."
            )

    # SEGUNDA PESTAÑA: MAPA DE OPORTUNIDADES
    with tab_map:
        st.subheader("Detección de Inmuebles Subvaluados (Precio por debajo del Cuantil P15)")
        f_alc = st.multiselect(
            "Filtrar Alcaldías",
            options=list(df_oof["alcaldia"].unique()),
            default=list(df_oof["alcaldia"].unique()),
        )
        solo_oportunidades = st.checkbox("Mostrar únicamente inmuebles Subvaluados (Oportunidad)", value=False)

        df_map = df_oof[df_oof["alcaldia"].isin(f_alc)].copy()
        if solo_oportunidades:
            df_map = df_map[df_map["clasificacion_mercado"] == "Subvaluado (Oportunidad)"]

        # Colores RGB según clasificación de mercado
        color_map = {
            "Subvaluado (Oportunidad)": [16, 185, 129, 200],  # Verde esmeralda
            "Precio Justo": [59, 130, 246, 140],              # Azul
            "Sobrevaluado": [239, 68, 68, 190],               # Rojo
        }
        df_map["color"] = df_map["clasificacion_mercado"].map(color_map)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position="[longitud, latitud]",
            get_fill_color="color",
            get_radius=45,
            pickable=True,
        )
        view_state = pdk.ViewState(
            latitude=float(df_map["latitud"].mean()),
            longitude=float(df_map["longitud"].mean()),
            zoom=11.5,
            pitch=35,
        )
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "text": "{alcaldia} (AGEB {CVEGEO})\nPublicado: ${precio_m2}/m² | Predicción: ${pred_precio_m2}/m²\nDescuento: {descuento_pct}%\nClasificación: {clasificacion_mercado}"
                },
            )
        )

        st.markdown("**Top 10 Oportunidades Detectadas con Mayor Descuento vs. Valor Multimodal**")
        cols_tabla = [
            "id_anuncio",
            "alcaldia",
            "CVEGEO",
            "m2_construidos",
            "precio_m2",
            "pred_precio_m2",
            "descuento_pct",
            "pct_vegetacion",
            "estacion_cercana",
        ]
        top_deals = (
            df_map[df_map["clasificacion_mercado"] == "Subvaluado (Oportunidad)"]
            .sort_values("descuento_pct", ascending=False)
            .head(10)[cols_tabla]
        )
        st.dataframe(top_deals, use_container_width=True)

    # ================= PESTAÑA 3: ESTUDIO DE ABLACIÓN =================
    with tab_abl:
        st.subheader("Resultados del Estudio de Ablación (GroupKFold Espacial por AGEB)")
        st.markdown(
            "Esta tabla demuestra que agregar microdatos del **INEGI** y variables visuales de **SegFormer + CLIP** "
            "reduce el error de predicción en cuadras no vistas durante el entrenamiento:"
        )
        st.dataframe(df_ablation, use_container_width=True)

        st.markdown("**Importancia Global de Variables (SHAP Medio Absoluto en MXN/m²)**")
        df_imp = bundle["shap_importance"].reset_index()
        df_imp.columns = ["Variable", "SHAP_abs_MXN_m2"]
        st.bar_chart(df_imp.set_index("Variable")["SHAP_abs_MXN_m2"], horizontal=True)


if __name__ == "__main__":
    main()