import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

# Claves oficiales INEGI por alcaldía 
CLAVES_INEGI = {
    "Benito Juárez": {"cve_mun": "014", "graproes_base": 14.2, "pct_auto_base": 0.68, "pct_net_base": 0.88},
    "Cuauhtémoc":    {"cve_mun": "015", "graproes_base": 13.1, "pct_auto_base": 0.45, "pct_net_base": 0.80},
    "Coyoacán":      {"cve_mun": "003", "graproes_base": 12.8, "pct_auto_base": 0.58, "pct_net_base": 0.78},
    "Iztapalapa":    {"cve_mun": "007", "graproes_base": 10.1, "pct_auto_base": 0.34, "pct_net_base": 0.59},
}

# Coordenadas reales de nodos clave de Metro/Metrobús en las 4 alcaldías
ESTACIONES_TRANSPORTE = [
    ("Metro Insurgentes", 19.4235, -99.1628),
    ("Metro Bellas Artes", 19.4362, -99.1419),
    ("Metro Centro Médico", 19.4066, -99.1557),
    ("Metro Zapata", 19.3708, -99.1652),
    ("Metro Mixcoac", 19.3760, -99.1873),
    ("Metro Etiopía", 19.3956, -99.1563),
    ("Metro Coyoacán", 19.3614, -99.1708),
    ("Metro Copilco", 19.3359, -99.1768),
    ("Metro Taxqueña", 19.3437, -99.1404),
    ("Metro Constitución de 1917", 19.3458, -99.0639),
    ("Metro Iztapalapa", 19.3578, -99.0935),
    ("Metro UAM-I", 19.3563, -99.0746),
    ("Metrobús Poliforum", 19.3934, -99.1725),
    ("Metrobús Reforma", 19.4326, -99.1545),
    ("Metrobús Perisur", 19.3040, -99.1860),
]


def clean_inegi_census_table(df_censo: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia microdatos del Censo INEGI preservando ceros a la izquierda en CVEGEO
    e imputando los datos censurados con la mediana municipal
    """
    df = df_censo.copy()

    # Reconstruir CVEGEO a 13 caracteres 
    if "CVEGEO" in df.columns:
        df["CVEGEO"] = df["CVEGEO"].astype(str).str.strip().str.zfill(13)
    else:
        df["CVEGEO"] = (
            df["ENTIDAD"].astype(str).str.zfill(2)
            + df["MUN"].astype(str).str.zfill(3)
            + df["LOC"].astype(str).str.zfill(4)
            + df["AGEB"].astype(str).str.zfill(4)
        )

    df["CVE_MUN"] = df["CVEGEO"].str[2:5]

    # Convertir columnas censuradas a NaN numérico
    cols_numericas = ["POBTOT", "TVIVPARHAB", "GRAPROES", "VPH_AUTOM", "VPH_INTER"]
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("*", "", regex=False).str.replace("N/D", "", regex=False),
                errors="coerce",
            )
            # Imputar faltantes con la mediana de su propia alcaldía (nunca con 0 global)
            df[col] = df.groupby("CVE_MUN")[col].transform(lambda s: s.fillna(s.median()))

    # Construir tasas relativas en lugar de conteos absolutos
    viv_validas = df["TVIVPARHAB"].replace(0, np.nan)
    df["pct_viv_auto"] = (df["VPH_AUTOM"] / viv_validas).clip(0, 1).fillna(0.45)
    df["pct_viv_internet"] = (df["VPH_INTER"] / viv_validas).clip(0, 1).fillna(0.70)
    df["escolaridad_prom"] = df["GRAPROES"].fillna(12.0)

    return df[["CVEGEO", "CVE_MUN", "POBTOT", "escolaridad_prom", "pct_viv_auto", "pct_viv_internet"]]


def build_bootstrap_ageb_grid(df_listings: pd.DataFrame, seed: int = 42) -> gpd.GeoDataFrame:
    """
    Construye una malla de polígonos AGEB sintéticos pero topológicamente reales (~600m x 600m)
    sobre las 4 alcaldías con datos del Censo 2020 para ejecutar el pipeline sin el shapefile crudo.
    """
    rng = np.random.default_rng(seed)
    polygons = []
    step = 0.006  # aprox 650 metros por cuadrante AGEB

    for alc, meta in CLAVES_INEGI.items():
        sub = df_listings[df_listings["alcaldia"] == alc]
        if sub.empty:
            continue
        lat_min, lat_max = sub["latitud"].min() - step, sub["latitud"].max() + step
        lon_min, lon_max = sub["longitud"].min() - step, sub["longitud"].max() + step

        lats = np.arange(lat_min, lat_max, step)
        lons = np.arange(lon_min, lon_max, step)
        ageb_counter = 1

        for lat in lats:
            for lon in lons:
                cvegeo = f"09{meta['cve_mun']}0001{ageb_counter:04d}"
                ageb_counter += 1
                poly = box(lon, lat, lon + step, lat + step)

                # Simular valores crudos con algunos '*' como vienen en el INEGI real
                graproes_val = round(rng.normal(meta["graproes_base"], 0.9), 2)
                viv_hab = int(rng.integers(400, 1800))
                vph_auto = "*" if rng.random() < 0.04 else str(int(viv_hab * np.clip(rng.normal(meta["pct_auto_base"], 0.10), 0.1, 0.95)))
                vph_inter = str(int(viv_hab * np.clip(rng.normal(meta["pct_net_base"], 0.08), 0.2, 0.98)))

                polygons.append({
                    "CVEGEO": cvegeo,
                    "POBTOT": int(viv_hab * 3.1),
                    "TVIVPARHAB": viv_hab,
                    "GRAPROES": str(graproes_val),
                    "VPH_AUTOM": vph_auto,
                    "VPH_INTER": vph_inter,
                    "densidad_comercial_denue": int(rng.integers(15, 190)),
                    "geometry": poly,
                })

    gdf_raw = gpd.GeoDataFrame(polygons, crs="EPSG:4326")
    df_cleaned_attrs = clean_inegi_census_table(gdf_raw.drop(columns=["geometry"]))
    gdf_clean = gdf_raw[["CVEGEO", "densidad_comercial_denue", "geometry"]].merge(df_cleaned_attrs, on="CVEGEO")
    return gdf_clean


def enrich_with_geospatial_features(df_listings: pd.DataFrame, gdf_ageb: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Cruza anuncios con polígonos AGEB y calcula distancias exactas en metros usando UTM 14N (EPSG:32614).
    """
    # Convertir anuncios a GeoDataFrame en WGS84 y proyectar a metros 
    gdf_props = gpd.GeoDataFrame(
        df_listings.copy(),
        geometry=gpd.points_from_xy(df_listings["longitud"], df_listings["latitud"]),
        crs="EPSG:4326",
    ).to_crs(epsg=32614)

    gdf_ageb_utm = gdf_ageb.to_crs(epsg=32614)

    # Calcular área real del AGEB en km2 para obtener densidad poblacional
    gdf_ageb_utm["area_km2"] = gdf_ageb_utm.geometry.area / 1e6
    gdf_ageb_utm["densidad_pob_km2"] = (gdf_ageb_utm["POBTOT"] / gdf_ageb_utm["area_km2"]).round(0)

    # Spatial Join (Punto dentro de Polígono AGEB)
    cols_ageb = [
        "CVEGEO",
        "escolaridad_prom",
        "pct_viv_auto",
        "pct_viv_internet",
        "densidad_pob_km2",
        "densidad_comercial_denue",
        "geometry",
    ]
    gdf_joined = gpd.sjoin(gdf_props, gdf_ageb_utm[cols_ageb], how="left", predicate="within").drop(
        columns=["index_right"], errors="ignore"
    )

    # Si un punto cayó exactamente en el borde de una calle entre dos AGEB, asignar el más cercano
    if gdf_joined["CVEGEO"].isna().any():
        faltantes_idx = gdf_joined[gdf_joined["CVEGEO"].isna()].index
        reemplazo = gpd.sjoin_nearest(
            gdf_props.loc[faltantes_idx], gdf_ageb_utm[cols_ageb], how="left"
        ).drop(columns=["index_right"], errors="ignore")
        reemplazo = reemplazo[~reemplazo.index.duplicated(keep="first")]
        for col in cols_ageb[:-1]:
            gdf_joined.loc[faltantes_idx, col] = reemplazo[col]

    # Distancia en metros reales a la estación de transporte masivo más cercana
    gdf_transporte = gpd.GeoDataFrame(
        ESTACIONES_TRANSPORTE,
        columns=["estacion_cercana", "lat", "lon"],
        geometry=[Point(lon, lat) for _, lat, lon in ESTACIONES_TRANSPORTE],
        crs="EPSG:4326",
    ).to_crs(epsg=32614)

    gdf_final = gpd.sjoin_nearest(
        gdf_joined,
        gdf_transporte[["estacion_cercana", "geometry"]],
        how="left",
        distance_col="dist_transporte_m",
    ).drop(columns=["index_right"], errors="ignore")

    # Eliminar posibles duplicados por equidistancia y redondear metros
    gdf_final = gdf_final[~gdf_final.index.duplicated(keep="first")]
    gdf_final["dist_transporte_m"] = gdf_final["dist_transporte_m"].round(1)

    return pd.DataFrame(gdf_final.drop(columns=["geometry"]))


def main():
    parser = argparse.ArgumentParser(description="Enriquecimiento geoespacial con INEGI y Transporte CDMX")
    parser.add_argument("--bootstrap", action="store_true", help="Construye malla AGEB calibrada si no hay shapefile local")
    args = parser.parse_args()

    in_path = Path("data/processed/listings_clean.parquet")
    out_path = Path("data/processed/listings_geo.parquet")
    ageb_shp_path = Path("data/raw/inegi_ageb_cdmx.gpkg")

    if not in_path.exists():
        raise FileNotFoundError("Primero ejecuta el Módulo 1: python src/extract/listings.py --bootstrap")

    df_listings = pd.read_parquet(in_path)

    if not args.bootstrap and ageb_shp_path.exists():
        print(f"[INEGI] Cargando archivo oficial desde {ageb_shp_path}...")
        gdf_ageb = gpd.read_file(ageb_shp_path)
    else:
        print("[Modo Bootstrap] Generando malla geoespacial AGEB con formato oficial INEGI...")
        gdf_ageb = build_bootstrap_ageb_grid(df_listings)

    df_geo = enrich_with_geospatial_features(df_listings, gdf_ageb)
    df_geo.to_parquet(out_path, index=False)

    print(f"\n[Éxito] Propiedades enriquecidas: {len(df_geo)} | AGEBs únicos detectados: {df_geo['CVEGEO'].nunique()}")
    cols_resumen = ["escolaridad_prom", "pct_viv_internet", "dist_transporte_m", "precio_m2"]
    print("\nPromedios por Alcaldía tras el cruce espacial:")
    print(df_geo.groupby("alcaldia")[cols_resumen].mean().round(2))
    print(f"\nArchivo guardado en: {out_path}")


if __name__ == "__main__":
    main()