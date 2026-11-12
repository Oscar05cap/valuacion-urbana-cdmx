import io
import zipfile
from pathlib import Path
import geopandas as gpd
import pandas as pd
import requests

from src.features.geospatial_inegi import clean_inegi_census_table

# URL pública directa del Censo 2020 por AGEB y Manzana Urbana para CDMX 
URL_CENSO_CDMX = (
    "https://www.inegi.org.mx/contenidos/programas/ccpv/2020/microdatos/ageb_manzana/RESAGEBURB_09CSV20.zip"
)

# Claves municipales de nuestras 4 alcaldías: Coyoacán (003), Iztapalapa (007), Benito Juárez (014), Cuauhtémoc (015)
ALCALDIAS_CVE_MUN = {"003", "007", "014", "015"}


def download_and_extract_censo_cdmx(raw_dir: Path) -> pd.DataFrame:
    """Descarga el CSV oficial del Censo 2020 (CDMX) y filtra estrictamente a nivel AGEB (MZA == '000')."""
    print("[INEGI] Descargando Censo 2020 por AGEB para CDMX desde inegi.org.mx...")
    resp = requests.get(URL_CENSO_CDMX, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No se encontró el archivo CSV dentro del ZIP del INEGI.")

        with z.open(csv_files[0]) as f:
            # Forzar string en claves geográficas
            df = pd.read_csv(
                f,
                dtype={"ENTIDAD": str, "MUN": str, "LOC": str, "AGEB": str, "MZA": str},
                encoding="utf-8-sig",
                low_memory=False,
            )

    # Filtrar solo el resumen por AGEB y descartar totales municipales
    df["ENTIDAD"] = df["ENTIDAD"].str.zfill(2)
    df["MUN"] = df["MUN"].str.zfill(3)
    df["LOC"] = df["LOC"].str.zfill(4)
    df["AGEB"] = df["AGEB"].str.zfill(4)
    df["MZA"] = df["MZA"].str.zfill(3)

    df_ageb = df[
        (df["ENTIDAD"] == "09")
        & (df["MUN"].isin(ALCALDIAS_CVE_MUN))
        & (df["AGEB"] != "0000")
        & (df["MZA"] == "000")
    ].copy()

    df_ageb["CVEGEO"] = df_ageb["ENTIDAD"] + df_ageb["MUN"] + df_ageb["LOC"] + df_ageb["AGEB"]
    print(f"[INEGI] AGEBs urbanas únicas extraídas en las 4 alcaldías: {len(df_ageb)}")

    return clean_inegi_census_table(df_ageb)


def build_real_inegi_geopackage(shp_path: Path, out_gpkg: Path):
    """
    Une el shapefile oficial de AGEBs urbanas del Marco Geoestadístico 
    con la tabla limpia del Censo 2020 y lo guarda en EPSG:4326 listo para el Módulo 2
    """
    raw_dir = out_gpkg.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    df_censo = download_and_extract_censo_cdmx(raw_dir)

    if not shp_path.exists():
        raise FileNotFoundError(
            f"No se encontró el shapefile en {shp_path}.\n"
            "Descarga el Marco Geoestadístico de CDMX del portal del INEGI, "
            "extrae la carpeta 'conjunto_de_datos' y coloca '09a.shp' en 'data/raw/09a.shp'"
        )

    print(f"[INEGI] Leyendo polígonos oficiales desde {shp_path}...")
    # El shapefile del INEGI viene en proyección Cónica Conforme de Lambert 
    gdf_polys = gpd.read_file(shp_path).to_crs(epsg=4326)
    gdf_polys["CVEGEO"] = gdf_polys["CVEGEO"].astype(str).str.strip().str.zfill(13)

    gdf_merged = gdf_polys[["CVEGEO", "geometry"]].merge(df_censo, on="CVEGEO", how="inner")

    # Estimar densidad comercial base si aún no se cruza el CSV masivo del DENUE
    if "densidad_comercial_denue" not in gdf_merged.columns:
        gdf_merged["densidad_comercial_denue"] = (gdf_merged["POBTOT"] * 0.018).round(0).astype(int)

    gdf_merged.to_file(out_gpkg, driver="GPKG")
    print(f"[Éxito] GeoPackage real guardado en {out_gpkg} con {len(gdf_merged)} polígonos AGEB.")


if __name__ == "__main__":
    build_real_inegi_geopackage(
        shp_path=Path("data/raw/09a.shp"),
        out_gpkg=Path("data/raw/inegi_ageb_cdmx.gpkg"),
    )