import argparse
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from playwright.sync_api import sync_playwright

# Límites geográficos para validar coordenadas en las 4 alcaldías de CDMX
ALCALDIAS_BBOX = {
    "Benito Juárez": {"lat_min": 19.350, "lat_max": 19.405, "lon_min": -99.195, "lon_max": -99.130, "base_m2": 46000},
    "Cuauhtémoc":    {"lat_min": 19.400, "lat_max": 19.465, "lon_min": -99.185, "lon_max": -99.120, "base_m2": 49000},
    "Coyoacán":      {"lat_min": 19.295, "lat_max": 19.360, "lon_min": -99.205, "lon_max": -99.100, "base_m2": 36000},
    "Iztapalapa":    {"lat_min": 19.310, "lat_max": 19.395, "lon_min": -99.125, "lon_max": -98.980, "base_m2": 19500},
}

# Palabras clave que delatan remates bancarios o litigios
REGEX_REMATES = re.compile(
    r"(remate|bancario|recuperaci[oó]n|cesi[oó]n de derechos|no cr[eé]ditos|"
    r"solo contado|recursos propios|adjudicaci[oó]n|litigio|oportunidad inversionistas)",
    flags=re.IGNORECASE,
)

# Caché en memoria para no repetir consultas idénticas a Nominatim
_GEOCODE_CACHE: dict[str, tuple[float, float]] = {}


def detect_bank_foreclosure(text: str) -> bool:
    """Devuelve True si el título o descripción sugiere un remate bancario"""
    if not isinstance(text, str):
        return False
    return bool(REGEX_REMATES.search(text))


def geocode_address_cdmx(
    direccion_texto: str,
    alcaldia: str,
    bbox: dict,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Convierte la colonia/calle del anuncio en coordenadas reales dentro de la alcaldía.
    Usa caché en memoria y añade una micro-dispersión para evitar que anuncios
    que solo reportan la colonia colapsen en el mismo punto exacto
    """
    if rng is None:
        rng = np.random.default_rng()

    clean_dir = (direccion_texto or "").strip()
    query = f"{clean_dir}, {alcaldia}, Ciudad de México, México".strip(", ")

    if query not in _GEOCODE_CACHE:
        coord_found = None
        if clean_dir:
            try:
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": query, "format": "json", "limit": 1},
                    headers={"User-Agent": "valuacion-urbana-cdmx-portfolio/1.0"},
                    timeout=5,
                )
                time.sleep(1.05) 
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                        if bbox["lat_min"] <= lat <= bbox["lat_max"] and bbox["lon_min"] <= lon <= bbox["lon_max"]:
                            coord_found = (lat, lon)
            except Exception:
                pass
        if coord_found is None:
            coord_found = (
                (bbox["lat_min"] + bbox["lat_max"]) / 2.0,
                (bbox["lon_min"] + bbox["lon_max"]) / 2.0,
            )

        _GEOCODE_CACHE[query] = coord_found

    base_lat, base_lon = _GEOCODE_CACHE[query]
    lat_jitter = float(np.clip(base_lat + rng.normal(0, 0.00035), bbox["lat_min"], bbox["lat_max"]))
    lon_jitter = float(np.clip(base_lon + rng.normal(0, 0.00035), bbox["lon_min"], bbox["lon_max"]))
    return lat_jitter, lon_jitter


def clean_and_validate_listings(df: pd.DataFrame, usd_to_mxn: float = 20.0) -> pd.DataFrame:
    """Limpia anuncios crudos aplicando reglas de negocio del mercado mexicano."""
    df = df.copy()
    initial_count = len(df)

    # Filtrar remates bancarios por texto
    df["es_remate"] = df["titulo"].apply(detect_bank_foreclosure) | df["descripcion"].apply(detect_bank_foreclosure)
    df = df[~df["es_remate"]].drop(columns=["es_remate"])

    # Normalizar moneda 
    df["precio_mxn"] = np.where(
        df["moneda"].str.upper().isin(["USD", "U$S", "US$"]),
        df["precio_raw"] * usd_to_mxn,
        df["precio_raw"],
    )

    # Filtrar errores de captura
    df = df[(df["m2_construidos"] >= 30) & (df["m2_construidos"] <= 600)]
    df = df[(df["precio_mxn"] >= 850_000) & (df["precio_mxn"] <= 45_000_000)]

    # Precio por metro cuadrado
    df["precio_m2"] = df["precio_mxn"] / df["m2_construidos"]

    # Recortar outliers extremos de precio_m2
    df = df[(df["precio_m2"] >= 10_000) & (df["precio_m2"] <= 130_000)]

    # Validar coordenadas dentro de la CDMX
    df = df[
        (df["latitud"] >= 19.25) & (df["latitud"] <= 19.55) &
        (df["longitud"] >= -99.30) & (df["longitud"] <= -98.95)
    ]

    # Eliminar duplicados espaciales y de características 
    df["lat_round"] = df["latitud"].round(4)
    df["lon_round"] = df["longitud"].round(4)
    df["m2_round"] = (df["m2_construidos"] / 5).round() * 5
    df = df.drop_duplicates(subset=["lat_round", "lon_round", "m2_round", "recamaras"]).drop(
        columns=["lat_round", "lon_round", "m2_round"]
    )

    print(f"[Limpieza] Registros iniciales: {initial_count} -> Registros válidos: {len(df)}")
    return df.reset_index(drop=True)


def scrape_mercadolibre_zone(alcaldia: str, max_pages: int = 3) -> list[dict]:
    """Extrae anuncios públicos de departamentos/casas en venta y geocodifica su ubicación real."""
    slug = alcaldia.lower().replace(" ", "-").replace("á", "a").replace("é", "e").replace("ó", "o")
    bbox = ALCALDIAS_BBOX[alcaldia]
    rng = np.random.default_rng(42)
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for page_num in range(1, max_pages + 1):
            offset = (page_num - 1) * 48 + 1
            url = f"https://inmuebles.mercadolibre.com.mx/departamentos/venta/distrito-federal/{slug}/_Desde_{offset}_NoIndex_True"
            print(f"[Scraper] Consultando {alcaldia} (Página {page_num})...")
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                time.sleep(2.5)
                cards = page.query_selector_all(".ui-search-layout__item")
                if not cards:
                    break

                for idx, card in enumerate(cards):
                    title_el = card.query_selector(".poly-component__title, .ui-search-item__title")
                    price_el = card.query_selector(".andes-money-amount__fraction")
                    currency_el = card.query_selector(".andes-money-amount__currency-symbol")
                    loc_el = card.query_selector(".poly-component__location, .ui-search-item__group__element--location")
                    attrs = card.query_selector_all(".poly-attributes-list__item, .ui-search-card-attributes__attribute")

                    if not title_el or not price_el:
                        continue

                    titulo = title_el.inner_text().strip()
                    precio_str = price_el.inner_text().replace(",", "").replace(".", "").strip()
                    moneda = "USD" if (currency_el and "U$S" in currency_el.inner_text()) else "MXN"
                    direccion_texto = loc_el.inner_text().strip() if loc_el else ""

                    m2, recamaras, banos = np.nan, 2, 1
                    for attr in attrs:
                        txt = attr.inner_text().lower()
                        nums = re.findall(r"\d+", txt)
                        if not nums:
                            continue
                        if "m²" in txt or "m2" in txt:
                            m2 = float(nums[0])
                        elif "rec" in txt or "dorm" in txt:
                            recamaras = int(nums[0])
                        elif "bañ" in txt:
                            banos = int(nums[0])

                    lat, lon = geocode_address_cdmx(direccion_texto, alcaldia, bbox, rng=rng)

                    results.append({
                        "id_anuncio": f"ML-{alcaldia[:3].upper()}-{page_num}-{idx}",
                        "alcaldia": alcaldia,
                        "titulo": titulo,
                        "descripcion": f"{titulo} - {direccion_texto}",
                        "precio_raw": float(precio_str) if precio_str.isdigit() else np.nan,
                        "moneda": moneda,
                        "m2_construidos": m2,
                        "recamaras": recamaras,
                        "banos": banos,
                        "estacionamientos": 1 if recamaras >= 2 else 0,
                        "tipo_propiedad": "departamento",
                        "latitud": lat,
                        "longitud": lon,
                    })
            except Exception as e:
                print(f"[Aviso] Error en página {page_num} de {alcaldia}: {e}")
                break

        browser.close()
    return results


def generate_bootstrap_dataset(n_samples: int = 2400, seed: int = 42) -> pd.DataFrame:
    """
    Genera un dataset semilla calibrado con la distribución real de CDMX para probar el pipeline y correr pytest sin bloqueos
    """
    rng = np.random.default_rng(seed)
    rows = []
    alcaldias = list(ALCALDIAS_BBOX.keys())

    for i in range(n_samples):
        alc = rng.choice(alcaldias, p=[0.30, 0.30, 0.20, 0.20])
        bbox = ALCALDIAS_BBOX[alc]
        lat = rng.uniform(bbox["lat_min"], bbox["lat_max"])
        lon = rng.uniform(bbox["lon_min"], bbox["lon_max"])

        m2 = float(rng.integers(45, 220))
        recamaras = int(np.clip(round(m2 / 45), 1, 4))
        banos = int(np.clip(recamaras, 1, 3))
        estac = int(rng.choice([0, 1, 2], p=[0.25, 0.55, 0.20]))

        spatial_factor = 1.0 + 0.18 * ((lat - bbox["lat_min"]) / (bbox["lat_max"] - bbox["lat_min"]))
        precio_m2_real = bbox["base_m2"] * spatial_factor * rng.normal(1.0, 0.14)
        precio_total = precio_m2_real * m2

        es_remate = rng.random() < 0.08
        titulo = (
            f"Remate bancario departamento en {alc} solo contado no créditos"
            if es_remate
            else f"Departamento en venta en {alc} excelente ubicación"
        )
        if es_remate:
            precio_total *= 0.45

        rows.append({
            "id_anuncio": f"CDMX-{i:05d}",
            "alcaldia": alc,
            "titulo": titulo,
            "descripcion": titulo,
            "precio_raw": round(precio_total, -3),
            "moneda": "MXN",
            "m2_construidos": m2,
            "recamaras": recamaras,
            "banos": banos,
            "estacionamientos": estac,
            "tipo_propiedad": rng.choice(["departamento", "casa"], p=[0.82, 0.18]),
            "latitud": lat,
            "longitud": lon,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extracción y limpieza de anuncios inmobiliarios CDMX")
    parser.add_argument("--bootstrap", action="store_true", help="Usa datos semilla calibrados sin scraping en vivo")
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    if args.bootstrap:
        print("[Modo Bootstrap] Generando dataset con ruido realista de CDMX...")
        df_raw = generate_bootstrap_dataset(n_samples=2500)
    else:
        all_rows = []
        for alc in ALCALDIAS_BBOX:
            all_rows.extend(scrape_mercadolibre_zone(alc, max_pages=3))
        df_raw = pd.DataFrame(all_rows)
        if df_raw.empty:
            print("[Aviso] El portal bloqueó la consulta sin sesión. Usando modo bootstrap de respaldo...")
            df_raw = generate_bootstrap_dataset(n_samples=2500)

    raw_path = raw_dir / "listings_raw.parquet"
    df_raw.to_parquet(raw_path, index=False)

    df_clean = clean_and_validate_listings(df_raw)

    # Validar contrato de datos con Pandera antes de guardar
    try:
        from src.schemas import ListingsCleanSchema
        ListingsCleanSchema.validate(df_clean)
        print("[Pandera] Contrato de datos ListingsCleanSchema validado correctamente.")
    except ImportError:
        pass

    clean_path = proc_dir / "listings_clean.parquet"
    df_clean.to_parquet(clean_path, index=False)

    print("\nResumen por Alcaldía (Precio mediano por m2 en MXN):")
    resumen = df_clean.groupby("alcaldia")["precio_m2"].agg(["count", "median", "std"]).round(0)
    print(resumen)
    print(f"\nArchivo guardado en: {clean_path}")


if __name__ == "__main__":
    main()