import pandera as pa
from pandera import Column, Check, DataFrameSchema

# Esquema para validar la salida del Módulo 1 
ListingsCleanSchema = DataFrameSchema(
    {
        "id_anuncio": Column(str, unique=True, nullable=False),
        "alcaldia": Column(
            str,
            Check.isin(["Benito Juárez", "Cuauhtémoc", "Coyoacán", "Iztapalapa"]),
            nullable=False,
        ),
        "precio_mxn": Column(float, Check.in_range(850_000, 45_000_000), nullable=False),
        "m2_construidos": Column(float, Check.in_range(30, 600), nullable=False),
        "precio_m2": Column(float, Check.in_range(10_000, 130_000), nullable=False),
        "recamaras": Column(int, Check.in_range(1, 10), nullable=False),
        "banos": Column(int, Check.in_range(1, 10), nullable=False),
        "latitud": Column(float, Check.in_range(19.25, 19.55), nullable=False),
        "longitud": Column(float, Check.in_range(-99.30, -98.95), nullable=False),
    },
    strict=False,
    coerce=True,
)

# Esquema para validar la salida del Módulo 2 
ListingsGeoSchema = ListingsCleanSchema.add_columns(
    {
        # Debe ser exactamente de 13 caracteres y empezar con '09' (CDMX)
        "CVEGEO": Column(str, Check.str_matches(r"^09\d{7}[0-9A-Z]{4}$"), nullable=False),
        "escolaridad_prom": Column(float, Check.in_range(5.0, 22.0), nullable=False),
        "pct_viv_auto": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "pct_viv_internet": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "dist_transporte_m": Column(float, Check.ge(0.0), nullable=False),
    }
)

# Esquema para validar la salida del Módulo 3
ListingsMultimodalSchema = ListingsGeoSchema.add_columns(
    {
        "pct_vegetacion": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "pct_banqueta": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "clip_urban_score": Column(float, Check.in_range(-1.0, 1.0), nullable=False),
    }
)