# Valuador de departamentos en CDMX

Proyecto para estimar el precio por metro cuadrado (`MXN/m²`) de departamentos en cuatro alcaldías de la CDMX (Benito Juárez, Cuauhtémoc, Coyoacán e Iztapalapa) cruzando anuncios inmobiliarios, datos del Censo del INEGI y fotos de la calle procesadas con visión por computadora.

La idea nació porque los modelos típicos que solo usan metros cuadrados, número de recámaras y alcaldía fallan mucho en la CDMX por dos razones:
1. En los portales hay muchos remates bancarios mezclados con ventas normales (anunciados a mitad de precio porque son litigios), lo que sesga cualquier regresión si no se limpian antes.
2. Dentro de una misma colonia, el precio cambia bastante dependiendo de si la calle está arbolada y tranquila o si es una avenida deteriorada.

## Qué hay en el repositorio

El código está dividido por etapas para no tener todo en un solo notebook:

- `src/extract/listings.py`: Limpia los anuncios crudos. Filtra remates bancarios usando expresiones regulares (buscando frases como "cesión de derechos" o "no créditos"), pasa precios en dólares a pesos y quita anuncios duplicados por ubicación y tamaño.
- `src/features/geospatial_inegi.py`: Cruza cada departamento con su AGEB del INEGI (Censo 2020 y DENUE) y calcula la distancia al Metro o Metrobús más cercano. Para medir distancias reales en metros reprogramé la proyección a UTM Zona 14N (`EPSG:32614`). También corrige el problema clásico de los CSV del INEGI donde se borra el `0` inicial de la clave `09` de la CDMX y vienen asteriscos (`*`) por confidencialidad.
- `src/features/vision_street.py`: Descarga fotos de la calle en un radio de 120m usando la API de Mapillary. Con SegFormer saca el porcentaje de vegetación, banqueta, cielo y edificio en la foto, y con CLIP calcula un índice visual de la calle.
- `src/models/train_ablation.py`: Entrena un modelo LightGBM para predecir el precio por m² y dos modelos cuantílicos (P15 y P85) para armar un rango de precio y detectar departamentos subvaluados.
- `app/streamlit_app.py`: Interfaz en Streamlit con mapa interactivo y gráficas SHAP para ver cuántos pesos por m² suma o resta cada variable.

## Resultados

Para no hacer trampa en la validación (que dos departamentos del mismo edificio caigan uno en train y otro en test), usé `GroupKFold` agrupando por la clave `CVEGEO` del AGEB. Así el modelo siempre se evalúa en cuadras que no vio durante el entrenamiento.

Además, para que las 512 dimensiones de CLIP no sobreajustaran los árboles de decisión, las comprimí a 3 componentes con PCA calculado adentro de cada fold.

| Variables usadas | RMSE (MXN/m²) | MAPE (%) | R² |
| :--- | :---: | :---: | :---: |
| 1. Solo datos del anuncio | $4,912 | 11.42% | 0.781 |
| 2. + INEGI y distancia al Metro | $4,186 | 9.68% | 0.841 |
| 3. + Fotos de la calle (SegFormer + CLIP) | $3,640 | 8.35% | 0.880 |

## Cómo correrlo

1. Clonar e instalar dependencias:

```bash
git clone [https://github.com/O5car0ckap/valuacion-urbana-cdmx.git](https://github.com/O5car0ckap/valuacion-urbana-cdmx.git)
cd valuacion-urbana-cdmx
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Correr los scripts en orden. Dejé una bandera `--bootstrap` que genera una base ya calibrada para poder probar todo el flujo en un minuto sin que los portales bloqueen tu IP ni tengas que bajar los shapefiles completos del INEGI de entrada:

```bash
python src/extract/listings.py --bootstrap
python src/features/geospatial_inegi.py --bootstrap
python src/features/vision_street.py --bootstrap
python src/models/train_ablation.py
```

3. Abrir el tablero en Streamlit:

```bash
streamlit run app/streamlit_app.py
```

## Notas y limitaciones

- Los precios con los que entrena el modelo son precios de lista publicados en portales, no precios finales de escrituración (en México ese dato no es público y suele haber un margen de negociación del 5% al 10%).
- El módulo de visión evalúa el exterior de la cuadra, pero no analiza las fotos del interior del departamento (no distingue si está recién remodelado o si tiene acabados viejos).