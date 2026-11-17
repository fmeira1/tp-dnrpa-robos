"""
Herramienta interactiva del TP: cotizador de riesgo de robo de un vehículo.
Requiere haber corrido antes `python tp_dnrpa.py` (genera el modelo y base_limpia.csv).

Ejecutar:  python -m streamlit run app_dnrpa.py
"""

import pandas as pd
import streamlit as st

from tp_dnrpa import ARCHIVO_FINAL, ANIO_ACTUAL, cargar_cotizador

st.set_page_config(page_title="Riesgo de robo vehicular", page_icon="🚗", layout="wide")


@st.cache_resource
def cargar():
    return cargar_cotizador(), pd.read_csv(ARCHIVO_FINAL)


cotizar, df = cargar()

st.title("🚗 Scoring de riesgo de robo vehicular")
st.caption("Datos abiertos DNRPA - Ministerio de Justicia | Modelo XGBoost + score de severidad")

# --- Selección del vehículo -------------------------------------------------
col1, col2, col3 = st.columns(3)
marca = col1.selectbox("Marca", sorted(df["marca"].unique()),
                       index=sorted(df["marca"].unique()).index("VOLKSWAGEN"))
modelos_marca = df[df["marca"] == marca]["modelo"]
modelos = sorted(modelos_marca.unique())
# arranca en el modelo con más trámites de la marca (no en el primero alfabético,
# que a veces es un código suelto como "0")
modelo_auto = col2.selectbox("Modelo", modelos, index=modelos.index(modelos_marca.mode()[0]))

# a partir de acá, todo se filtra por marca + modelo: un Bora no se vende desde 2014,
# así que no tiene sentido dejar elegir 2024, ni mostrarlo como "Nacional" si acá
# siempre vino importado. Las opciones se recalculan cada vez que cambia el modelo.
sel_modelo = df[(df["marca"] == marca) & (df["modelo"] == modelo_auto)]
clave = f"{marca}_{modelo_auto}"  # para que el año no arrastre un valor de otro modelo

anios_vistos = sel_modelo["anio_modelo"].dropna().astype(int)
if len(anios_vistos):
    anio_min, anio_max = int(anios_vistos.min()), int(anios_vistos.max())
else:
    anio_min, anio_max = 1980, ANIO_ACTUAL
anio = col3.number_input("Año del modelo", min_value=anio_min, max_value=anio_max,
                          value=anio_max, key=f"anio_{clave}")

col4, col5, col6 = st.columns(3)
provincia = col4.selectbox("Provincia de radicación", sorted(df["provincia"].unique()))
tipos = sorted(sel_modelo["tipo_vehiculo"].unique()) or sorted(df["tipo_vehiculo"].unique())
tipo = col5.selectbox("Tipo de vehículo", tipos, key=f"tipo_{clave}")
origenes = sorted(sel_modelo["origen"].unique()) or sorted(df["origen"].unique())
origen = col6.selectbox("Origen", origenes, key=f"origen_{clave}")

if st.button("Calcular riesgo", type="primary"):
    r = cotizar(marca, tipo, origen, provincia, int(anio))
    a, b, c = st.columns(3)
    a.metric("Probabilidad de recupero", f"{r['prob_recupero'] * 100:.1f}%")
    b.metric("Score de severidad (0-1)", f"{r['score']:.2f}")
    c.metric("Categoría de riesgo", r["categoria"])
    st.info(f"**Prima:** {r['prima']}  |  **Rastreador satelital:** {r['rastreador']}")

# --- Contexto del modelo elegido -------------------------------------------
st.divider()
st.subheader(f"Historial de {marca} {modelo_auto}")
sel = sel_modelo
m1, m2, m3 = st.columns(3)
m1.metric("Trámites registrados", f"{len(sel):,}")
m2.metric("Recuperos", f"{int(sel['es_recuperado'].sum()):,}")
m3.metric("Tasa de recupero", f"{sel['es_recuperado'].mean() * 100:.1f}%" if len(sel) else "-")
st.bar_chart(sel.groupby("provincia").size().sort_values(ascending=False).head(10))
