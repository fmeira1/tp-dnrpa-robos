"""
TP Integrador - Modelizado de Sistemas e IA
Sistema de scoring de riesgo vehicular y tarificación para aseguradoras
usando los datos abiertos de la DNRPA (Ministerio de Justicia).

Dataset: "DNRPA. Robos y recuperos de autos"
         https://datos.jus.gob.ar/dataset/robos-y-recuperos-de-autos

Uso (desde la carpeta donde está este archivo):
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy requests joblib streamlit
    python tp_dnrpa.py            -> descarga, limpia, hace EDA, entrena y muestra cotizaciones de ejemplo
    python -m streamlit run app_dnrpa.py    -> abre la herramienta interactiva (cotizador)

No requiere base de datos: todo se hace con archivos CSV.
"""

import difflib
import io
import os
import re
import zipfile

import joblib
import matplotlib
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

matplotlib.use("Agg")  # graficar a archivo, sin abrir ventanas
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
API_PORTAL = "https://datos.jus.gob.ar/api/3/action/package_show?id=robos-y-recuperos-de-autos"
ANIO_DESDE = 2021          # se usan los últimos años (acordado en el grupo: ~5 años)
CARPETA_DATOS = "datos"    # acá se guardan los archivos descargados (caché)
ARCHIVO_LIMPIO = "dataset_limpio.csv"
ARCHIVO_MODELO = "modelo_scoring_recupero.joblib"
ANIO_ACTUAL = 2026

FEATURES_CAT = ["marca", "tipo_vehiculo", "origen", "provincia"]
FEATURES = FEATURES_CAT + ["antiguedad_vehiculo", "mes_tramite"]


# ----------------------------------------------------------------------------
# 1. DESCARGA AUTOMÁTICA DESDE EL PORTAL (sin bajar archivos a mano)
# ----------------------------------------------------------------------------
def descargar_datos(anio_desde=ANIO_DESDE):
    """Baja del portal los ZIP anuales y los CSV mensuales >= anio_desde.
    Si el archivo ya está en CARPETA_DATOS no se vuelve a bajar."""
    os.makedirs(CARPETA_DATOS, exist_ok=True)
    recursos = requests.get(API_PORTAL, timeout=60).json()["result"]["resources"]

    for r in recursos:
        # el nombre del recurso termina en el período: "... - 2024" o "... - 202608"
        periodo = r["name"].split("-")[-1].strip()
        if not periodo.isdigit() or int(periodo[:4]) < anio_desde:
            continue
        destino = os.path.join(CARPETA_DATOS, r["url"].split("/")[-1])
        if os.path.exists(destino):
            continue
        print(f"  descargando {os.path.basename(destino)} ...")
        resp = requests.get(r["url"], timeout=300)
        resp.raise_for_status()
        with open(destino, "wb") as f:
            f.write(resp.content)


def leer_csv(fuente):
    """Lee un CSV (ruta o archivo abierto) probando las dos codificaciones habituales."""
    contenido = fuente.read() if hasattr(fuente, "read") else open(fuente, "rb").read()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(contenido), encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo leer el CSV")


def cargar_datos():
    """Unifica todos los archivos descargados en un único DataFrame.
    Los CSV mensuales pisan a los del ZIP anual si el mes está repetido."""
    por_mes = {}  # nombre de csv -> DataFrame
    for nombre in sorted(os.listdir(CARPETA_DATOS)):
        ruta = os.path.join(CARPETA_DATOS, nombre)
        if nombre.endswith(".zip"):
            with zipfile.ZipFile(ruta) as z:
                for interno in z.namelist():
                    if interno.endswith(".csv"):
                        por_mes[interno] = leer_csv(z.open(interno))
        elif nombre.endswith(".csv"):
            por_mes[nombre] = leer_csv(ruta)
    df = pd.concat(por_mes.values(), ignore_index=True)
    print(f"  {len(por_mes)} archivos mensuales unificados -> {len(df):,} registros")
    return df


# ----------------------------------------------------------------------------
# 2. LIMPIEZA Y FEATURE ENGINEERING
# ----------------------------------------------------------------------------
# Los nombres vienen escritos distinto según el mes: se unifican con reglas.
CORRECCION_MARCAS = {
    r"^MERCEDES.*|^M\.?\s?BENZ.*": "MERCEDES-BENZ",
    r"^VW.*|^VOLKSW.*": "VOLKSWAGEN",
    r"^CHEVR.*": "CHEVROLET",
    r"^CITRO.*": "CITROEN",
    r"^GENERAL MOTORS.*|^G\.MOTORS.*|^GM\b.*": "CHEVROLET",  # en Argentina, GM vende como Chevrolet
    # variantes de "no se registró la marca": no son errores de tipeo entre sí, pero
    # dicen lo mismo, así que se agrupan en una sola categoría
    r"^SIN MARCA.*|^MARCA INVALIDA.*|^NO POSEE$|^NO CONSTA$|^\*$": "SIN ESPECIFICAR",
}


def limpiar_marca(valor):
    texto = str(valor).strip().upper()
    texto = re.sub(r"\s*[\(\[\{].*?[\)\]\}]", "", texto)   # "CHEVROLET (024)" -> "CHEVROLET"
    texto = re.sub(r"^[-\d\s]+", "", texto).strip("- .")   # "-136- FORD" -> "FORD"
    texto = re.sub(r"\s+", " ", texto)                     # "RENAULT     R" -> "RENAULT R"
    for patron, correcta in CORRECCION_MARCAS.items():
        if re.search(patron, texto):
            return correcta
    return texto


def limpiar_tipo(valor):
    """Normaliza el tipo de vehículo: el campo tiene ~1200 variantes de lo que en
    realidad son unas 15 categorías (código al inicio, puntuación, abreviaturas como
    'C/', 'VID.', 'PTAS' escritas de mil formas). 'SEDAN 5 PTAS' y 'SEDAN 5 PUERTAS'
    terminan siendo el mismo tipo."""
    texto = str(valor).strip().upper()
    texto = re.sub(r"^\d+\s*-\s*", "", texto)              # "12 - SEDAN 4 PUERTAS" -> "SEDAN 4 PUERTAS"
    texto = re.sub(r"^[A-ZÑ]{1,2}\s*-\s*", "", texto)       # "ZZ- FURGON VIDRIADO" -> "FURGON VIDRIADO"
    texto = texto.replace("-", " ").replace("/", " ")       # "PICK-UP" y "PICK UP" -> mismo texto
    texto = re.sub(r"\s+", " ", texto).strip(" .")
    texto = re.sub(r"(\d)([A-ZÑ])", r"\1 \2", texto)        # "5PUERTAS" -> "5 PUERTAS"
    texto = re.sub(r"([A-ZÑ])(\d)", r"\1 \2", texto)

    palabras = []
    for tok in texto.split():
        if re.fullmatch(r"PTAS?|PTS?|PUERTA", tok):
            tok = "PUERTAS"
        elif tok == "C":
            tok = "CON"
        elif tok.startswith("VID") and len(tok) < 8:         # VID/VIDR/VIDRIA/VIDRIAD... -> VIDRIADO
            tok = "VIDRIADO"
        elif tok.startswith("ASI") and len(tok) < 8:         # ASI/ASIE/ASIENT... -> ASIENTOS
            tok = "ASIENTOS"
        elif tok == "CAB":
            tok = "CABINA"
        palabras.append(tok)
    return " ".join(palabras)


# Los alias de marca que a veces aparecen repetidos al principio del campo modelo
# (p. ej. "VOLKSWAGEN VENTO 2.5", "VW SENDA", "V.W. GACEL")
ALIAS_MARCA = {"VOLKSWAGEN": ["VOLKSWAGEN", "VW", "V.W.", "V.W"], "CHEVROLET": ["CHEVROLET", "CHEVY"],
               "MERCEDES-BENZ": ["MERCEDES-BENZ", "MERCEDES BENZ", "MB"]}

# modelos cuyo código real incluye un guion seguido de números (Ford F-100, F-1000, etc.):
# se conservan enteros en vez de cortarlos en la primera palabra.
MODELO_CON_GUION = re.compile(r"^[A-ZÑ]+-\d+")
# códigos cortos donde el número ES el modelo, no la motorización (Citroën C3/C4,
# Chevrolet S10, Audi A4, BMW X1...): se conservan enteros.
MODELO_CODIGO_CORTO = re.compile(r"^[A-ZÑ]{1,2}\d{1,2}$")
# palabras que anteceden al nombre real del modelo ("NUEVO SANDERO" -> "SANDERO")
CALIFICADORES = {"NUEVO", "NUEVA", "NEW"}


def limpiar_modelo(marca, valor):
    """Nombre base del modelo (nameplate): la primera palabra reconocible del campo,
    sin el código de trámite, la marca repetida, ni la motorización/versión pegada.
    El campo mezcla modelo + versión + motor sin un separador fijo (p. ej. 'GOL TREND
    1.6 MSI', 'GOLF1.6', '-202-GOL'), así que en vez de intentar reconocer cada sigla de
    equipamiento (lista interminable) nos quedamos con la primera palabra: es una regla
    única, aplicable a cualquier marca, aunque a costa de perder la variante/versión."""
    texto = str(valor).strip().upper()
    texto = re.sub(r"\s*[\(\[\{].*?[\)\]\}]", "", texto)        # "GOL (024)" -> "GOL"
    texto = re.sub(r"^-+\d+-+\s*", "", texto).strip(" .")       # "-202-GOL" -> "GOL"
    # OJO: no se saca un dígito inicial suelto ("9", "206 ROLAND GARROS"),
    # porque en varios modelos el número ES el nombre (Renault 9, Peugeot 206...).
    for alias in ALIAS_MARCA.get(marca, [marca]):
        if texto == alias or texto.startswith(alias + " "):
            texto = texto[len(alias):].strip(" .-")
            break

    if not texto:              # el campo era solo un código o quedó vacío tras limpiar
        return "SIN ESPECIFICAR"

    palabras = texto.split()
    if palabras[0] in CALIFICADORES and len(palabras) > 1:
        palabras = palabras[1:]                                  # "NUEVO SANDERO" -> "SANDERO"

    primer_token = palabras[0]
    if MODELO_CON_GUION.match(primer_token) or MODELO_CODIGO_CORTO.match(primer_token):
        return primer_token                # "F-100" y "C3"/"S10" se mantienen enteros
    letras = re.match(r"[A-ZÑÁÉÍÓÚÜ]+", primer_token)
    if letras:
        return letras.group()                                     # "GOLF1.6" -> "GOLF"
    digitos = re.match(r"\d+", primer_token)
    return digitos.group() if digitos else primer_token            # "504SRDTCA" -> "504"


# ----------------------------------------------------------------------------
# Corrección de errores de tipeo por similitud de texto
# ----------------------------------------------------------------------------
UMBRAL_FRECUENTE = 30    # a partir de cuántas apariciones un texto se considera "correcto"
UMBRAL_SIMILITUD = 0.80  # qué tan parecido tiene que ser un texto raro a uno frecuente


def corregir_por_similitud(serie, umbral_frecuente=UMBRAL_FRECUENTE, umbral_similitud=UMBRAL_SIMILITUD,
                            proteger=frozenset()):
    """Detecta errores de tipeo comparando cada valor poco frecuente contra los valores
    frecuentes de la misma serie: si uno aparece una sola vez (p. ej. 'AMAROX') y es casi
    idéntico a otro que aparece miles de veces ('AMAROK'), el raro es casi con certeza un
    typo del frecuente y se reemplaza. Devuelve la serie corregida.
    No usa un catálogo externo de modelos (no lo tenemos): asume que la ortografía
    correcta es, dentro del propio dataset, la que más se repite.

    Resguardos para no fusionar valores que en realidad son distintos:
    - se ignoran los valores de 3 caracteres o menos (demasiado ambiguos: 'GL' y 'LK'
      son modelos Mercedes-Benz reales, no un typo de 'GLK').
    - si el valor raro tiene números, solo se compara contra frecuentes con exactamente
      los mismos números ('F-350' nunca se corrige a 'F-150': son pickups distintas,
      aunque el texto se parezca).
    - 'proteger' es un conjunto de valores que nunca se tocan, ni como origen ni como
      destino de una corrección (p. ej. 'OMNIBUS', que por similitud de texto podría
      fundirse con 'MINIBUS' aunque son categorías de vehículo distintas)."""
    conteos = serie.value_counts()
    frecuentes = [f for f in conteos[conteos >= umbral_frecuente].index if f not in proteger]
    raros = [v for v in conteos[conteos < umbral_frecuente].index if v not in proteger]
    if not frecuentes or not raros:
        return serie

    correccion = {}
    for valor in raros:
        if len(valor) <= 3:
            continue

        # regla exacta (no probabilística): si el texto empieza con el nombre completo
        # de un valor frecuente, se corta ahí. "NISSAN PATHFINDER" -> "NISSAN",
        # "FORD ARGENTINA S.C.A" -> "FORD" (quedó el modelo o la razón social pegados
        # al nombre de la marca).
        prefijos = [f for f in frecuentes if re.match(rf"^{re.escape(f)}\b", valor)]
        if prefijos:
            correccion[valor] = max(prefijos, key=len)
            continue

        digitos_valor = re.findall(r"\d+", valor)
        candidatos = [f for f in frecuentes if not digitos_valor or re.findall(r"\d+", f) == digitos_valor]
        if not candidatos:
            continue
        mejor = max(candidatos, key=lambda f: difflib.SequenceMatcher(None, valor, f).ratio())
        ratio = difflib.SequenceMatcher(None, valor, mejor).ratio()
        if ratio >= umbral_similitud:
            correccion[valor] = mejor
    return serie.replace(correccion)


def preparar_dataset(df):
    """Devuelve un DataFrame limpio con la variable objetivo y las features."""
    out = pd.DataFrame()

    # Variable binaria: 1 = comunicación de recupero, 0 = denuncia de robo
    out["es_recuperado"] = df["tramite_tipo"].str.upper().str.contains("RECUPERO").astype(int)
    out["fecha"] = pd.to_datetime(df["tramite_fecha"], errors="coerce")
    out["marca"] = df["automotor_marca_descripcion"].map(limpiar_marca)
    # umbral más bajo que el de modelo: hay pocas marcas reales (~30 frecuentes) y muy
    # alejadas entre sí en texto, así que el riesgo de confundir dos marcas distintas es
    # bajo. Con 0.80 quedaban afuera casos como "REANUT" -> "RENAULT" (0.77 de similitud).
    out["marca"] = corregir_por_similitud(out["marca"], umbral_similitud=0.75)

    out["modelo"] = [limpiar_modelo(m, v) for m, v in
                      zip(out["marca"], df["automotor_modelo_descripcion"])]
    # la corrección de modelo se hace marca por marca: un typo raro de Ford no debe
    # confundirse con un modelo distinto y legítimo de otra marca
    out["modelo"] = out.groupby("marca")["modelo"].transform(corregir_por_similitud)
    out["tipo_vehiculo"] = df["automotor_tipo_descripcion"].map(limpiar_tipo)
    # umbral más alto que el de marca/modelo: el texto es más largo y variado, así que
    # hace falta más similitud para confiar en la corrección. "OMNIBUS" y "MINIBUS" son
    # categorías distintas (colectivo grande vs. chico) aunque el texto se parezca.
    out["tipo_vehiculo"] = corregir_por_similitud(out["tipo_vehiculo"], umbral_similitud=0.85,
                                                   proteger={"OMNIBUS"})
    out["origen"] = df["automotor_origen"].astype(str).str.strip().str.upper()
    out["provincia"] = (df["registro_seccional_provincia"].astype(str).str.strip()
                        .replace({"Ciudad Autónoma de Bs.As.": "Ciudad Autónoma de Buenos Aires"}))
    out["anio_modelo"] = pd.to_numeric(df["automotor_anio_modelo"], errors="coerce")

    # Features derivadas
    out["anio_tramite"] = out["fecha"].dt.year
    out["mes_tramite"] = out["fecha"].dt.month
    out["antiguedad_vehiculo"] = out["anio_tramite"] - out["anio_modelo"]

    # Descartar registros inválidos: sin fecha, marcas genéricas, años de modelo imposibles
    out = out.dropna(subset=["fecha", "anio_modelo"])
    out = out[~out["marca"].isin(["OTROS", "DESCONOCIDO", "NAN", "NONE", ""])]
    # campos de texto vacíos (según la versión de pandas llegan como NaN o como "nan")
    columnas_texto = ["modelo", "tipo_vehiculo", "origen", "provincia"]
    out = out.dropna(subset=columnas_texto)
    vacios = out[columnas_texto].apply(lambda col: col.str.upper().isin(["NAN", "NONE", ""])).any(axis=1)
    out = out[~vacios]
    out = out[(out["antiguedad_vehiculo"] >= 0) & (out["antiguedad_vehiculo"] <= 50)]
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. ANÁLISIS EXPLORATORIO + PRUEBA DE LA HIPÓTESIS
# ----------------------------------------------------------------------------
def analisis_exploratorio(df):
    print("\n=== RESUMEN GENERAL ===")
    print(f"Registros: {len(df):,} | Período: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    robos = (df["es_recuperado"] == 0).sum()
    recup = (df["es_recuperado"] == 1).sum()
    print(f"Denuncias de robo: {robos:,} | Comunicaciones de recupero: {recup:,} "
          f"({recup / len(df) * 100:.2f}%)")
    print(f"Antigüedad promedio del vehículo: {df['antiguedad_vehiculo'].mean():.1f} años")

    print("\nTop 10 marcas con más trámites:")
    print(df["marca"].value_counts().head(10).to_string())

    print("\nMarcas (>=2000 trámites) con mayor tasa de recupero:")
    por_marca = df.groupby("marca")["es_recuperado"].agg(total="count", tasa="mean")
    print((por_marca[por_marca["total"] >= 2000].sort_values("tasa", ascending=False)
           .head(5).assign(tasa=lambda x: (x["tasa"] * 100).round(2)).to_string()))

    print("\nProvincias con más trámites y su tasa de recupero (%):")
    por_prov = df.groupby("provincia")["es_recuperado"].agg(total="count", tasa="mean")
    print(por_prov.sort_values("total", ascending=False).head(8)
          .assign(tasa=lambda x: (x["tasa"] * 100).round(2)).to_string())

    # --- HIPÓTESIS H1 ---
    # H1: los vehículos de hasta 5 años de antigüedad tienen mayor tasa de recupero
    #     que los de más de 15 años (misma fuente, misma ventana temporal).
    nuevos = df[df["antiguedad_vehiculo"] <= 5]["es_recuperado"]
    viejos = df[df["antiguedad_vehiculo"] > 15]["es_recuperado"]
    tabla = [[nuevos.sum(), len(nuevos) - nuevos.sum()], [viejos.sum(), len(viejos) - viejos.sum()]]
    chi2, p, _, _ = chi2_contingency(tabla)
    print("\n=== HIPÓTESIS H1 (antigüedad vs. recupero) ===")
    print(f"Tasa de recupero vehículos <=5 años : {nuevos.mean() * 100:.2f}%  (n={len(nuevos):,})")
    print(f"Tasa de recupero vehículos >15 años : {viejos.mean() * 100:.2f}%  (n={len(viejos):,})")
    print(f"Test chi-cuadrado: chi2={chi2:.1f}, p-valor={p:.2e} -> "
          f"{'diferencia significativa' if p < 0.05 else 'sin diferencia significativa'}")

    # Gráficos
    sns.set_theme(style="whitegrid")
    fig, ejes = plt.subplots(1, 3, figsize=(17, 4.5))
    sns.histplot(df["antiguedad_vehiculo"], bins=25, ax=ejes[0], color="skyblue")
    ejes[0].set(title="Antigüedad del vehículo", xlabel="Años", ylabel="Trámites")
    tasa_ant = df.groupby("antiguedad_vehiculo")["es_recuperado"].mean() * 100
    ejes[1].plot(tasa_ant.index[:30], tasa_ant.values[:30], marker="o", color="crimson")
    ejes[1].set(title="Tasa de recupero vs. antigüedad", xlabel="Años", ylabel="% recuperado")
    top = df["marca"].value_counts().head(10)
    sns.barplot(x=top.values, y=top.index, ax=ejes[2], color="steelblue")
    ejes[2].set(title="Top 10 marcas", xlabel="Trámites", ylabel="")
    plt.tight_layout()
    plt.savefig("eda_resultados.png", dpi=130)
    print("Gráficos guardados en eda_resultados.png")


# ----------------------------------------------------------------------------
# 4. MODELO PREDICTIVO (XGBoost)
# ----------------------------------------------------------------------------
def entrenar_modelo(df):
    """Entrena un clasificador que estima la probabilidad de que el registro
    de un vehículo sea un recupero. Guarda el modelo y las categorías usadas."""
    data = df[FEATURES + ["es_recuperado"]].dropna().copy()

    # Las categorías se guardan para codificar igual al cotizar
    categorias = {}
    for col in FEATURES_CAT:
        data[col] = data[col].astype("category")
        categorias[col] = list(data[col].cat.categories)
        data[col] = data[col].cat.codes

    X, y = data[FEATURES], data["es_recuperado"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # La clase "recuperado" es minoritaria (~3.5%). No se pondera artificialmente
    # para que predict_proba devuelva probabilidades reales (calibradas).
    modelo = XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42)
    modelo.fit(X_tr, y_tr)

    proba = modelo.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    print("\n=== MODELO XGBoost ===")
    print(f"Entrenamiento: {len(X_tr):,} | Test: {len(X_te):,}")
    print(f"ROC-AUC en test: {auc:.4f}   (0.5 = azar, 1.0 = perfecto)")
    print(f"Average precision: {average_precision_score(y_te, proba):.4f}   "
          f"(referencia si fuera azar: {y_te.mean():.4f})")
    # Los 20% de registros con mayor probabilidad: qué proporción de los recuperos capturan
    corte = np.quantile(proba, 0.80)
    captura = y_te[proba >= corte].sum() / y_te.sum()
    print(f"El 20% de casos con mayor probabilidad concentra el {captura * 100:.1f}% de los recuperos")

    importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Importancia de variables:")
    print(importancias.round(3).to_string())

    joblib.dump({"modelo": modelo, "categorias": categorias, "auc": auc}, ARCHIVO_MODELO)
    print(f"Modelo guardado en {ARCHIVO_MODELO}")
    return auc


# ----------------------------------------------------------------------------
# 5. COTIZADOR (usa el modelo + frecuencia de siniestros)
# ----------------------------------------------------------------------------
def cargar_cotizador():
    """Carga modelo y dataset limpio; devuelve la función cotizar()."""
    paquete = joblib.load(ARCHIVO_MODELO)
    modelo, categorias = paquete["modelo"], paquete["categorias"]
    ref = pd.read_csv(ARCHIVO_LIMPIO)
    freq_marca = ref["marca"].value_counts(normalize=True).to_dict()
    freq_prov = ref["provincia"].value_counts(normalize=True).to_dict()

    def codigo(col, valor):
        cats = categorias[col]
        return cats.index(valor) if valor in cats else 0

    def a_matriz(marca, tipo_vehiculo, origen, provincia, anio_modelo, mes):
        return pd.DataFrame([{
            "marca": codigo("marca", marca),
            "tipo_vehiculo": codigo("tipo_vehiculo", tipo_vehiculo),
            "origen": codigo("origen", origen),
            "provincia": codigo("provincia", provincia),
            "antiguedad_vehiculo": ANIO_ACTUAL - anio_modelo,
            "mes_tramite": mes,
        }])

    # Distribución de referencia: probabilidades de recupero de una muestra real.
    # Sirve para convertir cada cotización en un percentil (0 a 1).
    muestra = ref.dropna(subset=FEATURES).sample(min(20000, len(ref)), random_state=1)
    ref_prob = np.sort(modelo.predict_proba(
        pd.DataFrame({c: muestra[c].map(lambda v, c=c: codigo(c, v)) for c in FEATURES_CAT}
                     | {"antiguedad_vehiculo": muestra["antiguedad_vehiculo"],
                        "mes_tramite": muestra["mes_tramite"]})[FEATURES])[:, 1])
    ref_freq = np.sort(muestra["marca"].map(freq_marca) * 0.6 + muestra["provincia"].map(freq_prov) * 0.4)

    def percentil(valor, distribucion):
        return float(np.searchsorted(distribucion, valor) / len(distribucion))

    def cotizar(marca, tipo_vehiculo, origen, provincia, anio_modelo, mes=9):
        marca, tipo_vehiculo, origen = marca.upper(), tipo_vehiculo.upper(), origen.upper()
        prob_recupero = float(modelo.predict_proba(
            a_matriz(marca, tipo_vehiculo, origen, provincia, anio_modelo, mes))[0][1])
        factor_frecuencia = freq_marca.get(marca, 0.0) * 0.6 + freq_prov.get(provincia, 0.0) * 0.4

        # Score de severidad (0 a 1) = combinación de dos percentiles:
        #   - riesgo de NO recuperar el auto (menor prob. de recupero -> mayor riesgo)
        #   - frecuencia de robo de esa marca y provincia
        # SUPUESTO del grupo: los pesos 0.6 / 0.4 y los cortes siguientes son criterios
        # de negocio elegidos por nosotros; no se estimaron con datos.
        riesgo_no_recupero = 1 - percentil(prob_recupero, ref_prob)
        riesgo_frecuencia = percentil(factor_frecuencia, ref_freq)
        score = 0.6 * riesgo_no_recupero + 0.4 * riesgo_frecuencia

        # Matriz de decisión. SUPUESTO: los recargos son ilustrativos.
        if score >= 0.75:
            categoria, prima, rastreador = "CRÍTICO", "+35% sobre prima técnica", "Obligatorio"
        elif score >= 0.55:
            categoria, prima, rastreador = "ALTO", "+18% sobre prima técnica", "Obligatorio (bonificado 50%)"
        elif score >= 0.35:
            categoria, prima, rastreador = "MEDIO", "Tarifa estándar", "Recomendado"
        else:
            categoria, prima, rastreador = "BAJO", "-5% (descuento por bajo riesgo)", "No requiere"

        return {"prob_recupero": prob_recupero, "score": score, "categoria": categoria,
                "prima": prima, "rastreador": rastreador}

    return cotizar


def mostrar_cotizacion(datos, res):
    print("-" * 55)
    print(f"{datos[0]} {datos[4]} | {datos[1]} | {datos[3]}")
    print(f"  Prob. de recupero (ML): {res['prob_recupero'] * 100:.2f}%")
    print(f"  Score de severidad    : {res['score']:.3f}  -> {res['categoria']}")
    print(f"  Prima: {res['prima']} | Rastreador: {res['rastreador']}")


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("1) Descargando y unificando datos ...")
    descargar_datos()
    bruto = cargar_datos()

    print("\n2) Limpiando y generando variables ...")
    limpio = preparar_dataset(bruto)
    limpio.to_csv(ARCHIVO_LIMPIO, index=False)
    print(f"  {len(limpio):,} registros válidos guardados en {ARCHIVO_LIMPIO}")

    print("\n3) Análisis exploratorio ...")
    analisis_exploratorio(limpio)

    print("\n4) Entrenando modelo ...")
    entrenar_modelo(limpio)

    print("\n5) Ejemplos de cotización ...")
    cotizar = cargar_cotizador()
    ejemplos = [
        ("VOLKSWAGEN", "SEDAN 5 PUERTAS", "Nacional", "Buenos Aires", 2012),
        ("TOYOTA", "SEDAN 4 PUERTAS", "Importado", "Ciudad Autónoma de Buenos Aires", 2022),
        ("HONDA", "SEDAN 4 PUERTAS", "Importado", "Entre Ríos", 2020),
    ]
    for e in ejemplos:
        mostrar_cotizacion(e, cotizar(*e))
