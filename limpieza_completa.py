"""
====================================================================================================
TRABAJO PRÁCTICO IFTS - PIPELINE DEFINITIVO DE LIMPIEZA Y TRANSFORMACIÓN DE DATOS VEHICULARES
====================================================================================================
Archivo: limpieza_completa.py
Dataset de Entrada: dataset_limpio.csv (227.312 registros de trámites de la DNRPA)
Archivos de Salida: base_limpia.xlsx (hoja 'base_limpia') y base_limpia.csv (UTF-8 con BOM)

Autores: Alumnos de la Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial (IFTS)
Fecha: Noviembre 2026
====================================================================================================
"""

import os
import re
import time
import pandas as pd
import numpy as np

# ==================================================================================================
# 1. CATÁLOGOS Y DICCIONARIOS DE DOMINIO AUTOMOTOR
# ==================================================================================================

# 1.1 Unificación de marcas y corrección de codificación / variantes
REEMPLAZOS_MARCA = {
    'MERCEDES': 'MERCEDES-BENZ',
    'MERCEDES BENZ': 'MERCEDES-BENZ',
    'MERCEDES-BENZ': 'MERCEDES-BENZ',
    'DEUTZ-AGRALE': 'DEUTZ AGRALE',
    'DEUTZ - AGRALE': 'DEUTZ AGRALE',
    'DEUTZ': 'DEUTZ AGRALE',
    'SAAB SCANIA': 'SCANIA',
    'PRATI-FRUEHAUF': 'FRUEHAUF',
    'A.F.F': 'AFF',
    'AST PRA': 'ASTPRA',
    'AST-PRA': 'ASTPRA',
    'R E N A U L T': 'RENAULT',
    'NIÃ‘O JESUS DE PRAGA': 'NIÑO JESUS DE PRAGA',
    'NIÑO JESUS DE PRAGA': 'NIÑO JESUS DE PRAGA',
}

# 1.2 Modelos invertidos cargados erróneamente en la columna 'marca'
MODELOS_EN_MARCA = {
    'CHEVETTE': ('CHEVROLET', 'CHEVETTE'),
    'GMC CHEVETTE': ('CHEVROLET', 'CHEVETTE'),
    'G.M.C. CHEVETTE': ('CHEVROLET', 'CHEVETTE'),
    'DUNA W.E. 1.6': ('FIAT', 'DUNA WEEKEND'),
    'PICASSO': ('CITROEN', 'XSARA PICASSO'),
    'TORINO': ('RENAULT', 'TORINO'),
    'RASTROJERO DIESEL': ('IME', 'RASTROJERO'),
}

# 1.3 Corrección de abreviaturas del registro y desambiguación (marca, modelo_original)
CORRECCIONES_MODELO = {
    # --- PEUGEOT ---
    # Deshace el bug donde PUP se asociaba a Partner; PUP es históricamente la 504 Pick-Up
    ('PEUGEOT', 'PUP'): '504 PICK UP',
    ('PEUGEOT', 'PU'): '504 PICK UP',
    ('PEUGEOT', 'PARPADDAA'): 'PARTNER',
    ('PEUGEOT', 'PARFUD'): 'PARTNER',
    ('PEUGEOT', 'PARPAD'): 'PARTNER',
    ('PEUGEOT', 'PAR'): 'PARTNER',
    ('PEUGEOT', 'PART'): 'PARTNER',
    ('PEUGEOT', 'PARPANDAADAB'): 'PARTNER',
    ('PEUGEOT', 'ROLAND'): '206 ROLAND GARROS',

    # --- VOLKSWAGEN ---
    # Niveles de equipamiento comercializados del VW Up! agrupados bajo el modelo oficial
    ('VOLKSWAGEN', 'HIGH'): 'UP!',
    ('VOLKSWAGEN', 'MOVE'): 'UP!',
    ('VOLKSWAGEN', 'TAKE'): 'UP!',
    ('VOLKSWAGEN', 'WHITE'): 'UP!',
    ('VOLKSWAGEN', 'PEPPER'): 'UP!',
    ('VOLKSWAGEN', 'BLACK'): 'UP!',
    ('VOLKSWAGEN', 'CROSS'): 'CROSS UP!',
    ('VOLKSWAGEN', 'T'): 'T-CROSS',
    ('VOLKSWAGEN', 'THE'): 'THE BEETLE',

    # --- CITROEN ---
    ('CITROEN', 'BERMUL'): 'BERLINGO MULTISPACE',
    ('CITROEN', 'BERFUD'): 'BERLINGO',
    ('CITROEN', 'BERMUNDPLDAB'): 'BERLINGO',
    ('CITROEN', 'GRAND'): 'GRAND C4 PICASSO',
    ('CITROEN', 'C'): 'C-ELYSEE',
    ('CITROEN', 'CE'): 'C-ELYSEE',
    ('CITROEN', 'CITROENXSARA'): 'XSARA',
    ('CITROEN', 'FAM'): 'BERLINGO MULTISPACE',
    ('CITROEN', 'FUR'): 'BERLINGO',
    ('CITROEN', 'X'): 'XSARA PICASSO',

    # --- CHEVROLET ---
    ('CHEVROLET', 'S'): 'S10',
    ('CHEVROLET', 'GRAND'): 'GRAND VITARA',
    ('CHEVROLET', '/10'): 'C-10',
    ('CHEVROLET', '/10/1975'): 'C-10',
    ('CHEVROLET', 'C-10SIL'): 'C-10',
    ('CHEVROLET', 'C-10CUSD4'): 'C-10',
    ('CHEVROLET', 'C-10CUS'): 'C-10',
    ('CHEVROLET', 'C10'): 'C-10',
    ('CHEVROLET', 'D20'): 'D-20',
    ('CHEVROLET', 'GMT'): 'SILVERADO',
    ('CHEVROLET', 'G'): 'GRAND VITARA',
    ('CHEVROLET', 'CHEVROLET'): 'SIN ESPECIFICAR',

    # --- DAIHATSU ---
    # Saneamiento del guion inicial (-CUORE) que provocaba error #NAME? en Power Query
    ('DAIHATSU', 'CUORE'): 'CUORE',
    ('DAIHATSU', '-CUORE'): 'CUORE',

    # --- FORD ---
    ('FORD', 'FOCUSEXE'): 'FOCUS EXE',
    ('FORD', 'S'): 'S-MAX',
    ('FORD', 'K'): 'KA',

    # --- DESAMBIGUACIÓN DEL NOMBRE 'GRAND' SEGÚN LA MARCA ---
    ('FIAT', 'GRAND'): 'GRAND SIENA',
    ('FIAT', 'WEEKEND'): 'PALIO WEEKEND',
    ('FIAT', 'SPAZIO'): '147 SPAZIO',
    ('FIAT', 'VIVACE'): '147 VIVACE',
    ('FIAT', 'BRIO'): '147 BRIO',
    ('FIAT', 'EUROPA'): '147 EUROPA',
    ('FIAT', 'SUPEURO'): '147 SUPER EUROPA',
    ('FIAT', '1'): 'UNO',
    ('FIAT', 'SE'): 'SIENA',

    ('JEEP', 'GRAND'): 'GRAND CHEROKEE',
    ('JEEP', 'GR'): 'GRAND CHEROKEE',
    ('SUZUKI', 'GRAND'): 'GRAND VITARA',
    ('SUZUKI', 'G'): 'GRAND VITARA',
    ('SUZUKI', 'SE'): 'SWIFT',
    ('CHRYSLER', 'GRAND'): 'GRAND CARAVAN',
    ('CHRYSLER', 'PT'): 'PT CRUISER',
    ('KIA', 'GRAND'): 'GRAND CARNIVAL',
    ('KIA', 'K'): 'K2500',
    ('KIA', 'CEE'): "CEE'D",
    ('HYUNDAI', 'GRAND'): 'GRAND I10',
    ('HYUNDAI', 'SANTA'): 'SANTA FE',
    ('HYUNDAI', 'H'): 'H-100',
    ('HYUNDAI', 'HI'): 'H-1',
    ('HYUNDAI', 'H1'): 'H-1',
    ('HYUNDAI', 'I'): 'I10',
    ('RENAULT', 'GRAND'): 'GRAND SCENIC',
    ('DODGE', 'GRAND'): 'GRAND CARAVAN',

    # --- MERCEDES-BENZ ---
    ('MERCEDES-BENZ', 'A'): 'CLASE A',
    ('MERCEDES-BENZ', 'B'): 'CLASE B',
    ('MERCEDES-BENZ', 'C'): 'CLASE C',
    ('MERCEDES-BENZ', 'E'): 'CLASE E',
    ('MERCEDES-BENZ', 'ML'): 'CLASE ML',
    ('MERCEDES-BENZ', 'GL'): 'CLASE GL',
    ('MERCEDES-BENZ', 'SPORTCOUPE'): 'CLASE C SPORTCOUPE',

    # --- RENAULT ---
    ('RENAULT', 'R'): '12',
    ('RENAULT', 'R-12'): '12',
    ('RENAULT', 'R12'): '12',
    ('RENAULT', 'R9'): '9',
    ('RENAULT', 'RENAULT-9'): '9',
    ('RENAULT', 'R19'): '19',
    ('RENAULT', 'R21'): '21',
    ('RENAULT', 'R18'): '18',
    ('RENAULT', 'CLI'): 'CLIO',
    ('RENAULT', 'KANGOOCONFORT'): 'KANGOO',
    ('RENAULT', 'REANULT'): '19',
    ('RENAULT', 'RENAUL'): '19',
    ('RENAULT', 'RENAUTL'): '19',
    ('RENAULT', 'RENAUKT'): '12',
    ('RENAULT', 'RENALT'): '9',
    ('RENAULT', 'RE'): '9',

    # --- TOYOTA ---
    ('TOYOTA', 'RAV'): 'RAV4',
    ('TOYOTA', 'LAND'): 'LAND CRUISER',
    ('TOYOTA', '4'): '4RUNNER',
    ('TOYOTA', 'RUNNER'): '4RUNNER',
    ('TOYOTA', 'TOYOYA'): 'COROLLA',

    # --- MITSUBISHI, HONDA, NISSAN, AUDI, BMW ---
    ('MITSUBISHI', 'L'): 'L200',
    ('HONDA', 'HR'): 'HR-V',
    ('HONDA', 'CR'): 'CR-V',
    ('HONDA', 'CRV'): 'CR-V',
    ('HONDA', 'WR'): 'WR-V',
    ('HONDA', 'ZR'): 'ZR-V',
    ('NISSAN', 'X'): 'X-TRAIL',
    ('NISSAN', 'NP'): 'NP300 FRONTIER',
    ('NISSAN', 'DOBLE'): 'FRONTIER',
    ('NISSAN', 'CABINA'): 'FRONTIER',
    ('AUDI', 'A'): 'A3',
    ('AUDI', 'A-3'): 'A3',
    ('BMW', 'M'): 'SERIE M',
}

# 1.4 Fabricantes exclusivos de remolques y acoplados (100% unidades de carga sin tracción propia)
FABRICANTES_ACOPLADOS = {
    'HERMANN', 'SALTO', 'PETINARI', 'BONANO', 'SOLA Y BRUSA', 'RANDON', 'MONTENEGRO',
    'CORMETAL', 'HELVETICA', 'MALDONADO', 'RUTERA', 'MANQUE', 'ASTPRA', 'AST-PRA',
    'AST PRA', 'GOMATRO', 'GRASSANI', 'OMBU', 'PATRONELLI', 'PINCEN', 'LAMBERT',
    'SATURNI', 'VULCANO', 'GUERRA', 'NOMA', 'MACTRAIL', 'ULTRANS', 'DEPAOLI', 'FRUEHAUF',
    'SABINO', 'PLUSCARGA', 'GOLONDRINA', 'EL FIERRO', 'MORALES', 'ACOPLAR', 'ASTIVIA',
    'BARBARO', 'BIANCHI', 'CRESCENTE', 'DANES', 'DARWIN', 'DELL ORO', 'DI POLVERE',
    'DIMATAN', 'F.H', 'FEDERAL', 'FERBUS', 'GAMMA', 'IC TRAILERS', 'IDEROL', 'INDECAR',
    'INDUSTRIAS AHT ALZUARTE', 'INDUTEK', 'LUIS IGLESIAS', 'M.CORDANI', 'METAGRO',
    'METALURGICA FRANCK', 'METALURGICA MAURO', 'MIC', 'MILEI', 'MIRBAL', 'MM-MARTIN MEYER',
    'MONTEBRAS', 'NAKAI', 'NIÑO JESUS DE PRAGA', 'RIMEK', 'RUBIOLO', 'SAN MARINO',
    'SCHELL', 'STAHL', 'STAR', 'TAHNOS', 'TORRESI', 'TRAIL KING', 'TRENQUEMOLQUE'
}

# 1.5 Fabricantes exclusivos de camiones pesados
FABRICANTES_CAMIONES = {
    'SCANIA', 'FREIGHTLINER', 'INTERNATIONAL', 'BELAVTOMAZ', 'MACK', 'KENWORTH', 'HINO'
}

# 1.6 Catálogos de modelos puros para garantizar consistencia absoluta
MODELOS_CAMION = {
    'CARGO', 'F-4000', 'SERIE F CAMION', 'ATEGO', 'AXOR', 'ACTROS', 'ACCELO', 'ATRON',
    '710', '712', '709', '1215', '1634', 'LS-1634', '1720', '1624', 'L-1624', '17220',
    '18310', '17310', '17210', '17300', '15160', '9150', 'CONSTELLATION', 'DELIVERY',
    'STRALIS', 'TECTOR', 'EUROCARGO', 'TRAKKER', 'CURSOR', 'CAVALLINO', 'EUROTRAKKER',
    'FH', 'FM', 'VM', 'NH', 'NL', '13', '17', '19', '26', '31', '15', '180', '693', '695',
    '1214', '1418', '1420', '1618', '1633', '1718', '1722', '1728', '2318', '2423'
}

MODELOS_4X4 = {
    'HILUX', 'AMAROK', 'RANGER', 'S10', 'FRONTIER', 'NP300 FRONTIER', 'TORO', 'STRADA',
    'SAVEIRO', 'MONTANA', 'F-100', 'F-150', '1500', 'RAM', 'RAMPAGE', 'L200', 'ALASKAN',
    'SILVERADO', 'LUV', 'D-20', 'C-10', 'TFR', 'TFS', 'MAVERICK', 'COURIER', '504 PICK UP',
    'OROCH', 'TITANO', 'RANCHERO', 'D20', 'C10'
}

MODELOS_SUV = {
    'ECOSPORT', 'DUSTER', 'TRACKER', 'RENEGADE', 'COMPASS', 'SW4', 'CR-V', 'HR-V', 'WR-V', 'ZR-V',
    'RAV4', 'LAND CRUISER', '4RUNNER', 'TIGGO', 'TUCSON', 'KICKS', 'CAPTUR', 'NIVUS', 'TAOS',
    'T-CROSS', 'TERRITORY', 'JOURNEY', 'KUGA', '2008', '3008', '5008', '4008', 'Q2', 'Q3', 'Q5',
    'Q7', 'Q8', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'BLAZER', 'TRAILBLAZER', 'CAPTIVA', 'EQUINOX',
    'KOLEOS', 'SORENTO', 'SPORTAGE', 'SELTOS', 'SANTA FE', 'VITARA', 'GRAND VITARA',
    'GRAND CHEROKEE', 'CHEROKEE', 'PATRIOT', 'WRANGLER', 'FREELANDER', 'DISCOVERY', 'EXPLORER',
    'BRONCO', 'PULSE', 'FASTBACK', 'X-TRAIL', 'PATHFINDER', 'MURANO', 'GLK', 'GLA', 'GLB', 'GLC', 'GLE',
    'GLS', 'ML', 'GL', 'TIGUAN', 'TOUAREG', 'TERA', 'CROSSFOX', 'BASALT', 'AVALANCHE', 'STEPWAY',
    '500X', 'SIDEKICK', 'PASSPORT', 'TERRANO', 'MOHAVE', 'KODIAQ'
}

MODELOS_UTILITARIO = {
    'KANGOO', 'PARTNER', 'BERLINGO', 'BERLINGO MULTISPACE', 'FIORINO', 'SPRINTER',
    'MASTER', 'TRAFIC', 'DUCATO', 'BOXER', 'DAILY', 'TRANSIT', 'CADDY', 'EXPRESS',
    'QUBO', 'DOBLO', 'VITO', 'EXPERT', 'JUMPER', 'JUMPY', 'H-1', 'H-100', 'FOISON',
    'CARNIVAL', 'GRAND CARNIVAL', 'BESTA', 'PREGIO', 'INCA', 'TRANSPORTER', 'K2500', 'VIANO',
    'TOPIC', 'TOWNER', 'DAMAS', 'WINDSTAR', 'CARAVAN', 'GRAND CARAVAN', 'VOYAGER'
}

MODELOS_PASAJEROS = {
    'BMO', 'BM', 'OF', 'OH', 'OHL', 'LO-814', 'LO-914', 'LO-812', 'LO', 'OH-1518', 'OH-1315'
}

SEDANES_4P = {
    'SIENA', 'GRAND SIENA', 'VOYAGE', 'LOGAN', 'PRISMA', 'COBALT', 'COROLLA', 'CIVIC',
    'VENTO', 'BORA', 'FLUENCE', '408', '405', '406', '407', '504', '505', '508', 'DUNA',
    '9', '18', '21', 'FOCUS EXE', 'VIRTUS', 'CRONOS', 'VERSA', 'SENTRA', 'CITY', 'ACCORD',
    'LINEA', 'SYMBOL', 'C-ELYSEE', '301', 'CLASSIC', 'AVEO', 'ASTRA', 'MEGANE', 'PASSAT',
    'MONDEO', 'CORONA', 'CAMRY', '404', 'TORINO', 'REGATTA', 'TEMPRA', '128'
}


# ==================================================================================================
# 2. FUNCIONES DE LIMPIEZA Y TRANSFORMACIÓN
# ==================================================================================================

def clasificar_tipo_vehiculo(row):
    """
    Clasifica de manera jerárquica y determinística el tipo de vehículo.
    Resuelve inconsistencias causadas por abreviaturas atípicas del DNRPA.
    """
    marca = row['marca']
    mod = row['modelo']
    tipo_orig = str(row['_tipo_orig']).upper()

    # Nivel 1: Fabricantes exclusivos
    if marca in FABRICANTES_ACOPLADOS:
        return 'Acoplado / Remolque'
    if marca in FABRICANTES_CAMIONES:
        return 'Camion'

    # Nivel 2: Modelos con homologación automotriz indiscutible
    if mod in MODELOS_PASAJEROS or any(k in tipo_orig for k in ['OMNIBUS', 'MINIBUS', 'MIDIBUS', 'COLECTIVO', 'MICRO']):
        return 'Transporte de Pasajeros'
    if mod in MODELOS_CAMION or (marca in {'MERCEDES-BENZ', 'VOLKSWAGEN', 'FORD', 'IVECO'} and any(k in tipo_orig for k in ['CAMION', 'CAMIÓN', 'TRACT', 'CHASIS', 'CHAS.', 'CH.'])):
        if mod not in {'DAILY', 'K2500'}:
            return 'Camion'
    if mod in MODELOS_4X4:
        return '4x4'
    if mod in MODELOS_SUV:
        return 'SUV'
    if mod in MODELOS_UTILITARIO:
        return 'Utilitario / Furgon'

    # Nivel 3: Variantes comerciales de modelos con carrocerías mixtas de fábrica
    if mod in {'UNO', 'CORSA', 'CLASSIC'} and any(k in tipo_orig for k in ['FURGON', 'UTILITARIO', 'CARGO']):
        return 'Utilitario / Furgon'
    if mod == '504' and 'PICK' in tipo_orig:
        return '4x4'

    # Nivel 4: Reglas textuales de descarte para modelos poco comunes
    if any(k in tipo_orig for k in ['ACOPLAD', 'SEMIRREM', 'SEMI-REM', 'SEMIACOPL', 'REMOLQU', 'CARRETON', 'BATAN', 'TRAILER']):
        return 'Acoplado / Remolque'
    if any(k in tipo_orig for k in ['PICK UP', 'PICK-UP', 'CABINA DOBLE', 'CABINA SIMPLE', 'CABINA Y MEDIA']):
        return '4x4'
    if any(k in tipo_orig for k in ['TODO TERRENO', 'SUV', 'JEEP']):
        return 'SUV'
    if any(k in tipo_orig for k in ['FURGON', 'FOURGON', 'UTILITARIO', 'COMBI', 'KOMBI', 'VAN', 'FURGONETA']):
        return 'Utilitario / Furgon'

    return 'Auto'


def extraer_puertas(row):
    """
    Determina la cantidad de puertas del vehículo mediante:
    1. Extracción numérica explícita por regex sobre el texto.
    2. Imputación técnica estándar basada en el modelo y tipo de carrocería.
    3. Asignación lógica de NaN a unidades de arrastre sin cabina (Acoplado / Remolque).
    """
    texto = f"{row['modelo']} {row['_tipo_orig']}".upper()

    # Búsqueda explícita
    m = re.search(r'\b([2345])\s*(?:PUERTAS|PTAS|PTA|P)\b', texto)
    if m:
        return int(m.group(1))
    if '4 5 PUERTAS' in texto:
        return 5
    if '2 3 PUERTAS' in texto:
        return 3
    if 'CABINA DOBLE' in texto:
        return 4
    if 'CABINA SIMPLE' in texto or 'CABINA SIMPL' in texto:
        return 2
    if 'COUPE' in texto or 'DESCAPOTABLE' in texto or 'CABRIOLET' in texto:
        return 2

    # Imputación por dominio
    tipo = row['tipo_vehiculo_limpio']
    mod = row['modelo']

    if tipo == 'Acoplado / Remolque':
        return np.nan
    if tipo in {'Camion', 'Transporte de Pasajeros'}:
        return 2
    if tipo == 'SUV':
        return 5
    if tipo == '4x4':
        if any(k in texto for k in ['SIMPLE', 'SIMPL']) or mod in {'SAVEIRO', 'STRADA', 'MONTANA', 'COURIER', 'F-100', '504 PICK UP', 'LUV'}:
            return 2
        return 4
    if tipo == 'Utilitario / Furgon':
        return 4 if mod in {'KANGOO', 'PARTNER', 'BERLINGO', 'BERLINGO MULTISPACE', 'FIORINO', 'CADDY', 'QUBO', 'DOBLO'} else 3

    if mod in SEDANES_4P:
        return 4
    return 5


# ==================================================================================================
# 3. PIPELINE PRINCIPAL DE EJECUCIÓN
# ==================================================================================================

def ejecutar_pipeline(archivo_entrada='dataset_limpio.csv',
                      archivo_excel='base_limpia.xlsx',
                      archivo_csv='base_limpia.csv'):
    """
    Coordina la carga, transformaciones, validaciones y exportaciones.
    """
    t0 = time.time()
    print("=" * 80)
    print("   INICIANDO PIPELINE DE LIMPIEZA VEHICULAR - TP IFTS")
    print("=" * 80)

    # ----------------------------------------------------------------------------------------------
    # PASO 1: Carga de Datos
    # ----------------------------------------------------------------------------------------------
    print(f"\n[1/6] Cargando datos desde '{archivo_entrada}'...")
    if not os.path.exists(archivo_entrada):
        raise FileNotFoundError(f"Error crítico: No se encontró el archivo '{archivo_entrada}'.")

    df = pd.read_csv(archivo_entrada, encoding='utf-8')
    print(f"      Total de filas leídas: {len(df):,}")

    # Preservar el tipo original para contexto en regex de puertas y desambiguación
    tipo_vehiculo_original = df['tipo_vehiculo'].astype(str).str.strip().str.upper()

    # ----------------------------------------------------------------------------------------------
    # PASO 2: Limpieza y Normalización de MARCA
    # ----------------------------------------------------------------------------------------------
    print("\n[2/6] Limpiando y normalizando marcas...")
    df['marca'] = df['marca'].astype(str).str.strip().str.upper()
    df['marca'] = df['marca'].replace(REEMPLAZOS_MARCA)

    # Corrección de modelos ingresados como marca
    for marca_erronea, (marca_real, mod_real) in MODELOS_EN_MARCA.items():
        mask = df['marca'] == marca_erronea
        if mask.any():
            df.loc[mask, 'modelo'] = mod_real
            df.loc[mask, 'marca'] = marca_real

    # ----------------------------------------------------------------------------------------------
    # PASO 3: Limpieza y Normalización de MODELO
    # ----------------------------------------------------------------------------------------------
    print("\n[3/6] Limpiando y normalizando modelos...")
    # Saneamiento de espacios y símbolos iniciales (evita que Excel interprete fórmulas como -CUORE)
    df['modelo'] = df['modelo'].astype(str).str.strip().str.upper().str.lstrip('-+=@/').str.strip()

    # Aplicar diccionario de pares (marca, modelo)
    df['modelo'] = df.apply(
        lambda r: CORRECCIONES_MODELO.get((r['marca'], r['modelo']), r['modelo']),
        axis=1
    )

    # Desambiguación contextual: Ford 'F' -> F-100 Pick Up vs Camión Serie F
    mask_ford_f = (df['marca'] == 'FORD') & (df['modelo'] == 'F')
    mask_ford_f_camion = mask_ford_f & tipo_vehiculo_original.str.contains(
        'CAMION|CAMIÓN|CHASIS|CHAS|TRACTOR|PORTA|VOLQUET', case=False, na=False
    )
    df.loc[mask_ford_f & (~mask_ford_f_camion), 'modelo'] = 'F-100'
    df.loc[mask_ford_f_camion, 'modelo'] = 'SERIE F CAMION'

    # Desambiguación contextual: Peugeot 'P' -> 504 Pick Up vs Partner
    mask_peugeot_p = (df['marca'] == 'PEUGEOT') & (df['modelo'] == 'P')
    mask_peugeot_p_pup = mask_peugeot_p & tipo_vehiculo_original.str.contains('PICK|CAMIONETA', case=False, na=False)
    df.loc[mask_peugeot_p_pup, 'modelo'] = '504 PICK UP'
    df.loc[mask_peugeot_p & (~mask_peugeot_p_pup), 'modelo'] = 'PARTNER'

    # ----------------------------------------------------------------------------------------------
    # PASO 4: Estandarización Jerárquica de TIPO_VEHICULO
    # ----------------------------------------------------------------------------------------------
    print("\n[4/6] Estandarizando taxonomía de tipo de vehículo...")
    df['_tipo_orig'] = tipo_vehiculo_original
    df['tipo_vehiculo_limpio'] = df.apply(clasificar_tipo_vehiculo, axis=1)

    # ----------------------------------------------------------------------------------------------
    # PASO 5: Determinación de PUERTAS
    # ----------------------------------------------------------------------------------------------
    print("\n[5/6] Extrayendo e imputando cantidad de puertas...")
    df['Puertas'] = df.apply(extraer_puertas, axis=1)

    # ----------------------------------------------------------------------------------------------
    # PASO 6: Estructuración y Exportación Final
    # ----------------------------------------------------------------------------------------------
    print("\n[6/6] Exportando resultados finales...")

    # Generación de columna identificadora secuencial única (id: 1 a N)
    df['id'] = range(1, len(df) + 1)

    columnas_finales = [
        'id',
        'es_recuperado',
        'fecha',
        'marca',
        'modelo',
        'tipo_vehiculo_limpio',
        'Puertas',
        'origen',
        'provincia',
        'anio_modelo',
        'anio_tramite',
        'mes_tramite',
        'antiguedad_vehiculo'
    ]

    df_salida = df[columnas_finales].rename(columns={'tipo_vehiculo_limpio': 'tipo_vehiculo'})

    # Exportación a Excel con manejo de bloqueo de archivos y strings_to_formulas deshabilitado
    print(f"      Generando libro Excel '{archivo_excel}'...")
    try:
        with pd.ExcelWriter(archivo_excel, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_formulas': False}}) as writer:
            df_salida.to_excel(writer, sheet_name='base_limpia', index=False)
        print(f"      -> Archivo '{archivo_excel}' creado con éxito.")
    except PermissionError:
        archivo_alt = archivo_excel.replace('.xlsx', '_v2.xlsx')
        print(f"      [AVISO] '{archivo_excel}' está abierto en otra aplicación. Guardando como '{archivo_alt}'...")
        with pd.ExcelWriter(archivo_alt, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_formulas': False}}) as writer:
            df_salida.to_excel(writer, sheet_name='base_limpia', index=False)
        print(f"      -> Archivo alternativo '{archivo_alt}' creado con éxito.")

    # Exportación a CSV con codificación UTF-8 con BOM
    print(f"      Generando archivo CSV '{archivo_csv}'...")
    df_salida.to_csv(archivo_csv, index=False, encoding='utf-8-sig')
    print(f"      -> Archivo '{archivo_csv}' creado con éxito.")

    # Resumen de Métricas
    duracion = time.time() - t0
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETADO CON ÉXITO EN {duracion:.2f} SEGUNDOS")
    print(f"Total registros procesados: {len(df_salida):,}")
    print("=" * 80)
    print("\nDistribución Final por Tipo de Vehículo:")
    print(df_salida['tipo_vehiculo'].value_counts().to_string())
    print("\nDistribución Final por Cantidad de Puertas:")
    print(df_salida['Puertas'].value_counts(dropna=False).to_string())
    print("=" * 80)


if __name__ == '__main__':
    ejecutar_pipeline()
