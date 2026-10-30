# Configuración compartida por todos los notebooks y modelos.
# Nadie cambia estos valores después del primer día, así todos los modelos usan lo mismo.

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROC = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "Resultados"
FIGURAS = RAIZ / "Figuras"
MODELOS = RAIZ / "Modelos"

URL_DATOS = ("https://archive.ics.uci.edu/static/public/321/"
             "electricityloaddiagrams20112014.zip")
ARCHIVO_CRUDO = DATA_RAW / "LD2011_2014.txt"
URL_ESPEJO = ("https://huggingface.co/datasets/sxj1024/electricity_load_2011_2014/"
              "resolve/main/data/train-00000-of-00001.parquet")
ARCHIVO_ESPEJO = DATA_RAW / "hf" / "electricity_hf.parquet"
PANEL = DATA_PROC / "panel_horario.parquet"
CALENDARIO = DATA_PROC / "calendario.parquet"
CLIENTES = DATA_PROC / "clientes.json"
HORARIO_TODOS = DATA_PROC / "horario_todos.parquet"

SEMILLA = 42
N_CLIENTES = 100

# Periodo de estudio y bloques (todas las horas marcan el inicio de la hora)
INICIO = pd.Timestamp("2012-01-01 00:00")
FIN = pd.Timestamp("2014-12-31 23:00")
FIN_TRAIN = pd.Timestamp("2014-06-30 23:00")
INICIO_VAL = pd.Timestamp("2014-07-01 00:00")
FIN_VAL = pd.Timestamp("2014-09-30 23:00")
INICIO_TEST = pd.Timestamp("2014-10-01 00:00")
FIN_TEST = FIN

L = 168
H = 24
TEMPORADA_MASE = 24

# Pronóstico day-ahead, un origen por día a las 00:00 con datos hasta las 23:00 del día anterior
ORIGENES_VAL = pd.date_range(INICIO_VAL, "2014-09-30", freq="D")
ORIGENES_PRUEBA = pd.date_range(INICIO_TEST, "2014-12-31", freq="D")

# Semana de Navidad que se evalúa aparte
NAVIDAD = (pd.Timestamp("2014-12-22 00:00"), pd.Timestamp("2014-12-28 23:00"))

# Orígenes donde se reentrena con ventana expandible (inicio de cada mes de prueba)
ORIGENES_REENTRENO = [pd.Timestamp("2014-10-01"), pd.Timestamp("2014-11-01"),
                      pd.Timestamp("2014-12-01")]

COLUMNAS_CALENDARIO = ["hora_sin", "hora_cos", "dow_sin", "dow_cos",
                       "mes_sin", "mes_cos", "feriado"]

for _carpeta in [DATA_RAW, DATA_PROC, RESULTADOS / "pronosticos", RESULTADOS / "metricas",
                 RESULTADOS / "iteraciones", RESULTADOS / "historia", RESULTADOS / "reentreno",
                 RESULTADOS / "tiempos", FIGURAS, MODELOS]:
    _carpeta.mkdir(parents=True, exist_ok=True)
