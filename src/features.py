# Covariables de calendario conocidas a futuro.
# Hora, día de la semana y mes con seno y coseno (así la hora 23 queda cerca de la 0),
# y feriado de Portugal como 0/1.

import holidays
import numpy as np
import pandas as pd

from src import config as C


def feriados_portugal(anios=(2011, 2012, 2013, 2014, 2015)):
    return holidays.Portugal(years=list(anios))


def calendario(ds=None):
    if ds is None:
        ds = pd.date_range(C.INICIO, C.FIN, freq="h")
    ds = pd.DatetimeIndex(ds)
    fer = feriados_portugal(sorted(set(ds.year)))
    cal = pd.DataFrame({"ds": ds})
    cal["hora_sin"] = np.sin(2 * np.pi * ds.hour / 24)
    cal["hora_cos"] = np.cos(2 * np.pi * ds.hour / 24)
    cal["dow_sin"] = np.sin(2 * np.pi * ds.dayofweek / 7)
    cal["dow_cos"] = np.cos(2 * np.pi * ds.dayofweek / 7)
    cal["mes_sin"] = np.sin(2 * np.pi * (ds.month - 1) / 12)
    cal["mes_cos"] = np.cos(2 * np.pi * (ds.month - 1) / 12)
    cal["feriado"] = np.array([d in fer for d in ds.date], dtype=float)
    return cal


def construir_calendario():
    cal = calendario()
    cal.to_parquet(C.CALENDARIO, index=False)
    return cal


def cargar_calendario():
    if not C.CALENDARIO.exists():
        return construir_calendario()
    return pd.read_parquet(C.CALENDARIO)
