# Métricas en la escala original (kWh), agregadas sobre todos los clientes y orígenes.
# Todas reciben un DataFrame de pronósticos con unique_id, ds, origen, h, y, y_hat
# (y q10, q50, q90 si el modelo es probabilístico).

import numpy as np
import pandas as pd

from src import config as C

CUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


def escalas_mase(panel, m=C.TEMPORADA_MASE, hasta=C.FIN_TRAIN):
    # Denominador del MASE, MAE del seasonal naive de m horas dentro de entrenamiento, por serie
    ancho = panel[panel.ds <= hasta].pivot(index="ds", columns="unique_id", values="y")
    return (ancho - ancho.shift(m)).abs().mean()


def mae(df):
    return float((df.y - df.y_hat).abs().mean())


def rmse(df):
    return float(np.sqrt(((df.y - df.y_hat) ** 2).mean()))


def wape(df):
    return float((df.y - df.y_hat).abs().sum() / df.y.abs().sum())


def mape(df, eps=0.0):
    # Con clientes de consumo cercano a cero el denominador explota, por eso se reporta
    # también qué parte de las observaciones tiene y = 0 (esas quedan fuera del cálculo)
    ok = df.y.abs() > eps
    return float(((df.y - df.y_hat).abs() / df.y.abs())[ok].mean())


def mase(df, escalas):
    # Error escalado por punto y promedio. Con el mismo número de puntos por serie
    # equivale al promedio del MASE de cada cliente.
    esc = df.unique_id.map(escalas).to_numpy()
    return float(((df.y - df.y_hat).abs().to_numpy() / esc).mean())


def pinball(df):
    cols = [c for c in CUANTILES if c in df]
    if not cols:
        return None
    perdidas = []
    for c in cols:
        q = CUANTILES[c]
        e = df.y - df[c]
        perdidas.append(np.maximum(q * e, (q - 1) * e).mean())
    return float(np.mean(perdidas))


def cobertura(df):
    if not {"q10", "q90"} <= set(df.columns):
        return None
    return float(((df.y >= df.q10) & (df.y <= df.q90)).mean())


def por_cliente(df, escalas):
    g = df.assign(ae=(df.y - df.y_hat).abs(), se=(df.y - df.y_hat) ** 2).groupby("unique_id")
    t = pd.DataFrame({"MAE": g.ae.mean(), "RMSE": np.sqrt(g.se.mean())})
    t["MASE"] = t.MAE / escalas.reindex(t.index)
    return t


def evaluar(df, escalas):
    nav = df[(df.ds >= C.NAVIDAD[0]) & (df.ds <= C.NAVIDAD[1])]
    ae = (df.y - df.y_hat).abs()
    clientes = por_cliente(df, escalas)
    return {
        "MAE": mae(df), "RMSE": rmse(df), "WAPE": wape(df), "MAPE": mape(df),
        "MAPE_frac_ceros": float((df.y == 0).mean()), "MASE": mase(df, escalas),
        "MASE_navidad": mase(nav, escalas) if len(nav) else None,
        "MAE_navidad": mae(nav) if len(nav) else None,
        "pinball": pinball(df), "cobertura_p10_p90": cobertura(df),
        "mae_por_h": ae.groupby(df.h).mean().tolist(),
        "mase_por_h": (ae / df.unique_id.map(escalas)).groupby(df.h).mean().tolist(),
        "mae_por_hora": ae.groupby(df.ds.dt.hour).mean().tolist(),
        "mae_por_cliente": clientes.MAE.to_dict(),
        "rmse_por_cliente": clientes.RMSE.to_dict(),
        "mase_por_cliente": clientes.MASE.to_dict(),
    }
