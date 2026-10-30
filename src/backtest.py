# Protocolo de evaluación único para todos los modelos.
# Backtesting con origen móvil diario. En cada origen el modelo recibe solo la historia
# hasta las 23:00 del día anterior y devuelve las H horas siguientes.
#
# Un modelo es cualquier objeto con
#     predecir(historia, origen) -> DataFrame con unique_id, ds, y_hat (y q10, q50, q90 opcional)
# donde historia trae unique_id, ds, y con ds < origen.

import time

import pandas as pd

from src import config as C
from src.data import cargar_panel
from src.metrics import escalas_mase, evaluar

ORIGENES_PRUEBA = C.ORIGENES_PRUEBA
ORIGENES_VAL = C.ORIGENES_VAL

_CACHE = {}


def _panel_y_escalas(panel):
    if panel is None:
        if "panel" not in _CACHE:
            p = cargar_panel().sort_values(["unique_id", "ds"]).reset_index(drop=True)
            _CACHE["panel"] = p
            _CACHE["escalas"] = escalas_mase(p)
        return _CACHE["panel"], _CACHE["escalas"]
    return panel, escalas_mase(panel)


def origenes_validos(origenes, H, fin):
    # Solo los orígenes cuyo objetivo completo cae dentro del bloque
    return [o for o in origenes if o + pd.Timedelta(hours=H - 1) <= fin]


def backtest(modelo, origenes=ORIGENES_PRUEBA, nombre=None, H=C.H, panel=None,
             reentrenar=None, origenes_reentreno=C.ORIGENES_REENTRENO, verbose=False):
    # reentrenar(historia) -> modelo nuevo. Si se pasa, se llama en cada origen de
    # origenes_reentreno con toda la historia disponible (ventana expandible).
    panel, escalas = _panel_y_escalas(panel)
    fin = pd.Timestamp(origenes[-1]).normalize() + pd.Timedelta(hours=23)
    origenes = origenes_validos([pd.Timestamp(o) for o in origenes], H, fin)
    reales = panel.set_index(["unique_id", "ds"]).y
    ids = panel.unique_id.unique()

    salidas, t_inf, t_reent = [], 0.0, 0.0
    for o in origenes:
        historia = panel[panel.ds < o]
        if reentrenar is not None and o in set(origenes_reentreno):
            t0 = time.perf_counter()
            modelo = reentrenar(historia)
            t_reent += time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = modelo.predecir(historia, o)
        t_inf += time.perf_counter() - t0

        # Controles para que ningún modelo se salga del protocolo
        horas = pd.date_range(o, periods=H, freq="h")
        assert pred.ds.min() == horas[0] and pred.ds.max() == horas[-1], f"horas mal en {o}"
        assert len(pred) == len(ids) * H, f"faltan pronósticos en {o}"
        assert historia.ds.max() < o
        pred = pred.copy()
        pred["origen"] = o
        pred["h"] = ((pred.ds - o) / pd.Timedelta(hours=1)).astype(int) + 1
        salidas.append(pred)
        if verbose:
            print(o.date(), end=" ", flush=True)

    pron = pd.concat(salidas, ignore_index=True)
    pron["y"] = reales.reindex(pd.MultiIndex.from_frame(pron[["unique_id", "ds"]])).to_numpy()
    cols = ["unique_id", "ds", "origen", "h", "y", "y_hat"] + \
           [c for c in ["q10", "q50", "q90"] if c in pron]
    pron = pron[cols]
    res = evaluar(pron, escalas)
    res.update({"modelo": nombre, "H": H, "n_origenes": len(origenes),
                "t_inferencia_s": t_inf, "t_reentreno_s": t_reent if reentrenar else None,
                "pronosticos": pron})
    return res


class SeasonalNaive:
    # Referencia rápida para comprobar que el protocolo cuadra (el oficial lo hace B con StatsForecast)
    def __init__(self, m=24):
        self.m = m

    def predecir(self, historia, origen):
        ult = historia[historia.ds >= origen - pd.Timedelta(hours=self.m)]
        filas = []
        for uid, g in ult.groupby("unique_id"):
            vals = g.sort_values("ds").y.to_numpy()[-self.m:]
            for k in range(C.H):
                filas.append((uid, origen + pd.Timedelta(hours=k), vals[k % self.m]))
        return pd.DataFrame(filas, columns=["unique_id", "ds", "y_hat"])
