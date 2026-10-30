# Envoltorios de los modelos de librería (StatsForecast, MLForecast, NeuralForecast y Chronos)
# para pasarlos por backtest con el mismo protocolo que los modelos propios.
#
# Los modelos de Nixtla se corren una sola vez con cross_validation(refit=False) sobre los
# 92 orígenes y backtest recibe el pedazo de cada origen. Así no se entrena 92 veces cada modelo.

import json
import time

import numpy as np
import pandas as pd

from src import config as C
from src.backtest import ORIGENES_PRUEBA, ORIGENES_VAL, backtest
from src.data import cargar_panel
from src.features import cargar_calendario
from src.registro import guardar_resultado, hardware

CUANTILES = ["q10", "q50", "q90"]


def panel_con_calendario(panel=None):
    # unique_id, ds, y y las covariables de calendario, listo para MLForecast y NeuralForecast
    panel = cargar_panel() if panel is None else panel
    return panel.merge(cargar_calendario(), on="ds", how="left").sort_values(["unique_id", "ds"]).reset_index(drop=True)


def calendario_futuro(ids, inicio, fin):
    # Covariables de las horas a pronosticar, conocidas de antemano
    cal = cargar_calendario()
    cal = cal[(cal.ds >= inicio) & (cal.ds <= fin)]
    return pd.concat([cal.assign(unique_id=u) for u in ids], ignore_index=True)


class DesdeCV:
    # Modelo para backtest a partir de la salida de cross_validation (columnas unique_id, ds,
    # cutoff y una columna por modelo). El origen es una hora después del cutoff.
    def __init__(self, cv, columnas):
        # columnas es {columna_en_cv: nombre_en_backtest}, por ejemplo {"SeasonalNaive": "y_hat"}
        cv = cv.rename(columns=columnas).copy()
        cv["origen"] = pd.to_datetime(cv["cutoff"]) + pd.Timedelta(hours=1)
        cols = ["unique_id", "ds"] + list(columnas.values())
        self.por_origen = {o: g[cols].reset_index(drop=True) for o, g in cv.groupby("origen")}

    def predecir(self, historia, origen):
        return self.por_origen[pd.Timestamp(origen)]


def revisar_cortes(cv, origenes=ORIGENES_PRUEBA):
    # Los cortes de cross_validation deben ser las 23:00 del día anterior a cada origen
    origenes_cv = sorted((pd.to_datetime(cv.cutoff) + pd.Timedelta(hours=1)).unique())
    esperados = [pd.Timestamp(o) for o in origenes]
    assert origenes_cv == esperados, f"los cortes no coinciden con los orígenes: {origenes_cv[:2]}"
    assert (cv.groupby(["unique_id", "cutoff"]).size() == C.H).all()


def meta_libreria(familia, L, estrategia, covariables, t_entrenamiento_s, t_inferencia_s,
                  params_total=None, params_entrenables=None, **extra):
    return {"familia": familia, "L": L, "H": C.H, "estrategia": estrategia,
            "covariables": covariables, "params_total": params_total,
            "params_entrenables": params_entrenables, "t_entrenamiento_s": t_entrenamiento_s,
            "t_inferencia_s": t_inferencia_s, "hardware": hardware(), **extra}


def evaluar_cv(nombre, cv, columnas, meta, origenes=ORIGENES_PRUEBA, guardar=True):
    # Pasa un cross_validation por backtest y guarda pronósticos y métricas
    revisar_cortes(cv, origenes)
    res = backtest(DesdeCV(cv, columnas), origenes, nombre=nombre)
    return guardar_resultado(nombre, res, meta) if guardar else res


# ---------------------------------------------------------------------------------------
# Escalado por serie con estadísticos de entrenamiento (igual que los modelos propios)
# ---------------------------------------------------------------------------------------

class ZScore:
    # media y desviación por cliente hasta 2014-06-30. Se escala fuera de la librería para que
    # ningún modelo vea estadísticos de validación ni de prueba.
    def __init__(self, panel, hasta=C.FIN_TRAIN, desde=C.INICIO):
        tr = panel[(panel.ds <= hasta) & (panel.ds >= desde)]
        self.mu = tr.groupby("unique_id").y.mean()
        self.sd = tr.groupby("unique_id").y.std().clip(lower=1e-6)

    def escalar(self, df, cols=("y",)):
        df = df.copy()
        for c in cols:
            df[c] = (df[c] - df.unique_id.map(self.mu)) / df.unique_id.map(self.sd)
        return df

    def invertir(self, df, cols):
        df = df.copy()
        for c in cols:
            df[c] = df[c] * df.unique_id.map(self.sd) + df.unique_id.map(self.mu)
        return df


def inicio_historia(meses):
    # Los meses de historia terminan siempre el 2014-06-30 (igual que en src/entrenamiento.py)
    return max(C.INICIO, C.FIN_TRAIN + pd.Timedelta(hours=1) - pd.DateOffset(months=meses))


def ultimas_horas(historia, n):
    # Solo las últimas n horas de cada cliente, que es lo que necesitan los modelos con ventana
    return historia.groupby("unique_id", sort=False).tail(n)


def con_calendario(df):
    return df.merge(cargar_calendario(), on="ds", how="left")


# ---------------------------------------------------------------------------------------
# LightGBM global con MLForecast (rezagos 1 a 24 y 168 más calendario)
# ---------------------------------------------------------------------------------------

LAGS = [*range(1, 25), 168]
PARAMS_LGBM = {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 63, "objective": "l1",
               "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
               "colsample_bytree": 0.8, "verbose": -1}


class LightGBMModelo:
    # Un solo modelo recursivo para los 100 clientes. Se entrena con datos hasta 2014-06-30
    # y early stopping en el bloque de validación.
    def __init__(self, meses=30, params=None, semilla=C.SEMILLA):
        self.meses, self.semilla = meses, semilla
        self.params = {**PARAMS_LGBM, **(params or {})}

    def entrenar(self, panel=None, hasta=None, n_arboles=None):
        # Con hasta se reentrena sin validación y n_arboles fija el número de árboles
        import lightgbm as lgb
        from mlforecast import MLForecast
        panel = cargar_panel() if panel is None else panel
        self.z = ZScore(panel)
        fin = C.FIN_VAL if hasta is None else hasta
        ini = inicio_historia(self.meses)
        datos = self.z.escalar(con_calendario(panel[(panel.ds >= ini) & (panel.ds <= fin)]))
        t0 = time.perf_counter()
        self.fc = MLForecast(models={"lgbm": lgb.LGBMRegressor(n_estimators=1)}, freq="h",
                             lags=LAGS, num_threads=-1)
        prep = self.fc.preprocess(datos, static_features=[], dropna=True)
        feats = [c for c in prep.columns if c not in ("unique_id", "ds", "y")]
        params = {**self.params, "random_state": self.semilla, "n_jobs": -1}
        if hasta is None:
            tr, va = prep[prep.ds <= C.FIN_TRAIN], prep[prep.ds >= C.INICIO_VAL]
            modelo = lgb.LGBMRegressor(**params)
            modelo.fit(tr[feats], tr.y, eval_set=[(va[feats], va.y)],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.record_evaluation(h := {})])
            self.curva_val = [float(v) for v in next(iter(h["valid_0"].values()))]
            self.mejor_arbol = int(modelo.best_iteration_)
        else:
            modelo = lgb.LGBMRegressor(**{**params, "n_estimators": n_arboles})
            modelo.fit(prep[feats], prep.y)
            self.mejor_arbol = n_arboles
        self.fc.models_ = {"lgbm": modelo}
        self.modelo, self.feats = modelo, feats
        self.t_entrenamiento_s = time.perf_counter() - t0
        return self

    def predecir(self, historia, origen):
        hist = self.z.escalar(con_calendario(ultimas_horas(historia, 400)))
        ids = hist.unique_id.unique()
        fut = calendario_futuro(ids, origen, origen + pd.Timedelta(hours=C.H - 1))
        p = self.fc.predict(h=C.H, new_df=hist, X_df=fut).rename(columns={"lgbm": "y_hat"})
        return self.z.invertir(p, ["y_hat"])[["unique_id", "ds", "y_hat"]]

    def n_parametros(self):
        # Hojas totales de los árboles usados, que es lo más parecido a un conteo de parámetros
        arboles = self.modelo.booster_.trees_to_dataframe()
        return int((arboles.split_feature.isna()).sum())


# ---------------------------------------------------------------------------------------
# NeuralForecast (N-HiTS, TCN, PatchTST, TFT y DeepAR)
# ---------------------------------------------------------------------------------------

CAL = C.COLUMNAS_CALENDARIO
N_SERIES = C.N_CLIENTES
# PatchTST no acepta covariables futuras en NeuralForecast, el resto sí
NF_CON_CALENDARIO = {"NHITS": True, "TCN": True, "iTransformer": True, "DeepAR": True, "PatchTST": False}
NF_PROBABILISTICOS = ("DeepAR",)
NF_PESADOS = ("DeepAR",)


def construir_nf(nombre, L=C.L, max_steps=3000, semilla=C.SEMILLA, **kw):
    # Todos con h = 24, ventana L, pérdida MAE (DeepAR usa una distribución StudentT) y
    # early stopping sobre el bloque de validación. val_check_steps y la paciencia son iguales.
    from neuralforecast.losses.pytorch import MAE, DistributionLoss
    from neuralforecast.models import NHITS, TCN, DeepAR, PatchTST, iTransformer
    clases = {"NHITS": NHITS, "TCN": TCN, "iTransformer": iTransformer, "DeepAR": DeepAR,
              "PatchTST": PatchTST}
    comun = dict(h=C.H, input_size=L, max_steps=max_steps, val_check_steps=100,
                 early_stop_patience_steps=5, random_seed=semilla, scaler_type=None,
                 accelerator="gpu", devices=1, enable_progress_bar=False, logger=False,
                 enable_checkpointing=False, valid_loss=MAE(),
                 inference_windows_batch_size=1024, valid_batch_size=32)
    if NF_CON_CALENDARIO[nombre]:
        comun["futr_exog_list"] = CAL
    if nombre == "iTransformer":
        # Cada ventana trae los 100 clientes juntos, así que los lotes son más chicos
        comun.update(n_series=N_SERIES, windows_batch_size=64, inference_windows_batch_size=64)
    if nombre in NF_PESADOS:
        # En la validación cada ventana cuesta mucho en estos modelos. Con step_size = 24 las
        # ventanas empiezan siempre a las 00:00, que es la hora del origen de cada pronóstico.
        comun["step_size"] = C.H
    if nombre in NF_PROBABILISTICOS:
        comun["loss"] = DistributionLoss("StudentT", level=[80])
    else:
        comun["loss"] = MAE()
    return clases[nombre](**{**comun, **kw})


class NFModelo:
    # Un modelo de NeuralForecast para los 100 clientes. Entrena con datos hasta 2014-06-30 y
    # usa julio a septiembre como validación (val_size = 2208). En cada origen predice con la
    # historia recortada que le da backtest, así que el protocolo es el mismo de los demás.
    def __init__(self, nombre, L=C.L, meses=30, max_steps=3000, semilla=C.SEMILLA, **kw):
        self.nombre, self.L, self.meses = nombre, L, meses
        self.max_steps, self.semilla, self.kw = max_steps, semilla, kw

    def entrenar(self, panel=None, hasta=None, pasos=None):
        # Con hasta se reentrena sin validación (ventana expandible) y pasos fija los pasos de
        # entrenamiento, que son los que usó el early stopping de la corrida original
        import torch
        from neuralforecast import NeuralForecast
        panel = cargar_panel() if panel is None else panel
        self.z = ZScore(panel, hasta=C.FIN_TRAIN if hasta is None else hasta)
        ini = inicio_historia(self.meses)
        fin = C.FIN_VAL if hasta is None else hasta
        datos = self.z.escalar(con_calendario(panel[(panel.ds >= ini) & (panel.ds <= fin)]))
        cols = ["unique_id", "ds", "y"] + (CAL if NF_CON_CALENDARIO[self.nombre] else [])
        kw = dict(self.kw)
        if hasta is not None:
            kw["early_stop_patience_steps"] = -1
        modelo = construir_nf(self.nombre, self.L, pasos or self.max_steps, self.semilla, **kw)
        self.nf = NeuralForecast(models=[modelo], freq="h")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        self.nf.fit(df=datos[cols], val_size=C.ORIGENES_VAL.size * C.H if hasta is None else 0)
        self.t_entrenamiento_s = time.perf_counter() - t0
        self.memoria_pico_mb = torch.cuda.max_memory_allocated() / 2**20
        m = self.nf.models[0]
        self.params_total = sum(p.numel() for p in m.parameters())
        self.params_entrenables = sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.curva_train = [(int(s), float(v)) for s, v in getattr(m, "train_trajectories", [])]
        self.curva_val = [(int(s), float(v)) for s, v in getattr(m, "valid_trajectories", [])]
        return self

    def columnas_salida(self):
        return {"y_hat": self.nombre if self.nombre not in NF_PROBABILISTICOS
                else f"{self.nombre}-median",
                "q10": f"{self.nombre}-lo-80", "q90": f"{self.nombre}-hi-80"}

    def predecir(self, historia, origen):
        cols = ["unique_id", "ds", "y"] + (CAL if NF_CON_CALENDARIO[self.nombre] else [])
        hist = self.z.escalar(con_calendario(ultimas_horas(historia, self.L + 8)))[cols]
        ids = hist.unique_id.unique()
        fut = calendario_futuro(ids, origen, origen + pd.Timedelta(hours=self.kw.get("h", C.H) - 1)) \
            if NF_CON_CALENDARIO[self.nombre] else None
        p = self.nf.predict(df=hist, futr_df=fut)
        sal = {k: v for k, v in self.columnas_salida().items() if v in p.columns}
        out = p[["unique_id", "ds"] + list(sal.values())].rename(columns={v: k for k, v in sal.items()})
        out = self.z.invertir(out, list(sal))
        if "q10" in out:
            out["q50"] = out["y_hat"]
        return out


# ---------------------------------------------------------------------------------------
# Chronos-Bolt zero-shot
# ---------------------------------------------------------------------------------------

class ChronosModelo:
    # Sin entrenamiento. Recibe en kWh las últimas horas de cada cliente (Chronos hace su propia
    # normalización por serie) y devuelve P10, P50 y P90. El P50 es el pronóstico puntual.
    def __init__(self, repo="amazon/chronos-bolt-base", contexto=2048):
        import torch
        from chronos import BaseChronosPipeline
        self.repo, self.contexto = repo, contexto
        self.pipeline = BaseChronosPipeline.from_pretrained(
            repo, device_map="cuda" if torch.cuda.is_available() else "cpu")
        self.params_total = sum(p.numel() for p in self.pipeline.model.parameters())

    def predecir(self, historia, origen):
        import torch
        ult = ultimas_horas(historia, self.contexto)
        ids = list(ult.unique_id.unique())
        series = [torch.tensor(g.y.to_numpy(), dtype=torch.float32)
                  for _, g in ult.groupby("unique_id", sort=False)]
        cuantiles, _ = self.pipeline.predict_quantiles(series, prediction_length=C.H,
                                                       quantile_levels=[0.1, 0.5, 0.9])
        q = cuantiles.cpu().numpy()
        horas = pd.date_range(origen, periods=C.H, freq="h")
        return pd.DataFrame({"unique_id": np.repeat(ids, C.H), "ds": np.tile(horas, len(ids)),
                             "y_hat": q[:, :, 1].ravel(), "q10": q[:, :, 0].ravel(),
                             "q50": q[:, :, 1].ravel(), "q90": q[:, :, 2].ravel()})
