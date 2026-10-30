# Corridas de los modelos de librería (baselines, LightGBM, NeuralForecast y Chronos).
# Cada corrida queda guardada en Resultados/, así el notebook 4 solo carga resultados.
# Para correr todo desde la terminal:  python -m src.experimentos_libreria
# O una sola fase:                     python -m src.experimentos_libreria fase_lightgbm

import json
import logging
import sys
import warnings

import numpy as np
import pandas as pd

from src import config as C
from src.backtest import ORIGENES_PRUEBA, ORIGENES_VAL, SeasonalNaive, backtest
from src.data import cargar_panel
from src.modelos_libreria import (ChronosModelo, DesdeCV, LightGBMModelo, NFModelo,
                                  NF_PROBABILISTICOS, inicio_historia, meta_libreria,
                                  revisar_cortes)
from src.registro import (_escribir, cargar_iteracion, cargar_metricas, guardar_historia,
                          guardar_iteracion, guardar_resultado, hardware)

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

CPU = "Intel Core i5-12400 (CPU)"
MESES = [3, 12, 30]
CONTEXTO_MSTL = 24 * 7 * 8     # 8 semanas de historia en cada ajuste de MSTL
NOMBRES_NF = {"NHITS": "N-HiTS", "TCN": "TCN", "PatchTST": "PatchTST", "iTransformer": "iTransformer", "DeepAR": "DeepAR"}
FAMILIAS_NF = {"NHITS": "Residuales y convolucionales", "TCN": "Residuales y convolucionales",
               "PatchTST": "Atención", "iTransformer": "Atención", "DeepAR": "Probabilísticos"}
PASOS_NF = {"NHITS": 3000, "TCN": 3000, "PatchTST": 3000, "iTransformer": 3000, "DeepAR": 1500}
BASELINES = ["Naive", "SeasonalNaive24", "SeasonalNaive168", "MSTL"]

_PANEL = {}


def panel():
    if "p" not in _PANEL:
        _PANEL["p"] = cargar_panel().sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return _PANEL["p"]


def info_val(res, extra=None):
    return {"MASE_val": res["MASE"], "MAE_val": res["MAE"], "WAPE_val": res["WAPE"],
            "t_inferencia_s": res["t_inferencia_s"], **(extra or {})}


# ---------------------------------------------------------------------------------------
# Baselines de StatsForecast
# ---------------------------------------------------------------------------------------

def _cv_stats(modelos, df, h, n_windows, refit=False, input_size=None, n_jobs=1):
    import time
    from statsforecast import StatsForecast
    t0 = time.perf_counter()
    cv = StatsForecast(models=modelos, freq="h", n_jobs=n_jobs).cross_validation(
        df=df, h=h, step_size=C.H, n_windows=n_windows, refit=refit, input_size=input_size)
    return cv, time.perf_counter() - t0


def _modelos_stats():
    from statsforecast.models import MSTL, Naive
    from statsforecast.models import SeasonalNaive as SN
    return {"Naive": Naive(), "SeasonalNaive24": SN(24, alias="SeasonalNaive24"),
            "SeasonalNaive168": SN(168, alias="SeasonalNaive168"),
            "MSTL": MSTL(season_length=[24, 168], alias="MSTL")}


def fase_baselines():
    from statsforecast.models import MSTL
    modelos = _modelos_stats()
    p = panel()
    sencillos = {k: v for k, v in modelos.items() if k != "MSTL"}
    cv_test, t_test = _cv_stats(list(sencillos.values()), p, C.H, 92)
    cv_val, t_val = _cv_stats(list(sencillos.values()), p[p.ds <= C.FIN_VAL], C.H, 92)
    cv168, t168 = _cv_stats([sencillos["SeasonalNaive168"]], p, 168, 86)
    revisar_cortes(cv_test)
    revisar_cortes(cv_val, ORIGENES_VAL)
    for nombre in sencillos:
        if cargar_metricas(nombre) is None:
            res_val = backtest(DesdeCV(cv_val, {nombre: "y_hat"}), ORIGENES_VAL)
            guardar_iteracion(f"bl_{nombre}", {"id": f"bl_{nombre}", "config": {"modelo": nombre},
                                               **info_val(res_val, {"t_inferencia_s": t_val})})
            meta = meta_libreria("Baselines", 168 if nombre == "SeasonalNaive168" else 24
                                 if nombre != "Naive" else 1, "multi-salida", False, 0.0, t_test,
                                 0, 0, hardware=CPU, MASE_val=res_val["MASE"], id_iteracion=f"bl_{nombre}",
                                 config={"modelo": nombre})
            if nombre == "SeasonalNaive168":
                r168 = backtest(DesdeCV(cv168, {nombre: "y_hat"}), ORIGENES_PRUEBA, H=168)
                meta.update({"MASE_h168": r168["MASE"], "MAE_h168": r168["MAE"],
                             "n_origenes_h168": r168["n_origenes"]})
            res = backtest(DesdeCV(cv_test, {nombre: "y_hat"}), ORIGENES_PRUEBA, nombre=nombre)
            guardar_resultado(nombre, res, meta)
            print(nombre, "val", round(res_val["MASE"], 4), "prueba", round(res["MASE"], 4), flush=True)
    # MSTL se reajusta en cada origen con las últimas 8 semanas (el ajuste cuesta mucho más
    # que un naive y con refit=False no usaría la historia nueva)
    if cargar_metricas("MSTL") is None:
        mstl = {"MSTL": MSTL(season_length=[24, 168], alias="MSTL")}
        cvm_val, tm_val = _cv_stats(list(mstl.values()), p[p.ds <= C.FIN_VAL], C.H, 92, refit=True,
                                    input_size=CONTEXTO_MSTL, n_jobs=-1)
        res_val = backtest(DesdeCV(cvm_val, {"MSTL": "y_hat"}), ORIGENES_VAL)
        guardar_iteracion("bl_MSTL", {"id": "bl_MSTL", "config": {"modelo": "MSTL", "contexto_h": CONTEXTO_MSTL},
                                      **info_val(res_val, {"t_inferencia_s": tm_val})})
        cvm, tm = _cv_stats(list(mstl.values()), p, C.H, 92, refit=True, input_size=CONTEXTO_MSTL, n_jobs=-1)
        revisar_cortes(cvm)
        res = backtest(DesdeCV(cvm, {"MSTL": "y_hat"}), ORIGENES_PRUEBA, nombre="MSTL")
        meta = meta_libreria("Baselines", CONTEXTO_MSTL, "multi-salida", False, 0.0, tm, 0, 0,
                             hardware=CPU, MASE_val=res_val["MASE"], id_iteracion="bl_MSTL",
                             config={"modelo": "MSTL", "season_length": [24, 168], "contexto_h": CONTEXTO_MSTL})
        guardar_resultado("MSTL", res, meta)
        print("MSTL val", round(res_val["MASE"], 4), "prueba", round(res["MASE"], 4), flush=True)
    # Historia del seasonal naive (recorta el panel antes de pronosticar)
    from statsforecast.models import SeasonalNaive as SN
    for m in MESES:
        df = p[(p.ds <= C.FIN_VAL) & (p.ds >= inicio_historia(m) - pd.Timedelta(hours=0))]
        cv, _ = _cv_stats([SN(24, alias="SN")], df, C.H, 92)
        res = backtest(DesdeCV(cv, {"SN": "y_hat"}), ORIGENES_VAL)
        guardar_historia("SeasonalNaive24", m, {"MASE_val": res["MASE"], "MAE_val": res["MAE"]})


# ---------------------------------------------------------------------------------------
# LightGBM global
# ---------------------------------------------------------------------------------------

def _meta_lgbm(m, res_val, id_iter):
    return meta_libreria("Árboles", 168, "recursiva", True, m.t_entrenamiento_s, None,
                         m.n_parametros(), m.n_parametros(), hardware=CPU,
                         MASE_val=res_val["MASE"], id_iteracion=id_iter,
                         config={"modelo": "lightgbm", "lags": "1 a 24 y 168", "meses": m.meses,
                                 **m.params, "mejor_arbol": m.mejor_arbol})


def fase_lightgbm():
    p = panel()
    for meses in [30, 3, 12]:
        id_iter = "lgbm_base" if meses == 30 else f"lgbm_hist{meses}"
        if cargar_iteracion(id_iter) is not None:
            continue
        m = LightGBMModelo(meses=meses).entrenar(p)
        res_val = backtest(m, ORIGENES_VAL)
        guardar_iteracion(id_iter, {"id": id_iter, **info_val(res_val), "config": {"meses": meses, **m.params},
                                    "mejor_arbol": m.mejor_arbol, "curva_val": m.curva_val,
                                    "t_entrenamiento_s": m.t_entrenamiento_s, "params_total": m.n_parametros()})
        guardar_historia("LightGBM", meses, {"MASE_val": res_val["MASE"], "MAE_val": res_val["MAE"], "id": id_iter,
                                             "t_entrenamiento_s": m.t_entrenamiento_s})
        print(id_iter, "val", round(res_val["MASE"], 4), "arboles", m.mejor_arbol, flush=True)
        if meses == 30 and cargar_metricas("LightGBM") is None:
            res = backtest(m, ORIGENES_PRUEBA, nombre="LightGBM")
            meta = _meta_lgbm(m, res_val, id_iter)
            meta["t_inferencia_s"] = res["t_inferencia_s"]
            guardar_resultado("LightGBM", res, meta)
            print("LightGBM prueba", round(res["MASE"], 4), flush=True)


# ---------------------------------------------------------------------------------------
# NeuralForecast
# ---------------------------------------------------------------------------------------

def correr_nf(id_iter, nombre, **kw):
    # Entrena una configuración y la mide en validación. Guarda curvas y recursos de la iteración.
    previo = cargar_iteracion(id_iter)
    if previo is not None:
        return None, previo
    m = NFModelo(nombre, max_steps=PASOS_NF[nombre], **kw).entrenar(panel())
    res_val = backtest(m, ORIGENES_VAL)
    info = {"id": id_iter, **info_val(res_val),
            "config": {"modelo": nombre, "L": m.L, "meses": m.meses, "max_steps": m.max_steps,
                       "semilla": m.semilla, **{k: str(v) for k, v in m.kw.items()}},
            "params_total": m.params_total, "params_entrenables": m.params_entrenables,
            "t_entrenamiento_s": m.t_entrenamiento_s, "memoria_pico_mb": m.memoria_pico_mb,
            "curva_train": m.curva_train, "curva_val": m.curva_val,
            "pasos_entrenados": m.curva_train[-1][0] if m.curva_train else None}
    guardar_iteracion(id_iter, info)
    print(id_iter, "val MASE", round(res_val["MASE"], 4), "t", round(m.t_entrenamiento_s), "s", flush=True)
    return m, info


def evaluar_nf_final(nombre, id_iter, m, info):
    final = NOMBRES_NF[nombre]
    if cargar_metricas(final) is not None:
        return
    res = backtest(m, ORIGENES_PRUEBA, nombre=final)
    cfg = info["config"]
    meta = meta_libreria(FAMILIAS_NF[nombre], cfg["L"], "multi-salida" if nombre != "DeepAR" else "autorregresiva",
                         nombre != "PatchTST", info["t_entrenamiento_s"], res["t_inferencia_s"],
                         info["params_total"], info["params_entrenables"],
                         hardware=hardware(), MASE_val=info["MASE_val"], id_iteracion=id_iter, config=cfg,
                         memoria_pico_mb=info["memoria_pico_mb"], pasos_entrenados=info["pasos_entrenados"])
    guardar_resultado(final, res, meta)
    print(final, "prueba", round(res["MASE"], 4), flush=True)


def _entrenar_o_cargar(id_iter, nombre, **kw):
    # La prueba necesita el modelo entrenado en memoria. Si la iteración ya estaba guardada
    # (corrida anterior) se vuelve a entrenar solo cuando falta evaluar en prueba.
    m, info = correr_nf(id_iter, nombre, **kw)
    if m is None:
        m = NFModelo(nombre, max_steps=PASOS_NF[nombre], **kw).entrenar(panel())
    return m, info


def fase_nf(nombres=("NHITS", "TCN", "iTransformer", "DeepAR")):
    for nombre in nombres:
        final = NOMBRES_NF[nombre]
        id_iter = f"nf_{nombre}"
        if cargar_metricas(final) is not None:
            continue
        m, info = _entrenar_o_cargar(id_iter, nombre)
        evaluar_nf_final(nombre, id_iter, m, info)


def mejor_id_nf(nombre):
    cand = [i for i in [f"nf_{nombre}", f"nf_{nombre}_L48", f"nf_{nombre}_L336"] if cargar_iteracion(i)]
    return min(cand, key=lambda i: cargar_iteracion(i)["MASE_val"])


def fase_patchtst():
    # Iteración de L (48, 168, 336), se elige con validación. Después la historia (3, 12 y 30 meses)
    entrenados = {}
    for L, id_iter in [(168, "nf_PatchTST"), (48, "nf_PatchTST_L48"), (336, "nf_PatchTST_L336")]:
        m, _ = correr_nf(id_iter, "PatchTST", L=L)
        if m is not None:
            entrenados[id_iter] = m
    mejor = mejor_id_nf("PatchTST")
    base = cargar_iteracion(mejor)
    L = base["config"]["L"]
    for meses in [3, 12]:
        id_hist = f"nf_PatchTST_hist{meses}"
        correr_nf(id_hist, "PatchTST", L=L, meses=meses)
        h = cargar_iteracion(id_hist)
        guardar_historia("PatchTST", meses, {"MASE_val": h["MASE_val"], "MAE_val": h["MAE_val"], "id": id_hist,
                                             "L": L, "t_entrenamiento_s": h["t_entrenamiento_s"]})
    guardar_historia("PatchTST", 30, {"MASE_val": base["MASE_val"], "MAE_val": base["MAE_val"], "id": mejor,
                                      "L": L, "t_entrenamiento_s": base["t_entrenamiento_s"]})
    if cargar_metricas("PatchTST") is None:
        m = entrenados.get(mejor) or NFModelo("PatchTST", L=L, max_steps=PASOS_NF["PatchTST"]).entrenar(panel())
        evaluar_nf_final("PatchTST", mejor, m, base)


# ---------------------------------------------------------------------------------------
# Chronos-Bolt
# ---------------------------------------------------------------------------------------

def fase_chronos():
    if cargar_metricas("Chronos-Bolt") is not None:
        return
    m = ChronosModelo()
    res_val = backtest(m, ORIGENES_VAL)
    guardar_historia("Chronos", 0, {"MASE_val": res_val["MASE"], "MAE_val": res_val["MAE"],
                                    "nota": "zero-shot, sin entrenamiento, 2048 horas de contexto"})
    res = backtest(m, ORIGENES_PRUEBA, nombre="Chronos-Bolt")
    meta = meta_libreria("Fundacional", m.contexto, "multi-salida", False, 0.0, res["t_inferencia_s"],
                         m.params_total, 0, hardware=hardware(), MASE_val=res_val["MASE"],
                         config={"modelo": m.repo, "contexto": m.contexto})
    guardar_resultado("Chronos-Bolt", res, meta)
    print("Chronos val", round(res_val["MASE"], 4), "prueba", round(res["MASE"], 4), flush=True)


# ---------------------------------------------------------------------------------------
# Reentrenamiento mensual y mejor baseline
# ---------------------------------------------------------------------------------------

def mejor_baseline():
    # Entre los baselines de StatsForecast, elegido con el MASE de validación
    return min(BASELINES, key=lambda n: cargar_iteracion(f"bl_{n}")["MASE_val"])


def fase_reentreno_baseline():
    ruta = C.RESULTADOS / "reentreno" / "mejor_baseline.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    nombre = mejor_baseline()
    met = cargar_metricas(nombre)
    salida = {"modelo": nombre, "fijo": {k: met[k] for k in ["MAE", "RMSE", "WAPE", "MASE", "MASE_navidad"]}}
    if nombre.startswith("SeasonalNaive") or nombre == "Naive":
        # No tiene pesos que reentrenar, el pronóstico sale solo de la historia
        m = int(nombre.replace("SeasonalNaive", "")) if nombre != "Naive" else 1
        res = backtest(SeasonalNaive(m), ORIGENES_PRUEBA, reentrenar=lambda h: SeasonalNaive(m))
        salida["reentreno_mensual"] = {k: res[k] for k in ["MAE", "RMSE", "WAPE", "MASE", "MASE_navidad"]}
        salida["nota"] = "No tiene parámetros, entonces reentrenar cada mes no cambia nada"
    else:
        salida["reentreno_mensual"] = salida["fijo"]
        salida["nota"] = "MSTL se reajusta en cada origen con las últimas 8 semanas"
    _escribir(ruta, salida)
    return salida


def fase_reentreno_lightgbm():
    ruta = C.RESULTADOS / "reentreno" / "LightGBM.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    met = cargar_metricas("LightGBM")
    arboles = met["config"]["mejor_arbol"]
    tiempos = []

    def reentrenar(historia):
        m = LightGBMModelo(meses=30).entrenar(panel(), hasta=historia.ds.max(), n_arboles=arboles)
        tiempos.append({"hasta": str(historia.ds.max()), "t_s": m.t_entrenamiento_s})
        return m

    res = backtest(None, ORIGENES_PRUEBA, nombre="LightGBM_reentreno", reentrenar=reentrenar)
    pron = res.pop("pronosticos")
    pron.insert(2, "modelo", "LightGBM_reentreno")
    pron.to_parquet(C.RESULTADOS / "reentreno" / "LightGBM.parquet", index=False)
    salida = {"modelo": "LightGBM", "fijo": {k: met[k] for k in ["MAE", "RMSE", "WAPE", "MASE", "MASE_navidad"]},
              "reentreno_mensual": {k: res[k] for k in ["MAE", "RMSE", "WAPE", "MASE", "MASE_navidad"]},
              "mase_por_h_reentreno": res["mase_por_h"], "entrenamientos": tiempos, "arboles": arboles}
    _escribir(ruta, salida)
    return salida


# ---------------------------------------------------------------------------------------
# Mejor modelo de deep learning, semillas y reentrenamiento mensual
# ---------------------------------------------------------------------------------------

FAMILIAS_NO_DL = ("Baselines", "Árboles", "Fundacional")
SEMILLAS = [C.SEMILLA, 7, 123]


def mejor_deep_learning():
    # Se elige con el MASE de validación entre todos los modelos de redes (propios y de librería)
    filas = []
    for ruta in sorted((C.RESULTADOS / "metricas").glob("*.json")):
        d = json.loads(ruta.read_text(encoding="utf-8"))
        if d["familia"] not in FAMILIAS_NO_DL and d.get("MASE_val") is not None:
            filas.append((d["modelo"], d["MASE_val"]))
    return min(filas, key=lambda x: x[1])[0]


def _clase_nf(final):
    return {v: k for k, v in NOMBRES_NF.items()}.get(final)


def fase_semillas_nf():
    # Si el mejor modelo de deep learning es de NeuralForecast, se repite con 3 semillas
    final = mejor_deep_learning()
    nombre = _clase_nf(final)
    ruta = C.RESULTADOS / f"semillas_{final}.json"
    if nombre is None or ruta.exists():
        print("mejor deep learning", final, "(los propios ya tienen sus semillas)" if nombre is None else "")
        return
    met = cargar_metricas(final)
    L, filas = met["L"], []
    for s in SEMILLAS:
        m = NFModelo(nombre, L=L, max_steps=PASOS_NF[nombre], semilla=s).entrenar(panel())
        res_val = backtest(m, ORIGENES_VAL)
        res = backtest(m, ORIGENES_PRUEBA)
        filas.append({"semilla": s, "MASE_val": res_val["MASE"], "MASE_prueba": res["MASE"],
                      "MAE_prueba": res["MAE"], "WAPE_prueba": res["WAPE"],
                      "pasos": m.curva_train[-1][0] if m.curva_train else None})
        print(final, "semilla", s, "val", round(res_val["MASE"], 4), "prueba", round(res["MASE"], 4), flush=True)
    t = pd.DataFrame(filas)
    _escribir(ruta, {"modelo": final, "L": L, "corridas": filas,
                     "MASE_prueba_media": float(t.MASE_prueba.mean()),
                     "MASE_prueba_desv": float(t.MASE_prueba.std(ddof=1)),
                     "MASE_val_media": float(t.MASE_val.mean()), "MASE_val_desv": float(t.MASE_val.std(ddof=1))})


def fase_reentreno_nf():
    final = mejor_deep_learning()
    nombre = _clase_nf(final)
    ruta = C.RESULTADOS / "reentreno" / f"{final}.json"
    if nombre is None or ruta.exists():
        return
    met = cargar_metricas(final)
    pasos = met["pasos_entrenados"]
    tiempos = []

    def reentrenar(historia):
        m = NFModelo(nombre, L=met["L"], max_steps=pasos).entrenar(panel(), hasta=historia.ds.max(), pasos=pasos)
        tiempos.append({"hasta": str(historia.ds.max()), "t_s": m.t_entrenamiento_s})
        return m

    res = backtest(None, ORIGENES_PRUEBA, nombre=f"{final}_reentreno", reentrenar=reentrenar)
    pron = res.pop("pronosticos")
    pron.insert(2, "modelo", f"{final}_reentreno")
    pron.to_parquet(C.RESULTADOS / "reentreno" / f"{final}.parquet", index=False)
    salida = {"modelo": final, "fijo": {k: met[k] for k in ["MAE", "RMSE", "WAPE", "MASE", "MASE_navidad"]},
              "reentreno_mensual": {k: res[k] for k in ["MAE", "RMSE", "WAPE", "MASE", "MASE_navidad"]},
              "mase_por_h_reentreno": res["mase_por_h"], "entrenamientos": tiempos, "pasos": pasos}
    _escribir(ruta, salida)


def fase_h168():
    # MASE con H = 168 para N-HiTS y PatchTST (si son los que aplican). Se reentrena con h = 168
    for nombre in ["NHITS", "PatchTST"]:
        final = NOMBRES_NF[nombre]
        met = cargar_metricas(final)
        if met is None or met.get("MASE_h168") is not None:
            continue
        m = NFModelo(nombre, L=met["L"], max_steps=PASOS_NF[nombre], h=168).entrenar(panel())
        res = backtest(m, ORIGENES_PRUEBA, nombre=final, H=168)
        met["MASE_h168"], met["MAE_h168"], met["n_origenes_h168"] = res["MASE"], res["MAE"], res["n_origenes"]
        _escribir(C.RESULTADOS / "metricas" / f"{final}.json", met)
        print(final, "H=168", round(res["MASE"], 4), flush=True)


def fase_deepar():
    # La primera corrida de DeepAR (StudentT, h_train = 1) empeoraba en validación desde el inicio.
    # Se prueban variantes y el modelo final es la de menor MASE de validación.
    from neuralforecast.losses.pytorch import DistributionLoss
    variantes = {"nf_DeepAR": None,
                 "nf_DeepAR_n24": {"h_train": 24, "loss": DistributionLoss("Normal", level=[80])},
                 "nf_DeepAR_t24": {"h_train": 24, "loss": DistributionLoss("StudentT", level=[80])}}
    entrenados = {}
    for id_iter, kw in variantes.items():
        m, _ = correr_nf(id_iter, "DeepAR", **(kw or {}))
        if m is not None:
            entrenados[id_iter] = m
    mejor = min(variantes, key=lambda i: cargar_iteracion(i)["MASE_val"])
    met = cargar_metricas("DeepAR")
    if met is not None and met.get("id_iteracion") == mejor:
        return
    for ruta in [C.RESULTADOS / "metricas" / "DeepAR.json", C.RESULTADOS / "pronosticos" / "DeepAR.parquet"]:
        ruta.unlink(missing_ok=True)
    kw = variantes[mejor] or {}
    m = entrenados.get(mejor) or NFModelo("DeepAR", max_steps=PASOS_NF["DeepAR"], **kw).entrenar(panel())
    evaluar_nf_final("DeepAR", mejor, m, cargar_iteracion(mejor))


def todo():
    fase_baselines()
    fase_lightgbm()
    fase_chronos()
    fase_nf()
    fase_patchtst()
    fase_reentreno_baseline()
    fase_reentreno_lightgbm()


if __name__ == "__main__":
    pasos = sys.argv[1:] or ["todo"]
    for p in pasos:
        globals()[p]()
