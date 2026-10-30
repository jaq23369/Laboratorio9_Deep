# Todas las corridas de los modelos propios (secciones 5, 5.1, 5.2, 5.3 y semillas).
# Cada corrida queda guardada, así el notebook 3 solo carga resultados.
# Para correr todo desde la terminal:  python -m src.experimentos

import json
import sys

import numpy as np
import pandas as pd

from src import config as C
from src.backtest import ORIGENES_PRUEBA, backtest
from src.data import cargar_panel
from src.entrenamiento import (Datos, PronosticadorLocal, PronosticadorTorch, completar,
                               correr_iteracion, entrenar)
from src.metrics import escalas_mase, evaluar
from src.registro import _escribir, cargar_iteracion, cargar_metricas, guardar_historia, guardar_resultado

# Fase 1. Iteraciones de hiperparámetros, solo con validación (sección 5.2).
# La base de cada familia usa L = 168, 30 meses de historia y z-score por serie.
LIN = {"modelo": "dlinear"}
NL = {"modelo": "nlinear"}
MLP = {"modelo": "mlp", "oculto": 512, "capas": 2, "dropout": 0.1}
LS = {"modelo": "lstm"}

ITERACIONES = {
    # DLinear
    "dl_base": LIN,
    "dl_L48": {**LIN, "L": 48},
    "dl_L336": {**LIN, "L": 336},
    "dl_h3": {**LIN, "meses": 3},
    "dl_h12": {**LIN, "meses": 12},
    "dl_global": {**LIN, "norm": "global"},
    "dl_revin": {**LIN, "norm": "revin"},
    # NLinear
    "nl_base": NL,
    "nl_L336": {**NL, "L": 336},
    "nl_revin": {**NL, "norm": "revin"},
    # MLP
    "mlp_base": MLP,
    "mlp_o256_c3": {**MLP, "oculto": 256, "capas": 3},
    "mlp_d03": {**MLP, "dropout": 0.3},
    "mlp_L336": {**MLP, "L": 336},
    "mlp_revin": {**MLP, "norm": "revin"},
    # LSTM multi-salida
    "ls_base": LS,
    "ls_o128": {**LS, "oculto": 128},
    "ls_c1": {**LS, "capas": 1},
    "ls_c3": {**LS, "capas": 3},
    "ls_d0": {**LS, "dropout": 0.0},
    "ls_d03": {**LS, "dropout": 0.3},
    "ls_lr3e3": {**LS, "lr": 3e-3},
    "ls_lr3e4": {**LS, "lr": 3e-4},
    "ls_b256": {**LS, "batch": 256},
    "ls_L48": {**LS, "L": 48},
    "ls_L336": {**LS, "L": 336},
    "ls_h3": {**LS, "meses": 3},
    "ls_h12": {**LS, "meses": 12},
    "ls_global": {**LS, "norm": "global"},
    "ls_revin": {**LS, "norm": "revin"},
    # Combinaciones de los cambios que mejoraron a cada familia por separado
    "dl_L336_revin": {**LIN, "L": 336, "norm": "revin"},
    "nl_L336_revin": {**NL, "L": 336, "norm": "revin"},
    "mlp_L336_revin": {**MLP, "L": 336, "norm": "revin"},
    "ls_o128_lr3e3": {**LS, "oculto": 128, "lr": 3e-3},
}

# Modelos finales de la tabla comparativa y la familia de iteraciones de donde salen
FINALES = {
    "MLP": {"prefijo": "mlp_", "familia": "MLP y lineales", "estrategia": "multi-salida", "covariables": True},
    "DLinear": {"prefijo": "dl_", "familia": "MLP y lineales", "estrategia": "multi-salida", "covariables": False},
    "NLinear": {"prefijo": "nl_", "familia": "MLP y lineales", "estrategia": "multi-salida", "covariables": False},
    "LSTM": {"prefijo": "ls_", "familia": "Recurrentes", "estrategia": "multi-salida", "covariables": True},
    "LSTM_AR": {"prefijo": "ar_", "familia": "Recurrentes", "estrategia": "autorregresiva", "covariables": True},
    "Seq2seq": {"prefijo": "s2s_", "familia": "Encoder-decoder", "estrategia": "encoder-decoder", "covariables": True},
}

SEMILLAS = [C.SEMILLA, 7, 123]
MESES = [3, 12, 30]

_DATOS = {}


def datos():
    if "d" not in _DATOS:
        _DATOS["d"] = Datos()
    return _DATOS["d"]


def tabla_iteraciones(prefijo=None):
    filas = []
    for ruta in sorted((C.RESULTADOS / "iteraciones").glob("*.json")):
        info = json.loads(ruta.read_text())
        if prefijo and not info["id"].startswith(prefijo):
            continue
        c = info["config"]
        filas.append({"id": info["id"], "modelo": c["modelo"], "L": c["L"], "H": c["H"],
                      "meses": c["meses"], "norm": c["norm"], "oculto": c["oculto"],
                      "capas": c["capas"], "dropout": c["dropout"], "lr": c["lr"],
                      "batch": c["batch"], "tf": c.get("teacher_forcing"),
                      "epochs": len(info["historial"]["epoch"]), "mejor_epoch": info["mejor_epoch"],
                      "params": info["params_total"], "entrenables": info["params_entrenables"],
                      "t_epoch_s": info["t_epoch_prom_s"], "t_total_s": info["t_entrenamiento_s"],
                      "t_inferencia_s": info.get("t_inferencia_s"), "mem_mb": info["memoria_pico_mb"], "MAE_val": info.get("mae_val"),
                      "MASE_val": info.get("mase_val"),
                      "perdida_train_final": info["historial"]["perdida_train"][info["mejor_epoch"] - 1]})
    return pd.DataFrame(filas)


def mejor_id(prefijo):
    # Las corridas de semillas, H = 168 y reentreno no compiten en la selección
    t = tabla_iteraciones(prefijo)
    t = t[(t.H == C.H) & ~t.id.str.contains("_seed|_H168|_hist")]
    return t.sort_values("MASE_val").id.iloc[0]


def correr(ids_cfgs, verbose=True):
    for id_iter, cfg in ids_cfgs.items():
        correr_iteracion(id_iter, cfg, datos(), verbose=verbose)


def fase1():
    correr(ITERACIONES)


def config_de(id_iter):
    return cargar_iteracion(id_iter)["config"]


def fase2():
    # Estrategias multi-paso (5.1) con la misma configuración que la mejor LSTM multi-salida,
    # más dos variantes de teacher forcing para el seq2seq
    cfg = config_de(mejor_id("ls_"))
    comun = {k: cfg[k] for k in ["L", "meses", "norm", "oculto", "capas", "dropout", "lr", "batch"]}
    correr({
        "ar_base": {**comun, "modelo": "lstm_ar"},
        "s2s_base": {**comun, "modelo": "seq2seq", "teacher_forcing": 0.5},
        "s2s_tf0": {**comun, "modelo": "seq2seq", "teacher_forcing": 0.0},
        "s2s_tf1": {**comun, "modelo": "seq2seq", "teacher_forcing": 1.0},
    })


def fase_historia():
    # MASE de validación contra meses de historia para DLinear y la mejor recurrente
    salida = {}
    for nombre, prefijo in [("DLinear", "dl_"), ("LSTM", "ls_")]:
        id_mejor = mejor_id(prefijo)
        cfg = config_de(id_mejor)
        for m in MESES:
            id_iter = id_mejor if m == cfg["meses"] else f"{prefijo}hist{m}"
            _, _, info = correr_iteracion(id_iter, {**cfg, "meses": m}, datos())
            guardar_historia(nombre, m, {"MASE_val": info["mase_val"], "MAE_val": info["mae_val"],
                                         "id": id_iter, "t_entrenamiento_s": info["t_entrenamiento_s"]})
            salida[(nombre, m)] = info["mase_val"]
    return salida


def evaluar_final(nombre, forzar=False):
    # La prueba se usa una sola vez por modelo, con la mejor configuración de validación
    previo = cargar_metricas(nombre)
    if previo is not None and not forzar:
        return previo
    meta = FINALES[nombre]
    id_iter = mejor_id(meta["prefijo"])
    modelo, esc, info = correr_iteracion(id_iter, config_de(id_iter), datos())
    cfg = info["config"]
    res = backtest(PronosticadorTorch(modelo, esc, datos(), cfg), ORIGENES_PRUEBA, nombre=nombre)
    extra = {"familia": meta["familia"], "estrategia": meta["estrategia"],
             "covariables": meta["covariables"], "L": cfg["L"], "H": cfg["H"], "id_iteracion": id_iter,
             "config": cfg, "params_total": info["params_total"],
             "params_entrenables": info["params_entrenables"],
             "t_entrenamiento_s": info["t_entrenamiento_s"], "t_epoch_prom_s": info["t_epoch_prom_s"],
             "memoria_pico_mb": info["memoria_pico_mb"], "hardware": info["hardware"],
             "MASE_val": info["mase_val"], "MASE_h168": None}
    return guardar_resultado(nombre, res, extra)


def fase_h168():
    # MASE con H = 168 para los modelos lineales (los más baratos de reentrenar)
    for nombre, prefijo in [("DLinear", "dl_"), ("NLinear", "nl_"), ("MLP", "mlp_")]:
        met = cargar_metricas(nombre)
        if met.get("MASE_h168") is not None:
            continue
        cfg = {**config_de(mejor_id(prefijo)), "H": 168}
        modelo, esc, info = correr_iteracion(f"{prefijo}H168", cfg, datos())
        res = backtest(PronosticadorTorch(modelo, esc, datos(), info["config"]), ORIGENES_PRUEBA,
                       nombre=nombre, H=168)
        met["MASE_h168"] = res["MASE"]
        met["MAE_h168"] = res["MAE"]
        met["n_origenes_h168"] = res["n_origenes"]
        _escribir(C.RESULTADOS / "metricas" / f"{nombre}.json", met)


def mejor_propio():
    # El mejor modelo propio se elige con el MASE de validación, no con el de prueba
    filas = [(n, cargar_metricas(n)["MASE_val"]) for n in FINALES if cargar_metricas(n)]
    return min(filas, key=lambda x: x[1])[0]


def fase_semillas():
    ruta = C.RESULTADOS / "semillas.json"
    if ruta.exists():
        return json.loads(ruta.read_text())
    nombre = mejor_propio()
    base_id = cargar_metricas(nombre)["id_iteracion"]
    cfg = config_de(base_id)
    filas = []
    for s in SEMILLAS:
        id_iter = base_id if s == C.SEMILLA else f"{base_id}_seed{s}"
        modelo, esc, info = correr_iteracion(id_iter, {**cfg, "semilla": s}, datos())
        res = backtest(PronosticadorTorch(modelo, esc, datos(), info["config"]), ORIGENES_PRUEBA)
        filas.append({"semilla": s, "MASE_val": info["mase_val"], "MASE_prueba": res["MASE"],
                      "MAE_prueba": res["MAE"], "WAPE_prueba": res["WAPE"]})
    t = pd.DataFrame(filas)
    salida = {"modelo": nombre, "id_iteracion": base_id, "corridas": filas,
              "MASE_prueba_media": float(t.MASE_prueba.mean()),
              "MASE_prueba_desv": float(t.MASE_prueba.std(ddof=1)),
              "MASE_val_media": float(t.MASE_val.mean()), "MASE_val_desv": float(t.MASE_val.std(ddof=1))}
    _escribir(ruta, salida)
    return salida


def fase_reentreno():
    # Reentrenamiento al inicio de cada mes de prueba con ventana expandible.
    # Se entrena con todos los datos hasta el origen y el número de epochs que eligió
    # el early stopping original (ya no hay un bloque de validación limpio).
    nombre = mejor_propio()
    ruta = C.RESULTADOS / "reentreno" / f"{nombre}.json"
    if ruta.exists():
        return json.loads(ruta.read_text())
    met = cargar_metricas(nombre)
    info0 = cargar_iteracion(met["id_iteracion"])
    cfg = info0["config"]
    tiempos = []

    def reentrenar(historia):
        hasta = historia.ds.max()
        modelo, esc, info = entrenar(cfg, datos(), hasta=hasta, epochs_fijos=info0["mejor_epoch"],
                                     verbose=False)
        tiempos.append({"hasta": str(hasta), "t_s": info["t_entrenamiento_s"]})
        return PronosticadorTorch(modelo, esc, datos(), cfg)

    res = backtest(None, ORIGENES_PRUEBA, nombre=f"{nombre}_reentreno", reentrenar=reentrenar)
    pron = res.pop("pronosticos")
    pron.insert(2, "modelo", f"{nombre}_reentreno")
    pron.to_parquet(C.RESULTADOS / "reentreno" / f"{nombre}.parquet", index=False)
    salida = {"modelo": nombre, "fijo": {k: met[k] for k in ["MAE", "RMSE", "WAPE", "MASE", "MASE_navidad"]},
              "reentreno_mensual": {k: res[k] for k in ["MAE", "RMSE", "WAPE", "MASE", "MASE_navidad"]},
              "mase_por_h_reentreno": res["mase_por_h"], "entrenamientos": tiempos,
              "epochs": info0["mejor_epoch"]}
    _escribir(ruta, salida)
    return salida


def clientes_locales(n=10):
    rng = np.random.default_rng(C.SEMILLA)
    return sorted(rng.choice(datos().ids, size=n, replace=False).tolist())


def fase_local():
    # Sección 5.3. El mejor modelo propio entrenado por cliente (10 clientes) contra el global
    ruta = C.RESULTADOS / "local_global.json"
    if ruta.exists():
        return json.loads(ruta.read_text())
    nombre = mejor_propio()
    met = cargar_metricas(nombre)
    cfg = config_de(met["id_iteracion"])
    elegidos = clientes_locales()
    d = datos()
    pronos, tiempos = [], []
    for uid in elegidos:
        fila = d.ids.index(uid)
        modelo, esc, info = entrenar({**cfg, "batch": 256}, d, series=[fila], verbose=False)
        pronos.append(PronosticadorTorch(modelo, esc, d, cfg, ids=[uid]))
        tiempos.append(info["t_entrenamiento_s"])
        print(f"  local {uid} MASE val {info['mase_val']:.4f}", flush=True)
    panel = cargar_panel()
    sub = panel[panel.unique_id.isin(elegidos)].reset_index(drop=True)
    res_local = backtest(PronosticadorLocal(pronos), ORIGENES_PRUEBA, panel=sub)
    glob = pd.read_parquet(C.RESULTADOS / "pronosticos" / f"{nombre}.parquet")
    glob = glob[glob.unique_id.isin(elegidos)]
    res_global = evaluar(glob, escalas_mase(sub))
    salida = {"modelo": nombre, "clientes": elegidos,
              "MASE_local": res_local["MASE"], "MASE_global": res_global["MASE"],
              "MAE_local": res_local["MAE"], "MAE_global": res_global["MAE"],
              "mase_por_cliente_local": res_local["mase_por_cliente"],
              "mase_por_cliente_global": res_global["mase_por_cliente"],
              "t_entrenamiento_local_total_s": float(np.sum(tiempos))}
    _escribir(ruta, salida)
    return salida


def todo():
    fase1()
    fase2()
    for nombre in FINALES:
        evaluar_final(nombre)
    fase_historia()
    fase_h168()
    fase_semillas()
    fase_reentreno()
    fase_local()


if __name__ == "__main__":
    pasos = sys.argv[1:] or ["todo"]
    for p in pasos:
        globals()[p]()
