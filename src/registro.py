# Guardado de resultados con el formato que comparten A y B.
#   Resultados/pronosticos/<modelo>.parquet   unique_id, ds, origen, h, y, y_hat (q10, q50, q90)
#   Resultados/metricas/<modelo>.json         métricas de backtest + datos de la tabla comparativa
#   Resultados/iteraciones/<id>.json          una iteración de hiperparámetros (sección 5.2)
#   Resultados/historia/<modelo>_<meses>m.json  MASE de validación según meses de historia

import json
import platform

import numpy as np

from src import config as C

CAMPOS_META = ["modelo", "familia", "L", "H", "estrategia", "covariables", "params_total",
               "params_entrenables", "t_entrenamiento_s", "t_inferencia_s", "hardware",
               "MASE_h168"]


def hardware():
    import torch
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} (CUDA)"
    if torch.backends.mps.is_available():
        chip = "Apple Silicon"
        if platform.system() == "Darwin":
            import subprocess
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip() or chip
        return f"{chip} (MPS)"
    return f"{platform.processor() or 'CPU'} (CPU)"


def _limpiar(o):
    if isinstance(o, dict):
        return {str(k): _limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpiar(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def _escribir(ruta, datos):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(_limpiar(datos), indent=1, ensure_ascii=False), encoding="utf-8")


def guardar_resultado(nombre, resultado, meta):
    # resultado es lo que devuelve backtest, meta trae familia, L, parámetros, tiempos, etc.
    pron = resultado["pronosticos"].copy()
    pron.insert(2, "modelo", nombre)
    pron.to_parquet(C.RESULTADOS / "pronosticos" / f"{nombre}.parquet", index=False)
    datos = {k: v for k, v in resultado.items() if k != "pronosticos"}
    datos.update(meta)
    datos["modelo"] = nombre
    datos.setdefault("hardware", hardware())
    if datos.get("t_inferencia_s") is None:
        datos["t_inferencia_s"] = resultado.get("t_inferencia_s")
    _escribir(C.RESULTADOS / "metricas" / f"{nombre}.json", datos)
    return datos


def guardar_iteracion(id_iter, datos):
    _escribir(C.RESULTADOS / "iteraciones" / f"{id_iter}.json", datos)


def cargar_iteracion(id_iter):
    ruta = C.RESULTADOS / "iteraciones" / f"{id_iter}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def guardar_historia(modelo, meses, datos):
    _escribir(C.RESULTADOS / "historia" / f"{modelo}_{meses}m.json",
              {"modelo": modelo, "meses": meses, **datos})


def cargar_metricas(nombre):
    ruta = C.RESULTADOS / "metricas" / f"{nombre}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None
