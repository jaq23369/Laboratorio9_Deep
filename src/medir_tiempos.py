# Mide el tiempo por epoch de la configuración final de DLinear y de la LSTM en cualquier
# máquina, para sacar un factor entre la Mac (MPS) y la PC con NVIDIA (CUDA).
# Uso:  python -m src.medir_tiempos
# Guarda Resultados/tiempos/<hardware>.json. Con los dos archivos, el notebook de comparación
# puede pasar los tiempos de una máquina a la escala de la otra.

import json
import re

import numpy as np

from src import config as C
from src.entrenamiento import Datos, entrenar
from src.registro import _escribir, cargar_metricas, hardware

EPOCHS = 5


def config_final(nombre):
    # La configuración final queda en Resultados/metricas/<nombre>.json, que viene en el repo
    met = cargar_metricas(nombre)
    return met["id_iteracion"], met["config"]


def main():
    datos = Datos()
    filas = {}
    for nombre in ["DLinear", "LSTM"]:
        id_iter, cfg = config_final(nombre)
        cfg = {**cfg, "epochs": EPOCHS, "paciencia": EPOCHS + 1}
        _, _, info = entrenar(cfg, datos, verbose=False)
        # El primer epoch incluye el calentamiento de la GPU, se descarta
        t = float(np.mean(info["historial"]["t_epoch_s"][1:]))
        filas[nombre] = {"id_referencia": id_iter, "t_epoch_s": t,
                         "muestras_epoch": cfg["muestras_epoch"], "batch": cfg["batch"]}
        print(f"{nombre:8s} {t:.2f} s por epoch")
    hw = hardware()
    salida = {"hardware": hw, "epochs_medidos": EPOCHS - 1, "modelos": filas}
    nombre = re.sub(r"[^A-Za-z0-9]+", "_", hw).strip("_")
    _escribir(C.RESULTADOS / "tiempos" / f"{nombre}.json", salida)
    print(json.dumps(salida, indent=1))


if __name__ == "__main__":
    main()
