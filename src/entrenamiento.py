# Bucle de entrenamiento común para los modelos propios y su envoltorio para backtest.
# Selección de configuraciones y early stopping solo con el bloque de validación.

import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from src import config as C
from src.data import cargar_panel
from src.dataset import Escalador, VentanasDataset, indice, panel_a_matriz, t0_bloque
from src.features import cargar_calendario
from src.metrics import escalas_mase
from src.modelos_torch import construir, contar_parametros
from src.registro import cargar_iteracion, guardar_iteracion, hardware

BASE = {"modelo": "dlinear", "L": C.L, "H": C.H, "meses": 30, "norm": "serie",
        "oculto": 128, "capas": 2, "dropout": 0.1, "lr": 1e-3, "batch": 1024,
        "epochs": 30, "paciencia": 4, "muestras_epoch": 200_000, "semilla": C.SEMILLA,
        "teacher_forcing": 0.5}

# Los recurrentes son unas 100 veces más lentos por paso que los lineales en MPS,
# así que por defecto ven 100 mil ventanas por epoch en vez de 200 mil
RECURRENTES = ("lstm", "gru", "lstm_ar", "seq2seq")
BASE_RECURRENTE = {"oculto": 64, "muestras_epoch": 100_000, "epochs": 60, "paciencia": 5}


def completar(cfg):
    extra = BASE_RECURRENTE if cfg.get("modelo") in RECURRENTES else {}
    return {**BASE, **extra, **cfg}


def dispositivo():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def memoria_gpu():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def inicio_historia(meses):
    # Los meses de historia terminan siempre el 2014-06-30, así validación es la misma
    return max(C.INICIO, C.FIN_TRAIN + pd.Timedelta(hours=1) - pd.DateOffset(months=meses))


class Datos:
    # Panel en forma de matriz (n_series, T) alineado con el calendario
    def __init__(self, panel=None):
        panel = cargar_panel() if panel is None else panel
        self.Y, self.ids, self.ds = panel_a_matriz(panel)
        cal = cargar_calendario().set_index("ds").loc[self.ds]
        self.cal = cal[C.COLUMNAS_CALENDARIO].to_numpy()
        self.cal_df = cal[C.COLUMNAS_CALENDARIO]
        self.escalas = escalas_mase(panel).reindex(self.ids).to_numpy()
        self.k = self.cal.shape[1]


def _predecir_lotes(modelo, dset, n=1024):
    modelo.eval()
    salidas = []
    with torch.no_grad():
        for a in range(0, len(dset), n):
            k = torch.arange(a, min(a + n, len(dset)), device=dset.Y.device)
            X, Xf, y, s = dset.lote(k)
            salidas.append((modelo(X, Xf), y, s))
    p = torch.cat([o[0] for o in salidas]).cpu().numpy()
    y = torch.cat([o[1] for o in salidas]).cpu().numpy()
    s = torch.cat([o[2] for o in salidas]).cpu().numpy()
    return p, y, s


def _metricas_val(modelo, dset, esc, escalas):
    p, y, s = _predecir_lotes(modelo, dset)
    perdida = float(np.abs(p - y).mean())
    p_kwh = esc.invertir(p, s)
    y_kwh = esc.invertir(y, s)
    ae = np.abs(p_kwh - y_kwh)
    return perdida, float(ae.mean()), float((ae / escalas[s][:, None]).mean())


def entrenar(cfg, datos, series=None, hasta=None, epochs_fijos=None, verbose=True):
    # series limita el entrenamiento a algunos clientes (modelo local de la 5.3).
    # hasta y epochs_fijos se usan al reentrenar con ventana expandible, sin validación.
    cfg = completar(cfg)
    fijar_semilla(cfg["semilla"])
    dev = dispositivo()
    L, H = cfg["L"], cfg["H"]
    ih = inicio_historia(cfg["meses"])
    fin_train = C.FIN_TRAIN if hasta is None else hasta

    esc = Escalador("global" if cfg["norm"] == "global" else "serie")
    esc.ajustar(datos.Y, indice(datos.ds, ih), indice(datos.ds, fin_train))
    Yn = esc.transformar(datos.Y)

    t0_tr = t0_bloque(datos.ds, ih, fin_train, L, H, inicio_hist=ih)
    dtr = VentanasDataset(Yn, datos.cal, t0_tr, L, H, series=series, dispositivo=dev)
    dval = None
    if hasta is None:
        t0_val = t0_bloque(datos.ds, C.INICIO_VAL, C.FIN_VAL, L, H, solo_00=True)
        dval = VentanasDataset(Yn, datos.cal, t0_val, L, H, series=series, dispositivo=dev)

    modelo = construir(cfg, datos.k).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    total, entrenables = contar_parametros(modelo)
    muestras = min(cfg["muestras_epoch"], len(dtr))
    pasos = max(1, muestras // cfg["batch"])
    n_epochs = epochs_fijos or cfg["epochs"]

    hist = {"epoch": [], "perdida_train": [], "perdida_val": [], "mae_val": [],
            "mase_val": [], "t_epoch_s": []}
    mejor, mejor_estado, sin_mejora, mem = np.inf, None, 0, 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_inicio = time.perf_counter()
    for ep in range(1, n_epochs + 1):
        modelo.train()
        t_ep = time.perf_counter()
        acum = torch.zeros((), device=dev)
        for _ in range(pasos):
            k = torch.randint(0, len(dtr), (cfg["batch"],), device=dev)
            X, Xf, y, _ = dtr.lote(k)
            perdida = (modelo(X, Xf, y) - y).abs().mean()
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            acum += perdida.detach()
        p_train = float(acum / pasos)
        if dev.type == "mps":
            torch.mps.synchronize()
        hist["epoch"].append(ep)
        hist["perdida_train"].append(p_train)
        if dval is not None:
            p_val, mae_v, mase_v = _metricas_val(modelo, dval, esc, datos.escalas)
            hist["perdida_val"].append(p_val)
            hist["mae_val"].append(mae_v)
            hist["mase_val"].append(mase_v)
        hist["t_epoch_s"].append(time.perf_counter() - t_ep)
        mem = max(mem, memoria_gpu() or 0)
        if dev.type == "mps":
            torch.mps.empty_cache()
        if verbose:
            extra = f" val {p_val:.4f} MAE {mae_v:.2f} MASE {mase_v:.4f}" if dval else ""
            print(f"  ep {ep:2d} train {p_train:.4f}{extra} ({hist['t_epoch_s'][-1]:.1f}s)", flush=True)
        if dval is None:
            continue
        if p_val < mejor - 1e-5:
            mejor, mejor_estado, sin_mejora = p_val, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= cfg["paciencia"]:
                break
    t_total = time.perf_counter() - t_inicio
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    mejor_ep = int(np.argmin(hist["perdida_val"])) + 1 if hist["perdida_val"] else n_epochs
    info = {"config": cfg, "historial": hist, "params_total": total,
            "params_entrenables": entrenables, "t_entrenamiento_s": t_total,
            "t_epoch_prom_s": float(np.mean(hist["t_epoch_s"])), "memoria_pico_mb": mem,
            "mejor_epoch": mejor_ep, "hardware": hardware(),
            "ventanas_train": len(dtr), "ventanas_val": len(dval) if dval else 0}
    if hist["mase_val"]:
        info["mase_val"] = hist["mase_val"][mejor_ep - 1]
        info["mae_val"] = hist["mae_val"][mejor_ep - 1]
    return modelo, esc, info


class PronosticadorTorch:
    # Envoltorio para backtest. Con la historia hasta origen - 1 h arma la entrada de cada
    # cliente con sus últimas L horas y el calendario de las H horas siguientes.
    def __init__(self, modelo, esc, datos, cfg, ids=None):
        self.modelo, self.esc, self.datos = modelo.eval(), esc, datos
        self.L, self.H = cfg["L"], cfg["H"]
        self.ids = list(datos.ids if ids is None else ids)
        self.filas = np.array([datos.ids.index(i) for i in self.ids])
        self.dev = next(modelo.parameters()).device

    def predecir(self, historia, origen):
        desde = origen - pd.Timedelta(hours=self.L)
        ult = historia[(historia.ds >= desde) & historia.unique_id.isin(self.ids)]
        ancho = ult.pivot(index="ds", columns="unique_id", values="y")[self.ids]
        assert len(ancho) == self.L and ancho.index.max() < origen
        Yn = self.esc.transformar(ancho.to_numpy().T, self.filas)
        horas_f = pd.date_range(origen, periods=self.H, freq="h")
        cal_in = self.datos.cal_df.loc[ancho.index].to_numpy()
        cal_f = self.datos.cal_df.loc[horas_f].to_numpy()
        n = len(self.ids)
        X = np.concatenate([Yn[:, :, None], np.broadcast_to(cal_in, (n,) + cal_in.shape)], axis=-1)
        Xf = np.broadcast_to(cal_f, (n,) + cal_f.shape)
        with torch.no_grad():
            p = self.modelo(torch.tensor(X, dtype=torch.float32, device=self.dev),
                            torch.tensor(np.ascontiguousarray(Xf), dtype=torch.float32, device=self.dev))
        p = self.esc.invertir(p.cpu().numpy(), self.filas)
        return pd.DataFrame({"unique_id": np.repeat(self.ids, self.H),
                             "ds": np.tile(horas_f, n), "y_hat": p.ravel()})


class PronosticadorLocal:
    # Junta varios modelos locales (uno por cliente) en un solo objeto para backtest
    def __init__(self, pronosticadores):
        self.pronosticadores = pronosticadores

    def predecir(self, historia, origen):
        return pd.concat([p.predecir(historia, origen) for p in self.pronosticadores],
                         ignore_index=True)


def ruta_pesos(id_iter):
    return C.MODELOS / "iteraciones" / f"{id_iter}.pt"


def medir_inferencia(modelo, esc, datos, cfg):
    # Tiempo de inferencia de una iteración con backtest sobre los 92 orígenes de validación.
    # Se usa validación para que la prueba se toque una sola vez por modelo final.
    from src.backtest import ORIGENES_VAL, backtest
    res = backtest(PronosticadorTorch(modelo, esc, datos, cfg), ORIGENES_VAL, H=cfg["H"])
    return {"t_inferencia_s": res["t_inferencia_s"], "origenes_inferencia": "validación",
            "n_origenes_inferencia": res["n_origenes"]}


def correr_iteracion(id_iter, cfg, datos, forzar=False, verbose=True):
    # Si la iteración ya está guardada, carga sus pesos y su registro en vez de entrenar
    cfg = completar(cfg)
    info = cargar_iteracion(id_iter)
    ruta = ruta_pesos(id_iter)
    if info is not None and ruta.exists() and not forzar:
        modelo = construir(cfg, datos.k).to(dispositivo())
        guardado = torch.load(ruta, map_location=dispositivo(), weights_only=False)
        modelo.load_state_dict(guardado["estado"])
        if "t_inferencia_s" not in info:
            info.update(medir_inferencia(modelo, guardado["escalador"], datos, cfg))
            guardar_iteracion(id_iter, info)
        return modelo, guardado["escalador"], info
    if verbose:
        print(f"{id_iter}: {json.dumps({k: cfg[k] for k in cfg if cfg[k] != BASE.get(k)})}")
    modelo, esc, info = entrenar(cfg, datos, verbose=verbose)
    info["id"] = id_iter
    info.update(medir_inferencia(modelo, esc, datos, cfg))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"estado": modelo.state_dict(), "escalador": esc}, ruta)
    guardar_iteracion(id_iter, info)
    return modelo, esc, info
