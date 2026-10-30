# Ventanas deslizantes para PyTorch.
# Una ventana se identifica por (serie, t0), donde t0 es la primera hora del objetivo.
#   entrada   horas t0-L ... t0-1   (y normalizada + calendario)
#   objetivo  horas t0 ... t0+H-1   (y normalizada)
# Una ventana pertenece al bloque donde cae todo su objetivo. La entrada puede venir del
# bloque anterior porque es pasado y ya se conoce cuando se emite el pronóstico.

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config as C


class Escalador:
    # tipo "serie"  z-score por cliente con media y desviación de entrenamiento
    # tipo "global" una sola media y desviación para todo el panel (también de entrenamiento)
    def __init__(self, tipo="serie"):
        self.tipo = tipo

    def ajustar(self, Y, desde, hasta):
        # Y es (n_series, T) y solo se miran las columnas desde..hasta (índices incluidos)
        tramo = Y[:, desde:hasta + 1]
        if self.tipo == "global":
            self.media = np.full((Y.shape[0], 1), tramo.mean())
            self.desv = np.full((Y.shape[0], 1), tramo.std())
        else:
            self.media = tramo.mean(axis=1, keepdims=True)
            self.desv = tramo.std(axis=1, keepdims=True)
        self.desv = np.maximum(self.desv, 1e-6)
        self.rango = (desde, hasta)
        return self

    def transformar(self, Y, filas=None):
        m, s = (self.media, self.desv) if filas is None else (self.media[filas], self.desv[filas])
        return (Y - m) / s

    def invertir(self, Yn, filas=None):
        m, s = (self.media, self.desv) if filas is None else (self.media[filas], self.desv[filas])
        return Yn * s + m


def panel_a_matriz(panel):
    ancho = panel.pivot(index="ds", columns="unique_id", values="y").sort_index()
    return ancho.to_numpy().T.astype(np.float64), list(ancho.columns), ancho.index


def indice(ds_index, t):
    return int(ds_index.get_loc(pd.Timestamp(t)))


def t0_bloque(ds_index, inicio, fin, L=C.L, H=C.H, stride=1, inicio_hist=None, solo_00=False):
    # Todos los t0 cuyo objetivo completo cae en [inicio, fin] y cuya entrada no empieza
    # antes de inicio_hist (por defecto el inicio del panel)
    i0 = indice(ds_index, inicio)
    i1 = indice(ds_index, fin)
    ih = 0 if inicio_hist is None else indice(ds_index, inicio_hist)
    desde = max(i0, ih + L)
    t0s = np.arange(desde, i1 - H + 2, stride)
    if solo_00:
        t0s = t0s[ds_index[t0s].hour == 0]
    return t0s


class VentanasDataset(Dataset):
    # Yn es (n_series, T) ya normalizada, cal es (T, k) con las covariables de calendario.
    # __getitem__ devuelve X (L, 1 + k) y y (H,). Con futuro=True también devuelve el
    # calendario de las horas del objetivo, Xf (H, k), que se conoce de antemano.
    def __init__(self, Yn, cal, t0s, L=C.L, H=C.H, series=None, futuro=False, dispositivo="cpu"):
        self.Y = torch.as_tensor(Yn, dtype=torch.float32, device=dispositivo)
        self.cal = torch.as_tensor(np.array(cal, dtype=np.float32), device=dispositivo)
        series = np.arange(self.Y.shape[0]) if series is None else np.asarray(series)
        t0s = np.asarray(t0s)
        self.serie = torch.as_tensor(np.repeat(series, len(t0s)), device=dispositivo)
        self.t0 = torch.as_tensor(np.tile(t0s, len(series)), device=dispositivo)
        self.L, self.H, self.futuro = L, H, futuro
        self.rel_x = torch.arange(-L, 0, device=dispositivo)
        self.rel_y = torch.arange(0, H, device=dispositivo)

    def __len__(self):
        return len(self.t0)

    def lote(self, k):
        # Arma un lote completo con indexación avanzada, sin bucles de Python.
        # Devuelve X (B, L, 1+k), Xf (B, H, k), y (B, H) y el índice de serie de cada ventana.
        s, t0 = self.serie[k], self.t0[k]
        ix = t0[:, None] + self.rel_x
        iy = t0[:, None] + self.rel_y
        y_in = self.Y[s[:, None], ix].unsqueeze(-1)
        X = torch.cat([y_in, self.cal[ix]], dim=-1)
        return X, self.cal[iy], self.Y[s[:, None], iy], s

    def __getitem__(self, k):
        X, Xf, y, _ = self.lote(torch.as_tensor([k], device=self.Y.device))
        return (X[0], Xf[0], y[0]) if self.futuro else (X[0], y[0])


def contar_ventanas(ds_index, n_series, L=C.L, H=C.H, stride=1):
    bloques = {
        "entrenamiento": (C.INICIO, C.FIN_TRAIN),
        "validación": (C.INICIO_VAL, C.FIN_VAL),
        "prueba": (C.INICIO_TEST, C.FIN_TEST),
    }
    filas = []
    for nombre, (a, b) in bloques.items():
        t0s = t0_bloque(ds_index, a, b, L, H, stride)
        filas.append({"bloque": nombre, "ventanas_por_serie": len(t0s),
                      "ventanas_panel": len(t0s) * n_series,
                      "primer_objetivo": ds_index[t0s[0]], "ultimo_objetivo": ds_index[t0s[-1] + H - 1]})
    return pd.DataFrame(filas)
