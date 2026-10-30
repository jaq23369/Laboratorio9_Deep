# Estilo común de las gráficas (paleta Okabe-Ito, sin marco superior ni derecho,
# rejilla horizontal tenue, título a la izquierda con unidades).

import matplotlib.pyplot as plt
import numpy as np

from src import config as C

AZUL, NARANJA, VERDE, GRIS, GRIS_CLARO = "#0072B2", "#D55E00", "#009E73", "#999999", "#C9CCD6"
CUALITATIVA = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
FUENTE = "UCI ElectricityLoadDiagrams20112014, 100 clientes (semilla 42)"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                     "axes.titlesize": 12, "savefig.bbox": "tight"})


def estilo(ax, titulo, subtitulo=None, xlabel=None, ylabel=None):
    ax.set_title(titulo, loc="left", fontsize=12, pad=18 if subtitulo else 6)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, color="#777777", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E5E5EC", lw=0.8)
    ax.set_axisbelow(True)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax


def pie(fig, texto=FUENTE):
    fig.text(0.01, -0.07, f"Fuente: {texto}", color="#777777", fontsize=8)


def guardar(fig, nombre):
    fig.savefig(C.FIGURAS / f"{nombre}.png")


def etiquetas_finales(ax, xs, ys, textos, colores, fontsize=8, sep=0.05, dx_min=None):
    # Etiquetas directas al final de cada línea, separadas en vertical cuando quedan encimadas.
    # Llamar después de fijar los límites de los ejes.
    lo, hi = ax.get_ylim()
    paso = (hi - lo) * sep
    x0, x1 = ax.get_xlim()
    dx_min = (x1 - x0) * 0.12 if dx_min is None else dx_min
    puestas = []
    for i in np.argsort(ys):
        y = ys[i]
        for xp, yp in sorted(puestas, key=lambda t: t[1]):
            if abs(xs[i] - xp) < dx_min and y < yp + paso:
                y = yp + paso
        puestas.append((xs[i], y))
        ax.text(xs[i], y, textos[i], color=colores[i], fontsize=fontsize, va="center")
