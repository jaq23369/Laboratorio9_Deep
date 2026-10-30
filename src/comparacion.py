# Tabla comparativa y gráficas de la sección 6. Lee todo lo que dejaron los modelos propios
# (A) y los de librería (B) en Resultados/ y lo junta en un solo lugar.

import json

import numpy as np
import pandas as pd

from src import config as C
from src.graficas import AZUL, GRIS, NARANJA, VERDE, estilo, etiquetas_finales, guardar, pie

R = C.RESULTADOS

# Un color por familia. Los modelos de una misma familia se distinguen por el marcador.
COLOR_FAMILIA = {"Baselines": GRIS, "Árboles": "#CC79A7", "MLP y lineales": AZUL,
                 "Recurrentes": NARANJA, "Encoder-decoder": "#8C3A00",
                 "Residuales y convolucionales": "#56B4E9", "Atención": VERDE,
                 "Probabilísticos": "#E69F00", "Fundacional": "#000000"}
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]

# Modelos de A, entrenados en la Mac. Para compararlos con los de B (PC con NVIDIA) su tiempo
# se pasa a la escala de la GPU con el factor que sale de medir DLinear y la LSTM en las dos.
FACTOR_DE = {"MLP": "DLinear", "DLinear": "DLinear", "NLinear": "DLinear",
             "LSTM": "LSTM", "LSTM_AR": "LSTM", "Seq2seq": "LSTM"}


def cargar_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))


def factores_hardware():
    # Cociente entre el tiempo por epoch en la PC con NVIDIA y en la Mac, para DLinear y la LSTM
    archivos = {("Mac" if "MPS" in cargar_json(f)["hardware"] else "PC"): cargar_json(f)
                for f in (R / "tiempos").glob("*.json")}
    if {"Mac", "PC"} - set(archivos):
        return None
    return {m: archivos["PC"]["modelos"][m]["t_epoch_s"] / archivos["Mac"]["modelos"][m]["t_epoch_s"]
            for m in archivos["Mac"]["modelos"]}


def cargar_tabla():
    filas = []
    f = factores_hardware()
    for ruta in sorted((R / "metricas").glob("*.json")):
        m = cargar_json(ruta)
        t_ent = m.get("t_entrenamiento_s")
        t_norm = t_ent
        if f and m["modelo"] in FACTOR_DE and t_ent:
            t_norm = t_ent * f[FACTOR_DE[m["modelo"]]]
        filas.append({
            "Modelo": m["modelo"], "Familia": m["familia"], "L": m.get("L"),
            "Estrategia": m.get("estrategia"), "Covariables": "Sí" if m.get("covariables") else "No",
            "Parámetros": m.get("params_total"), "Entrenables": m.get("params_entrenables"),
            "t_entrenamiento_s": t_ent, "t_entrenamiento_gpu_s": t_norm,
            "t_inferencia_s": m.get("t_inferencia_s"), "Hardware": m.get("hardware"),
            "MAE": m["MAE"], "RMSE": m["RMSE"], "WAPE": m["WAPE"], "MASE": m["MASE"],
            "MASE_h168": m.get("MASE_h168"), "MASE_navidad": m.get("MASE_navidad"),
            "Pinball": m.get("pinball"), "Cobertura": m.get("cobertura_p10_p90"),
            "MASE_val": m.get("MASE_val"), "mae_por_h": m["mae_por_h"],
            "mase_por_h": m.get("mase_por_h"), "mae_por_hora": m["mae_por_hora"],
        })
    return pd.DataFrame(filas).sort_values("MASE").reset_index(drop=True)


def frontera_pareto(t, mase):
    # Puntos que ningún otro supera a la vez en tiempo y en error
    idx = []
    for i in range(len(t)):
        if not any((t[j] <= t[i] and mase[j] <= mase[i]) and (t[j] < t[i] or mase[j] < mase[i])
                   for j in range(len(t))):
            idx.append(i)
    return sorted(idx, key=lambda i: t[i])


def estilo_modelo(tabla):
    # color por familia y marcador por orden dentro de la familia, fijos para todas las gráficas
    est, contador = {}, {}
    for _, r in tabla.sort_values("Modelo").iterrows():
        k = contador.get(r.Familia, 0)
        contador[r.Familia] = k + 1
        est[r.Modelo] = (COLOR_FAMILIA.get(r.Familia, GRIS), MARCADORES[k % len(MARCADORES)])
    return est


def _colocar_etiquetas(fig, ax, xs, ys, textos, fontsize=8):
    # Pone cada etiqueta en la primera posición que no se monta con las anteriores ni con los puntos
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ocupadas = []
    pts = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    candidatos = [(7, 5), (7, -12), (-7, 5), (-7, -12), (7, 17), (7, -24), (-7, 17), (-7, -24), (0, 22), (0, -28)]
    for k in np.argsort(ys):
        for dx, dy in candidatos:
            t = ax.annotate(textos[k], (xs[k], ys[k]), textcoords="offset points", xytext=(dx, dy),
                            fontsize=fontsize, color="#333333", ha="left" if dx >= 0 else "right")
            caja = t.get_window_extent(rend).expanded(1.04, 1.1)
            choca = (any(caja.overlaps(o) for o in ocupadas)
                     or any(caja.contains(*p) for j, p in enumerate(pts) if j != k))
            if not choca:
                ocupadas.append(caja)
                break
            t.remove()
        else:
            t = ax.annotate(textos[k], (xs[k], ys[k]), textcoords="offset points", xytext=candidatos[0],
                            fontsize=fontsize, color="#333333")
            ocupadas.append(t.get_window_extent(rend))


def grafica_pareto(tabla, nombre="5_mase_vs_tiempo"):
    import matplotlib.pyplot as plt
    est = estilo_modelo(tabla)
    sin_entrenar = 0.3
    tabla = tabla.reset_index(drop=True)
    t = np.where(tabla.t_entrenamiento_gpu_s.fillna(0) > 0, tabla.t_entrenamiento_gpu_s.fillna(0), sin_entrenar)
    mase = tabla.MASE.to_numpy()
    visibles = mase <= 1.25
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    estilo(ax, "MASE de prueba contra tiempo de entrenamiento",
           "Menor es mejor en los dos ejes. Chronos-Bolt no entrena y tiene el menor MASE, o sea que domina a todos los demás.",
           "Tiempo de entrenamiento en segundos (escala log, tiempos de la Mac pasados a la GPU de la PC)",
           "MASE en prueba (H = 24)")
    for i, r in tabla.iterrows():
        if visibles[i]:
            c, mk = est[r.Modelo]
            ax.scatter(t[i], mase[i], s=62, color=c, marker=mk, zorder=3, edgecolor="white", linewidth=0.8)
    idx_todos = frontera_pareto(t, mase)
    entrenan = [i for i in range(len(t)) if t[i] > sin_entrenar and visibles[i]]
    sub = frontera_pareto(t[entrenan], mase[entrenan])
    idx = [entrenan[j] for j in sub]
    ax.plot(t[idx], mase[idx], color="#444444", lw=1.2, ls="--", zorder=2)
    ax.text(t[idx[-1]] * 1.15, mase[idx[-1]] - 0.035, "frontera entre los modelos que entrenan", fontsize=8,
            color="#444444", ha="right")
    ax.set_xscale("log")
    ax.set_xticks([sin_entrenar, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["sin" + chr(10) + "entrenar", "1", "10", "100", "1000", "10000"])
    ax.set_xlim(0.17, t.max() * 4)
    ax.set_ylim(0.70, 1.22)
    fuera = [f"{r.Modelo} ({r.MASE:.2f})" for i, r in tabla.iterrows() if not visibles[i]]
    if fuera:
        ax.text(0.99, 0.97, "Fuera de la escala: " + ", ".join(fuera), transform=ax.transAxes, ha="right",
                va="top", fontsize=8.5, color="#555555")
    ver = [i for i in range(len(tabla)) if visibles[i]]
    _colocar_etiquetas(fig, ax, t[ver], mase[ver], list(tabla.Modelo[ver]))
    pie(fig)
    guardar(fig, nombre)
    return fig, ([tabla.Modelo[i] for i in idx_todos], [tabla.Modelo[i] for i in idx])


def grafica_error_por_h(tabla, nombre="5_error_por_h"):
    import matplotlib.pyplot as plt
    est = estilo_modelo(tabla)
    pasos = np.arange(1, C.H + 1)
    ver = tabla[tabla.Modelo != "Naive"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    estilo(ax, "MAE por paso del horizonte",
           "kWh, promedio de los 100 clientes y los 92 orígenes de prueba. Naive queda fuera de la escala (de 52 a 290 kWh)",
           "Paso del horizonte h (horas después del origen)", "MAE (kWh)")
    ys = []
    for _, r in ver.iterrows():
        c, mk = est[r.Modelo]
        ax.plot(pasos, r.mae_por_h, color=c, lw=1.6, marker=mk, ms=4, alpha=0.95)
        ys.append(r.mae_por_h[-1])
    ax.set_xlim(1, C.H + 5.5)
    ax.set_ylim(15, max(ver.mae_por_h.map(max)) * 1.04)
    etiquetas_finales(ax, [C.H + 0.3] * len(ver), ys, list(ver.Modelo),
                      [est[m][0] for m in ver.Modelo], fontsize=7.5, sep=0.034)
    pie(fig)
    guardar(fig, nombre)
    return fig


def tabla_historia():
    filas = []
    for ruta in sorted((R / "historia").glob("*.json")):
        d = cargar_json(ruta)
        filas.append({"modelo": d["modelo"], "meses": d["meses"], "MASE_val": d["MASE_val"]})
    return pd.DataFrame(filas)


def grafica_historia(nombre="5_mase_vs_historia"):
    import matplotlib.pyplot as plt
    h = tabla_historia()
    chronos = h[h.modelo == "Chronos"].MASE_val.iloc[0] if (h.modelo == "Chronos").any() else None
    orden = ["SeasonalNaive24", "LightGBM", "DLinear", "LSTM", "PatchTST"]
    colores = {"SeasonalNaive24": GRIS, "LightGBM": "#CC79A7", "DLinear": AZUL, "LSTM": NARANJA, "PatchTST": VERDE}
    marc = {"SeasonalNaive24": "o", "LightGBM": "s", "DLinear": "^", "LSTM": "D", "PatchTST": "v"}
    fig, ax = plt.subplots(figsize=(8.6, 5))
    estilo(ax, "MASE de validación contra meses de historia de entrenamiento",
           "Cada punto es un modelo entrenado con esa cantidad de meses, siempre terminando el 2014-06-30",
           "Meses de historia de entrenamiento", "MASE en validación")
    finales = []
    for m in orden:
        d = h[h.modelo == m].sort_values("meses")
        if d.empty:
            continue
        ax.plot(d.meses, d.MASE_val, color=colores[m], lw=2, marker=marc[m], ms=7, markeredgecolor="white")
        finales.append((m, d.MASE_val.iloc[-1]))
    if chronos is not None:
        ax.axhline(chronos, color="#000000", lw=1.3, ls="--")
        ax.text(3, chronos, " Chronos-Bolt zero-shot (sin entrenamiento)", va="bottom", fontsize=8.5)
    ax.set_xticks([3, 12, 30])
    ax.set_xlim(2, 36)
    etiquetas_finales(ax, [30.7] * len(finales), [v for _, v in finales], [m for m, _ in finales],
                      [colores[m] for m, _ in finales], fontsize=8.5, sep=0.05)
    pie(fig)
    guardar(fig, nombre)
    return fig, h


def grafica_hora_y_navidad(tabla, nombre="5_hora_y_navidad"):
    import matplotlib.pyplot as plt
    est = estilo_modelo(tabla)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    estilo(ax, "MAE por hora del día", "kWh, horas del día pronosticado. Naive queda fuera de la escala",
           "Hora del día", "MAE (kWh)")
    ys = []
    tabla_h = tabla[tabla.Modelo != "Naive"]
    for _, r in tabla_h.iterrows():
        c, mk = est[r.Modelo]
        ax.plot(range(24), r.mae_por_hora, color=c, lw=1.4, alpha=0.9)
        ys.append(r.mae_por_hora[-1])
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(0, 28.5)
    ax.set_ylim(15, max(tabla_h.mae_por_hora.map(max)) * 1.04)
    etiquetas_finales(ax, [23.3] * len(tabla_h), ys, list(tabla_h.Modelo), [est[m][0] for m in tabla_h.Modelo],
                      fontsize=7, sep=0.034)
    ax = axes[1]
    t = tabla.sort_values("MASE_navidad")
    estilo(ax, "MASE en la semana del 22 al 28 de diciembre", "Línea punteada, MASE del modelo en todo el periodo de prueba")
    y = np.arange(len(t))
    ax.barh(y, t.MASE_navidad, color=[est[m][0] for m in t.Modelo], height=0.62)
    ax.scatter(t.MASE, y, color="#222222", s=14, zorder=3, label="MASE de todo el periodo")
    ax.set_yticks(y)
    ax.set_yticklabels(t.Modelo, fontsize=8)
    ax.invert_yaxis()
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color="#E5E5EC", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    pie(fig)
    guardar(fig, nombre)
    return fig


def pronosticos_prueba():
    # Archivo de entrega con unique_id, ds, modelo y y_hat de todos los modelos
    partes = []
    for ruta in sorted((R / "pronosticos").glob("*.parquet")):
        p = pd.read_parquet(ruta, columns=["unique_id", "ds", "modelo", "y_hat"])
        partes.append(p)
    return pd.concat(partes, ignore_index=True)


def tabla_presentacion(tabla):
    # La tabla de la sección 6 con los nombres de columna del enunciado, ordenada por MASE
    t = tabla.copy()
    t["t_entrenamiento_s"] = t.t_entrenamiento_s.fillna(0)
    out = pd.DataFrame({
        "Modelo": t.Modelo, "Familia": t.Familia, "L": t.L, "Estrategia multi-paso": t.Estrategia,
        "Covariables": t.Covariables, "Parámetros totales": t.Parámetros, "Parámetros entrenables": t.Entrenables,
        "Entrenamiento (s)": t.t_entrenamiento_s.round(1), "Inferencia (s)": t.t_inferencia_s.round(1),
        "Hardware": t.Hardware, "MAE": t.MAE.round(2), "RMSE": t.RMSE.round(2), "WAPE": t.WAPE.round(4),
        "MASE": t.MASE.round(4), "MASE H=168": t.MASE_h168.round(4), "MASE 22 al 28 dic": t.MASE_navidad.round(4),
        "Pinball": t.Pinball.round(2), "Cobertura P10-P90": t.Cobertura.round(3)})
    return out


def tabla_reentreno():
    filas = []
    for ruta in sorted((R / "reentreno").glob("*.json")):
        d = cargar_json(ruta)
        if "fijo" not in d:
            continue
        filas.append({"Modelo": d["modelo"], "MASE fijo": d["fijo"]["MASE"],
                      "MASE reentreno mensual": d["reentreno_mensual"]["MASE"],
                      "MAE fijo": d["fijo"]["MAE"], "MAE reentreno mensual": d["reentreno_mensual"]["MAE"],
                      "MASE Navidad fijo": d["fijo"]["MASE_navidad"],
                      "MASE Navidad reentreno": d["reentreno_mensual"]["MASE_navidad"]})
    return pd.DataFrame(filas)
