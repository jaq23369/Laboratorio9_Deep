# Pipeline de datos crudos a series horarias limpias.
# Pasos: descargar, leer, pasar de kW cada 15 min a kWh por hora, arreglar los días de
# cambio de horario, filtrar clientes completos desde 2012 y elegir el subconjunto.

import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src import config as C


def descargar(timeout=60):
    # Primero intenta el UCI como dice el enunciado. Si el servidor no responde (el 29-oct
    # bajaba a 2 KB/s), usa una copia en Hugging Face del mismo archivo en parquet.
    # Se comprobó que coincide con el UCI (diferencia relativa de 6e-8 por el float32).
    if C.ARCHIVO_CRUDO.exists():
        return C.ARCHIVO_CRUDO
    if C.ARCHIVO_ESPEJO.exists():
        return C.ARCHIVO_ESPEJO
    zip_path = C.DATA_RAW / "electricity.zip"
    try:
        with urllib.request.urlopen(C.URL_DATOS, timeout=timeout) as r, open(zip_path, "wb") as f:
            f.write(r.read())
        zipfile.ZipFile(zip_path).extract("LD2011_2014.txt", C.DATA_RAW)
        return C.ARCHIVO_CRUDO
    except Exception as e:
        print("UCI no respondió, se usa la copia de Hugging Face:", e)
        zip_path.unlink(missing_ok=True)
        C.ARCHIVO_ESPEJO.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(C.URL_ESPEJO, C.ARCHIVO_ESPEJO)
        return C.ARCHIVO_ESPEJO


def leer_crudo():
    # 140,256 filas (cada 15 min) por 370 clientes, valores en kW
    ruta = descargar()
    if ruta.suffix == ".parquet":
        crudo = pd.read_parquet(ruta)
        crudo.index = pd.to_datetime(crudo.pop("datetime"))
        return crudo.astype("float64")
    return pd.read_csv(ruta, sep=";", decimal=",", index_col=0, parse_dates=True)


def a_horario(crudo):
    # Cada registro es la potencia promedio en kW de los 15 min que terminan en su marca
    # (el archivo empieza en 2011-01-01 00:15 y termina en 2015-01-01 00:00).
    # La energía de ese cuarto de hora es kW * 0.25 h, y la de una hora es la suma de 4.
    # closed="right" y label="left" hacen que la hora 00:00 junte 00:15, 00:30, 00:45 y 01:00.
    return (crudo / 4.0).resample("h", closed="right", label="left").sum()


MARCAS_CAMBIO = ["01:00", "01:15", "01:30", "01:45"]


def dias_cambio_horario(anios):
    # En Portugal el reloj cambia el último domingo de marzo y el último domingo de octubre
    marzo, octubre = [], []
    for a in anios:
        marzo.append(max(d for d in pd.date_range(f"{a}-03-01", f"{a}-03-31") if d.dayofweek == 6))
        octubre.append(max(d for d in pd.date_range(f"{a}-10-01", f"{a}-10-31") if d.dayofweek == 6))
    return marzo, octubre


def corregir_cambio_horario(crudo):
    # Se corrige sobre los registros de 15 min, antes de agrupar por hora.
    # En los datos la anomalía cae en las marcas 01:00, 01:15, 01:30 y 01:45.
    # Marzo tiene 23 horas y esas marcas vienen en cero, se interpolan entre 00:45 y 02:00.
    # Octubre tiene 25 horas y esas marcas traen dos horas sumadas, se dividen entre 2.
    c = crudo.copy()
    marzo, octubre = dias_cambio_horario(sorted(set(c.index.year)))
    for d in marzo:
        marcas = [d + pd.Timedelta(m + ":00") for m in MARCAS_CAMBIO]
        antes, despues = d + pd.Timedelta("00:45:00"), d + pd.Timedelta("02:00:00")
        if antes in c.index and despues in c.index:
            for k, t in enumerate(marcas, start=1):
                c.loc[t] = c.loc[antes] + (c.loc[despues] - c.loc[antes]) * k / 5
    for d in octubre:
        marcas = [d + pd.Timedelta(m + ":00") for m in MARCAS_CAMBIO]
        if marcas[0] in c.index:
            c.loc[marcas] = c.loc[marcas] / 2
    return c


def racha_max_ceros(serie):
    es_cero = (serie.to_numpy() == 0).astype(np.int64)
    # largo de cada bloque de ceros seguidos
    bordes = np.diff(np.concatenate([[0], es_cero, [0]]))
    inicios, finales = np.where(bordes == 1)[0], np.where(bordes == -1)[0]
    return int((finales - inicios).max()) if len(inicios) else 0


def criterios_filtro(horario):
    # Tabla con los criterios de "datos completos desde 2012-01-01" para cada cliente
    periodo = horario.loc[C.INICIO:C.FIN]
    final = periodo.loc[C.FIN - pd.Timedelta(days=28):]
    return pd.DataFrame({
        "activo_1_ene_2012": (periodo.iloc[:24] > 0).any(),
        "racha_ceros_h": periodo.apply(racha_max_ceros),
        "frac_ceros": (periodo == 0).mean(),
        "activo_final": (final > 0).mean() > 0.5,
    })


def filtrar_clientes(horario):
    # Se queda con los clientes que consumen desde el primer día de 2012, que no tienen
    # rachas de ceros de más de una semana, que tienen menos de 10 % de horas en cero
    # y que siguen consumiendo en las últimas 4 semanas de 2014.
    t = criterios_filtro(horario)
    ok = t.activo_1_ene_2012 & (t.racha_ceros_h <= 168) & (t.frac_ceros < 0.10) & t.activo_final
    return list(t.index[ok])


def elegir_subconjunto(clientes, n=C.N_CLIENTES, semilla=C.SEMILLA):
    rng = np.random.default_rng(semilla)
    return sorted(rng.choice(clientes, size=n, replace=False).tolist())


def a_largo(ancho):
    largo = ancho.rename_axis("ds").reset_index().melt(id_vars="ds", var_name="unique_id",
                                                      value_name="y")
    return largo[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def a_ancho(largo):
    return largo.pivot(index="ds", columns="unique_id", values="y").sort_index()


def construir_panel(forzar=False):
    if C.PANEL.exists() and C.CLIENTES.exists() and C.HORARIO_TODOS.exists() and not forzar:
        return cargar_panel()
    horario = a_horario(corregir_cambio_horario(leer_crudo()))
    horario.to_parquet(C.HORARIO_TODOS)
    filtrados = filtrar_clientes(horario)
    elegidos = elegir_subconjunto(filtrados)
    C.CLIENTES.write_text(json.dumps({"filtrados": filtrados, "elegidos": elegidos,
                                      "semilla": C.SEMILLA}, indent=1), encoding="utf-8")
    a_largo(horario.loc[C.INICIO:C.FIN, elegidos]).to_parquet(C.PANEL, index=False)
    return cargar_panel()


def cargar_panel():
    return pd.read_parquet(C.PANEL)


def cargar_clientes():
    return json.loads(C.CLIENTES.read_text(encoding="utf-8"))


def cargar_horario_todos():
    return pd.read_parquet(C.HORARIO_TODOS)
