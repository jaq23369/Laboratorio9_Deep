# Laboratorio 9. Series de tiempo y forecasting

CC3092 Deep Learning y Sistemas Inteligentes. Pronóstico day-ahead (H = 24 horas) del consumo eléctrico por hora de clientes de Portugal con ElectricityLoadDiagrams20112014 del UCI.

## Estructura

```
src/
  config.py          fechas de los bloques, semilla, L, H, rutas y orígenes de backtest
  data.py            descarga, kW cada 15 min a kWh por hora, cambio de horario, filtro de clientes
  features.py        covariables de calendario (seno y coseno, feriados de Portugal)
  dataset.py         escalador por serie y VentanasDataset de PyTorch
  metrics.py         MAE, RMSE, WAPE, MAPE, MASE, pinball y cobertura
  backtest.py        backtest(modelo), protocolo único para todos los modelos
  registro.py        guardado de pronósticos, métricas e iteraciones
  modelos_torch.py   MLP, DLinear, NLinear, LSTM (3 estrategias), Seq2seq y RevIN
  entrenamiento.py   bucle de entrenamiento con early stopping y envoltorio para backtest
  experimentos.py    todas las corridas de los modelos propios
  medir_tiempos.py   tiempo por epoch de referencia para comparar máquinas
  graficas.py        estilo común de las figuras
  modelos_libreria.py      envoltorios de StatsForecast, MLForecast (LightGBM), NeuralForecast y Chronos
  experimentos_libreria.py todas las corridas de los modelos de librería
  comparacion.py     tabla comparativa y gráficas de la sección 6
notebooks/
  1_Datos_y_preparacion.ipynb    carga, limpieza, exploración, Dataset y auditoría de fugas
  2_Investigacion.ipynb          herramientas de forecasting en PyTorch, con comprobaciones de formas
  3_Modelos_propios.ipynb        protocolo, modelos propios, iteraciones, estrategias, local contra global
  4_Modelos_libreria.ipynb       baselines, LightGBM, N-HiTS, TCN, PatchTST, iTransformer, DeepAR y Chronos
  5_Comparacion_y_discusion.ipynb tabla comparativa, gráficas, pronósticos de prueba y discusión
data/processed/      panel_horario.parquet (unique_id, ds, y), calendario.parquet, clientes.json
Resultados/
  pronosticos/       pronósticos de prueba de cada modelo
  metricas/          métricas de prueba y datos de la tabla comparativa de cada modelo
  iteraciones/       registro de cada iteración de hiperparámetros (modelos propios, bl_, lgbm_ y nf_)
  historia/          MASE de validación según meses de historia
  reentreno/         pronósticos y métricas de los modelos reentrenados cada mes (MLP, N-HiTS, LightGBM y el mejor baseline)
  tiempos/           tiempo por epoch de referencia de cada máquina (src/medir_tiempos.py)
  semillas.json      MASE del mejor modelo propio (MLP) con 3 semillas
  semillas_N-HiTS.json  MASE del mejor modelo de deep learning de librería con 3 semillas
  local_global.json  modelo local contra global en 10 clientes
  tabla_comparativa.csv      tabla de la sección 6 ordenada por MASE
  pronosticos_prueba.csv.gz  unique_id, ds, modelo y y_hat de todos los modelos (comprimido, el CSV pesa más de 200 MB)
Figuras/             gráficas de los notebooks
```

## Cómo reproducir

1. Crear el entorno con Python 3.12.

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python -m ipykernel install --user --name lab9-dl --display-name "Python 3.12 (lab9)"
   ```

   En una PC con NVIDIA hay que instalar primero `torch` con CUDA desde [pytorch.org](https://pytorch.org/get-started/locally/).

2. Construir el panel. Descarga los datos (del UCI o, si no responde, de una copia en Hugging Face), corrige el cambio de horario, filtra los clientes y guarda `data/processed/`.

   ```bash
   python -c "from src.data import construir_panel; from src.features import construir_calendario; construir_panel(); construir_calendario()"
   ```

3. Entrenar los modelos propios. Todo queda guardado en `Resultados/`, y si una corrida ya existe se carga en vez de entrenarla otra vez. También se puede correr una sola fase, por ejemplo `python -m src.experimentos fase_semillas`.

   ```bash
   python -m src.experimentos
   ```

   En una Mac con M3 Max (MPS) son 47 iteraciones y unas 3 horas de entrenamiento, casi todo en las LSTM y el seq2seq.

4. Entrenar y evaluar los modelos de librería. También guarda todo en `Resultados/` y salta lo que ya existe, y se puede correr una sola fase, por ejemplo `python -m src.experimentos_libreria fase_nf`. Las fases son `fase_baselines`, `fase_lightgbm`, `fase_chronos`, `fase_nf`, `fase_deepar`, `fase_patchtst`, `fase_reentreno_baseline`, `fase_reentreno_lightgbm`, `fase_semillas_nf`, `fase_reentreno_nf` y `fase_h168`. En una PC con una RTX 4060 Ti de 8 GB y un i5-12400 toda la tanda tarda unas 2.5 horas, con MSTL (unos 17 minutos con los 12 hilos) y las redes de NeuralForecast como lo más largo. Chronos baja sus pesos de Hugging Face la primera vez (`amazon/chronos-bolt-base`, unos 800 MB).

   ```bash
   python -m src.experimentos_libreria
   python -m src.medir_tiempos
   ```

5. Ejecutar los notebooks en orden desde la carpeta `notebooks/`. El notebook 5 usa los tiempos de las dos máquinas de `Resultados/tiempos/` para pasar los de la Mac a la escala de la GPU.

## Decisiones principales

| Decisión | Valor |
| --- | --- |
| Clientes | 313 pasan el filtro de datos completos desde 2012-01-01, se usan 100 elegidos con semilla 42 |
| Bloques | entrenamiento hasta 2014-06-30, validación de julio a septiembre de 2014, prueba de octubre a diciembre de 2014 |
| Protocolo | origen móvil diario a las 00:00 con datos hasta las 23:00 del día anterior, 92 orígenes en prueba |
| Normalización | z-score por serie con estadísticos de entrenamiento |
| MASE | escala igual al MAE del seasonal naive de 24 h dentro de entrenamiento, por cliente |
| Modelos de librería | Se normaliza cada cliente con el z-score de entrenamiento antes de pasar los datos a la librería, así ningún estadístico de validación ni de prueba entra al modelo. Las redes de NeuralForecast se entrenan con `fit` (`val_size = 2208`) y pronostican origen por origen con `predict`, para pasar por el mismo `backtest` |
| Transformer elegido | iTransformer, porque TFT necesitó casi 10 GB de RAM en los primeros 100 pasos con las 218 mil ventanas de validación |
| Hardware | Los modelos propios se entrenaron en una Mac M3 Max (MPS), las redes de librería y Chronos en una PC con RTX 4060 Ti (CUDA), y los baselines y LightGBM en el CPU de la PC |
| Codificación | Todos los JSON se leen y se escriben en UTF-8 (en Windows `write_text` usa cp1252 por defecto) |

## Cómo evaluar un modelo nuevo

Cualquier modelo pasa por la misma función `backtest`. Solo necesita un método `predecir(historia, origen)` que devuelva `unique_id`, `ds` y `y_hat` (y `q10`, `q50`, `q90` si es probabilístico) para las 24 horas desde `origen`. La función revisa en cada origen que no haya fuga y que estén las 24 horas de los 100 clientes.

Con StatsForecast o NeuralForecast lo más rápido es correr `cross_validation` una sola vez y en `predecir` devolver el pedazo de cada origen. Los cortes de `cross_validation(h=24, step_size=24, n_windows=92)` caen a las 23:00 del día anterior a cada origen.

```python
import time
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from src.backtest import backtest, ORIGENES_PRUEBA
from src.data import cargar_panel
from src.registro import guardar_resultado

class DesdeCV:
    def __init__(self, cv, columna):
        cv = cv.rename(columns={columna: "y_hat"})
        cv["origen"] = cv.cutoff + pd.Timedelta(hours=1)
        self.por_origen = {o: g[["unique_id", "ds", "y_hat"]] for o, g in cv.groupby("origen")}

    def predecir(self, historia, origen):
        return self.por_origen[origen]

panel = cargar_panel()
t0 = time.perf_counter()
cv = StatsForecast(models=[SeasonalNaive(24)], freq="h").cross_validation(
    df=panel, h=24, step_size=24, n_windows=92, refit=False)
t_cv = time.perf_counter() - t0

res = backtest(DesdeCV(cv, "SeasonalNaive"), ORIGENES_PRUEBA)
guardar_resultado("SeasonalNaive24", res, {
    "familia": "Baselines", "L": 24, "H": 24, "estrategia": "multi-salida", "covariables": False,
    "params_total": 0, "params_entrenables": 0, "t_entrenamiento_s": 0, "t_inferencia_s": t_cv})
```

- Para NeuralForecast el calendario se une con `panel.merge(cargar_calendario(), on="ds")` y las columnas de `config.COLUMNAS_CALENDARIO` van en `futr_exog_list`. Con `val_size=2208` el early stopping usa justo el bloque de validación.
- En los modelos que pasan por `cross_validation`, `t_inferencia_s` debe ser el tiempo de esa llamada, porque `predecir` solo busca en un diccionario. Si `meta` no lo trae, se guarda el de `backtest`.
- Para la historia (3, 12 y 30 meses) se entrena con datos hasta 2014-06-30, se evalúa con `backtest(modelo, ORIGENES_VAL)` y se guarda con `guardar_historia(nombre, meses, {"MASE_val": res["MASE"]})`.
- Para el reentrenamiento mensual se pasa `reentrenar=fn`, que recibe la historia disponible el 1 de octubre, el 1 de noviembre y el 1 de diciembre y devuelve un modelo nuevo.
- `python -m src.medir_tiempos` guarda en `Resultados/tiempos/` el tiempo por epoch de DLinear y la LSTM finales en la máquina donde se corre. Con el archivo de la Mac y el de la PC con NVIDIA se saca el factor entre las dos.
- En un script (no en un notebook) con `n_jobs=-1` el código tiene que ir dentro de `if __name__ == "__main__":`, porque en macOS y Windows los procesos se lanzan con spawn.
