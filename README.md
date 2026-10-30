# Análisis de ECG 

Código del Trabajo de Fin de Grado de María del Pilar Albarrán Villalón (ETSI Industriales, Universidad Politécnica de Madrid), tutorizado por Manuel Mendoza de Haro.

El trabajo clasifica electrocardiogramas de 12 derivaciones de la base de datos PTB-XL en cinco superclases diagnósticas (Normal, Infarto, Isquemia/ST-T, Hipertrofia y Conducción) y compara qué aporta representar la señal como espectrograma STFT frente a usar directamente la señal temporal.

## Resultados principales

Conjunto de prueba de la partición oficial (fold 10, 2.158 ECG). Media y desviación típica de tres semillas.

| Modelo | Entrada | Parámetros | AUC macro | F1 macro |
|---|---|---|---|---|
| Modelo 05 (estudio de ablación) | Espectrograma | 105.317 | 0.879 ± 0.002 | 0.671 ± 0.002 |
| Tras la búsqueda de hiperparámetros | Espectrograma | 105.317 | 0.886 ± 0.002 | 0.672 ± 0.001 |
| Codificador por derivación + R5 | Espectrograma | 73.285 | 0.892 ± 0.0004 | 0.690 ± 0.003 |
| ResNet 1D + R5 | Señal temporal | 537.029 | 0.910 ± 0.006 | 0.719 ± 0.008 |

## Contenido

| Archivo | Contenido |
|---|---|
| `01` a `05` | Escalera de redes convolucionales del estudio de ablación |
| `06` a `08` | Modelos clásicos: Random Forest, XGBoost y regresión logística |
| `09` a `13` | Métricas ponderadas, validación cruzada, gráficas comparativas del estudio de ablación y remuestreo |
| `14_busqueda_experimental.ipynb` | Búsqueda de hiperparámetros (30 entrenamientos) |
| `15_arquitecturas_y_sobreajuste.ipynb` | Comparación de 15 arquitecturas, técnicas contra el sobreajuste y semillas (31 entrenamientos) |
| `16_evaluacion_final.ipynb` | Matrices de confusión, curvas ROC, métricas por superclase y diagnóstico de ECG concretos |
| `experimentos_tfg.py` | Lectura de PTB-XL, filtrado, STFT, entrenamiento y métricas |
| `arquitecturas_tfg.py` | Las 15 arquitecturas y las variantes contra el sobreajuste |
| `evaluacion_final.py` | Evaluación final y función de diagnóstico |
| `outputs/` | Resultados de cada entrenamiento (CSV y Excel), historiales y figuras |

## Datos

Los datos no se incluyen en el repositorio. PTB-XL (versión 1.0.3) es de acceso libre en PhysioNet:
https://physionet.org/content/ptb-xl/1.0.3/

Se usa la versión de 100 Hz y la partición oficial por paciente (folds 1-8 entrenamiento, 9 validación y 10 prueba).

## Cómo reproducirlo

1. Instalar Python 3.10 o superior y las librerías:
   ```
   pip install -r requirements.txt
   ```
2. Descargar PTB-XL y descomprimirlo.
3. Indicar la carpeta de los datos al principio de cada notebook, por ejemplo:
   ```python
   import arquitecturas_tfg as A
   A.configurar(r"C:\ruta\a\ptb-xl-1.0.3")
   ```
4. Ejecutar los notebooks en orden. Los notebooks 14 y 15 guardan cada entrenamiento en cuanto termina, así que si se interrumpen se pueden relanzar sin repetir lo ya hecho. El notebook 16 no entrena: carga los pesos guardados por el 15.

Todas las decisiones se toman con el conjunto de validación; el conjunto de prueba solo se usa para informar de los resultados. La semilla por defecto es 42.

## Aviso

Es un prototipo académico. No es un producto sanitario ni debe usarse para tomar decisiones clínicas.
