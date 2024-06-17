# New or used algorithm_MELI

Este proyecto abarca un análisis exploratorio de datos (EDA), preprocesamiento de datos, ingeniería de características y entrenamiento de modelos utilizando PyCaret para una tarea de clasificación. Se realizaron diversas transformaciones de datos, se exploraron correlaciones, se identificaron y manejarion valores faltantes, y se utilizó la capacidad de AutoML de PyCaret para seleccionar y ajustar el mejor modelo. Además, se visualizó el rendimiento del modelo y se generó un informe de lift para una evaluación adicional.

## Pasos Realizados

### 1. Análisis Exploratorio de Datos (EDA)

**Normalización y Renombrado:**

Se normalizaron las columnas anidadas JSON y se renombraron para mejorar la legibilidad.

**Eliminación de Columnas Innecesarias:**

Se eliminaron las columnas iniciales que fueron normalizadas.

**Concatenación de DataFrames:**

Se concatenaron los DataFrames normalizados en un solo DataFrame.

**Reemplazo de Valores:**

Se reemplazaron los valores 'new' y 'used' con valores binarios para la variable objetivo.

### 2. Ingeniería de Características

**Cálculo de Estadísticas de Grupo:**

Se calcularon estadísticas de grupo para varias columnas.

**Creación de Columnas Transformadas:**

Se crearon nuevas columnas con transformaciones como cuadrado, cubo, raíz cuadrada, logaritmo, etc.

**Extracción de Palabras de Títulos:**

Se extrajeron las primeras palabras de los títulos y se crearon nuevas columnas con estas palabras.

### 3. Manejo de Duplicados y Correlaciones

**Identificación de Columnas Duplicadas:**

Se identificaron las columnas duplicadas en el DataFrame.

**Eliminación de Características Altamente Correlacionadas:**

Se eliminaron las características altamente correlacionadas para mejorar la precisión del modelo.

### 4. División de Datos

**División de Datos en Conjuntos de Entrenamiento y Validación:**

Se dividió el conjunto de datos en conjuntos de entrenamiento y validación.

### 5. Entrenamiento y Evaluación del Modelo

**Inicialización del Experimento PyCaret y Entrenamiento de Modelos:**

Se utilizó PyCaret para comparar y seleccionar el mejor modelo basado en la precisión.

**Ajuste del Mejor Modelo Usando Validación Cruzada:**

Se ajustó el modelo seleccionado utilizando validación cruzada para mejorar su rendimiento.

**Evaluación del Modelo:**

Se evaluó el rendimiento del modelo mediante visualización de curvas y matrices de confusión.

### 6. Análisis de Lift

**Generación de Informe de Lift:**

Se generó un informe de lift para evaluar la efectividad del modelo.

## Resumen

* **Limpieza de Datos:** Se manejarion valores faltantes y se eliminaron columnas innecesarias.
* **Ingeniería de Características:** Se crearon nuevas características y términos de interacción.
* **Manejo de Duplicados y Correlaciones:** Se identificaron y eliminaron columnas duplicadas y características altamente correlacionadas.
* **División de Datos:** Se dividió el conjunto de datos en conjuntos de entrenamiento y validación.
* **Entrenamiento del Modelo:** Se utilizó PyCaret para comparar y seleccionar el mejor modelo basado en precisión.
* **Ajuste del Modelo:** Se ajustó el modelo seleccionado utilizando validación cruzada.
* **Evaluación:** Se visualizó el rendimiento del modelo utilizando varias gráficas.
* **Análisis de Lift:** Se generó un informe de lift para evaluar la efectividad del modelo.

Este README proporciona una visión general de los pasos realizados y el código utilizado en este proyecto, asegurando la reproducibilidad y la comprensión del flujo de trabajo.
