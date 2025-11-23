# Pontificia Universidad Javeriana - Facultad de Ciencias

**Proyecto:** Modelos de Comportamiento para la Predicción de Resultados por Municipios en Colombia de las Pruebas Saber 11 - ICFES

**Autor:** Santiago Andrés Ramírez Montero  
**Director:** Ing. John Corredor, PhD  
**Año:** 2025

---

## Descripción del Proyecto

Este proyecto desarrolla modelos predictivos para analizar y comparar el desempeño académico en las pruebas Saber 11 del ICFES entre municipios colombianos. El estudio se enfoca en identificar patrones de comportamiento y factores determinantes del rendimiento estudiantil mediante técnicas de aprendizaje automático y análisis estadístico avanzado.

### Objetivos Principales

- **Seleccionar** dos municipios con características poblacionales similares pero con desempeño académico contrastante
- **Analizar** los datos históricos de las pruebas Saber 11 (periodo 2014-2024) mediante técnicas de procesamiento de datos masivos
- **Desarrollar** modelos predictivos que expliquen las diferencias en el rendimiento académico

---

## Metodología

El proyecto se estructura en cuatro etapas principales, cada una implementada en un notebook Jupyter independiente:

### 1. Selección de Municipios (`Create_csv.ipynb`)
- Análisis exploratorio de datos del periodo 2023-2024
- Aplicación de criterios consecutivos de selección:
  - Top 2% de municipios por población estudiantil
  - Similitud poblacional (diferencia relativa ≤ 10%)
  - Máximo contraste en puntaje promedio
- Selección final: **Bucaramanga** y **Santa Marta**
- Consolidación del dataset completo (2014-2024)

### 2. Análisis Exploratorio (`Analisis_Preparacion_datos.ipynb`)
- Inicialización de sesión Apache Spark para procesamiento distribuido
- Análisis de calidad de datos (nulos, outliers, inconsistencias)
- Visualizaciones comparativas por municipio
- Análisis de tendencias temporales
- Preparación del dataset para limpieza profunda

### 3. Limpieza de Datos (`Limpieza_Saber11.ipynb`)
- Pipeline de limpieza estructurado en 9 etapas:
  - Normalización de categorías
  - Manejo de valores nulos
  - Estandarización de formatos
  - Eliminación de duplicados
  - Validación de rangos
  - Tratamiento de outliers
  - Creación de variables derivadas
  - Análisis de correlaciones
  - Exportación del dataset limpio
- Dataset final optimizado para modelado predictivo

### 4. Modelado Predictivo (`Modelos.ipynb`)
- Implementación de 4 algoritmos de Machine Learning:
  - **Regresión Lineal**: Modelo base de referencia
  - **Modelo Lineal Generalizado (GLM)**: Con distribución Gaussiana
  - **Random Forest**: Ensamble de árboles de decisión
  - **Gradient Boosted Trees (GBT)**: Boosting con árboles
- Validación cruzada con 3 folds
- Evaluación mediante múltiples métricas (RMSE, MAE, R², MSE)
- Análisis comparativo de desempeño
- Análisis de errores y residuales

---

## Estructura del Repositorio

```
Modelos_Prediccion_Icfes-11/
│
├── datos/                                    # Directorio de datos
│   ├── Examen_Saber_11_20141.txt           # Datos originales ICFES
│   ├── Examen_Saber_11_20142.txt
│   ├── ...
│   ├── Examen_Saber_11_20242.txt
│   ├── municipios_bucaramanga_santamarta.csv  # Dataset filtrado
│   ├── saber_11_limpio_final.csv            # Dataset limpio
│   └── saber_11_para_modelos.csv            # Dataset para modelado
│
├── Create_csv.ipynb                         # [1] Selección de municipios
├── Analisis_Preparacion_datos.ipynb        # [2] Análisis exploratorio
├── Limpieza_Saber11.ipynb                   # [3] Pipeline de limpieza
├── Modelos.ipynb                            # [4] Modelado predictivo
├── requirements.txt                         # Dependencias del proyecto
└── README.md                                # Este archivo
```

---

## Tecnologías Utilizadas

### Procesamiento de Datos
- **Apache Spark (PySpark)**: Procesamiento distribuido de grandes volúmenes de datos
- **Pandas**: Manipulación y análisis de datos estructurados
- **NumPy**: Operaciones numéricas y matriciales

### Machine Learning
- **PySpark MLlib**: Librería de aprendizaje automático distribuido
  - Regresión Lineal
  - Generalized Linear Models (GLM)
  - Random Forest Regressor
  - Gradient Boosted Trees

### Visualización
- **Matplotlib**: Gráficas estáticas personalizables
- **Seaborn**: Visualizaciones estadísticas avanzadas

### Infraestructura
- **Jupyter Notebook**: Entorno de desarrollo interactivo
- **Python 3.x**: Lenguaje de programación principal

---

## Instalación y Configuración

### Requisitos Previos
- Python 3.8 o superior
- Java 8 o superior (requerido por Spark)
- 8 GB RAM mínimo (recomendado 16 GB)

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/Sanandramirezm/Modelos_Prediccion_Icfes-11.git
cd Modelos_Prediccion_Icfes-11

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración de Apache Spark

#### Opción 1: Modo Local (Desarrollo)

El proyecto incluye configuración automática de Spark en modo local. Para entornos Windows, se requiere configuración adicional de Hadoop:

```python
# La configuración se maneja automáticamente en los notebooks
# Ver celdas de inicialización en cada notebook
spark = SparkSession.builder \
    .appName("Analisis Saber 11") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
```

#### Opción 2: Cluster Apache Spark (Producción)

Para ejecutar en un cluster Spark existente (Standalone, YARN o Mesos), modificar la configuración en las celdas de inicialización de cada notebook:

**Configuración para Spark Standalone:**

```python
spark = SparkSession.builder \
    .appName("Analisis Saber 11") \
    .master("spark://master-node:7077") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.instances", "10") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()
```

**Configuración para YARN:**

```python
spark = SparkSession.builder \
    .appName("Analisis Saber 11") \
    .master("yarn") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.instances", "10") \
    .config("spark.driver.memory", "4g") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.shuffle.service.enabled", "true") \
    .getOrCreate()
```

**Sumisión de jobs vía spark-submit:**

```bash
# Para ejecutar en cluster, convertir notebooks a scripts Python y usar:
spark-submit \
    --master spark://master-node:7077 \
    --deploy-mode cluster \
    --executor-memory 8g \
    --executor-cores 4 \
    --num-executors 10 \
    --driver-memory 4g \
    --py-files dependencies.zip \
    script_analisis.py
```

**Consideraciones para Cluster:**

1. **Datos en HDFS**: Los archivos deben estar en un sistema de archivos distribuido
   ```python
   # Cambiar rutas locales por HDFS
   df = spark.read.csv("hdfs://namenode:9000/user/datos/saber11/*.txt")
   ```

2. **Gestión de Dependencias**: Empaquetar librerías Python necesarias
   ```bash
   zip -r dependencies.zip libs/
   ```

3. **Configuración de Recursos**: Ajustar según disponibilidad del cluster
   - `executor.memory`: Memoria por executor (2-8 GB recomendado)
   - `executor.cores`: Cores por executor (2-4 recomendado)
   - `executor.instances`: Número de executors (depende del cluster)

4. **Particionamiento**: Para datasets grandes, aumentar particiones
   ```python
   .config("spark.sql.shuffle.partitions", "400")  # Default es 200
   ```

---

## Uso del Proyecto

### Ejecución Secuencial Recomendada

1. **Generar Dataset Filtrado**
   ```bash
   jupyter notebook Create_csv.ipynb
   ```
   - Ejecutar todas las celdas secuencialmente
   - Output: `datos/municipios_bucaramanga_santamarta.csv`

2. **Análisis Exploratorio**
   ```bash
   jupyter notebook Analisis_Preparacion_datos.ipynb
   ```
   - Revisar visualizaciones y estadísticas descriptivas
   - Identificar patrones iniciales en los datos

3. **Limpieza de Datos**
   ```bash
   jupyter notebook Limpieza_Saber11.ipynb
   ```
   - Ejecutar pipeline de limpieza completo
   - Output: `datos/saber_11_limpio_final.csv`

4. **Modelado Predictivo**
   ```bash
   jupyter notebook Modelos.ipynb
   ```
   - Entrenar y evaluar modelos
   - Comparar desempeño de algoritmos
   - Analizar resultados y conclusiones

---

## Resultados Principales

### Dataset Final
- **Periodo cubierto**: 2014-1 a 2024-2 (21 periodos)
- **Municipios**: Bucaramanga y Santa Marta
- **Registros totales**: Variable según disponibilidad de datos
- **Variables predictoras**: ~150 características después de limpieza

### Comparación de Modelos

Los modelos fueron evaluados mediante validación cruzada (3-fold) con las siguientes métricas:

| Modelo | RMSE | MAE | R² | MSE |
|--------|------|-----|-----|-----|
| Regresión Lineal | Baseline | Baseline | Baseline | Baseline |
| GLM (Gaussian) | Comparable | Comparable | Comparable | Comparable |
| Random Forest | Mejor | Mejor | Mejor | Mejor |
| Gradient Boosted Trees | Óptimo | Óptimo | Óptimo | Óptimo |

*Los valores específicos se encuentran detallados en `Modelos.ipynb`*

### Hallazgos Clave

1. **Diferencias Municipales**: Se identificó una brecha significativa en el desempeño académico entre Bucaramanga y Santa Marta
2. **Variables Importantes**: Los modelos de árbol (Random Forest y GBT) identificaron las variables más influyentes en el desempeño
3. **Tendencias Temporales**: Análisis longitudinal reveló patrones de evolución en ambos municipios
4. **Capacidad Predictiva**: Los modelos de ensamble (Random Forest y GBT) mostraron el mejor desempeño predictivo

---

## Contribuciones y Desarrollo Futuro

### Posibles Extensiones

1. **Ampliación Geográfica**: Incluir más municipios en el análisis comparativo
2. **Variables Adicionales**: Incorporar datos socioeconómicos y demográficos
3. **Modelos Avanzados**: Implementar redes neuronales y deep learning
4. **Análisis Causal**: Aplicar técnicas de inferencia causal
5. **Dashboard Interactivo**: Desarrollar visualizaciones dinámicas con Dash/Streamlit

### Líneas de Investigación Futuras

- Análisis de equidad educativa regional
- Predicción de tendencias a largo plazo
- Identificación de políticas públicas efectivas
- Análisis de factores institucionales y pedagógicos

---

## Contacto

**Autor:** Santiago Andrés Ramírez Montero  
**Institución:** Pontificia Universidad Javeriana  
**Facultad:** Ciencias  
**Email:** sa-ramirezm"javeriana.edu.co

**Director:** John Jairo Corredor Franco 

---

## Licencia

Este proyecto es desarrollado como trabajo de grado para la Pontificia Universidad Javeriana. Los datos utilizados provienen del ICFES y están sujetos a sus términos de uso.

---

## Agradecimientos

- Instituto Colombiano para la Evaluación de la Educación (ICFES) por proporcionar los datos
- Pontificia Universidad Javeriana por el apoyo institucional
- Director de proyecto John Jairo Corredor Franco por su guía y asesoría
- Comunidad de Apache Spark y Python por las herramientas open-source

---

## Referencias

- Instituto Colombiano para la Evaluación de la Educación (ICFES). (2014-2024). *Resultados Pruebas Saber 11*.
- Apache Software Foundation. (2024). *Apache Spark Documentation*.
- Scikit-learn Development Team. (2024). *Machine Learning in Python*.

---
