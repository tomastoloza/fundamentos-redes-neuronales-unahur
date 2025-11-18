# Trabajo Pr ́actico: Redes Neuronales

# Convolucionales

## Tecnicatura en Inteligencia Artificial

## Universidad Nacional de Hurlingham

## Docente: Emiliano Churruca

## Asistente: Nicolas Seivane

## Fecha de entrega: 11 y 18 de Noviembre

## Objetivos del Trabajo Pr ́actico

El objetivo de este trabajo pr ́actico es entrenar una red neuronal convolucional
(CNN) con una salida de 8 neuronas, donde cada neurona representar ́a una
categor ́ıa de animal. El trabajo consiste en los siguientes puntos:

1. Implementar una arquitectura de red convolucional para clasificar las
    im ́agenes de animales que se encuentran en el drive, se deben utilizar
    todas las categor ́ıas, no solo las que recolecto el grupo.
2. Deben dividir los conjuntos de datos en conjuntos de entrenamiento, prueba
    y validaci ́on. Recomendaci ́on: Busquen im ́agenes que no sean esas para
    la validaci ́on, pueden ser caricaturas o dibujos de los animales, sean cre-
    ativos.
3. Establecer una salida de 8 neuronas, donde cada una corresponde a una
    categor ́ıa de animal (por ejemplo: perro, gato, p ́ajaro, etc.). La salida
    debe ser one-hot, lo que significa que la representaci ́on de la clase de salida
    ser ́a un vector de 8 elementos en el que todas las posiciones, excepto una,
    tendr ́an valor 0. La ́unica posici ́on con valor 1 indicar ́a la categor ́ıa de
    animal predicha por la red. Por ejemplo, si la red clasifica una imagen
    como un ”gato”, el vector de salida ser ́a [0, 1 , 0 , 0 , 0 , 0 , 0 , 0], donde el 1 en
    la segunda posici ́on indica que la clase predicha es ”gato”.
4. Proponer una estructura de red adecuada y evaluar los par ́ametros uti-
    lizados para entrenar la red de manera efectiva. Guarden todas las arqui-
    tecturas que prueben as ́ı hacen un informe detallado.
5. Generar un c ́odigo donde se observe la imagen de entrada y se muestre la
    predicci ́on resultante.
6. Analizar los resultados obtenidos en el proceso de clasificaci ́on y comparar
    el rendimiento con diferentes configuraciones de la red, utilicen m ́etricas
    correspondientes para este punto.
7. Discutir las dificultades encontradas durante el desarrollo e implementaci ́on
    de la red.
Formato de Presentaci ́on El trabajo debe ser presentado en formato Pow-
erPoint o cualquier programa similar, si es c ́odigo que sea legible y debe poder
ser explicado. En la presentaci ́on deber ́an figurar los siguientes elementos:

- T ́ıtulo del trabajo
- Nombre de la materia
- Nombre de los integrantes del grupo
- Fecha de la presentaci ́on

Para cada ́ıtem del trabajo pr ́actico, deber ́an comentarse las decisiones tomadas
en el dise ̃no y la implementaci ́on del modelo, los par ́ametros utilizados, y las
dificultades encontradas durante el proceso. Finalmente, se presentar ́an los re-
sultados obtenidos y se discutir ́an las conclusiones del trabajo.

### 2

Módulo de redes neuronales convolucionales (CNN) para clasificación multiclase de animales

## Estructura del Proyecto

```
tp4/
├── __init__.py                    # Módulo principal
├── configuraciones.py             # Configuraciones de arquitecturas CNN y entrenamiento
├── constructor_modelos_cnn.py     # Constructor de arquitecturas CNN
├── procesador_imagenes.py         # Carga y procesamiento multiclase de imágenes
├── entrenador.py                 # Entrenamiento de CNN para clasificación
├── explorador.py                 # Exploración de predicciones y matriz de confusión
├── grid_search.py                # Búsqueda sistemática de hiperparámetros CNN
├── datos/
│   ├── suricatas/                # Dataset de imágenes de suricatas
│   └── [otros_animales]/         # Datasets de otros animales (8 clases total)
├── modelos/                      # Modelos CNN entrenados guardados
└── resultados/                   # Resultados y análisis de clasificación
```

## Características Principales

### 🖼️ Procesamiento de Imágenes Multiclase
- Carga automática desde múltiples carpetas de animales
- Redimensionamiento configurable (principalmente 200x200)
- Normalización a valores [0,1]
- Conversión RGB automática
- División automática en train/validation/test
- Estadísticas por clase y balanceo

### 🧠 Arquitecturas CNN
- **simple_200x200**: CNN básica con 3 capas convolucionales
- **profunda_200x200**: CNN profunda con 5 capas convolucionales
- **compacta_200x200**: CNN ligera con 2 capas convolucionales
- **ligera_200x200**: CNN optimizada para velocidad
- **ancha_200x200**: CNN con más filtros por capa
- **experimental_200x200**: CNN con kernels variables

### 🔍 Exploración de Clasificación
- Matriz de confusión interactiva
- Visualización de predicciones por clase
- Análisis de errores y confusiones
- Navegación por imágenes con predicciones
- Visualización de activaciones de capas

### 📊 Grid Search Avanzado
- Búsqueda exhaustiva de hiperparámetros
- Evaluación automática de modelos
- Análisis estadístico de resultados
- Exportación a CSV para análisis posterior

## Uso Rápido

### 1. Entrenar un Modelo CNN
```bash
# Entrenamiento básico
python3 -m tp4.entrenador --config simple_200x200 --entrenamiento normal

# Con evaluación automática
python3 -m tp4.entrenador --config profunda_200x200 --entrenamiento exhaustivo --evaluar

# Limitar número de imágenes por clase para pruebas
python3 -m tp4.entrenador --config compacta_200x200 --max-imagenes 50
```

### 2. Explorar Predicciones
```bash
# Exploración interactiva con matriz de confusión
python3 -m tp4.explorador modelo_cnn.keras

# Solo matriz de confusión
python3 -m tp4.explorador modelo_cnn.keras --matriz

# Navegación interactiva
python3 -m tp4.explorador modelo_cnn.keras --interactivo

# Listar modelos disponibles
python3 -m tp4.explorador --listar
```

### 3. Grid Search CNN
```bash
# Búsqueda completa de arquitecturas
python3 -m tp4.grid_search --completo --tamaño 200x200

# Arquitecturas específicas
python3 -m tp4.grid_search --arquitecturas simple_200x200 profunda_200x200

# Analizar resultados
python3 -m tp4.grid_search --analizar resultados_cnn.csv
```

## Configuraciones Disponibles

### Arquitecturas CNN
- `simple_200x200`: 3 capas conv [32, 64, 128] + 2 dense [256, 128] → 8 clases
- `profunda_200x200`: 5 capas conv [32, 64, 128, 256, 512] + 3 dense [512, 256, 128] → 8 clases
- `compacta_200x200`: 2 capas conv [32, 64] + 1 dense [128] → 8 clases
- `ligera_200x200`: 3 capas conv [16, 32, 64] + 1 dense [128] → 8 clases
- `ancha_200x200`: 3 capas conv [64, 128, 256] + 2 dense [512, 256] → 8 clases
- `experimental_200x200`: Kernels variables [7x7, 5x5, 3x3, 3x3] → 8 clases

### Entrenamientos
- `rapido`: 50 épocas, paciencia 20
- `normal`: 100 épocas, paciencia 30
- `exhaustivo`: 200 épocas, paciencia 50

## Dataset

El módulo utiliza un dataset de **8 tipos de animales** organizados en carpetas separadas en `tp4/datos/`. Cada carpeta contiene ~200 imágenes de un animal específico. Las imágenes son:
- Automáticamente redimensionadas a 200x200 píxeles
- Convertidas a RGB si es necesario
- Normalizadas a valores [0,1]
- Balanceadas automáticamente por clase
- Divididas en entrenamiento/validación/prueba

## Métricas de Evaluación

### Métricas de Clasificación
- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica de precision y recall
- **Matriz de Confusión**: Visualización de errores por clase

### Métricas de Entrenamiento
- **Loss de entrenamiento y validación**: categorical_crossentropy
- **Convergencia**: Detección de early stopping
- **Tiempo de entrenamiento**
- **Número de parámetros**

## Integración con TP3

El módulo TP4 reutiliza la infraestructura común de TP3:
- `EntrenadorBase`: Funcionalidad base de entrenamiento
- `GridSearchBase`: Sistema de búsqueda de hiperparámetros
- `ExploradorBase`: Interfaz de exploración interactiva
- `ConstructorModelos`: Creación de arquitecturas
- `CargadorModelos`: Persistencia de modelos

## Ejemplos de Uso Avanzado

### Entrenamiento Personalizado
```python
from tp4 import EntrenadorCNN

entrenador = EntrenadorCNN()
entrenador.cargar_datos(tamaño_imagen=(200, 200), max_imagenes_por_clase=100)
modelo, historia, nombre = entrenador.entrenar('profunda_200x200', 'exhaustivo')
```

### Exploración Programática
```python
from tp4 import ExploradorCNN

explorador = ExploradorCNN('modelo_cnn.keras')
explorador.cargar_datos(tamaño_imagen=(200, 200))
explorador.mostrar_matriz_confusion()
explorador.mostrar_predicciones_por_clase()
```

### Grid Search Personalizado
```python
from tp4 import GridSearchCNN

grid_search = GridSearchCNN()
resultados = grid_search.ejecutar_grid_search_arquitecturas(
    ['simple_200x200', 'profunda_200x200'], 
    'normal',
    tamaño_imagen=(200, 200)
)
```

## Respuestas a las Consignas del Trabajo Práctico

### 1. Implementación de Arquitectura CNN para 8 Categorías de Animales ✅

Se implementaron múltiples arquitecturas CNN optimizadas para clasificar 8 categorías de animales:

**Arquitecturas Desarrolladas:**
- `simple_200x200`: CNN básica de 3 capas convolucionales
- `profunda_200x200`: CNN profunda de 5 capas convolucionales  
- `ligera_200x200`: CNN optimizada para velocidad y eficiencia
- `optimizada_desde_cero`: **Mejor arquitectura estable (61.87% accuracy)**
- `balanceada_200x200`: Arquitectura con Max Pooling y Batch Normalization
- `moderada_profundidad_200x200`: Arquitectura de profundidad moderada con BN

**Categorías de Animales Implementadas:**
1. Carpinchos
2. Gatos  
3. Monos
4. Perros
5. Sapos
6. Suricatas
7. Tigres
8. Tortugas

### 2. División de Conjuntos de Datos ✅

**Implementación Automática de División:**
- **Entrenamiento**: 70% del dataset
- **Validación**: 20% del dataset (para early stopping y ajuste de hiperparámetros)
- **Prueba**: 10% del dataset (para evaluación final)

**Características del Dataset:**
- ~200 imágenes por categoría (~1600 imágenes totales)
- Redimensionamiento automático a 200x200 píxeles
- Normalización a valores [0,1]
- Balanceo automático por clase

### 3. Salida One-Hot de 8 Neuronas ✅

**Implementación:**
- Capa de salida: `Dense(8, activation='softmax')`
- Función de pérdida: `categorical_crossentropy`
- Codificación one-hot automática con `to_categorical()`

**Ejemplo de Salida:**
```
Predicción "Gato": [0, 1, 0, 0, 0, 0, 0, 0]
Predicción "Perro": [0, 0, 0, 1, 0, 0, 0, 0]
```

### 4. Evaluación de Parámetros y Arquitecturas ✅

**Grid Search Sistemático Implementado:**
- **Arquitecturas evaluadas**: 6+ configuraciones diferentes
- **Parámetros de entrenamiento**: 3 configuraciones (rápido, normal, exhaustivo)
- **Data Augmentation**: 5 configuraciones diferentes
- **Ejecución paralela**: Hasta 8 workers simultáneos
- **Logging incremental**: Resultados guardados en CSV en tiempo real

**Mejores Resultados Obtenidos:**
1. `optimizada_desde_cero` + `normal` + sin augmentation: **61.87% accuracy**
2. `ligera_200x200` + `rapido` + sin augmentation: **58.33% accuracy**
3. `simple_200x200` + `normal` + `animales_optimizado`: **55.00% accuracy**

### 5. Código de Predicción Visual ✅

**Explorador Interactivo Implementado:**
```bash
# Visualización de predicciones
python3 -m tp4.explorador modelo_cnn.keras --interactivo

# Matriz de confusión
python3 -m tp4.explorador modelo_cnn.keras --matriz
```

**Funcionalidades:**
- Visualización de imagen de entrada
- Predicción con probabilidades por clase
- Navegación interactiva por el dataset
- Matriz de confusión detallada
- Análisis de errores por clase

### 6. Análisis de Resultados y Métricas ✅

**Métricas Implementadas:**
- **Accuracy**: Precisión general del modelo
- **Precision, Recall, F1-Score**: Por clase individual
- **Matriz de Confusión**: Visualización de errores
- **Curvas de Loss**: Entrenamiento vs validación
- **Tiempo de entrenamiento**: Eficiencia computacional

**Análisis Comparativo Realizado:**
- Comparación de 6+ arquitecturas diferentes
- Evaluación de impacto de data augmentation
- Análisis de convergencia y overfitting
- Identificación de clases problemáticas (Monos, Carpinchos, Sapos)

### 7. Dificultades Encontradas y Soluciones ✅

**Principales Desafíos:**

1. **Dataset Limitado (~1600 imágenes)**
   - **Problema**: Overfitting con arquitecturas profundas
   - **Solución**: Arquitecturas más ligeras, dropout, early stopping

2. **Clases Ambiguas (Monos vs Carpinchos vs Sapos)**
   - **Problema**: Confusión entre especies similares
   - **Solución**: Data augmentation específico, arquitecturas especializadas

3. **Data Augmentation Contraproducente**
   - **Problema**: Augmentation agresivo reduce accuracy
   - **Solución**: Configuraciones sutiles específicas para animales

4. **Convergencia Lenta**
   - **Problema**: Learning rates altos causan inestabilidad
   - **Solución**: Learning rates bajos (0.0002) con entrenamiento exhaustivo

## 🛠️ Estrategia de Mejora para `optimizada_desde_cero`

### Análisis Actual
La arquitectura **`optimizada_desde_cero`** alcanzó **61.87% de accuracy**, siendo el mejor modelo estable sin Transfer Learning.

### 1. ⏳ Optimización del Entrenamiento (Experimento G)

**Propuesta de Mejora Inmediata:**
- **CNN**: `optimizada_desde_cero` (baseline estable confirmado)
- **Entrenamiento**: `exhaustivo` (200 épocas, paciencia 50)
- **Razón**: Learning rate bajo (0.0002) necesita más épocas para converger
- **Data Augmentation**: `animales_optimizado` o `agresivo`

### 2. 🖼️ Optimización de Representación de Datos

**A. Regularización de Entrada:**
- Añadir `GaussianNoise(stddev=0.1)` antes de la primera capa convolucional
- Fuerza a la red a aprender características esenciales vs memorización

**B. Mejora Arquitectural Mínima:**
- Aumentar última capa convolucional de 128 a 192 filtros
- Mayor capacidad de distinción sin inestabilidad

### 3. ⚠️ Límite Identificado
Si el experimento exhaustivo no supera **65% accuracy**, el límite es la cantidad de datos. La mejora significativa (>75%) requeriría **Transfer Learning**.

## Requisitos

- TensorFlow/Keras
- PIL (Pillow)
- NumPy
- Matplotlib
- Pandas
- scikit-learn
- Infraestructura TP3 (módulo `tp3.comun`)