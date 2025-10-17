# TP4 - Autocodificadores para Imágenes

Módulo de autocodificadores aplicados a imágenes de suricatas, siguiendo la misma arquitectura limpia y principios SOLID del proyecto TP3.

## Estructura del Proyecto

```
tp4/
├── __init__.py                 # Módulo principal
├── configuraciones.py          # Configuraciones de arquitecturas y entrenamiento
├── procesador_imagenes.py      # Carga y procesamiento de imágenes
├── entrenador.py              # Entrenamiento de autocodificadores
├── explorador.py              # Exploración interactiva del espacio latente
├── grid_search.py             # Búsqueda sistemática de hiperparámetros
├── datos/
│   └── suricatas/             # Dataset de imágenes de suricatas
├── modelos/                   # Modelos entrenados guardados
└── resultados/                # Resultados y análisis
```

## Características Principales

### 🖼️ Procesamiento de Imágenes
- Carga automática de imágenes desde directorio
- Redimensionamiento configurable (32x32, 64x64, 128x128)
- Normalización a valores [0,1]
- Conversión RGB automática
- Estadísticas del dataset

### 🧠 Arquitecturas de Autocodificadores
- **simple_64x64**: Arquitectura básica para imágenes 64x64
- **profundo_64x64**: Red más profunda con mejor capacidad
- **compacto_32x32**: Optimizado para imágenes pequeñas
- **ultra_profundo_128x128**: Para imágenes de alta resolución
- **ancho_64x64**: Capas anchas para mayor capacidad
- **tanh_64x64**: Usando activación tanh

### 🔍 Exploración Interactiva
- Visualización del espacio latente en 2D
- Navegación por imágenes con teclado
- Comparación original vs reconstruida
- Métricas de calidad en tiempo real
- Interpolación en espacio latente

### 📊 Grid Search Avanzado
- Búsqueda exhaustiva de hiperparámetros
- Evaluación automática de modelos
- Análisis estadístico de resultados
- Exportación a CSV para análisis posterior

## Uso Rápido

### 1. Entrenar un Modelo
```bash
# Entrenamiento básico
python3 -m tp4.entrenador --config simple_64x64 --entrenamiento normal

# Con evaluación automática
python3 -m tp4.entrenador --config profundo_64x64 --entrenamiento exhaustivo --evaluar

# Limitar número de imágenes para pruebas
python3 -m tp4.entrenador --config compacto_32x32 --max-imagenes 100
```

### 2. Explorar Espacio Latente
```bash
# Exploración interactiva
python3 -m tp4.explorador modelo_entrenado.keras

# Listar modelos disponibles
python3 -m tp4.explorador --listar

# Con tamaño específico
python3 -m tp4.explorador modelo.keras --tamaño 128x128
```

### 3. Grid Search
```bash
# Búsqueda completa
python3 -m tp4.grid_search --completo --tamaño 64x64

# Arquitecturas específicas
python3 -m tp4.grid_search --arquitecturas simple_64x64 profundo_64x64

# Analizar resultados
python3 -m tp4.grid_search --analizar resultados.csv
```

## Configuraciones Disponibles

### Arquitecturas
- `simple_64x64`: [128, 64, 32] → 32D → [32, 64, 128]
- `profundo_64x64`: [256, 128, 64, 32] → 16D → [32, 64, 128, 256]
- `compacto_32x32`: [64, 32] → 8D → [32, 64]
- `ultra_profundo_128x128`: [512, 256, 128, 64, 32, 16] → 8D
- `ancho_64x64`: [512, 256] → 64D → [256, 512]
- `tanh_64x64`: Igual que simple pero con activación tanh

### Entrenamientos
- `rapido`: 50 épocas, paciencia 20
- `normal`: 100 épocas, paciencia 30
- `exhaustivo`: 200 épocas, paciencia 50

## Dataset

El módulo utiliza un dataset de **imágenes de suricatas** ubicado en `tp4/datos/suricatas/`. Las imágenes son:
- Automáticamente redimensionadas al tamaño configurado
- Convertidas a RGB si es necesario
- Normalizadas a valores [0,1]
- Cargadas eficientemente con PIL

## Métricas de Evaluación

### Métricas de Reconstrucción
- **MSE (Mean Squared Error)**: Error cuadrático medio
- **MAE (Mean Absolute Error)**: Error absoluto medio
- **Calidad visual**: Comparación lado a lado

### Métricas de Entrenamiento
- **Loss de entrenamiento y validación**
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
from tp4 import EntrenadorAutocodificadorImagenes

entrenador = EntrenadorAutocodificadorImagenes()
entrenador.cargar_datos(tamaño_imagen=(128, 128), max_imagenes=500)
modelo, historia, nombre = entrenador.entrenar('ultra_profundo_128x128', 'exhaustivo')
```

### Exploración Programática
```python
from tp4 import ExploradorEspacioLatenteImagenes

explorador = ExploradorEspacioLatenteImagenes('modelo.keras')
explorador.cargar_datos(tamaño_imagen=(64, 64))
explorador.interpolar_en_espacio_latente(0, 10, num_pasos=20)
```

### Grid Search Personalizado
```python
from tp4 import GridSearchAutocodificadorImagenes

grid_search = GridSearchAutocodificadorImagenes()
resultados = grid_search.ejecutar_grid_search_arquitecturas(
    ['simple_64x64', 'profundo_64x64'], 
    'normal',
    tamaño_imagen=(64, 64)
)
```

## Requisitos

- TensorFlow/Keras
- PIL (Pillow)
- NumPy
- Matplotlib
- Pandas
- Infraestructura TP3 (módulo `tp3.comun`)

## Notas de Implementación

- **Sin comentarios**: Código auto-documentado siguiendo Clean Code
- **Principios SOLID**: Separación clara de responsabilidades
- **Reutilización**: Aprovecha la infraestructura existente de TP3
- **Extensibilidad**: Fácil agregar nuevas arquitecturas y métricas
- **Robustez**: Manejo de errores y validación de datos
