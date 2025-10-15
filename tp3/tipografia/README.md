# Autocodificadores para Tipografía Unicode

Sistema completo de autocodificadores para aprender representaciones latentes de símbolos tipográficos Unicode (32x32 píxeles).

## Características

- **Datos**: 30 símbolos Unicode predefinidos renderizados como matrices binarias 32x32 (1024 píxeles)
- **Arquitecturas**: 9 configuraciones desde minimalistas hasta ultra profundas
- **Dimensiones latentes**: 2D, 5D, 8D, 10D
- **Explorador interactivo**: Visualización y generación en espacio latente 2D
- **Grid search**: Búsqueda sistemática de mejores configuraciones

## Estructura de Archivos

```
tp3/tipografia/
├── generador_bitmap_tipografia.py    # Generación de bitmaps desde Unicode
├── procesador_datos_tipografia.py    # Procesamiento de datos tipográficos
├── configuraciones.py                # Configuraciones de modelos y entrenamiento
├── entrenador.py                     # Entrenador de autocodificadores
├── explorador.py                     # Explorador interactivo del espacio latente
├── grid_search.py                    # Búsqueda sistemática de configuraciones
└── README.md                         # Esta documentación
```

## Uso

### 1. Entrenamiento Individual

```bash
# Entrenamiento rápido con arquitectura simple 2D
python3 -m tp3.tipografia.entrenador simple_2d rapido

# Entrenamiento exhaustivo con arquitectura profunda
python3 -m tp3.tipografia.entrenador profundo_2d exhaustivo

# Sin visualización de gráficos
python3 -m tp3.tipografia.entrenador minimo_2d normal --sin-graficos

# Listar configuraciones disponibles
python3 -m tp3.tipografia.entrenador --listar
```

### 2. Explorador Interactivo

```bash
# Explorar modelo entrenado (solo modelos 2D)
python3 -m tp3.tipografia.explorador tp3_tipografia_lat2_ep500_lr0_001

# Con tamaño de imagen personalizado
python3 -m tp3.tipografia.explorador tp3_tipografia_lat2_ep500_lr0_001 --tamaño 32
```

**Controles del explorador:**
- **Click** en el espacio latente para generar símbolos
- **Sliders** para ajustar coordenadas latentes manualmente
- Visualización de símbolo original más cercano
- Visualización de símbolo generado

### 3. Grid Search

```bash
# Ejecutar búsqueda completa de configuraciones
python3 -m tp3.tipografia.grid_search
```

Esto ejecutará todas las combinaciones de:
- 9 arquitecturas de autocodificador
- 3 configuraciones de entrenamiento
- Total: 27 experimentos

## Configuraciones Disponibles

### Arquitecturas de Autocodificador

| Nombre | Dimensión Latente | Encoder | Decoder | Activación |
|--------|------------------|---------|---------|------------|
| `simple_2d` | 2 | [512, 256, 128] | [128, 256, 512] | ReLU |
| `profundo_2d` | 2 | [768, 512, 384, 256, 128] | [128, 256, 384, 512, 768] | ReLU |
| `minimo_2d` | 2 | [256, 128] | [128, 256] | ReLU |
| `ultra_profundo_2d` | 2 | [896, 768, 640, 512, 384, 256, 128, 64] | [64, 128, 256, 384, 512, 640, 768, 896] | ReLU |
| `ultra_ancho_2d` | 2 | [1536, 768] | [768, 1536] | ReLU |
| `tanh_2d` | 2 | [640, 384, 128] | [128, 384, 640] | Tanh |
| `compacto_5d` | 5 | [512, 256, 128] | [128, 256, 512] | ReLU |
| `profundo_8d` | 8 | [768, 512, 384, 256] | [256, 384, 512, 768] | ReLU |
| `ancho_10d` | 10 | [1024, 512] | [512, 1024] | ReLU |

### Configuraciones de Entrenamiento

| Nombre | Epochs | Patience | Validation Split |
|--------|--------|----------|------------------|
| `rapido` | 500 | 150 | 0.15 |
| `normal` | 1500 | 200 | 0.15 |
| `exhaustivo` | 3000 | 300 | 0.15 |

## Diferencias con tp3/simbolos

### Datos de Entrada
- **simbolos**: 35 píxeles (7x5) - caracteres hexadecimales
- **tipografia**: 1024 píxeles (32x32) - símbolos Unicode renderizados

### Arquitecturas
- **simbolos**: Capas más pequeñas (20, 10, 2) para 35 píxeles
- **tipografia**: Capas más grandes (512, 256, 128, 2) para 1024 píxeles

### Complejidad
- **simbolos**: Patrones simples, rápida convergencia
- **tipografia**: Patrones complejos, requiere más capacidad de red

## Símbolos Predefinidos

30 símbolos Unicode de diferentes categorías:
```
■ ● ▲ ▼ ◆ ★ ☀ ☁ ☂ ☃ ☎ ☕ ✈ ✉ ✏ ✒ ✓ ✗ ✚ ✝ ✠ ✦ ✪ ✰ ✶ ✿ ❤ ❥ ❦ ❧
```

## Resultados Esperados

### Métricas Típicas
- **Precisión**: 0.85 - 0.95 (85-95% de píxeles correctos)
- **MSE**: 0.01 - 0.05 (error cuadrático medio)
- **Convergencia**: 200-1000 epochs dependiendo de la arquitectura

### Modelos Guardados
Los modelos se guardan en `tp3/modelos/` con el formato:
```
tp3_tipografia_lat{dimension}_ep{epochs}_lr{learning_rate}.keras
```

Ejemplo: `tp3_tipografia_lat2_ep500_lr0_001.keras`

## Uso Programático

```python
from tp3.tipografia import (
    EntrenadorTipografia,
    ExploradorEspacioLatenteTipografia,
    GridSearchTipografia,
    obtener_configuracion,
    obtener_configuracion_entrenamiento
)

# Entrenamiento
entrenador = EntrenadorTipografia(tamaño_imagen=32)
config_modelo = obtener_configuracion('simple_2d')
config_entrenamiento = obtener_configuracion_entrenamiento('rapido')
modelo, datos, historial, metricas = entrenador.entrenar_modelo(
    config_modelo, 
    config_entrenamiento
)

# Exploración
explorador = ExploradorEspacioLatenteTipografia('tp3_tipografia_lat2_ep500_lr0_001')
explorador.explorar_interactivo()

# Grid search
grid_search = GridSearchTipografia(tamaño_imagen=32)
resultados, mejores = grid_search.ejecutar_grid_search_completo()
```

## Requisitos

- TensorFlow/Keras
- NumPy
- Matplotlib
- Python 3.8+

## Notas Técnicas

1. **Tamaño de imagen**: Configurable, default 32x32 píxeles
2. **Early stopping**: Activado por defecto con paciencia configurable
3. **Batch size**: Ajustado a 8-16 para dataset pequeño (30 muestras)
4. **Learning rate**: 0.001 para todas las configuraciones
5. **Activación salida**: Sigmoid para valores binarios [0, 1]
