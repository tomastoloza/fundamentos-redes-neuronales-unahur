# Instrucciones de Entrenamiento - TP3 Autocodificadores

## Resumen de Cambios

### ✅ Código Duplicado Eliminado
- Creado módulo `utilidades_entrenamiento.py` con funciones comunes
- Refactorizados 3 archivos para usar utilidades compartidas
- Reducción de ~100 líneas de código duplicado

### ✅ Arquitecturas Expandidas
- **40 arquitecturas diferentes** organizadas en 6 categorías
- Patrones: Compactas, Amplias, Piramidales, Profundas, Residual-like, Híbridas, Experimentales

### ✅ Configuraciones de Entrenamiento
- **Épocas**: [3000, 5000, 8000]
- **Learning rates**: [0.0001, 0.0005, 0.001, 0.002]
- **Dimensiones latentes**: [2, 3, 4, 5, 6, 8, 10]
- **Total combinaciones posibles**: 3,360 experimentos

---

## Opciones de Entrenamiento

### 1. Entrenamiento Completo (TODAS las configuraciones)

**Script**: `entrenar_todas_configuraciones.py`

```bash
cd /Users/ttoloza/git/personal/unahur/fundamentos-redes-neuronales
python -m tp3.src.entrenar_todas_configuraciones
```

**Características**:
- Entrena TODAS las combinaciones (40 arquitecturas × 7 dimensiones × 3 épocas × 4 LRs)
- **Total**: 3,360 experimentos
- **Tiempo estimado**: VARIOS DÍAS (depende del hardware)
- Genera reporte completo con análisis estadístico
- Crea visualizaciones comparativas
- Guarda todos los modelos entrenados

**⚠️ ADVERTENCIA**: Esto tomará MUCHO tiempo. El script pedirá confirmación antes de comenzar.

---

### 2. Entrenador Grid Search Completo (PARALELO)

**Script**: `entrenador_grid_search_completo.py`

```bash
python -m tp3.src.entrenador_grid_search_completo
```

**Características**:
- ✅ **EJECUCIÓN PARALELA** usando múltiples CPUs
- ✅ **GRID SEARCH COMPLETO**: Prueba TODAS las combinaciones
- Entrena todas las arquitecturas con todas las dimensiones latentes, épocas y learning rates
- **Total**: 3 arquitecturas × 5 dimensiones × 3 épocas × 3 LRs = **135 experimentos**
- **Tiempo estimado**: Depende del hardware (con 8 cores ~4-6 horas)
- Genera reporte completo con mejores modelos por dimensión
- Identifica configuración óptima de hiperparámetros

**Uso recomendado**: Para encontrar la mejor combinación de hiperparámetros por arquitectura

---

### 3. Búsqueda de Arquitecturas Óptimas (PARALELO)

**Script**: `buscar_arquitecturas_optimas.py`

```bash
python -m tp3.src.buscar_arquitecturas_optimas
```

**Características**:
- ✅ **EJECUCIÓN PARALELA** usando múltiples CPUs
- Entrena arquitecturas experimentales específicas
- Configuración optimizada para alta precisión
- **Total**: ~14 experimentos (configurable en el script)
- **Tiempo estimado**: Mucho más rápido con paralelización
- Busca modelos con precisión > 90%
- Genera reporte de mejores modelos

**Uso recomendado**: Para encontrar la mejor arquitectura rápidamente

---

### 4. Comparación de Arquitecturas (PARALELO)

**Script**: `comparador_arquitecturas_autocodificador.py`

```bash
python -m tp3.src.comparador_arquitecturas_autocodificador
```

**Características**:
- ✅ **EJECUCIÓN PARALELA** usando múltiples CPUs
- Entrena todas las arquitecturas con configuración actual
- **Total**: Depende de configuraciones en `configuraciones_entrenamiento.py`
- **Tiempo estimado**: Más rápido que opciones secuenciales
- Usa ProcessPoolExecutor para paralelización
- Genera reporte markdown con resultados

**Uso recomendado**: Para comparar arquitecturas rápidamente usando todos los cores

---

## Personalización de Configuraciones

### Modificar Arquitecturas a Entrenar

Edita `tp3/src/configuraciones_arquitecturas.py`:

```python
def obtener_arquitecturas_disponibles():
    return {
        'mi_arquitectura': {
            'codificador': [30, 15, 8],
            'activacion': 'tanh',
            'batch_norm': True,
            'descripcion': 'Mi arquitectura personalizada'
        }
    }
```

### Modificar Parámetros de Entrenamiento

Edita `tp3/src/configuraciones_entrenamiento.py`:

```python
def obtener_configuracion_entrenamiento():
    return {
        'epocas_lista': [5000],              # Épocas a probar
        'tasas_aprendizaje': [0.001],        # Learning rates
        'dimensiones_latentes': [2, 5],      # Dimensiones latentes
        'error_objetivo': 0.12,              # Umbral de convergencia
        'paciencia': 300,                    # Early stopping patience
        'tipo_perdida': 'bce',               # 'bce' o 'mse'
        'usar_scheduler': True,              # ReduceLROnPlateau
        'batch_size': 32
    }
```

---

## Entrenar Arquitecturas Específicas

### Opción A: Modificar el script principal

Edita `entrenar_todas_configuraciones.py` o cualquier otro script:

```python
arquitecturas_a_entrenar = [
    'compacta_doble',
    'piramidal_media',
    'amplia_moderada'
]

for arquitectura in arquitecturas_a_entrenar:
    # ... código de entrenamiento
```

### Opción B: Usar la consola interactiva

```bash
python -m tp3.src.consola_interactiva

Permite seleccionar arquitectura, dimensión latente y parámetros interactivamente.

---

### Estructura de Salida

### Modelos Guardados

Ubicación: `/modelos/`

{{ ... }}
### Reportes Generados

Ubicación: `/tp3/resultados/`

Archivos:
- `comparacion_dimensiones_latentes.csv` - **Datos en CSV para análisis**
- `comparacion_dimensiones_latentes.md` - Reporte markdown completo
- `comparacion_dimensiones_latentes.png` - Dashboard con 9 gráficos
- `heatmap_mse.png` - Heatmaps de MSE por arquitectura
- `heatmap_precision.png` - Heatmaps de precisión por arquitectura
- `analisis_detallado.png` - Análisis comparativo detallado
- `entrenamiento_completo_{timestamp}.md` - Reporte completo
- `entrenamiento_completo_{timestamp}.png` - Visualizaciones
- `arquitecturas_experimentales.md` - Búsqueda de óptimos
- `arquitecturas_experimentales.png` - Resultados experimentales

---

### Para Exploración Rápida (⚡ PARALELO)
```bash
python -m tp3.src.buscar_arquitecturas_optimas
```
- ~14 experimentos en paralelo
- Usa todos los cores disponibles
- Tiempo reducido significativamente

### Para Grid Search Completo (⚡ PARALELO)
```bash
python -m tp3.src.entrenador_grid_search_completo
```
- 280 experimentos en paralelo
- Usa todos los cores disponibles
- Tiempo reducido significativamente

### Para Comparación de Arquitecturas (⚡ PARALELO)
```bash
python -m tp3.src.comparador_arquitecturas_autocodificador
```
- Configuración personalizable
- Usa todos los cores disponibles
- Máxima velocidad

### Para Búsqueda Exhaustiva (DÍAS)
```bash
python -m tp3.src.entrenar_todas_configuraciones
```
- 3,360 experimentos secuenciales
- Análisis completo y exhaustivo

---

## Monitoreo del Entrenamiento

Cada script muestra:
- Progreso del experimento actual
- Métricas en tiempo real (MSE, precisión, convergencia)
- Tiempo estimado restante
- Resumen al finalizar

Ejemplo de salida (modo paralelo):
```
============================================================
COMPARACIÓN DE DIMENSIONES LATENTES (PARALELO)
============================================================

Dimensiones latentes a probar: [2, 3, 4, 5, 6, 8, 10]
Arquitecturas: ['compacta_simple', 'compacta_doble', ...]
Total experimentos: 280
Workers paralelos: 8

🚀 Iniciando entrenamiento paralelo...

✅ [15/280] L=5, piramidal_media
   MSE: 0.0856, Precisión: 88.3%, Convergió: Sí

✅ [16/280] L=3, amplia_moderada
   MSE: 0.1124, Precisión: 84.1%, Convergió: Sí

✅ [17/280] L=8, profunda_uniforme
   MSE: 0.0789, Precisión: 90.2%, Convergió: Sí
...
```

**Nota**: Los experimentos se completan en orden variable debido a la ejecución paralela.

---

## Troubleshooting

### Error: "Out of Memory"
- Reduce `batch_size` en configuraciones
- Entrena menos arquitecturas simultáneamente
- Usa dimensiones latentes más pequeñas

### Error: "No module named 'tp3'"
```bash
cd /Users/ttoloza/git/personal/unahur/fundamentos-redes-neuronales
python -m tp3.src.{script_name}
```

### Entrenamiento muy lento
- Usa `comparador_arquitecturas_autocodificador.py` (paralelo)
- Reduce número de épocas en configuraciones
- Entrena solo arquitecturas específicas

### Modelos no convergen
- Aumenta `paciencia` en configuraciones
- Reduce `error_objetivo`
- Prueba diferentes `learning_rates`

---

### Análisis de Resultados

Después del entrenamiento, revisa:

1. **CSV**: `comparacion_dimensiones_latentes.csv` - Importa en Excel/Python para análisis personalizado
2. **Reporte markdown**: Tabla completa de resultados con mejores modelos
3. **Gráficos PNG**: 4 archivos con visualizaciones completas
4. **Modelos guardados**: En `/modelos/` para uso posterior

Los mejores modelos se identifican automáticamente en los reportes.

### Generar Gráficos Adicionales

Si ya ejecutaste el entrenamiento y tienes el CSV, puedes generar gráficos adicionales:

```bash
python -m tp3.src.graficar_resultados
```

**Genera**:
- Heatmaps de MSE por arquitectura
- Heatmaps de precisión por arquitectura
- Análisis comparativo detallado
- Top 15 mejores modelos
