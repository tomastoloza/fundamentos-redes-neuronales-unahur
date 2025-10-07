# Exploradores Interactivos del Espacio Latente

## 📊 Tres Versiones Disponibles

### 1. **Explorador Simple** (`explorador_interactivo_espacio_latente.py`)
**Para dimensión latente = 2 únicamente**

```bash
python -m tp3.src.explorador_interactivo_espacio_latente
```

**Características**:
- ✅ Visualización completa del espacio 2D
- ✅ Generación directa en cualquier punto
- ✅ Interfaz simple y precisa
- ❌ Solo funciona con L=2

---

### 2. **Explorador Multidimensional** (`explorador_interactivo_multidimensional.py`)
**Para cualquier dimensión latente (2, 5, 6, 8, 10, etc.)**

```bash
python -m tp3.src.explorador_interactivo_multidimensional --dim 10 --modelo nombre_modelo
```

**Características**:
- ✅ Soporta L > 2
- ✅ Proyección 2D de espacios de alta dimensión
- ✅ Detección de proximidad a caracteres entrenados
- ✅ Completa dimensiones 3+ con ceros
- ⚠️ Requiere especificar modelo manualmente

**Argumentos**:
- `--dim`: Dimensión latente (2, 3, 4, 5, 8, 10)
- `--modelo`: Nombre del modelo (opcional)
- `--listar`: Lista modelos disponibles

---

### 3. **Explorador Avanzado** (`explorador_interactivo_avanzado.py`) ⭐ RECOMENDADO
**Selector dinámico de modelos con todas las dimensiones**

```bash
python -m tp3.src.explorador_interactivo_avanzado
```

**Características**:
- ✅ **Selector de modelos en tiempo real** con RadioButtons
- ✅ **Escaneo automático** de todos los modelos en `/modelos/`
- ✅ **Soporta TODAS las dimensiones** latentes
- ✅ **Cambio de modelo sin reiniciar** la aplicación
- ✅ **Completa dimensiones faltantes** automáticamente
- ✅ **Distingue caracteres** entrenados vs generados
- ✅ **Información completa** del modelo (arquitectura, épocas, LR)

**Ventajas**:
- No necesitas saber qué modelos tienes
- Compara modelos diferentes en tiempo real
- Interfaz gráfica intuitiva con panel de selección
- Muestra hasta 20 modelos simultáneamente

---

## 🎯 ¿Cuál usar?

### Usa **Explorador Simple** si:
- ✅ Trabajas con **L=2** exclusivamente
- ✅ Quieres **máxima precisión** en la generación
- ✅ Prefieres **simplicidad**

### Usa **Explorador Multidimensional** si:
- ✅ Sabes exactamente qué modelo quieres explorar
- ✅ Necesitas control por línea de comandos
- ✅ Trabajas con **L > 2**

### Usa **Explorador Avanzado** si: ⭐
- ✅ Quieres **comparar múltiples modelos**
- ✅ No sabes qué modelos tienes disponibles
- ✅ Quieres **cambiar de modelo sin reiniciar**
- ✅ Prefieres **interfaz gráfica completa**
- ✅ Trabajas con **cualquier dimensión**

---

## 🚀 Guía de Uso del Explorador Avanzado

### Paso 1: Entrenar Modelos

Primero necesitas modelos entrenados:

```bash
python -m tp3.src.entrenador_grid_search_completo
```

Esto generará modelos en `/modelos/` con nombres como:
```
tp3_BalancedAE_lat10_ep4000_lr0_001_autocodificador.keras
tp3_TinyAE_lat5_ep2000_lr0_0001_autocodificador.keras
```

### Paso 2: Ejecutar Explorador

```bash
python -m tp3.src.explorador_interactivo_avanzado
```

### Paso 3: Interactuar

**Panel Derecho**: Selector de modelos
- Radio buttons con todos los modelos disponibles
- Formato: `L={dim} {arquitectura} ep{épocas} lr{learning_rate}`
- Ejemplo: `L=10 BalancedAE ep4000 lr0.001`

**Panel Izquierdo**: Mapa del espacio latente
- Puntos azules con letras: caracteres entrenados
- Click **CERCA** de un carácter (< 0.15): usa vector latente completo
- Click **LEJOS**: genera nuevo con dims 3+ = 0

**Panel Central**: Carácter generado
- Muestra el carácter en 7×5 píxeles
- Información del vector latente
- Estadísticas (píxeles activos, promedio, clicks)

---

## 🎨 Funcionamiento del Completado de Dimensiones

### Para L=2
```python
Click en (1.2, 0.5) → Vector: [1.2, 0.5]
```

### Para L=5
```python
Click CERCA de 'A' → Vector: [1.2, 0.5, 0.8, -0.3, 1.1] ✅ (completo)
Click LEJOS       → Vector: [1.2, 0.5, 0.0, 0.0, 0.0]   (dims 3-5 = 0)
```

### Para L=10
```python
Click CERCA de 'B' → Vector: [0.9, -0.2, 1.1, ..., 0.4] ✅ (10 dims completas)
Click LEJOS       → Vector: [0.9, -0.2, 0, 0, 0, 0, 0, 0, 0, 0]
```

---

## 📋 Colores y Símbolos

| Color | Significado |
|---|---|
| 🔵 Azul | Caracteres entrenados en el mapa |
| 🔴 Rojo | Último click |
| 🟢 Verde | Clicks en caracteres entrenados (historial) |
| 🟠 Naranja | Clicks en puntos nuevos (historial) |

| Símbolo | Significado |
|---|---|
| █ | Píxel activo (> 0.5) |
| · | Píxel inactivo (≤ 0.5) |
| ✅ | Carácter de entrenamiento |

---

## 🔍 Ejemplos de Uso

### Ejemplo 1: Comparar Arquitecturas

```bash
python -m tp3.src.explorador_interactivo_avanzado
```

1. Selecciona `L=10 BalancedAE ep4000 lr0.001`
2. Observa la distribución de caracteres
3. Selecciona `L=10 TinyAE ep2000 lr0.0001`
4. Compara las diferencias en el espacio latente

### Ejemplo 2: Explorar Dimensiones

```bash
python -m tp3.src.explorador_interactivo_avanzado
```

1. Selecciona `L=2 BalancedAE ...`
2. Observa el espacio completo
3. Selecciona `L=10 BalancedAE ...`
4. Observa la proyección 2D del espacio 10D

### Ejemplo 3: Generar Caracteres Nuevos

```bash
python -m tp3.src.explorador_interactivo_avanzado
```

1. Selecciona cualquier modelo
2. Haz click **entre** dos caracteres
3. Observa la interpolación
4. Haz click **lejos** de todos
5. Observa el carácter generado con dims 3+ = 0

---

## ⚙️ Requisitos

- Python 3.8+
- TensorFlow/Keras
- Matplotlib
- NumPy
- Modelos entrenados en `/modelos/`

---

## 🐛 Troubleshooting

### "No se encontraron modelos"
```bash
python -m tp3.src.entrenador_grid_search_completo
```

### "Error al cargar modelo"
Verifica que existan los 3 archivos:
- `*_autocodificador.keras`
- `*_codificador.keras`
- `*_decodificador.keras`

### "Ventana no responde"
- Asegúrate de tener backend gráfico de matplotlib
- En macOS: `export MPLBACKEND=TkAgg`

---

## 📊 Comparación de Exploradores

| Característica | Simple | Multidimensional | Avanzado ⭐ |
|---|:---:|:---:|:---:|
| Dimensiones soportadas | Solo 2 | 2-10 | 2-∞ |
| Selector de modelos | ❌ | ❌ | ✅ |
| Cambio en tiempo real | ❌ | ❌ | ✅ |
| Escaneo automático | ❌ | ❌ | ✅ |
| Completado de dims | N/A | ✅ | ✅ |
| Interfaz gráfica | Básica | Media | Completa |
| Complejidad | Baja | Media | Alta |
| Recomendado para | L=2 | CLI | Exploración |

---

## 🎓 Conceptos Clave

### Proyección 2D
Cuando L > 2, solo se muestran las primeras 2 dimensiones del espacio latente. Las dimensiones 3+ existen pero no se visualizan.

### Completado con Ceros
Al hacer click en un punto nuevo (lejos de caracteres entrenados), las dimensiones 3+ se rellenan con 0. Esto permite explorar el espacio manteniendo las primeras 2 dimensiones controladas.

### Detección de Proximidad
Si haces click a menos de 0.15 unidades de un carácter entrenado, se usa su vector latente completo en lugar de generar uno nuevo.

---

## 📝 Notas

- El explorador avanzado muestra hasta 20 modelos en el selector
- Los modelos se ordenan por: arquitectura → épocas → learning rate
- El historial de clicks se mantiene al cambiar de modelo
- Click derecho limpia el historial en cualquier explorador
