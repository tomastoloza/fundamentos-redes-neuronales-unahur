# Verificación de Reconstrucción de Caracteres

## 📊 Script Mejorado

El script `verificar_reconstruccion_multidimensional.py` ha sido actualizado para soportar el nuevo sistema de escaneo de modelos.

## 🚀 Uso

### 1. Listar Modelos Disponibles

```bash
python -m tp3.src.verificar_reconstruccion_multidimensional --listar
```

**Salida**:
```
============================================================
MODELOS DISPONIBLES
============================================================

Total: 135 modelos encontrados

============================================================
Dimensión Latente = 2 (27 modelos)
============================================================
  BalancedAE      ep2000   lr0.0001     | tp3_BalancedAE_lat2_ep2000_lr0_0001
  BalancedAE      ep2000   lr0.0005     | tp3_BalancedAE_lat2_ep2000_lr0_0005
  ...
```

---

### 2. Verificar Modelo Específico por Dimensión

```bash
python -m tp3.src.verificar_reconstruccion_multidimensional --dim 10
```

Verifica el primer modelo encontrado para L=10.

---

### 3. Verificar por Dimensión y Arquitectura

```bash
python -m tp3.src.verificar_reconstruccion_multidimensional --dim 10 --arquitectura BalancedAE
```

Verifica el primer modelo BalancedAE con L=10.

---

### 4. Verificar Modelo por Nombre Completo

```bash
python -m tp3.src.verificar_reconstruccion_multidimensional --modelo tp3_BalancedAE_lat10_ep4000_lr0_001
```

Verifica un modelo específico por su nombre completo.

---

### 5. Comparar Todas las Dimensiones

```bash
python -m tp3.src.verificar_reconstruccion_multidimensional --comparar
```

Compara la reconstrucción de la primera arquitectura encontrada para cada dimensión latente.

**Genera**:
- Gráfico MSE vs Dimensión Latente
- Gráfico Precisión vs Dimensión Latente
- Archivo: `comparacion_reconstruccion_dimensiones.png`

---

### 6. Comparar por Arquitectura Específica

```bash
python -m tp3.src.verificar_reconstruccion_multidimensional --comparar --arquitectura BalancedAE
```

Compara solo modelos BalancedAE en todas las dimensiones.

---

### 7. Verificar TODOS los Modelos

```bash
python -m tp3.src.verificar_reconstruccion_multidimensional --todos
```

⚠️ **ADVERTENCIA**: Esto generará una imagen de verificación para cada modelo (135 imágenes si tienes todos los modelos del grid search).

**Genera**:
- Un archivo PNG por modelo: `verificacion_{nombre_modelo}.png`
- Ubicación: `/tp3/resultados/`

---

## 📋 Argumentos Disponibles

| Argumento | Tipo | Descripción |
|---|---|---|
| `--listar` | flag | Lista todos los modelos disponibles |
| `--dim` | int | Dimensión latente (2, 5, 6, 8, 10, etc.) |
| `--arquitectura` | str | Arquitectura específica (TinyAE, BalancedAE, DeepSparseAE) |
| `--modelo` | str | Nombre completo del modelo |
| `--comparar` | flag | Compara múltiples dimensiones |
| `--todos` | flag | Verifica todos los modelos |

---

## 📊 Salida del Script

### Verificación Individual

**Genera**:
1. **Imagen PNG**: Comparación original vs reconstruido (7×5 píxeles)
   - 8 columnas × N filas
   - Original arriba, reconstruido abajo
   - MSE y precisión por carácter

2. **Estadísticas en consola**:
   ```
   ============================================================
   ESTADÍSTICAS GLOBALES
   ============================================================
   Modelo: tp3_BalancedAE_lat10_ep4000_lr0_001
   Dimensión Latente: 10
   MSE promedio: 0.0565
   Precisión binaria promedio: 94.12%
   
   ============================================================
   DETALLE POR CARÁCTER
   ============================================================
     | MSE: 0.0234 | Precisión:  97.1% | Latente: [1.234, -0.567, 0.891...]
   ! | MSE: 0.0456 | Precisión:  94.3% | Latente: [0.789, 0.234, -0.456...]
   ...
   ```

---

## 🎨 Formato de Imagen Generada

```
┌─────────────────────────────────────────────────────────┐
│ Comparación Original vs Reconstruido (L=10)             │
│ Modelo: BalancedAE | Épocas: 4000 | LR: 0.001          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Original ' ']  [Original '!']  [Original '"']  ...    │
│  █████           █                █                     │
│  █   █           █                █                     │
│  █████           █                █                     │
│                                                         │
│  [Recons ' ']    [Recons '!']    [Recons '"']   ...    │
│  █████           █                █                     │
│  █   █           █                █                     │
│  █████           █                █                     │
│  MSE:0.023       MSE:0.045       MSE:0.034             │
│  Prec:97.1%      Prec:94.3%      Prec:95.7%            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 Ejemplos de Uso

### Ejemplo 1: Verificar Mejor Modelo

Según el grid search, el mejor modelo es `BalancedAE` con L=10:

```bash
python -m tp3.src.verificar_reconstruccion_multidimensional \
  --dim 10 \
  --arquitectura BalancedAE
```

---

### Ejemplo 2: Comparar Arquitecturas

Compara BalancedAE en todas las dimensiones:

```bash
python -m tp3.src.verificar_reconstruccion_multidimensional \
  --comparar \
  --arquitectura BalancedAE
```

---

### Ejemplo 3: Verificar Modelo Específico del Grid Search

```bash
python -m tp3.src.verificar_reconstruccion_multidimensional \
  --modelo tp3_BalancedAE_lat10_ep4000_lr0_001
```

---

## 📈 Interpretación de Resultados

### MSE (Mean Squared Error)
- **< 0.05**: Excelente reconstrucción
- **0.05 - 0.10**: Buena reconstrucción
- **0.10 - 0.15**: Reconstrucción aceptable
- **> 0.15**: Reconstrucción pobre

### Precisión Binaria
- **> 95%**: Excelente
- **90% - 95%**: Muy buena
- **85% - 90%**: Buena
- **< 85%**: Necesita mejora

---

## 🎯 Mejoras Implementadas

### Antes
```bash
python -m tp3.src.verificar_reconstruccion_multidimensional --dim 2
```
- Solo soportaba dimensiones predefinidas (2, 3, 4, 5, 8)
- Nombre de modelo manual
- Escaneo simple de archivos

### Ahora
```bash
python -m tp3.src.verificar_reconstruccion_multidimensional --dim 10 --arquitectura BalancedAE
```
- ✅ Soporta **cualquier dimensión** latente
- ✅ **Escaneo inteligente** con glob
- ✅ Detecta **arquitectura, épocas, learning rate**
- ✅ Filtrado por arquitectura
- ✅ Verificación de **todos los modelos** con `--todos`
- ✅ Nombres de archivo descriptivos
- ✅ Información completa en títulos

---

## 📁 Archivos Generados

### Verificación Individual
```
/tp3/resultados/verificacion_tp3_BalancedAE_lat10_ep4000_lr0_001.png
```

### Comparación de Dimensiones
```
/tp3/resultados/comparacion_reconstruccion_dimensiones.png
```

### Verificación de Todos (--todos)
```
/tp3/resultados/verificacion_tp3_BalancedAE_lat2_ep2000_lr0_0001.png
/tp3/resultados/verificacion_tp3_BalancedAE_lat2_ep2000_lr0_0005.png
...
(135 archivos si tienes todos los modelos)
```

---

## 💡 Recomendaciones

1. **Usa `--listar`** primero para ver qué modelos tienes
2. **Verifica el mejor modelo** del grid search:
   ```bash
   python -m tp3.src.verificar_reconstruccion_multidimensional \
     --modelo tp3_BalancedAE_lat10_ep4000_lr0_001
   ```
3. **Compara arquitecturas** para ver diferencias:
   ```bash
   python -m tp3.src.verificar_reconstruccion_multidimensional --comparar
   ```
4. **Evita `--todos`** a menos que necesites verificar cada modelo (genera muchos archivos)

---

## 🔗 Scripts Relacionados

- `entrenador_grid_search_completo.py`: Entrena modelos con grid search
- `explorador_interactivo_avanzado.py`: Explora espacio latente interactivamente
- `graficar_resultados.py`: Genera gráficos adicionales del CSV
