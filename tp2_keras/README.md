# TP2 Keras - Implementación con TensorFlow/Keras

## Descripción

Este directorio contiene implementaciones de los objetivos del TP2 utilizando **TensorFlow/Keras**, diseñadas para comparar directamente con las implementaciones personalizadas del perceptrón multicapa. Incluye análisis comparativo detallado entre ambos enfoques.

## Estructura del Proyecto

```
tp2_keras/
├── xor_keras.py                      # Problema XOR con Keras
├── discriminacion_pares_keras.py     # Discriminación números pares con Keras  
├── clasificacion_10_clases_keras.py  # Clasificación 10 clases con Keras
├── comparador_implementaciones.py    # Comparador entre implementaciones
├── requirements.txt                  # Dependencias de TensorFlow
├── README.md                        # Esta documentación
└── __init__.py                      # Configuración del módulo
```

## Experimentos Implementados

### 🔥 Ejercicio 1: Función Lógica XOR
- **Archivo**: `xor_keras.py`
- **Arquitectura**: [2, 4, 1] (2 entradas, 4 neuronas ocultas, 1 salida)
- **Activación**: Sigmoid en todas las capas
- **Optimizador**: SGD con learning rate 0.1
- **Objetivo**: Demostrar resolución de problemas no linealmente separables

### 🧠 Ejercicio 2: Discriminación de Números Pares
- **Archivo**: `discriminacion_pares_keras.py`
- **Arquitecturas múltiples**: MINIMA, COMPACTA, DIRECTA_ORIGINAL, BALANCEADA
- **Entrada**: 35 píxeles (5x7) de dígitos decimales
- **Salida**: Clasificación binaria (par=1, impar=0)
- **Datos**: Entrenamiento (0,2,4,6,1,3), Prueba (8,5,7,9)

### 📊 Ejercicio 3: Clasificación 10 Clases
- **Archivo**: `clasificacion_10_clases_keras.py`
- **Arquitectura**: [35, 20, 15, 10] con activación softmax en salida
- **Codificación**: One-hot encoding para 10 clases
- **División**: Entrenamiento (0-6), Prueba (7-9)
- **Evaluación con ruido**: Probabilidad 0.02 de intercambio de bits

## Instalación y Configuración

### Requisitos
```bash
pip install -r requirements.txt
```

### Dependencias principales:
- `tensorflow>=2.10.0`
- `numpy>=1.21.0`
- `matplotlib>=3.5.0` (para visualizaciones futuras)
- `seaborn>=0.11.0` (para análisis estadístico)
- `scikit-learn>=1.0.0` (para métricas adicionales)

## Uso

### Ejecutar experimentos individuales:

```python
# XOR con Keras
from tp2_keras.xor_keras import ejecutar_experimento_xor_keras
resultados_xor = ejecutar_experimento_xor_keras()

# Discriminación de pares con Keras
from tp2_keras.discriminacion_pares_keras import ejecutar_experimento_discriminacion_keras
resultados_pares = ejecutar_experimento_discriminacion_keras()

# Clasificación 10 clases con Keras
from tp2_keras.clasificacion_10_clases_keras import ejecutar_experimento_clasificacion_10_clases_keras
resultados_10_clases = ejecutar_experimento_clasificacion_10_clases_keras()
```

### Ejecutar comparación completa:

```python
from tp2_keras.comparador_implementaciones import ComparadorImplementaciones

comparador = ComparadorImplementaciones()
comparador.ejecutar_comparacion_completa()
```

### Desde línea de comandos:

```bash
# Experimento XOR
python tp2_keras/xor_keras.py

# Discriminación de pares (múltiples arquitecturas)
python tp2_keras/discriminacion_pares_keras.py

# Clasificación 10 clases con evaluación de ruido
python tp2_keras/clasificacion_10_clases_keras.py

# Comparación completa
python tp2_keras/comparador_implementaciones.py
```

## Características Técnicas

### Configuración de TensorFlow
- **Semillas fijas**: Reproducibilidad garantizada (`tf.random.set_seed(42)`)
- **Warnings suprimidos**: Output limpio para análisis
- **Optimización**: SGD con learning rates específicos por problema

### Arquitecturas Implementadas
```python
ARQUITECTURAS = {
    'XOR': [2, 4, 1],
    'PARES_MINIMA': [35, 10, 1],
    'PARES_COMPACTA': [35, 15, 8, 1], 
    'PARES_DIRECTA': [35, 20, 10, 1],
    'PARES_BALANCEADA': [35, 25, 12, 1],
    'CLASIFICACION_10': [35, 20, 15, 10]
}
```

### Funciones de Activación
- **Sigmoid**: Para capas ocultas y problemas binarios
- **Softmax**: Para capa de salida en clasificación multiclase
- **Criterios de parada**: Early stopping basado en tolerancia de error

## Resultados Esperados

### 🎯 XOR
- **Convergencia**: ✅ Sí (típicamente 100-500 épocas)
- **Precisión**: 100% en todos los patrones
- **Comparación**: Rendimiento similar a implementación personalizada

### 🎯 Discriminación de Pares
- **Mejor arquitectura**: MINIMA [35, 10, 1] (~75% precisión)
- **Sobreajuste**: Presente en arquitecturas complejas
- **Comparación**: Keras ligeramente más eficiente

### 🎯 Clasificación 10 Clases  
- **Entrenamiento**: 100% (memorización perfecta)
- **Prueba**: 0-25% (sobreajuste severo)
- **Robustez al ruido**: Excelente en patrones conocidos
- **Comparación**: Comportamiento idéntico a implementación personalizada

## Análisis Comparativo

### Ventajas de TensorFlow/Keras:
✅ **API más simple y directa**
✅ **Optimizaciones automáticas**
✅ **Mejor manejo de memoria**
✅ **Callbacks y herramientas integradas**
✅ **Ecosistema maduro**

### Ventajas de Implementación Personalizada:
✅ **Control total sobre el algoritmo**
✅ **Comprensión profunda del funcionamiento**
✅ **Flexibilidad para modificaciones específicas**
✅ **Valor educativo superior**
✅ **Transparencia completa**

### Similitudes:
⚖️ **Resultados comparables en todos los experimentos**
⚖️ **Mismos problemas fundamentales (sobreajuste)**
⚖️ **Confirman principios teóricos**
⚖️ **Demuestran importancia de arquitectura y datos**

## Conclusiones Principales

1. **Equivalencia Funcional**: Ambas implementaciones producen resultados prácticamente idénticos
2. **Eficiencia**: Keras es más eficiente en tiempo de desarrollo y ejecución
3. **Educación**: La implementación personalizada ofrece mayor valor pedagógico
4. **Problemas Fundamentales**: El sobreajuste persiste independientemente de la implementación
5. **Validación**: Los resultados validan la correctitud de la implementación personalizada

## Troubleshooting

### Error: "No module named tensorflow"
```bash
pip install tensorflow>=2.10.0
```

### Error: "Cannot import tp2 modules"
Asegúrate de ejecutar desde el directorio raíz del proyecto:
```bash
cd /path/to/fundamentos-redes-neuronales
python tp2_keras/comparador_implementaciones.py
```

### Warnings de TensorFlow
Los warnings están suprimidos por defecto. Para habilitarlos:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Mostrar todos los logs
```

## Contribución

Para mantener la consistencia:

1. **Seguir el patrón de nomenclatura** establecido
2. **Usar configuración reproducible** (semillas fijas)
3. **Mantener compatibilidad** con el cargador de datos existente
4. **Documentar en español** con ejemplos claros
5. **Incluir análisis comparativo** en nuevas implementaciones

## Archivos de Datos

Los experimentos utilizan los mismos datos que las implementaciones personalizadas:
- `tp2/datos/numeros_decimales.txt`: Patrones de dígitos 5x7 píxeles

## Próximas Mejoras

- [ ] Visualización de curvas de aprendizaje
- [ ] Análisis de gradientes y pesos
- [ ] Implementación con diferentes optimizadores (Adam, RMSprop)
- [ ] Regularización (Dropout, L1/L2)
- [ ] Validación cruzada
- [ ] Métricas adicionales (F1-score, matriz de confusión detallada)

---

**Nota**: Esta implementación complementa el TP2 original, proporcionando una perspectiva comparativa entre implementaciones desde cero y frameworks profesionales.
