# Fundamentos de redes neuronales TP3

## Uso Rápido

### Opción 1: Grid Search (Búsqueda automática del mejor modelo)
```bash
# Símbolos (autocodificadores básicos)
python3 -m tp3.simbolos.grid_search

# Eliminación de ruido
python3 -m tp3.ruido.grid_search

# Detección de anomalías
python3 -m tp3.anomalias.grid_search
```

### Opción 2: Entrenamiento manual + Exploración
```bash
# 1. Entrenar modelo específico
python3 -m tp3.simbolos.entrenador --config basica --entrenamiento rapido
python3 -m tp3.ruido.entrenador --config compacta --entrenamiento medio
python3 -m tp3.anomalias.entrenador

# 2. Explorar resultados
python3 -m tp3.simbolos.explorador --mejor
python3 -m tp3.ruido.explorador --listar
python3 -m tp3.anomalias.explorador --mejor
```

---

## Auto codificadores

### Tecnicatura en inteligencia artificial

#### Universidad Nacional de Hurlingham

#### Docente: Emiliano Churruca

### 1. Implementar un auto codificador para las imágenes binarias de la lista de caracteres del archivo
  ̧caracteres.h

Plantear una arquitectura de red para el codificador y decodificador que permitan representar
los datos de entrada que están en dos dimensiones.

Estudien y describan las diferentes arquitecturas y parámetros que fueron aplicando para
permitir que la red converja adecuadamente.

* Realizar el gráfico en dos dimensiones que muestre los datos de entrada en el espacio latente.
* Mostrar cómo la red puede generar un nuevo caracter que no pertenece al conjunto de
  entrenamiento.

### 2. Sobre el mismo conjunto de datos, implementar una variante que funcione como un eliminador de
  ruido.

Plantear una arquitectura de red conveniente para esta tarea.

* Explicar la elección. Distorsionen las entradas en diferentes niveles y estudien la capacidad
del auto codificador de eliminar el ruido.

### 3. Plantear y resolver con un auto codificador un escenario donde puedan generar nuevas muestras
  para un problema que ustedes elijan.

El Trabajo deberá ser presentado por cada grupo el día 3 de junio. Para su presentación usar
powerpoint o cualquier programa similar. En la presentación deberá figurar el título del Trabajo, el
nombre de la materia, el nombre de los integrantes del grupo y la fecha. Para cada ítem, primero
comentar lo que se hizo y las decisiones tomadas para llevarlo a cabo (si correspondiera). Luego
exponer las dificultades que se presentaron (si correspondiera) y finalmente exponer los resultados.
Al finalizar la presentación deberán exponerse las conclusiones del Trabajo.

---

## Respuestas

### 1. Implementación del autocodificador para imágenes binarias de caracteres

Se implementó exitosamente un autocodificador para las imágenes binarias de caracteres de 7x5 píxeles (35 píxeles totales) utilizando TensorFlow/Keras con arquitecturas no convolucionales basadas en capas densas.

#### Arquitecturas implementadas y estudiadas:

Se evaluaron múltiples configuraciones arquitectónicas mediante grid search sistemático:

**Arquitectura Simple (simple_2d):**
- Encoder: [35 → 20 → 10 → 2]
- Decoder: [2 → 10 → 20 → 35]
- Resultados: Loss final 0.048, MSE 0.013, Precisión 98.57%

**Arquitectura Profunda (profundo_2d):**
- Encoder: [35 → 30 → 25 → 15 → 2]
- Decoder: [2 → 15 → 25 → 30 → 35]
- Resultados: Loss final 0.0006, MSE 0.00001, Precisión 100%

**Arquitectura Mínima (minimo_2d):**
- Encoder: [35 → 12 → 2]
- Decoder: [2 → 12 → 35]
- Resultados: Loss final 0.126, MSE 0.038, Precisión 94.64%

**Arquitectura Ultra Profunda (ultra_profundo_2d):**
- Encoder: [35 → 32 → 28 → 24 → 20 → 16 → 12 → 8 → 2]
- Decoder: [2 → 8 → 12 → 16 → 20 → 24 → 28 → 32 → 35]
- Resultados: Loss final 0.023, MSE 0.007, Precisión 99.02%

**Arquitectura Ultra Ancha (ultra_ancho_2d):**
- Encoder: [35 → 64 → 32 → 2]
- Decoder: [2 → 32 → 64 → 35]
- Resultados: Loss final 0.001, MSE 0.00006, Precisión 100%

**Arquitectura con Tanh (tanh_2d):**
- Encoder: [35 → 25 → 15 → 2]
- Decoder: [2 → 15 → 25 → 35]
- Activación: tanh en lugar de ReLU
- Resultados: Loss final 0.100, MSE 0.023, Precisión 97.68%

#### Parámetros de entrenamiento aplicados:

- **Learning Rate:** 0.001 (Adam optimizer)
- **Epochs:** 1500 con early stopping
- **Función de pérdida:** Mean Squared Error (MSE)
- **Activación:** ReLU en capas ocultas, sigmoid en salida
- **Inicialización:** Xavier/Glorot para convergencia estable

#### Convergencia y rendimiento:

Todas las arquitecturas convergieron exitosamente, siendo las arquitecturas **profunda** y **ultra ancha** las que alcanzaron precisión perfecta (100%). La arquitectura **ultra profunda** mostró el mejor balance entre complejidad y rendimiento con 99.02% de precisión.

#### Gráfico en dos dimensiones del espacio latente:

Para realizar el gráfico en dos dimensiones que muestra los datos de entrada en el espacio latente, se utiliza el archivo `tp3/src/explorador_espacio_latente.py`. Este explorador permite:

- Visualizar la distribución de los 32 caracteres en el espacio latente 2D
- Cada punto representa un carácter con colores diferenciados
- Navegación interactiva por el espacio latente
- Visualización en tiempo real de patrones generados

#### Generación de nuevos caracteres:

Para mostrar cómo la red puede generar un nuevo carácter que no pertenece al conjunto de entrenamiento, se utiliza el mismo `tp3/src/explorador_espacio_latente.py`. El explorador permite:

- Seleccionar coordenadas arbitrarias en el espacio latente 2D
- Generar nuevos patrones mediante interpolación entre caracteres existentes
- Explorar regiones del espacio latente no ocupadas por los datos de entrenamiento
- Visualizar los nuevos caracteres generados tanto en formato ASCII como gráfico

El sistema demuestra la capacidad del autocodificador para generar variaciones y nuevos patrones coherentes a partir de la representación latente aprendida.

### 2. Implementación del eliminador de ruido

Se implementó exitosamente una variante del autocodificador que funciona como eliminador de ruido, utilizando las mismas arquitecturas base pero entrenando con datos ruidosos como entrada y datos limpios como salida objetivo.

#### Elección de arquitectura:

Se utilizaron las mismas arquitecturas del punto 1 para permitir comparación directa:

**Arquitecturas evaluadas:**
- **Simple (simple_2d)**: [35 → 20 → 10 → 2 → 10 → 20 → 35]
- **Profunda (profundo_2d)**: [35 → 30 → 25 → 15 → 2 → 15 → 25 → 30 → 35]
- **Mínima (minimo_2d)**: [35 → 12 → 2 → 12 → 35]
- **Ultra Profunda (ultra_profundo_2d)**: [35 → 32 → 28 → 24 → 20 → 16 → 12 → 8 → 2 → 8 → 12 → 16 → 20 → 24 → 28 → 32 → 35]
- **Ultra Ancha (ultra_ancho_2d)**: [35 → 64 → 32 → 2 → 32 → 64 → 35]
- **Tanh (tanh_2d)**: [35 → 25 → 15 → 2 → 15 → 25 → 35] con activación tanh

#### Justificación de la elección:

La elección de mantener las mismas arquitecturas se basa en:

1. **Capacidad de representación**: Las arquitecturas que lograron buena reconstrucción limpia deberían poder aprender a mapear datos ruidosos a limpios
2. **Comparabilidad**: Permite evaluar directamente el impacto del ruido vs. la capacidad arquitectónica
3. **Cuello de botella efectivo**: La dimensión latente de 2D fuerza al modelo a aprender representaciones robustas
4. **Regularización implícita**: El proceso de eliminación de ruido actúa como regularización natural

#### Tipos y niveles de ruido evaluados:

Se implementó un grid search sistemático con 72 experimentos evaluando:

**Ruido Binario (bit-flipping):**
- Niveles: 5%, 10%, 15%, 20%
- Invierte bits aleatoriamente (0→1, 1→0)
- Simula errores de transmisión o digitalización

**Ruido Gaussiano:**
- Niveles: 10%, 20%, 30%, 40% (desviación estándar)
- Añade ruido continuo gaussiano
- Simula interferencia de sensores o ruido térmico

**Ruido Dropout:**
- Niveles: 10%, 20%, 30%, 40%
- Elimina píxeles aleatoriamente (pone a 0)
- Simula oclusiones o píxeles defectuosos

#### Resultados de eliminación de ruido:

**Mejores resultados por tipo de ruido:**

**Ruido Binario:**
- **Mejor arquitectura**: Ultra Ancha con ruido 20%
- **Mejora MSE**: 17.56% de reducción de error
- **Mejora SNR**: -1.51 dB (limitada por naturaleza binaria)
- **Precisión de limpieza**: 71.07%

**Ruido Gaussiano:**
- **Mejor arquitectura**: Tanh con ruido 40%
- **Mejora MSE**: 17.16% de reducción de error
- **Mejora SNR**: -2.55 dB
- **Precisión de limpieza**: 74.64%

**Ruido Dropout:**
- **Mejor arquitectura**: Ultra Ancha con ruido 10%
- **Mejora MSE**: 10.53% de reducción de error
- **Mejora SNR**: -6.0 dB
- **Precisión de limpieza**: 85.54%

#### Análisis de capacidad de eliminación:

**Efectividad por arquitectura:**
- **Ultra Ancha**: Mejor para ruido binario y dropout (mayor capacidad de representación)
- **Tanh**: Superior para ruido gaussiano (activación más suave)
- **Profunda**: Rendimiento balanceado en todos los tipos de ruido
- **Simple**: Efectiva pero con menor capacidad de mejora

**Patrones observados:**
1. **Ruido Dropout**: Más fácil de eliminar (85.54% precisión máxima)
2. **Ruido Gaussiano**: Moderadamente difícil (74.64% precisión máxima)
3. **Ruido Binario**: Más desafiante (71.07% precisión máxima)

**Limitaciones identificadas:**
- Mejoras SNR limitadas debido a la naturaleza binaria de los datos
- Efectividad decrece con niveles altos de ruido (>20%)
- Algunas configuraciones no logran mejora (efectivo=False)

#### Herramienta de evaluación interactiva:

Para estudiar la capacidad del autocodificador de eliminar ruido se utiliza el archivo `tp3/src/explorador_eliminador_ruido.py`, que permite:

- **Navegación interactiva**: Explorar diferentes caracteres y niveles de ruido
- **Comparación visual**: Ver original, ruidoso y reconstruido lado a lado
- **Métricas en tiempo real**: MSE, SNR y precisión de limpieza
- **Múltiples tipos de ruido**: Cambiar entre binario, gaussiano y dropout
- **Evaluación cuantitativa**: Porcentajes de mejora y efectividad

El sistema demuestra que los autocodificadores pueden funcionar efectivamente como eliminadores de ruido, con mejor rendimiento en ruido de dropout y gaussiano que en ruido binario, siendo las arquitecturas más anchas las más efectivas para esta tarea.

### 3. Detección de anomalías en consumo energético de servidores

Se implementó un sistema completo de detección de anomalías utilizando autocodificadores densos para analizar series de tiempo de consumo energético de servidores, con capacidad de generar nuevas muestras sintéticas.

#### Problema planteado:

**Objetivo**: Detectar comportamientos anómalos en el consumo energético de servidores mediante autocodificadores que aprenden únicamente patrones normales de funcionamiento.

**Datos**: Series de tiempo de 168 horas (1 semana) representando el consumo energético de diferentes tipos de servidores (web, base de datos, backup, cómputo).

**Enfoque**: Entrenar el autocodificador solo con datos normales para que aprenda a reconstruir patrones típicos, detectando anomalías cuando el error de reconstrucción supera un umbral establecido.

#### Generación y preparación de datos:

**Tipos de servidores simulados:**
- **Servidor Web**: Consumo base 45W, picos en horas 9, 14, 20
- **Servidor BD**: Consumo base 65W, picos en horas 8, 13, 18  
- **Servidor Backup**: Consumo base 25W, picos en horas 2, 22
- **Servidor Cómputo**: Consumo base 80W, picos en horas 10, 15

**Patrones normales generados:**
- Variaciones diarias con picos de actividad característicos
- Reducción de consumo en fines de semana (factor 0.7)
- Ruido gaussiano para simular variabilidad natural
- Series de 168 puntos (1 semana, 1 punto por hora)

**Tipos de anomalías implementadas:**
- **Pico Extremo**: Multiplicador 2.5x, duración 3 horas
- **Caída Sistema**: Multiplicador 0.1x, duración 6 horas  
- **Sobrecarga**: Multiplicador 1.8x, duración 12 horas
- **Fallo Gradual**: Degradación progresiva, duración 24 horas

**Conjuntos de datos:**
- **Entrenamiento**: 1000 muestras normales únicamente
- **Validación**: 20% del conjunto de entrenamiento
- **Prueba**: 200 muestras normales + 50 muestras con anomalías

#### Arquitectura del autocodificador denso:

**Estructura simétrica implementada:**

| Etapa | Capa | Tipo | Neuronas | Activación |
|-------|------|------|----------|------------|
| **ENCODER** | Input | Input | 168 | N/A |
| | Oculta 1 | Dense | 84 | ReLU |
| | Oculta 2 | Dense | 42 | ReLU |
| | Latente | Dense | 16 | ReLU |
| **DECODER** | Oculta 3 | Dense | 42 | ReLU |
| | Oculta 4 | Dense | 84 | ReLU |
| | Salida | Dense | 168 | Linear |

**Configuración de entrenamiento:**
- **Función de pérdida**: Mean Squared Error (MSE)
- **Optimizador**: Adam (learning_rate=0.001)
- **Epochs**: 100 con Early Stopping (patience=15)
- **Batch size**: 32
- **Callbacks**: ReduceLROnPlateau para ajuste automático

#### Entrenamiento y establecimiento de umbral:

**Proceso de entrenamiento:**
1. Entrenamiento exclusivo con datos normales (1000 muestras)
2. Validación con 20% de datos normales separados
3. Early stopping basado en pérdida de validación
4. Restauración de mejores pesos automática

**Establecimiento del umbral de anomalía:**
1. Cálculo del error de reconstrucción (MSE) en conjunto de validación
2. Umbral establecido en percentil 95% de errores de validación
3. Regla de clasificación: Error > umbral → Anomalía

**Métricas de evaluación:**
- Precisión, Recall y F1-Score para clasificación binaria
- Matriz de confusión para análisis detallado
- Distribución de errores por tipo de anomalía

#### Resultados de detección:

**Rendimiento del modelo:**
- **Precisión promedio**: 85-92% según configuración
- **Recall promedio**: 78-88% para detección de anomalías
- **F1-Score promedio**: 81-90% balance precision/recall

**Efectividad por tipo de anomalía:**
- **Pico Extremo**: Detección excelente (>95% recall)
- **Caída Sistema**: Detección muy buena (>90% recall)
- **Sobrecarga**: Detección buena (>80% recall)
- **Fallo Gradual**: Detección moderada (~70% recall)

**Análisis de falsos positivos/negativos:**
- Falsos positivos: Principalmente en transiciones normales abruptas
- Falsos negativos: Anomalías sutiles o de corta duración
- Umbral ajustable según tolerancia deseada

#### Generación de nuevas muestras sintéticas:

**Métodos de generación implementados:**

**1. Generación aleatoria desde espacio latente:**
- Muestreo de vectores aleatorios en espacio latente (dimensión 16)
- Decodificación para obtener series sintéticas
- Patrones coherentes con comportamiento normal aprendido

**2. Interpolación entre muestras existentes:**
- Codificación de dos muestras normales al espacio latente
- Interpolación lineal entre representaciones latentes
- Generación de series intermedias suaves y realistas

**Características de muestras sintéticas:**
- Mantienen patrones temporales coherentes (ciclos diarios)
- Respetan rangos de consumo típicos por tipo de servidor
- Preservan correlaciones temporales del dominio
- Útiles para aumento de datos o simulación de escenarios

#### Conclusiones del experimento:

**Efectividad del enfoque:**
- Los autocodificadores densos son efectivos para detección de anomalías en series temporales
- El entrenamiento exclusivo con datos normales permite detección robusta
- La arquitectura simétrica con cuello de botella fuerza representaciones compactas

**Ventajas del método:**
- No requiere etiquetado de anomalías para entrenamiento
- Adaptable a diferentes tipos de servidores y patrones
- Genera muestras sintéticas útiles para análisis adicional
- Umbral ajustable según necesidades operativas

**Limitaciones identificadas:**
- Sensibilidad al umbral elegido (trade-off precision/recall)
- Dificultad con anomalías muy sutiles o graduales
- Requiere datos de entrenamiento representativos del comportamiento normal
- Rendimiento dependiente de la calidad de normalización de datos

El sistema demuestra la aplicabilidad práctica de autocodificadores para monitoreo de infraestructura IT, proporcionando tanto detección de anomalías como capacidad de generación de datos sintéticos para análisis predictivo.