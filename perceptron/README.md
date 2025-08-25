# Perceptrón Simple - Implementación Modular

Este proyecto implementa el algoritmo del perceptrón siguiendo el pseudocódigo proporcionado, aplicado a diferentes problemas de aprendizaje automático.

## 📁 Estructura de Archivos

### Archivos Principales

- **`main.py`** - Programa principal con menú interactivo
- **`perceptron_unificado.py`** - Implementación unificada del algoritmo del perceptrón
- **`compuerta_and.py`** - Implementación de la compuerta lógica AND
- **`compuerta_or.py`** - Implementación de la compuerta lógica OR  
- **`tp1.py`** - Implementación para regresión (TP1-EJ2)

### Archivos de Datos

- **`TP1-ej2-Conjunto-entrenamiento.txt`** - Datos de entrenamiento para TP1
- **`TP1-ej2-Salida-deseada.txt`** - Salidas esperadas para TP1

## 🚀 Uso

### Ejecutar el Programa Principal

```bash
python3 main.py
```

### Ejecutar Módulos Individuales

```bash
# Compuerta AND (lineal)
python3 compuerta_and.py

# Compuerta AND (no lineal)
python3 compuerta_and.py no_lineal

# Compuerta OR (lineal)
python3 compuerta_or.py

# Compuerta OR (no lineal)
python3 compuerta_or.py no_lineal

# Comparar ambos tipos
python3 compuerta_and.py comparar

# TP1 (Regresión)
python3 tp1.py
```

## 🔧 Algoritmo Implementado

El perceptrón unificado utiliza el siguiente algoritmo base con selección aleatoria de ejemplos:

```python
while True:
    # Calcular error global para todo el conjunto
    error_global = calcular_error(entradas, salidas_deseadas, funcion_activacion)
    
    # Criterio de parada
    if error_global < error_min or epoca >= max_epocas:
        break
    
    # Selección aleatoria de un ejemplo
    indice = random.randint(0, len(entradas))
    entrada = append(entradas[indice], 1)  # Agregar sesgo
    
    # Calcular salida
    h = dot(entrada, pesos)
    O = funcion_activacion(h)
    
    # Calcular error
    M = salidas_deseadas[indice] - O
    
    # Actualizar pesos (gradiente descendente)
    delta_W = tasa_aprendizaje * M * derivada_activacion(h) * entrada
    pesos += delta_W
```

### Funciones de Activación Disponibles

- **`escalon`**: f(x) = 1 if x ≥ 0 else 0 (perceptrón lineal clásico)
- **`sigmoide`**: f(x) = 1/(1 + e^(-2x)) (perceptrón no lineal)
- **`lineal`**: f(x) = x (regresión)

## 📋 TRABAJO PRÁCTICO - ENUNCIADO COMPLETO

### 1. Perceptrón Simple con Función de Activación Escalón

**Implementar el algoritmo de perceptrón simple con función de activación escalón y utilizarlo para aprender los siguientes problemas:**

#### Problema 1: Función Lógica 'Y' (AND)
- **Entradas**: x = {{-1, 1}, {1, -1}, {-1, -1}, {1, 1}}
- **Salida esperada**: y = {-1, -1, -1, 1}

#### Problema 2: Función Lógica 'O exclusivo' (XOR)
- **Entradas**: x = {{-1, 1}, {1, -1}, {-1, -1}, {1, 1}}
- **Salida esperada**: y = {1, 1, -1, -1}

**Pregunta de Análisis:**
¿Qué puede decir acerca de los problemas que puede resolver el perceptrón simple escalón en relación a la resolución de los problemas que se le pidió que haga que el perceptrón aprenda?

### 2. Perceptrón Simple Lineal y No Lineal

**Implementar el algoritmo del perceptrón simple lineal y perceptrón simple no lineal y utilizarlos para aprender el problema especificado en los archivos:**
- `TP1-ej2-Conjunto-entrenamiento.txt`
- `TP1-ej2-Salida-deseada.txt`

#### Evaluaciones Requeridas:

- **Evaluar la capacidad** del perceptrón simple lineal y perceptrón simple no lineal para aprender la función cuyas muestras están presentes en los archivos indicados.

- **Evaluar la capacidad de generalización** del perceptrón simple no lineal utilizando, de los datos provistos, un subconjunto de ellos para entrenar y otro subconjunto para testear.

#### Preguntas de Análisis:

- **¿Cómo podría escoger el mejor conjunto de entrenamiento?**
- **¿Cómo podría evaluar la máxima capacidad de generalización del perceptrón para este conjunto de datos?**

### 3. Presentación del Trabajo

**Fecha de Presentación:** 8 de abril

**Formato:** PowerPoint o programa similar

**Contenido Requerido:**
- Título del trabajo
- Nombre de la materia
- Nombre de los integrantes del grupo
- Fecha

**Estructura de la Presentación:**

Para cada ítem solicitado:
1. **Comentar lo que se hizo** y las decisiones tomadas para llevarlo a cabo
2. **Exponer las dificultades** que se presentaron (si correspondiera)
3. **Exponer los resultados**
4. **Conclusiones del trabajo** al finalizar la presentación

## 🎯 Implementación Realizada

### 1. Funciones Lógicas con Perceptrón Escalón

#### Función AND (Y)
- ✅ **Implementada** en `compuerta_and.py`
- ✅ **Convergencia**: Típica en 10-50 épocas
- ✅ **Resultado**: 100% de precisión (problema linealmente separable)

#### Función XOR (O exclusivo)
- ⚠️ **Limitación conocida**: El perceptrón simple NO puede resolver XOR
- 📝 **Razón**: XOR no es linealmente separable
- 🔍 **Análisis**: Requiere perceptrón multicapa o funciones no lineales

### 2. Perceptrón para Regresión (TP1-EJ2)

#### Perceptrón Lineal (Regresión)
- ✅ **Implementado** con función de activación lineal
- ✅ **Datos**: 200 ejemplos, 3 características
- ✅ **División**: 80% entrenamiento, 20% prueba
- ✅ **Normalización**: Z-score para mejorar convergencia

#### Perceptrón No Lineal (Sigmoide)
- ✅ **Implementado** con función sigmoide
- ✅ **Gradiente descendente** con derivada de sigmoide
- ✅ **Convergencia** por umbral de error MSE

### 3. Capacidad de Generalización

#### Estrategias Implementadas:
- **División aleatoria** de datos (80/20)
- **Validación cruzada** posible con múltiples ejecuciones
- **Normalización** de datos para estabilidad
- **Early stopping** para evitar sobreajuste

#### Métricas de Evaluación:
- **Error cuadrático medio (MSE)**
- **Convergencia en épocas**
- **Comparación entre tipos de perceptrón**

## 📊 Resultados Típicos

### Compuertas Lógicas
- **AND Lineal**: Convergencia en 10-50 épocas, error = 0.000000
- **AND No Lineal**: Convergencia en 300-600 épocas, error < 0.01
- **OR Lineal**: Convergencia en 3-10 épocas, error = 0.000000
- **XOR**: No converge (limitación teórica del perceptrón simple)

### TP1-EJ2 (Regresión)
- **Perceptrón Lineal**: MSE final ~50-100 (dependiente de datos)
- **Perceptrón No Lineal**: Mejor capacidad de ajuste a patrones complejos
- **Generalización**: Evaluada mediante conjunto de prueba separado

## 🔍 Análisis de Limitaciones

### Perceptrón Simple Escalón
- ✅ **Puede resolver**: Problemas linealmente separables (AND, OR)
- ❌ **No puede resolver**: Problemas no linealmente separables (XOR)
- 📝 **Implicación**: Limitado a funciones booleanas linealmente separables

### Selección del Mejor Conjunto de Entrenamiento
- **Diversidad**: Representar toda la distribución de datos
- **Tamaño**: Balance entre información suficiente y eficiencia
- **Aleatoriedad**: Evitar sesgos en la selección
- **Estratificación**: Mantener proporciones de clases/rangos

### Evaluación de Máxima Capacidad de Generalización
- **Validación cruzada k-fold**
- **Curvas de aprendizaje** (error vs. tamaño del conjunto)
- **Análisis de sesgo-varianza**
- **Pruebas con múltiples divisiones aleatorias**

## 🛠️ Dependencias

```bash
pip install numpy
```

## 🎓 Conclusiones del Trabajo

1. **Limitaciones del Perceptrón Simple**: No puede resolver problemas no linealmente separables como XOR
2. **Ventajas del Perceptrón No Lineal**: Mayor capacidad de modelado para patrones complejos
3. **Importancia de la Preparación de Datos**: Normalización y división adecuada mejoran significativamente el rendimiento
4. **Trade-off Sesgo-Varianza**: El perceptrón lineal tiene mayor sesgo pero menor varianza que el no lineal
5. **Aplicabilidad**: El perceptrón simple es efectivo para problemas de clasificación binaria linealmente separables y regresión lineal básica