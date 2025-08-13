# Semana 1: Perceptrón Simple
## Conceptos Fundamentales (Notas de Clase)

### Geometría del Perceptrón
- **Hiperplano**: Subvariedad de dimensión (n-1) en un espacio de n dimensiones que separa dicho espacio en dos mitades
  - En 2D: una línea separa el plano
  - En 3D: un plano separa el espacio
  - En nD: un hiperplano separa el espacio n-dimensional

- **Convexo cerrado**: Polígono que contiene todos los puntos de una clase
  - Los datos linealmente separables forman regiones convexas
  - El perceptrón encuentra el hiperplano que separa estas regiones

### Operaciones Matemáticas Clave
- **Producto escalar/producto interno**: w^T x = w₁x₁ + w₂x₂ + ... + wₙxₙ
- **Proyección de a sobre b**: Componente del vector a en la dirección de b
  - Utilizado en el cálculo de la entrada neta del perceptrón

### Arquitectura Interna del Perceptrón
- **h (excitación)**: Información que llega a la neurona
  - h = Σ(wᵢ × xᵢ) + b
  - Es el producto de cada entrada por su peso correspondiente más el sesgo
  
- **O (función de activación)**: Transforma la excitación en la salida final
  - Para perceptrón simple: función escalón
  - Decide si la neurona "se activa" o no

- **Umbral**: Valor crítico que determina la activación
  - En perceptrón simple, umbral = 0
  - Si h ≥ umbral → salida = 1, sino → salida = 0

### Proceso de Entrenamiento
- **Pesos automáticos**: Los pesos se calculan automáticamente durante el entrenamiento
  - Inicialización: valores aleatorios pequeños
  - Actualización: regla del perceptrón basada en errores
  
- **Época**: Ciclo completo donde todos los ejemplos de entrenamiento han pasado por el perceptrón al menos una vez
  - Una época = una pasada completa por el dataset
  - El entrenamiento puede requerir múltiples épocas para convergir


## Introducción


El perceptrón simple es la unidad básica de procesamiento de las redes neuronales artificiales, inspirado en el funcionamiento de las neuronas biológicas. Fue propuesto por Frank Rosenblatt en 1957 y representa el primer modelo de neurona artificial.

## ¿Qué es un Perceptrón?

Un perceptrón es un clasificador binario lineal que puede aprender a separar dos clases de datos mediante un hiperplano en el espacio de características. Es la forma más simple de una red neuronal artificial.

### Componentes del Perceptrón

1. **Entradas (inputs)**: x₁, x₂, ..., xₙ
2. **Pesos (weights)**: w₁, w₂, ..., wₙ
3. **Sesgo (bias)**: b
4. **Función de activación**: f(z)
5. **Salida**: y

## Formulación Matemática

### Función de Entrada Neta
La entrada neta del perceptrón se calcula como:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

O en forma vectorial:
```
z = w^T x + b
```

### Función de Activación
Para el perceptrón simple, se utiliza la función escalón (step function):

```
f(z) = {
  1  si z ≥ 0
  0  si z < 0
}
```

### Salida del Perceptrón
```
y = f(w^T x + b)
```

## Algoritmo de Aprendizaje

El perceptrón aprende mediante un proceso iterativo que ajusta los pesos basándose en los errores de clasificación.

### Regla de Actualización de Pesos

Para cada ejemplo de entrenamiento (xᵢ, tᵢ):

1. Calcular la salida: yᵢ = f(w^T xᵢ + b)
2. Calcular el error: eᵢ = tᵢ - yᵢ
3. Actualizar pesos: w = w + η × eᵢ × xᵢ
4. Actualizar sesgo: b = b + η × eᵢ

Donde:
- η (eta) es la tasa de aprendizaje
- tᵢ es la salida deseada
- yᵢ es la salida obtenida

### Pseudocódigo del Algoritmo

```
1. Inicializar pesos w y sesgo b aleatoriamente
2. Establecer tasa de aprendizaje η
3. Repetir hasta convergencia:
   a. Para cada ejemplo (xᵢ, tᵢ):
      - Calcular yᵢ = f(w^T xᵢ + b)
      - Si yᵢ ≠ tᵢ:
        * w = w + η(tᵢ - yᵢ)xᵢ
        * b = b + η(tᵢ - yᵢ)
4. Retornar w y b finales
```

## Limitaciones del Perceptrón Simple

### Teorema de Convergencia
- El perceptrón converge en un número finito de pasos si los datos son **linealmente separables**
- Si los datos no son linealmente separables, el algoritmo no converge

### Problema XOR
El perceptrón simple no puede resolver problemas no linealmente separables como la función XOR:

| x₁ | x₂ | XOR |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |

## Ejemplo Práctico: Compuerta AND

Vamos a entrenar un perceptrón para implementar la función lógica AND.

### Datos de Entrenamiento
| x₁ | x₂ | AND |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 0  |
| 1  | 0  | 0  |
| 1  | 1  | 1  |

### Implementación Paso a Paso

**Inicialización:**
- w₁ = 0.5, w₂ = 0.5, b = -0.7
- η = 0.1

**Iteración 1:**
1. (0,0) → z = 0.5×0 + 0.5×0 - 0.7 = -0.7 → y = 0 ✓
2. (0,1) → z = 0.5×0 + 0.5×1 - 0.7 = -0.2 → y = 0 ✓
3. (1,0) → z = 0.5×1 + 0.5×0 - 0.7 = -0.2 → y = 0 ✓
4. (1,1) → z = 0.5×1 + 0.5×1 - 0.7 = 0.3 → y = 1 ✓

El perceptrón ya clasifica correctamente la función AND.

## Ejercicios Propuestos

### Ejercicio 1: Compuerta OR
Entrena un perceptrón para implementar la función lógica OR.

**Datos:**
| x₁ | x₂ | OR |
|----|----|----|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 1 |

### Ejercicio 2: Clasificación de Puntos
Dado el conjunto de puntos:
- Clase A: (1,1), (2,2), (3,3)
- Clase B: (-1,-1), (-2,-2), (-3,-3)

Entrena un perceptrón para separar estas dos clases.

### Ejercicio 3: Análisis de Convergencia
¿Por qué el perceptrón simple no puede resolver el problema XOR? Demuestra gráficamente que no existe una línea recta que separe las clases.

## Aplicaciones Históricas

1. **Reconocimiento de caracteres**: Uno de los primeros usos del perceptrón
2. **Clasificación de patrones simples**: Separación de objetos en imágenes
3. **Filtros de spam**: Clasificación básica de correos electrónicos

## Conexión con Temas Futuros

El perceptrón simple es la base para entender:
- **Perceptrón multicapa** (Semana 3-4)
- **Algoritmo de retropropagación**
- **Redes neuronales profundas**
- **Funciones de activación más complejas**

## Lecturas Recomendadas

1. Rosenblatt, F. (1958). "The perceptron: a probabilistic model for information storage and organization in the brain"
2. Minsky, M. & Papert, S. (1969). "Perceptrons" - Capítulo sobre limitaciones
3. Haykin, S. "Neural Networks and Learning Machines" - Capítulo 1

## Resumen Ejecutivo para Estudio

### Puntos Clave que Debes Recordar 📋

1. **Ecuación fundamental**: y = f(w^T x + b)
2. **Regla de aprendizaje**: w = w + η(t - y)x
3. **Condición de convergencia**: datos linealmente separables
4. **Limitación principal**: no resuelve XOR (problema no lineal)

### Conceptos Críticos para Examen ⭐

| Concepto | Definición | Importancia |
|----------|------------|-------------|
| Hiperplano | Frontera de decisión en n-1 dimensiones | Separar clases |
| Época | Una pasada completa por todos los datos | Medir progreso |
| Excitación (h) | Suma ponderada de entradas | Input de activación |
| Linealmente separable | Clases separables por línea recta | Condición convergencia |

### Fórmulas Esenciales 📐

```
Entrada neta:     z = Σ(wᵢxᵢ) + b
Función escalón:  f(z) = 1 si z≥0, 0 si z<0
Actualización:    wᵢ(nuevo) = wᵢ(viejo) + η(target - output)xᵢ
Error:           e = target - output
```

### Algoritmo del Perceptrón (Resumen) 🔄

1. **Inicializar** pesos aleatoriamente
2. **Para cada ejemplo**:
   - Calcular salida
   - Si hay error → actualizar pesos
3. **Repetir** hasta convergencia o máximo épocas

### Casos de Uso vs Limitaciones ✅❌

**✅ Puede resolver:**
- AND, OR, NOT
- Clasificación binaria lineal
- Reconocimiento de patrones simples

**❌ NO puede resolver:**
- XOR (requiere perceptrón multicapa)
- Problemas no lineales
- Clasificación multi-clase directa

## Preguntas de Repaso

1. ¿Cuál es la diferencia entre un perceptrón y una neurona biológica?
2. ¿Qué significa que un conjunto de datos sea "linealmente separable"?
3. ¿Por qué es importante la tasa de aprendizaje η?
4. ¿Cómo afecta la inicialización de pesos al proceso de aprendizaje?
5. ¿Cuándo está garantizada la convergencia del algoritmo del perceptrón?

## Preguntas Adicionales de Estudio

6. ¿Por qué el perceptrón simple no puede resolver XOR?
7. ¿Qué papel juega el sesgo (bias) en la capacidad de clasificación?
8. ¿Cómo se relaciona el hiperplano con la frontera de decisión?
9. ¿Qué sucede si la tasa de aprendizaje es muy alta o muy baja?
10. ¿Cuál es la diferencia entre época e iteración?

## Trucos de Estudio y Mnemotecnias 🧠

### Recordar la Regla de Actualización
**"WETη"**: **W**eights = **E**rror × **T**arget × **η**
- Si error > 0: incrementar peso
- Si error < 0: decrementar peso
- Si error = 0: no cambiar peso

### Visualización del Hiperplano
- **2D**: Imagina una línea que separa círculos rojos de azules
- **3D**: Imagina un papel que separa pelotas rojas de azules
- **nD**: Generalización matemática del concepto anterior

### Nemotecnia para Convergencia
**"LINEAL"**: **L**os datos **I**nealmente separables **N**ecesitan **E**l **A**lgoritmo para **L**ograra convergencia

### Regla de Oro del Perceptrón
> "Si no lo puedes separar con una línea recta, el perceptrón simple no lo puede aprender"

## Ejercicios Rápidos de Repaso ⚡

### Test de 5 Minutos
1. Dibuja un perceptrón con 2 entradas
2. Escribe la ecuación de la entrada neta
3. ¿Qué pasa si w₁=0.5, w₂=0.3, b=-0.2 y x=[1,1]?
4. ¿El punto (1,1) pertenece a la clase 1 o 0?

### Respuestas Rápidas
1. [Entrada1]→w₁→[+]→f()→[Salida]
   [Entrada2]→w₂→↗  ↑
   [Bias]→b→-----→ 
2. z = w₁x₁ + w₂x₂ + b
3. z = 0.5(1) + 0.3(1) + (-0.2) = 0.6
4. Clase 1 (porque z ≥ 0)
