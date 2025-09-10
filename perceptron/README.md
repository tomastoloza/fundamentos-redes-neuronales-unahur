**![][image1]**

**UNIVERSIDAD NACIONAL DE HURLINGHAM**  

**CARRERA:** LICENCIATURA UNIVERSITARIA EN INFORMÁTICA

**ASIGNATURA:** Fundamentos de Redes Neuronales

**AÑO:** 2025

**Trabajo Práctico 1: Perceptrón simple**  

**Profesores:**   
* Emiliano Churruca  

**Alumnos:**  
* Sebastían Brandariz  
* Mauricio Challiol  
* Tomás Toloza

---

## **ENUNCIADOS DEL TRABAJO PRÁCTICO**

**1. Implemente el algoritmo de perceptrón simple con función de activación escalón y utilícelo para aprender los siguientes problemas:**

- **Función lógica 'Y'**
  
  Entradas:  
  `x = { {−1, 1}, {1, −1}, {−1, −1}, {1, 1} }  `
  
  Salida esperada:  
  `y = {−1, −1, −1, 1}`

- **Función lógica 'O exclusivo' (XOR)**

  Entradas:  
  `x = { {−1, 1}, {1, −1}, {−1, −1}, {1, 1} }  `
  Salida esperada:
  `y = {1, 1, −1, −1}`

¿Qué puede decir acerca de los problemas que puede resolver el perceptrón simple escalón en relación a la resolución de los problemas que se le pidió que aprenda?

---

**2. Implemente el algoritmo del perceptrón simple lineal y del perceptrón simple no lineal, y utilícelos para aprender el problema especificado en los archivos [TP1-ej2-Conjunto-entrenamiento.txt](TP1-ej2-Conjunto-entrenamiento.txt) y [TP1-ej2-Salida-deseada.txt](TP1-ej2-Salida-deseada.txt).**

- Evalúe la capacidad del perceptrón simple lineal y perceptrón simple no lineal para aprender la función cuyas muestras están presentes en los archivos indicados.
- Evalúe la capacidad de generalización del perceptrón simple no lineal utilizando, de los datos provistos, un subconjunto de ellos para entrenar y otro subconjunto para testear.
- ¿Cómo podría escoger el mejor conjunto de entrenamiento?
- ¿Cómo podría evaluar la máxima capacidad de generalización del perceptrón para este conjunto de datos?

---

* El trabajo deberá ser presentado por cada grupo el día 8 de abril. 
* Para su presentación usar PowerPoint o cualquier programa similar. 
* En la presentación deberá figurar el título del trabajo, el nombre de la materia, el nombre de los integrantes del grupo y la fecha. 
* La presentación podría estar dividida de acuerdo a los ítems solicitados, indicando en cada uno qué es lo que se pide. 
* Para cada ítem, primero comentar lo que se hizo y las decisiones tomadas para llevarlo a cabo (si correspondiera). 
* Luego exponer las dificultades que se presentaron (si correspondiera) y finalmente exponer los resultados. 
* Al finalizar la presentación deberán exponerse las conclusiones del trabajo.

---

## **DESARROLLO Y RESULTADOS**

### **1. Perceptrón Simple Escalon: Compuertas AND y XOR.**

Se desarrolló una implementación unificada [perceptron_unificado.py](perceptron.py) que permite entrenar diferentes tipos de perceptrones. El perceptrón simple con función de activación escalón puede resolver la función lógica 'Y' (AND), ya que se trata de un problema linealmente separable: existe una línea recta que puede separar las clases en el espacio de características, lo que permite que el perceptrón converja rápidamente encontrando los pesos correctos. La función AND es un ejemplo clásico de separabilidad lineal. 


En cambio, la función lógica 'O exclusivo' (XOR) no puede ser resuelta por el perceptrón simple, ya que es un problema no linealmente separable: no existe ninguna línea recta que pueda separar correctamente las clases y, por lo tanto, el algoritmo del perceptrón no converge para este caso. 

La limitación fundamental del perceptrón simple escalón es que está matemáticamente restringido a crear fronteras de decisión lineales (hiperplanos), lo que significa que solo puede resolver problemas donde las clases pueden ser separadas por una línea recta en dos dimensiones o un hiperplano en dimensiones superiores.

### **2. Perceptrón Simple Lineal y No Lineal**

#### **Perceptrón Lineal**
- **Fortalezas**: Modelo simple y estable que encuentra la mejor aproximación lineal
- **Limitaciones**: Restringido a relaciones lineales entre entradas y salidas
- **Comportamiento esperado**: MSE más alto debido a la limitación de ajustar solo funciones lineales

#### **Perceptrón No Lineal utilizando funcion Sigmoide**  
- **Fortalezas**: Capacidad de modelar relaciones no lineales complejas
- **Flexibilidad**: Mayor capacidad de ajuste a patrones complejos en los datos
- **Comportamiento esperado**: MSE significativamente menor al capturar mejor la naturaleza no lineal de los datos

#### **¿Cómo escoger el mejor conjunto de entrenamiento?**

En un problema de aprendizaje supervisado no se trata de escoger un conjunto ya dado, sino de construir un conjunto de entrenamiento adecuado para que el modelo aprenda de manera representativa y sin sesgos.

**Criterios principales:**

- Representatividad: el conjunto debe reflejar la distribución real de los datos y cubrir todas las clases o situaciones posibles.

- Aleatoriedad: la selección de los ejemplos de entrenamiento debe hacerse de manera aleatoria, evitando patrones ocultos que sesguen el aprendizaje.

- Tamaño suficiente: se necesita una cantidad adecuada de datos para capturar la complejidad del problema. Si el conjunto es demasiado pequeño, el perceptrón puede “memorizar” en lugar de generalizar.

- Estratificación (en clasificación): cuando hay varias clases, es importante que estén representadas en la misma proporción que en el conjunto total.

- Calidad y preprocesamiento: los datos deben estar normalizados/escalados para asegurar estabilidad en el entrenamiento y eliminar ruido que no aporte al modelo.

**Aplicado a este TP:**

- Se utilizó una división 80% entrenamiento – 20% prueba, garantizando tanto representatividad como un conjunto independiente para la evaluación.

- Se aplicó normalización de atributos para asegurar estabilidad numérica en el entrenamiento.

- De este modo, el conjunto de entrenamiento construido cumple con los requisitos para que el perceptrón pueda aprender de manera robusta.

###**¿Cómo evaluar la máxima capacidad de generalización?**

La capacidad de generalización es la habilidad del perceptrón de predecir correctamente ejemplos no vistos durante el entrenamiento.

- **Método recomendado:** Validación Cruzada (k-fold)

- El conjunto de datos se divide en k particiones del mismo tamaño.

- Se entrena k veces, usando k-1 particiones para entrenar y 1 para validar.

- Se promedian los resultados para obtener una estimación robusta del rendimiento del modelo.

**Ventajas:**

- Se utilizan todos los datos tanto para entrenamiento como para validación.

- Se obtiene una medida más estable que una única división entrenamiento/prueba.

- Permite detectar sobreajuste (overfitting) de manera más confiable.

**Alternativas complementarias:**

- **Validación anidada:** útil cuando se ajustan hiperparámetros del perceptrón.

- **Bootstrap:** permite estimar incertidumbre en las métricas de desempeño.

- **División entrenamiento/validación/test:** en problemas grandes, se recomienda tener tres subconjuntos diferenciados.

**Implementación en este TP:**

Se realizaron pruebas dividiendo los datos entre conjunto de entrenamiento y conjunto de testeo.

La validación cruzada se identificó como la técnica más robusta para evaluar la máxima capacidad de generalización del perceptrón no lineal con función sigmoide.
No se trata de "escoger" un conjunto, sino de **crearlo correctamente**:

**Principios fundamentales:**
- **Representatividad**: Debe reflejar la distribución completa de los datos
- **Aleatoriedad**: Selección aleatoria para evitar sesgos sistemáticos
- **Tamaño adecuado**: Suficientemente grande para capturar la complejidad del problema
- **Estratificación**: En problemas de clasificación, mantener proporciones de clases

**Implementación en este TP:**
- División aleatoria 80/20 garantiza representatividad
- Normalización previa asegura estabilidad numérica
- Validación en conjunto independiente confirma calidad

#### **¿Cómo evaluar la máxima capacidad de generalización?**
Matriz de validacion

**Técnica óptima: Validación Cruzada (k-fold)**

**Proceso:**
1. Dividir datos en k particiones (folds)
2. Entrenar k modelos, usando k-1 folds para entrenamiento y 1 para validación
3. Promediar resultados de las k iteraciones
4. Obtener estimación robusta del rendimiento

**Ventajas sobre división simple:**
- Utiliza todos los datos para entrenamiento y validación
- Reduce varianza en la estimación del rendimiento
- Detecta mejor el sobreajuste
- Proporciona intervalos de confianza

**Alternativas complementarias:**
- **Validación anidada**: Para selección de hiperparámetros
- **Bootstrap**: Para estimación de incertidumbre
- **Validación temporal**: Para datos con dependencia temporal

---

## **CONCLUSIONES**

En el ejercicio 1, se demostró que el perceptrón simple es capaz de aprender funciones lógicamente linealmente separables como AND y OR, mostrando una convergencia rápida gracias a esta característica; además, se observaron diferencias entre las funciones de activación, donde el escalón produce salidas discretas y una convergencia veloz, siendo ideal para clasificación binaria, mientras que la sigmoide ofrece salidas continuas y una convergencia más gradual, útil para estimar probabilidades. 

En el ejercicio 2, se evidenció que el modelo lineal está limitado a problemas linealmente separables, aunque resulta robusto e interpretable, y que la introducción de la función sigmoide permite modelar relaciones más complejas con mayor precisión, aunque los modelos más complejos requieren especial atención para evitar el sobreajuste. Respecto a la metodología de evaluación, se destacó la importancia de la validación independiente para asegurar la capacidad de generalización del modelo, la robustez de la validación cruzada como técnica estándar para una evaluación rigurosa y la necesidad de normalizar los datos para garantizar la estabilidad y la convergencia de los algoritmos de gradiente.

En términos generales, se aprendió que la separabilidad lineal es fundamental en problemas como las compuertas AND y OR, que el perceptrón es especialmente útil para clasificación binaria y regresión simple, que la elección de la función de activación debe responder al tipo de problema y salida deseada, que una evaluación rigurosa requiere múltiples métricas y técnicas de validación, y que la preparación adecuada de los datos es tan relevante como la selección del modelo, especialmente en tareas de regresión.

[image1]: <data:image/png;base64,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>
