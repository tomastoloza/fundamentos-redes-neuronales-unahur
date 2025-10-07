### 1. ¿Qué tipo de arquitectura de red neuronal artificial tiene un auto codificador?

Un auto codificador es una red neuronal artificial con una arquitectura de **perceptrón multicapa**. La dimensión de la entrada coincide con la dimensión de la salida. Tiene una o más capas ocultas, en un número impar, con una capa oculta central llamada **capa latente**. Por lo general, la capa latente tiene una dimensión *k* que es menor que la dimensión de la entrada *d*.

***

### 2. ¿Cómo se usa un auto codificador para detectar datos atípicos?

Para detectar datos atípicos, se entrena un auto codificador con un conjunto de datos *X*. Cuando se prueba con un conjunto de datos *T* que tiene la misma distribución que *X*, la diferencia entre la entrada y la salida del auto codificador es muy pequeña. Sin embargo, si un dato $t^*$ en *T* es atípico, la diferencia entre la entrada y la salida será significativamente mayor.

***

### 3. ¿Cuál es el espacio latente?

El **espacio latente** es un espacio de dimensión *k* en el que se representa cada ejemplo de entrenamiento una vez que la red ha aprendido. Esta representación es una **compresión** de los datos de entrada, que podría tener alguna pérdida de información.

***

### 4. ¿Qué utilidad tiene un auto codificador lineal?

Un auto codificador lineal puede ser interpretado como un método para realizar el **Análisis de Componentes Principales (ACP)**. A través de él se pueden encontrar las direcciones en las que los datos tienen mayor variabilidad.

***

### 5. ¿Por qué se puede pensar un auto codificador como una forma de realizar análisis de componentes principales no lineal?

Un auto codificador se puede considerar una forma de realizar un ACP no lineal si se utilizan funciones de activación **no lineales** (como sigmoidea o relu) en lugar de lineales. En ambos casos, se obtiene una representación en la capa latente de dimensión *k*, con $k<d$.

***

### 6. Dado un conjunto de datos X, ¿Cómo se obtiene la matriz de covarianzas?

La matriz de covarianza se puede obtener a partir de la matriz de datos **X**. Si las variables ya están estandarizadas, es decir, a cada variable $x_i$ se le resta su media y se divide por la desviación estándar, la matriz de covarianza será también la matriz de correlación. La covarianza entre dos variables $x_i$ y $x_j$ se calcula con la fórmula:

$covar(x_i, x_j) = \frac{1}{n-1}\sum_{k=1}^{n}(x_i^k - \overline{x_i})(x_j^k - \overline{x_j})$ 

***

### 7. ¿Que significa estandarizar los datos en un conjunto de datos?

Estandarizar los datos significa restar la media muestral de cada variable y dividirla por la desviación estándar (la raíz cuadrada de su varianza). Esto se puede representar con la fórmula:

$z_i^j = \frac{x_i^j - \overline{x_i}}{\sqrt{s_{ii}}}$ 

***

### 8. Sean x_1 y x_2 dos variables aleatorias que son parte de un conjunto datos multivariado, y al calcular la covar(xi, xj) su resultado es 1,6. ¿Qué puede decir acerca de si están muy o poco relacionadas?

Un valor de covarianza positivo y alto, como 1,6, indica un **alto grado de correlación positiva** entre las variables $x_i$ y $x_j$. Si el valor fuera cercano a cero, no estarían relacionadas.

***

### 9. ¿Cómo se usa un auto codificador para eliminar ruido?

Para eliminar ruido, se entrena un auto codificador para reproducir una salida sin ruido a partir de una entrada que sí tiene ruido. Esto se puede lograr con ruido gaussiano o de otros tipos, como el ruido de "Sal y pimienta".

***

### 10. Sea un conjunto de datos X con n ejemplos, ¿cómo haría para generar dato nuevo que no pertenezca a X pero que siga la distribución de datos de donde se obtuvo X?

Para generar un nuevo ejemplo, se puede tomar un punto $z_i$ del espacio latente que no corresponda a ninguno de los ejemplos de entrenamiento. Al aplicar este punto al decodificador del auto codificador, la salida será un nuevo ejemplo que sigue la misma distribución que los datos originales.