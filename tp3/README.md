# Fundamentos de redes neuronales TP3

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

Por que la capa latente tiene que ser al menos 5?
Solo hay 32 clases (caracteres).
Para distinguir 32 ítems se necesita un mínimo de ⌈log2(32)⌉=5 bits (o dimensiones latentes).

