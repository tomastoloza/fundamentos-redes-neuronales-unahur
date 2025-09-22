# TP1 - Perceptrón Simple

## Descripción

Implementación del perceptrón simple siguiendo principios **Clean Code** y **SOLID**. Este módulo incluye experimentos con compuertas lógicas y análisis de problemas linealmente separables vs no separables.

## Estructura del Proyecto

```
tp1/
├── src/                          # Código fuente
│   ├── perceptron_simple.py      # Implementación del perceptrón simple
│   ├── cargador_datos.py         # Carga y preparación de datos (SRP)
│   ├── entrenador_compuertas.py  # Entrenamiento de compuertas lógicas (SRP)
│   ├── visualizador_resultados.py # Visualización de resultados (SRP)
│   ├── main_tp1.py              # Módulo principal de ejecución
│   └── __init__.py              # Configuración del módulo
├── datos/                        # Archivos de datos
│   ├── TP1-ej2-Conjunto-entrenamiento.txt
│   └── TP1-ej2-Salida-deseada.txt
├── resultados/                   # Resultados de experimentos
├── tests/                        # Tests unitarios
└── README.md                     # Esta documentación
```

## Principios Aplicados

### Clean Code
- **Eliminación de números mágicos**: Todas las constantes centralizadas en `comun/constantes/`
- **Nombres descriptivos**: Variables y métodos con nombres claros y explicativos
- **Funciones pequeñas**: Cada función tiene una responsabilidad específica
- **Comentarios significativos**: Documentación clara en español

### SOLID
- **SRP (Single Responsibility Principle)**: Cada clase tiene una única responsabilidad
  - `PerceptronSimple`: Solo lógica del perceptrón
  - `CargadorDatos`: Solo carga y preparación de datos
  - `EntrenadorCompuertas`: Solo entrenamiento de compuertas
  - `VisualizadorResultados`: Solo visualización y presentación

- **OCP (Open/Closed Principle)**: Extensible sin modificar código existente
- **DIP (Dependency Inversion Principle)**: Inyección de dependencias

## Experimentos Incluidos

### Ejercicio 1: Compuertas Lógicas
- **AND**: Problema linealmente separable ✅
- **OR**: Problema linealmente separable ✅  
- **XOR**: Problema NO linealmente separable ❌

### Ejercicio 2: Datos desde Archivo
- **Perceptrón Lineal**: Función de activación lineal
- **Perceptrón No Lineal**: Función de activación sigmoide
- **Análisis de generalización**: Validación cruzada

## Uso Rápido

```python
from tp1.src.main_tp1 import EjecutorTP1

# Ejecutar todos los experimentos
ejecutor = EjecutorTP1()
ejecutor.ejecutar_todos_los_experimentos()

# Ejecutar experimento específico
ejecutor.ejecutar_ejercicio_1_compuertas_logicas()
```

## Uso Avanzado

```python
from tp1.src.perceptron_simple import PerceptronSimple
from tp1.src.cargador_datos import CargadorDatos

# Crear perceptrón personalizado
perceptron = PerceptronSimple(num_entradas=2, funcion_activacion='sigmoide')

# Cargar datos personalizados
cargador = CargadorDatos()
entradas, salidas = cargador.cargar_datos_compuerta_logica('and')

# Entrenar
convergencia, epoca = perceptron.entrenar(entradas, salidas)

# Evaluar
metricas = perceptron.evaluar(entradas, salidas)
```

## Funciones de Activación Disponibles

- **Escalón**: Para clasificación binaria discreta
- **Sigmoide**: Para salidas continuas entre 0 y 1
- **Lineal**: Para problemas de regresión
- **Tanh**: Para salidas entre -1 y 1

## Resultados Esperados

### Compuertas Lógicas (Función Escalón)
- **AND**: Convergencia rápida (~10-50 épocas)
- **OR**: Convergencia rápida (~10-50 épocas)
- **XOR**: No converge (problema no linealmente separable)

### Análisis de Generalización
- **Perceptrón Lineal**: MSE más alto, pero estable
- **Perceptrón Sigmoide**: MSE menor, mayor flexibilidad

## Conclusiones Principales

1. **Separabilidad Lineal**: El perceptrón simple solo resuelve problemas linealmente separables
2. **Función de Activación**: La elección impacta significativamente el rendimiento
3. **Generalización**: La validación independiente es crucial para evaluar el modelo
4. **Limitaciones**: XOR requiere arquitecturas más complejas (perceptrón multicapa)

## Dependencias

- `numpy`: Operaciones matemáticas
- `typing`: Type hints para mejor documentación del código

## Ejecución

```bash
# Desde el directorio raíz del proyecto
cd tp1/src
python main_tp1.py
```

## Contribución

El código sigue estrictamente los principios de Clean Code y SOLID. Para contribuir:

1. Mantener la separación de responsabilidades
2. Usar las constantes centralizadas
3. Documentar en español
4. Seguir los patrones de naming establecidos
5. Agregar tests para nuevas funcionalidades
