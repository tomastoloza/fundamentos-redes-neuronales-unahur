# TP2 - Perceptrón Multicapa

## Descripción

Implementación del perceptrón multicapa con retropropagación siguiendo principios **Clean Code** y **SOLID**. Este módulo incluye experimentos con XOR, discriminación de números pares y clasificación multiclase de dígitos.

## Estructura del Proyecto

```
tp2/
├── src/                              # Código fuente
│   ├── perceptron_multicapa.py       # Implementación del perceptrón multicapa
│   ├── cargador_datos_digitos.py     # Carga de datos de dígitos (SRP)
│   ├── entrenador_tp2.py            # Entrenamiento de problemas TP2 (SRP)
│   ├── main_tp2.py                  # Módulo principal de ejecución
│   └── __init__.py                  # Configuración del módulo
├── datos/                            # Archivos de datos
│   └── numeros_decimales.txt         # Patrones de dígitos 5x7 píxeles
├── resultados/                       # Resultados de experimentos
├── tests/                            # Tests unitarios
└── README.md                         # Esta documentación
```
## Experimentos Incluidos

### Ejercicio 1: Función Lógica XOR
- **Arquitectura**: [2, 4, 1] (2 entradas, 4 neuronas ocultas, 1 salida)
- **Objetivo**: Demostrar que el perceptrón multicapa resuelve problemas no linealmente separables
- **Resultado esperado**: Convergencia exitosa donde el perceptrón simple falla

### Ejercicio 2: Discriminación de Números Pares
- **Entrada**: Dígitos representados como matrices 5x7 píxeles (35 entradas)
- **Salida**: Clasificación binaria (par/impar)
- **Entrenamiento**: Dígitos 0,2,4,6,1,3 (4 pares, 2 impares)
- **Prueba**: Dígitos 5,7,8,9 (1 par, 3 impares)
- **Análisis**: Capacidad de generalización de la red

### Ejercicio 3: Clasificación de Dígitos (10 Clases)
- **Arquitectura**: [35, 20, 15, 10] (35 entradas, 2 capas ocultas, 10 salidas)
- **Codificación**: One-hot encoding para 10 clases
- **Entrenamiento**: Dígitos 0-6
- **Prueba**: Dígitos 7-9
- **Evaluación con ruido**: Probabilidad 0.02 de intercambio de bits

## Arquitecturas Predefinidas

```python
ARQUITECTURAS_TP2 = {
    'MINIMA': [35, 10, 1],              # Arquitectura mínima
    'COMPACTA': [35, 15, 8, 1],         # Arquitectura compacta
    'DIRECTA_ORIGINAL': [35, 20, 10, 1], # Arquitectura original
    'PROFUNDA': [35, 30, 20, 15, 10, 1], # Arquitectura profunda
    'BALANCEADA': [35, 25, 12, 1],      # Arquitectura balanceada
    'CLASIFICACION_10_CLASES': [35, 20, 15, 10] # Para 10 clases
}
```

## Uso Rápido

```python
from tp2.src.main_tp2 import EjecutorTP2

# Ejecutar todos los experimentos
ejecutor = EjecutorTP2()
ejecutor.ejecutar_todos_los_experimentos()

# Ejecutar experimento específico
ejecutor.ejecutar_ejercicio_1_xor()
ejecutor.ejecutar_ejercicio_2_discriminacion_pares()
ejecutor.ejecutar_ejercicio_3_clasificacion_10_clases()
```

## Uso Avanzado

```python
from tp2.src.perceptron_multicapa import PerceptronMulticapa
from tp2.src.cargador_datos_digitos import CargadorDatosDigitos

# Crear red personalizada
red = PerceptronMulticapa(
    arquitectura=[35, 20, 10],
    funciones_activacion=['sigmoide', 'sigmoide']
)

# Cargar datos de dígitos
cargador = CargadorDatosDigitos()
entradas, salidas = cargador.cargar_datos_tp2()

# Entrenar
convergencia, epoca = red.entrenar(entradas, salidas)

# Evaluar
metricas = red.evaluar(entradas, salidas, tipo_problema='clasificacion_multiclase')
```

## Características Técnicas

### Algoritmo de Entrenamiento
- **Retropropagación**: Implementación completa del algoritmo backpropagation
- **Inicialización Xavier**: Inicialización inteligente de pesos para mejor convergencia
- **Funciones de activación**: Sigmoide, tanh, lineal (extensible)

### Evaluación de Rendimiento
- **Métricas múltiples**: Precisión, error cuadrático medio, matrices de confusión
- **Evaluación con ruido**: Robustez ante perturbaciones en los datos
- **Análisis de sobreajuste**: Comparación entre entrenamiento y validación

### Visualización de Patrones
```python
# Visualizar un dígito
cargador = CargadorDatosDigitos()
patron = cargador.obtener_patron_digito(digito=7)
visualizacion = cargador.visualizar_patron(patron)
print(visualizacion)
```

## Resultados Esperados

### XOR
- **Convergencia**: ✅ Sí (típicamente 100-500 épocas)
- **Precisión**: 100% en todos los patrones
- **Conclusión**: Demuestra superioridad sobre perceptrón simple

### Discriminación de Pares
- **Arquitectura MINIMA**: ~75% precisión (mejor rendimiento)
- **Arquitecturas complejas**: Menor precisión por sobreajuste
- **Conclusión**: Más parámetros no siempre es mejor

### Clasificación 10 Clases
- **Entrenamiento**: 100% (memorización perfecta)
- **Prueba**: 0-25% (sobreajuste severo)
- **Con ruido**: Robustez excelente en patrones conocidos
- **Conclusión**: Datos limitados causan sobreajuste

## Análisis de Sobreajuste

El proyecto incluye análisis detallado de sobreajuste:

- **Detección**: Diferencia significativa entre precisión de entrenamiento y prueba
- **Causas**: Datos limitados (1 patrón por dígito), arquitecturas complejas
- **Soluciones sugeridas**: Más datos, regularización, arquitecturas más simples

## Evaluación con Ruido

```python
# Evaluar robustez al ruido
resultado_ruido = entrenador.evaluar_robustez_ruido(
    nombre_experimento='clasificacion_10_clases',
    probabilidad_ruido=0.02
)
```

## Dependencias

- `numpy`: Operaciones matemáticas y arrays
- `typing`: Type hints para documentación del código

## Ejecución

```bash
# Desde el directorio raíz del proyecto
cd tp2/src
python main_tp2.py
```

## Conclusiones Principales

1. **Superioridad del Multicapa**: Resuelve problemas no linealmente separables (XOR)
2. **Arquitectura vs Rendimiento**: Arquitecturas simples pueden generalizar mejor
3. **Problema de Sobreajuste**: Datos limitados causan memorización sin generalización
4. **Robustez al Ruido**: Redes bien entrenadas son robustas a pequeñas perturbaciones
5. **Importancia de Validación**: Evaluación independiente es crucial

## Contribución

Para mantener la calidad del código:

1. Seguir principios SOLID establecidos
2. Usar constantes centralizadas para configuración
3. Mantener separación clara de responsabilidades
4. Documentar en español con ejemplos claros
5. Agregar tests para nuevas funcionalidades
