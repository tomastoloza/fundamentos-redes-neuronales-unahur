# Fundamentos de Redes Neuronales - UNAHUR

## Descripción

Repositorio reorganizado siguiendo principios **Clean Code** y **SOLID** para los trabajos prácticos de la materia Fundamentos de Redes Neuronales de la Universidad Nacional de Hurlingham (UNAHUR).

## 🏗️ Nueva Estructura Organizada

```
fundamentos-redes-neuronales/
├── comun/                           # Módulos compartidos
│   ├── constantes/                  # Constantes centralizadas
│   │   ├── constantes_redes_neuronales.py
│   │   └── __init__.py
│   ├── src/                         # Código fuente compartido
│   │   ├── funciones_activacion.py  # Funciones de activación (SRP)
│   │   ├── utilidades_matematicas.py # Utilidades matemáticas (SRP)
│   │   ├── evaluador_rendimiento.py # Evaluación de rendimiento (SRP)
│   │   └── __init__.py
│   └── __init__.py
├── tp1/                             # Perceptrón Simple
│   ├── src/                         # Código fuente TP1
│   │   ├── perceptron_simple.py     # Implementación perceptrón simple
│   │   ├── cargador_datos.py        # Carga de datos (SRP)
│   │   ├── entrenador_compuertas.py # Entrenamiento compuertas (SRP)
│   │   ├── visualizador_resultados.py # Visualización (SRP)
│   │   ├── main_tp1.py             # Ejecutor principal
│   │   └── __init__.py
│   ├── datos/                       # Datos de TP1
│   ├── resultados/                  # Resultados de experimentos
│   ├── tests/                       # Tests unitarios
│   └── README.md                    # Documentación TP1
├── tp2/                             # Perceptrón Multicapa
│   ├── src/                         # Código fuente TP2
│   │   ├── perceptron_multicapa.py  # Implementación multicapa
│   │   ├── cargador_datos_digitos.py # Carga datos dígitos (SRP)
│   │   ├── entrenador_tp2.py        # Entrenamiento TP2 (SRP)
│   │   ├── main_tp2.py             # Ejecutor principal
│   │   └── __init__.py
│   ├── datos/                       # Datos de TP2
│   ├── resultados/                  # Resultados de experimentos
│   ├── tests/                       # Tests unitarios
│   └── README.md                    # Documentación TP2
└── README.md                        # Esta documentación
```

## 🎯 Principios Aplicados

### Clean Code
- ✅ **Eliminación de números mágicos**: Todas las constantes centralizadas
- ✅ **Nombres descriptivos**: Variables y métodos con nombres claros en español
- ✅ **Funciones pequeñas**: Cada función tiene una responsabilidad específica
- ✅ **Separación de concerns**: Lógica separada por responsabilidades
- ✅ **Código autodocumentado**: Nombres explicativos y documentación clara

### Principios SOLID

#### Single Responsibility Principle (SRP)
- `PerceptronSimple` / `PerceptronMulticapa`: Solo lógica de redes neuronales
- `CargadorDatos` / `CargadorDatosDigitos`: Solo carga y preparación de datos
- `EntrenadorCompuertas` / `EntrenadorTP2`: Solo lógica de entrenamiento
- `VisualizadorResultados`: Solo presentación y visualización
- `EvaluadorRendimiento`: Solo métricas y evaluación
- `FuncionesActivacion`: Solo funciones de activación
- `UtilidadesMatematicas`: Solo operaciones matemáticas

#### Open/Closed Principle (OCP)
- Extensible para nuevas funciones de activación sin modificar código existente
- Nuevas arquitecturas de red pueden agregarse sin cambiar implementaciones base
- Nuevos tipos de problemas pueden añadirse extendiendo los entrenadores

#### Dependency Inversion Principle (DIP)
- Inyección de dependencias entre componentes principales
- Abstracciones no dependen de detalles concretos
- Fácil intercambio de implementaciones para testing

## 🚀 Uso Rápido

### TP1 - Perceptrón Simple
```python
from tp1.src.main_tp1 import EjecutorTP1

# Ejecutar todos los experimentos del TP1
ejecutor = EjecutorTP1()
ejecutor.ejecutar_todos_los_experimentos()
```

### TP2 - Perceptrón Multicapa
```python
from tp2.src.main_tp2 import EjecutorTP2

# Ejecutar todos los experimentos del TP2
ejecutor = EjecutorTP2()
ejecutor.ejecutar_todos_los_experimentos()
```

## 📚 Experimentos Incluidos

### TP1 - Perceptrón Simple
- **Compuertas Lógicas**: AND, OR, XOR (análisis de separabilidad lineal)
- **Funciones de Activación**: Escalón, sigmoide, lineal
- **Datos desde Archivo**: Análisis de generalización
- **Comparación**: Perceptrón lineal vs no lineal

### TP2 - Perceptrón Multicapa
- **Problema XOR**: Demostración de capacidad no lineal
- **Discriminación Pares**: Clasificación binaria de dígitos
- **Clasificación 10 Clases**: Reconocimiento multiclase de dígitos
- **Evaluación con Ruido**: Análisis de robustez
- **Comparación Arquitecturas**: Diferentes configuraciones de red

## 🔧 Componentes Compartidos

### Constantes Centralizadas
```python
from comun.constantes.constantes_redes_neuronales import (
    TASA_APRENDIZAJE_DEFECTO,
    EPOCAS_MAXIMAS_DEFECTO,
    ARQUITECTURAS_TP2
)
```

### Utilidades Matemáticas
```python
from comun.src.utilidades_matematicas import UtilidadesMatematicas

# Inicialización de pesos
pesos = UtilidadesMatematicas.inicializar_pesos_xavier(filas, columnas)

# División de datos
train_x, train_y, test_x, test_y = UtilidadesMatematicas.dividir_datos_entrenamiento_prueba(X, y)
```

### Funciones de Activación
```python
from comun.src.funciones_activacion import FuncionesActivacion

# Obtener función y su derivada
funcion, derivada = FuncionesActivacion.obtener_funcion_y_derivada('sigmoide')
```

## 📊 Resultados y Análisis

### Conclusiones TP1
- **Separabilidad Lineal**: Perceptrón simple solo resuelve problemas linealmente separables
- **XOR**: Requiere arquitecturas más complejas (multicapa)
- **Generalización**: Validación independiente es crucial

### Conclusiones TP2
- **Superioridad Multicapa**: Resuelve problemas no linealmente separables
- **Sobreajuste**: Arquitecturas complejas con datos limitados memorizan sin generalizar
- **Robustez**: Redes bien entrenadas son resistentes a ruido pequeño

## 🛠️ Instalación y Dependencias

```bash
# Clonar repositorio
git clone <repository-url>
cd fundamentos-redes-neuronales

# Instalar dependencias
pip install numpy

# Ejecutar experimentos
python tp1/src/main_tp1.py
python tp2/src/main_tp2.py
```

## 📈 Mejoras Implementadas

### Antes de la Reorganización
- ❌ Código monolítico en archivos grandes
- ❌ Números mágicos dispersos por el código
- ❌ Responsabilidades mezcladas en una sola clase
- ❌ Difícil mantenimiento y extensión
- ❌ Duplicación de código entre proyectos

### Después de la Reorganización
- ✅ **Separación clara de responsabilidades** (SRP)
- ✅ **Constantes centralizadas** (eliminación de números mágicos)
- ✅ **Código reutilizable** entre TP1 y TP2
- ✅ **Fácil extensión** para nuevas funcionalidades (OCP)
- ✅ **Mantenimiento simplificado** (cada clase tiene un propósito claro)
- ✅ **Testing independiente** de componentes
- ✅ **Documentación completa** en español

## 🧪 Testing

```python
# Ejemplo de test unitario para componente individual
from tp1.src.perceptron_simple import PerceptronSimple

def test_perceptron_and():
    perceptron = PerceptronSimple(num_entradas=2, funcion_activacion='escalon')
    # Test específico para compuerta AND
    assert perceptron is not None
```

## 🤝 Contribución

Para mantener la calidad del código reorganizado:

1. **Seguir principios SOLID**: Cada clase debe tener una responsabilidad única
2. **Usar constantes centralizadas**: No agregar números mágicos al código
3. **Documentar en español**: Mantener consistencia en el idioma
4. **Separar responsabilidades**: No mezclar lógica de datos, entrenamiento y visualización
5. **Agregar tests**: Para nuevas funcionalidades

## 📝 Licencia

Proyecto académico - Universidad Nacional de Hurlingham (UNAHUR)
Materia: Fundamentos de Redes Neuronales

## 👥 Autores

- **Estudiantes**: Sebastián Brandariz, Mauricio Challiol, Tomás Toloza
- **Docente**: Emiliano Churruca
- **Reorganización**: Aplicación de principios Clean Code y SOLID

---

## 🎓 Valor Académico

Esta reorganización demuestra:

- **Aplicación práctica** de principios de ingeniería de software
- **Código mantenible** y extensible para proyectos académicos
- **Separación de concerns** en proyectos de machine learning
- **Reutilización de código** entre diferentes experimentos
- **Documentación profesional** de proyectos técnicos

El código resultante es más **legible**, **mantenible** y **extensible**, siguiendo las mejores prácticas de la industria del software.
