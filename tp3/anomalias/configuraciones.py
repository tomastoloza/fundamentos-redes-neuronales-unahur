"""
Configuraciones para grid search de detección de anomalías en consumo energético.
"""

# Configuraciones de arquitectura para autocodificadores de anomalías
CONFIGURACIONES_ARQUITECTURA = {
    'basica': {
        'longitud_serie': 168,
        'dimension_latente': 16,
        'activacion_salida': 'linear'
    },
    'compacta': {
        'longitud_serie': 168,
        'dimension_latente': 8,
        'activacion_salida': 'linear'
    },
    'amplia': {
        'longitud_serie': 168,
        'dimension_latente': 32,
        'activacion_salida': 'linear'
    },
    'minima': {
        'longitud_serie': 168,
        'dimension_latente': 4,
        'activacion_salida': 'linear'
    },
    'profunda': {
        'longitud_serie': 168,
        'dimension_latente': 24,
        'activacion_salida': 'linear'
    }
}

# Configuraciones de entrenamiento
CONFIGURACIONES_ENTRENAMIENTO = {
    'rapido': {
        'epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 32,
        'patience': 10
    },
    'medio': {
        'epochs': 100,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'patience': 15
    },
    'completo': {
        'epochs': 200,
        'learning_rate': 0.0001,
        'batch_size': 16,
        'patience': 20
    },
    'intensivo': {
        'epochs': 300,
        'learning_rate': 0.00005,
        'batch_size': 16,
        'patience': 25
    }
}

# Configuraciones de datos para grid search
CONFIGURACIONES_DATOS = {
    'pequeno': {
        'num_entrenamiento': 500,
        'num_prueba_normal': 100,
        'num_prueba_anomala': 25
    },
    'medio': {
        'num_entrenamiento': 1000,
        'num_prueba_normal': 200,
        'num_prueba_anomala': 50
    },
    'grande': {
        'num_entrenamiento': 2000,
        'num_prueba_normal': 400,
        'num_prueba_anomala': 100
    }
}

# Configuraciones de evaluación
CONFIGURACIONES_EVALUACION = {
    'basica': {
        'percentil_umbral': 95,
        'mostrar_visualizaciones': False
    },
    'estricta': {
        'percentil_umbral': 99,
        'mostrar_visualizaciones': False
    },
    'permisiva': {
        'percentil_umbral': 90,
        'mostrar_visualizaciones': False
    }
}
