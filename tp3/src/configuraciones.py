CONFIGURACIONES_AUTOCODIFICADOR = {
    'simple_2d': {
        'capas_encoder': [20, 10],
        'dimension_latente': 2,
        'capas_decoder': [10, 20],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 16
    },
    
    'profundo_2d': {
        'capas_encoder': [30, 25, 15],
        'dimension_latente': 2,
        'capas_decoder': [15, 25, 30],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 16
    },
    
    'minimo_2d': {
        'capas_encoder': [12],
        'dimension_latente': 2,
        'capas_decoder': [12],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 16
    },
    
    'ultra_profundo_2d': {
        'capas_encoder': [32, 28, 24, 20, 16, 12, 8],
        'dimension_latente': 2,
        'capas_decoder': [8, 12, 16, 20, 24, 28, 32],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 16
    },
    
    'ultra_ancho_2d': {
        'capas_encoder': [64, 32],
        'dimension_latente': 2,
        'capas_decoder': [32, 64],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 32
    },
    
    'tanh_2d': {
        'capas_encoder': [25, 15],
        'dimension_latente': 2,
        'capas_decoder': [15, 25],
        'activacion': 'tanh',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 16
    }
}

CONFIGURACIONES_ENTRENAMIENTO = {
    # 'rapido': {
    #     'epochs': 300,
    #     'patience': 100,
    #     'validation_split': 0.15
    # },
    
    # 'normal': {
    #     'epochs': 800,
    #     'patience': 100,
    #     'validation_split': 0.15
    # },
    
    'exhaustivo': {
        'epochs': 1500,
        'patience': 100,
        'validation_split': 0.15
    }
}

def obtener_configuracion(nombre):
    if nombre in CONFIGURACIONES_AUTOCODIFICADOR:
        return CONFIGURACIONES_AUTOCODIFICADOR[nombre]
    else:
        raise ValueError(f"Configuración '{nombre}' no encontrada")

def obtener_configuracion_entrenamiento(nombre):
    if nombre in CONFIGURACIONES_ENTRENAMIENTO:
        return CONFIGURACIONES_ENTRENAMIENTO[nombre]
    else:
        raise ValueError(f"Configuración de entrenamiento '{nombre}' no encontrada")

def listar_configuraciones():
    print("Configuraciones de autocodificador disponibles:")
    for nombre, config in CONFIGURACIONES_AUTOCODIFICADOR.items():
        print(f"  - {nombre}: {config['dimension_latente']}D, encoder {config['capas_encoder']}")
    
    print("\nConfiguraciones de entrenamiento disponibles:")
    for nombre, config in CONFIGURACIONES_ENTRENAMIENTO.items():
        print(f"  - {nombre}: {config['epochs']} épocas, paciencia {config['patience']}")
