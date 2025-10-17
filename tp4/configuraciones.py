CONFIGURACIONES_AUTOCODIFICADOR = {
    'simple_64x64': {
        'capas_encoder': [128, 64, 32],
        'dimension_latente': 32,
        'capas_decoder': [32, 64, 128],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 16,
        'tamaño_imagen': (64, 64)
    },
    
    'profundo_64x64': {
        'capas_encoder': [256, 128, 64, 32],
        'dimension_latente': 16,
        'capas_decoder': [32, 64, 128, 256],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 16,
        'tamaño_imagen': (64, 64)
    },
    
    'compacto_32x32': {
        'capas_encoder': [64, 32],
        'dimension_latente': 8,
        'capas_decoder': [32, 64],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 32,
        'tamaño_imagen': (32, 32)
    },
    
    'ultra_profundo_128x128': {
        'capas_encoder': [512, 256, 128, 64, 32, 16],
        'dimension_latente': 8,
        'capas_decoder': [16, 32, 64, 128, 256, 512],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.0005,
        'batch_size': 8,
        'tamaño_imagen': (128, 128)
    },
    
    'ancho_64x64': {
        'capas_encoder': [512, 256],
        'dimension_latente': 64,
        'capas_decoder': [256, 512],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 16,
        'tamaño_imagen': (64, 64)
    },
    
    'tanh_64x64': {
        'capas_encoder': [128, 64, 32],
        'dimension_latente': 16,
        'capas_decoder': [32, 64, 128],
        'activacion': 'tanh',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 16,
        'tamaño_imagen': (64, 64)
    },
    
    'simple_200x200': {
        'capas_encoder': [1024, 512, 256],
        'dimension_latente': 64,
        'capas_decoder': [256, 512, 1024],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.0005,
        'batch_size': 8,
        'tamaño_imagen': (200, 200)
    },
    
    'profundo_200x200': {
        'capas_encoder': [2048, 1024, 512, 256, 128],
        'dimension_latente': 32,
        'capas_decoder': [128, 256, 512, 1024, 2048],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.0003,
        'batch_size': 4,
        'tamaño_imagen': (200, 200)
    },
    
    'compacto_200x200': {
        'capas_encoder': [512, 256],
        'dimension_latente': 128,
        'capas_decoder': [256, 512],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.0005,
        'batch_size': 8,
        'tamaño_imagen': (200, 200)
    }
}

CONFIGURACIONES_ENTRENAMIENTO = {
    'rapido': {
        'epochs': 50,
        'patience': 20,
        'validation_split': 0.2
    },
    
    'normal': {
        'epochs': 100,
        'patience': 30,
        'validation_split': 0.2
    },
    
    'exhaustivo': {
        'epochs': 200,
        'patience': 50,
        'validation_split': 0.2
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

def obtener_configuraciones_disponibles():
    return {
        'autocodificador': list(CONFIGURACIONES_AUTOCODIFICADOR.keys()),
        'entrenamiento': list(CONFIGURACIONES_ENTRENAMIENTO.keys())
    }

def listar_configuraciones():
    print("Configuraciones de autocodificador para imágenes disponibles:")
    for nombre, config in CONFIGURACIONES_AUTOCODIFICADOR.items():
        tamaño = config['tamaño_imagen']
        print(f"  - {nombre}: {config['dimension_latente']}D, {tamaño[0]}x{tamaño[1]}, encoder {config['capas_encoder']}")
    
    print("\nConfiguraciones de entrenamiento disponibles:")
    for nombre, config in CONFIGURACIONES_ENTRENAMIENTO.items():
        print(f"  - {nombre}: {config['epochs']} épocas, paciencia {config['patience']}")
