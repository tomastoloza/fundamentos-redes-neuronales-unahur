CONFIGURACIONES_AUTOCODIFICADOR = {
    'simple_2d': {
        'capas_encoder': [512, 256, 128],
        'dimension_latente': 2,
        'capas_decoder': [128, 256, 512],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 8
    },
    
    'profundo_2d': {
        'capas_encoder': [768, 512, 384, 256, 128],
        'dimension_latente': 2,
        'capas_decoder': [128, 256, 384, 512, 768],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 8
    },
    
    'minimo_2d': {
        'capas_encoder': [256, 128],
        'dimension_latente': 2,
        'capas_decoder': [128, 256],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 8
    },
    
    'ultra_profundo_2d': {
        'capas_encoder': [896, 768, 640, 512, 384, 256, 128, 64],
        'dimension_latente': 2,
        'capas_decoder': [64, 128, 256, 384, 512, 640, 768, 896],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 8
    },
    
    'ultra_ancho_2d': {
        'capas_encoder': [1536, 768],
        'dimension_latente': 2,
        'capas_decoder': [768, 1536],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 16
    },
    
    'tanh_2d': {
        'capas_encoder': [640, 384, 128],
        'dimension_latente': 2,
        'capas_decoder': [128, 384, 640],
        'activacion': 'tanh',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 8
    },
    
    'compacto_5d': {
        'capas_encoder': [512, 256, 128],
        'dimension_latente': 5,
        'capas_decoder': [128, 256, 512],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 8
    },
    
    'profundo_8d': {
        'capas_encoder': [768, 512, 384, 256],
        'dimension_latente': 8,
        'capas_decoder': [256, 384, 512, 768],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 8
    },
    
    'ancho_10d': {
        'capas_encoder': [1024, 512],
        'dimension_latente': 10,
        'capas_decoder': [512, 1024],
        'activacion': 'relu',
        'activacion_salida': 'sigmoid',
        'learning_rate': 0.001,
        'batch_size': 8
    }
}

CONFIGURACIONES_ENTRENAMIENTO = {
    'rapido': {
        'epochs': 500,
        'patience': 150,
        'validation_split': 0.15,
        'early_stopping': True,
        'monitor': 'loss',
        'verbose': 1
    },
    
    'normal': {
        'epochs': 1500,
        'patience': 200,
        'validation_split': 0.15,
        'early_stopping': True,
        'monitor': 'loss',
        'verbose': 1
    },
    
    'exhaustivo': {
        'epochs': 3000,
        'patience': 300,
        'validation_split': 0.15,
        'early_stopping': True,
        'monitor': 'loss',
        'verbose': 1
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
    print("Configuraciones de autocodificador disponibles:")
    for nombre, config in CONFIGURACIONES_AUTOCODIFICADOR.items():
        print(f"  - {nombre}: {config['dimension_latente']}D, encoder {config['capas_encoder']}")
    
    print("\nConfiguraciones de entrenamiento disponibles:")
    for nombre, config in CONFIGURACIONES_ENTRENAMIENTO.items():
        print(f"  - {nombre}: {config['epochs']} épocas, paciencia {config['patience']}")
