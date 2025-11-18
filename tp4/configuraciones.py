CONFIGURACIONES_CNN = {
    'simple_200x200': {
        'capas_conv': [
            {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'}
        ],
        'capas_dense': [256, 128],
        'num_clases': 8,
        'activacion_salida': 'softmax',
        'learning_rate': 0.001,
        'batch_size': 16,
        'tamaño_imagen': (200, 200),
        'dropout_rate': 0.5
    },
    
    'profunda_200x200': {
        'capas_conv': [
            {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 512, 'kernel_size': (3, 3), 'activation': 'relu'}
        ],
        'capas_dense': [512, 256, 128],
        'num_clases': 8,
        'activacion_salida': 'softmax',
        'learning_rate': 0.0005,
        'batch_size': 8,
        'tamaño_imagen': (200, 200),
        'dropout_rate': 0.5
    },
    
    'compacta_200x200': {
        'capas_conv': [
            {'filters': 32, 'kernel_size': (5, 5), 'activation': 'relu'},
            {'filters': 64, 'kernel_size': (5, 5), 'activation': 'relu'}
        ],
        'capas_dense': [128],
        'num_clases': 8,
        'activacion_salida': 'softmax',
        'learning_rate': 0.001,
        'batch_size': 32,
        'tamaño_imagen': (200, 200),
        'dropout_rate': 0.3
    },
    
    'ligera_200x200': {
        'capas_conv': [
            {'filters': 16, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'}
        ],
        'capas_dense': [128],
        'num_clases': 8,
        'activacion_salida': 'softmax',
        'learning_rate': 0.001,
        'batch_size': 32,
        'tamaño_imagen': (200, 200),
        'dropout_rate': 0.4
    },
    
    'ancha_200x200': {
        'capas_conv': [
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu'}
        ],
        'capas_dense': [512, 256],
        'num_clases': 8,
        'activacion_salida': 'softmax',
        'learning_rate': 0.0005,
        'batch_size': 16,
        'tamaño_imagen': (200, 200),
        'dropout_rate': 0.5
    },
    
    'experimental_200x200': {
        'capas_conv': [
            {'filters': 32, 'kernel_size': (7, 7), 'activation': 'relu'},
            {'filters': 64, 'kernel_size': (5, 5), 'activation': 'relu'},
            {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu'}
        ],
        'capas_dense': [512, 256],
        'num_clases': 8,
        'activacion_salida': 'softmax',
        'learning_rate': 0.0003,
        'batch_size': 8,
        'tamaño_imagen': (200, 200),
        'dropout_rate': 0.6
    },
    
    'balanceada_200x200': {
        'capas_conv': [
            {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu', 'pool': (2, 2), 'bn': True},
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu', 'pool': (2, 2), 'bn': True},
            {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu', 'pool': (2, 2), 'bn': True}
        ],
        'capas_dense': [256],
        'num_clases': 8,
        'activacion_salida': 'softmax',
        'learning_rate': 0.0001,
        'batch_size': 32,
        'tamaño_imagen': (200, 200),
        'dropout_rate': 0.3
    },
    
    'moderada_profundidad_200x200': {
        'capas_conv': [
            {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu', 'pool': (2, 2), 'bn': True},
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu', 'pool': (2, 2), 'bn': True},
            {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu', 'pool': (2, 2), 'bn': True},
            {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu', 'pool': (2, 2), 'bn': True}
        ],
        'capas_dense': [512],
        'num_clases': 8,
        'activacion_salida': 'softmax',
        'learning_rate': 0.0007,
        'batch_size': 32,
        'tamaño_imagen': (200, 200),
        'dropout_rate': 0.4
    }, 

    'optimizada_desde_cero': {
        'capas_conv': [
            {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu', 'bn': True, 'dropout_2d': 0.2, 'pool': (2, 2)},
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu', 'bn': True, 'dropout_2d': 0.2, 'pool': (2, 2)},
            {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu', 'bn': True, 'dropout_2d': 0.2, 'pool': (2, 2)}
        ],
        'capas_dense': [256], # Mantenemos la capa densa moderada
        'num_clases': 8,
        'activacion_salida': 'softmax',
        'learning_rate': 0.0002, # Un punto medio entre 0.0001 y 0.001
        'batch_size': 32,
        'tamaño_imagen': (200, 200),
        'dropout_rate': 0.4 # Dropout en la capa densa
    },

    'ancha_estabilizada': {
        'capas_conv': [
            # Bloque 1: Más ancho (64 filtros) + Estabilización completa
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu', 'bn': True, 'dropout_2d': 0.2, 'pool': (2, 2)},
            # Bloque 2: Más ancho (128 filtros) + Estabilización completa
            {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu', 'bn': True, 'dropout_2d': 0.2, 'pool': (2, 2)},
            # Bloque 3: Más ancho (256 filtros) + Estabilización completa
            {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu', 'bn': True, 'dropout_2d': 0.2, 'pool': (2, 2)}
        ],
        'capas_dense': [512, 256], # Usamos las dos capas densas originales para mayor capacidad de clasificación
        'num_clases': 8,
        'activacion_salida': 'softmax',
        'learning_rate': 0.0002, # Mantenemos el LR bajo que te dio estabilidad
        'batch_size': 32,
        'tamaño_imagen': (200, 200),
        'dropout_rate': 0.4 # Dropout en las capas densas
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

# https://gemini.google.com/u/0/app/96351ee83458b43e
CONFIGURACIONES_AUGMENTATION = {
    'ninguno': {
        # Sin augmentation - solo para comparación
    },
    
    'basico': {
        'horizontal_flip': True,
        'rotation_range': 10,
        'zoom_range': 0.1,
        'fill_mode': 'nearest'
    },
    
    'moderado': {
        'rotation_range': 20,
        'width_shift_range': 0.15,
        'height_shift_range': 0.15,
        'horizontal_flip': True,
        'zoom_range': 0.2,
        'brightness_range': [0.8, 1.2],
        'fill_mode': 'nearest'
    },
    
    'agresivo': {
        'rotation_range': 30,
        'width_shift_range': 0.25,
        'height_shift_range': 0.25,
        'horizontal_flip': True,
        'vertical_flip': False,  # Cuidado con animales - puede no ser natural
        'zoom_range': 0.3,
        'brightness_range': [0.7, 1.3],
        'shear_range': 0.2,
        'fill_mode': 'nearest'
    },
    
    'animales_optimizado': {
        # Configuración específica para clasificación de animales
        'rotation_range': 15,  # Rotación moderada
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'horizontal_flip': True,  # Los animales pueden estar en ambas direcciones
        'vertical_flip': False,  # Evitar voltear verticalmente (no natural)
        'zoom_range': 0.25,  # Simular diferentes distancias
        'brightness_range': [0.8, 1.2],  # Diferentes condiciones de luz
        'fill_mode': 'nearest'
    }
}

def obtener_configuracion(nombre):
    if nombre in CONFIGURACIONES_CNN:
        return CONFIGURACIONES_CNN[nombre]
    else:
        raise ValueError(f"Configuración '{nombre}' no encontrada")

def obtener_configuracion_entrenamiento(nombre):
    if nombre in CONFIGURACIONES_ENTRENAMIENTO:
        return CONFIGURACIONES_ENTRENAMIENTO[nombre]
    else:
        raise ValueError(f"Configuración de entrenamiento '{nombre}' no encontrada")

def obtener_configuracion_augmentation(nombre):
    if nombre in CONFIGURACIONES_AUGMENTATION:
        return CONFIGURACIONES_AUGMENTATION[nombre]
    else:
        raise ValueError(f"Configuración de augmentation '{nombre}' no encontrada")

def obtener_configuraciones_disponibles():
    return {
        'cnn': list(CONFIGURACIONES_CNN.keys()),
        'entrenamiento': list(CONFIGURACIONES_ENTRENAMIENTO.keys()),
        'augmentation': list(CONFIGURACIONES_AUGMENTATION.keys())
    }

def listar_configuraciones():
    print("Configuraciones de CNN para clasificación de animales disponibles:")
    for nombre, config in CONFIGURACIONES_CNN.items():
        tamaño = config['tamaño_imagen']
        num_conv = len(config['capas_conv'])
        num_dense = len(config['capas_dense'])
        print(f"  - {nombre}: {num_conv} capas conv, {num_dense} capas dense, {tamaño[0]}x{tamaño[1]}, {config['num_clases']} clases")
    
    print("\nConfiguraciones de entrenamiento disponibles:")
    for nombre, config in CONFIGURACIONES_ENTRENAMIENTO.items():
        print(f"  - {nombre}: {config['epochs']} épocas, paciencia {config['patience']}")
