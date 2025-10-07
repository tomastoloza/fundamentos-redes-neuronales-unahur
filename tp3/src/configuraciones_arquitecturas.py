from typing import Dict, Any


def obtener_arquitecturas_disponibles() -> Dict[str, Dict[str, Any]]:
    return {
        # 'TinyAE': {
        #     'codificador': [20, 10, 5],
        #     'activacion': 'relu',
        #     'descripcion': 'Mínima representación latente necesaria (5). Arquitectura poco profunda y máxima compresión.'
        # },
        'BalancedAE': {
            'codificador': [25, 15, 8],
            'activacion': 'tanh',
            'descripcion': 'Latente de 8 para mayor capacidad. Usa Tanh para evitar el "dead ReLU" y potencialmente mejores límites de decisión en el espacio latente.'
        },
        # 'DeepSparseAE': {
        #     'codificador': [30, 20, 15, 12], # 35 -> 30 -> 20 -> 15 -> 12
        #     'activacion': 'relu',
        #     'descripcion': 'Arquitectura más profunda (4 capas por lado). Latente de 12 para alta capacidad. Requiere la adición de regularización L1 o esparcimiento (sparse autoencoder) para funcionar bien.'
        # },
        'DeepBalancedAE': {
            'codificador': [30, 20, 10, 8], # 35 -> 30 -> 20 -> 10 -> 8
            'activacion': 'tanh',
            'descripcion': 'Arquitectura profunda (4 capas), similar a DeepSparseAE, pero con activación Tanh y menor latente (8) para estabilidad y menor riesgo de sobreajuste.'
        },
        'DeepReLU_L8': {
            'codificador': [30, 20, 10, 8], # 35 -> 30 -> 20 -> 10 -> 8
            'activacion': 'relu',
            'descripcion': 'Arquitectura profunda con latente medio (8), probando la activación ReLU en lugar de Tanh para ver si acelera la convergencia sin sacrificar estabilidad.'
        },

        'WideTanhAE': {
            'codificador': [25, 18, 12], # 35 -> 25 -> 18 -> 12
            'activacion': 'tanh',
            'descripcion': 'Arquitectura moderadamente profunda, pero con la máxima capacidad latente (12) y la activación Tanh para estabilidad. Busca la representación más rica posible.'
        },
        # Nueva variación del ganador: latente 9
        'BalancedAE_L9': {
            'codificador': [25, 15, 9], # 35 -> 25 -> 15 -> 9
            'activacion': 'tanh',
            'descripcion': 'Variación del modelo ganador (BalancedAE) con un latente ligeramente más pequeño (9) para probar el límite de compresión óptimo.'
        },

        # Prueba de fuego: La mejor estructura con la activación perdedora
        'WideReLU_L10': {
            'codificador': [25, 15, 10], # 35 -> 25 -> 15 -> 10
            'activacion': 'relu',
            'descripcion': 'La mejor arquitectura (capas 25, 15) con el latente ganador (10), pero forzando la activación ReLU para probar si acelera la convergencia sin sacrificar el MSE.'
        }
    }