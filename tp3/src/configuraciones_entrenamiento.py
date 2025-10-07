from typing import List


def obtener_configuracion_entrenamiento():
    return {
        # Aumentamos el máximo para asegurar tiempo para la convergencia lenta.
        'epocas_lista': [8000, 10000],

        # Enfocados en tasas muy precisas para evitar saltar el mínimo global.
        # Incluimos 0.00075 para mayor precisión.
        'tasas_aprendizaje': [0.0005, 0.00075, 0.001],

        # Las dos mejores dimensiones latentes de las pruebas anteriores.
        'dimensiones_latentes': [8, 9, 10],

        # Mantenemos el objetivo de error.
        'error_objetivo': 0.05,

        # Aumentamos la paciencia. Esto es CRÍTICO. Si la pérdida disminuye muy lentamente,
        # necesitamos más margen antes de que se active el Early Stopping.
        'paciencia': 400,

        'tipo_perdida': 'bce',

        'monitor': 'val_loss',

        'usar_scheduler': True,

        'batch_size': 32
    }
def obtener_epocas() -> List[int]:
    return obtener_configuracion_entrenamiento()['epocas_lista']


def obtener_dimensiones_latentes() -> List[int]:
    return obtener_configuracion_entrenamiento()['dimensiones_latentes']


def obtener_tasas_aprendizaje() -> List[float]:
    return obtener_configuracion_entrenamiento()['tasas_aprendizaje']


def obtener_error_objetivo() -> float:
    return obtener_configuracion_entrenamiento()['error_objetivo']


def obtener_paciencia() -> int:
    return obtener_configuracion_entrenamiento()['paciencia']


def obtener_tipo_perdida() -> str:
    return obtener_configuracion_entrenamiento()['tipo_perdida']


def obtener_monitor() -> str:
    return obtener_configuracion_entrenamiento()['monitor']


def obtener_usar_scheduler() -> bool:
    return obtener_configuracion_entrenamiento()['usar_scheduler']


def obtener_batch_size() -> int:
    return obtener_configuracion_entrenamiento()['batch_size']
