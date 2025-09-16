import numpy as np
from perceptron import Perceptron
from perceptron_printer import PerceptronPrinter
from training_config import get_config


def entrenar_compuerta_base(
    funcion_activacion, 
    funcion_derivada, 
    tipo_descripcion, 
    descripcion_entrenamiento, 
    es_entero, 
    salidas_esperadas,
    nombre_compuerta,
    config=None
):
    print(f"ENTRENANDO PERCEPTRÓN {descripcion_entrenamiento} PARA COMPUERTA {nombre_compuerta}")
    
    entradas = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    if config is None:
        config = get_config('gate')
    
    tasa_aprendizaje = config.tasa_aprendizaje
    max_epocas = config.max_epocas
    error_min = config.error_min

    printer = PerceptronPrinter()
    printer.imprimir_configuracion(tipo_descripcion, tasa_aprendizaje, max_epocas, error_min, len(entradas))

    perceptron = Perceptron(
        num_entradas=2,
        tasa_aprendizaje=tasa_aprendizaje,
        max_epocas=max_epocas,
        error_min=error_min,
        verbose=config.verbose,
        random_seed=config.random_seed
    )

    resultado = perceptron.entrenar(entradas, salidas_esperadas, funcion_activacion, funcion_derivada, tipo_descripcion)

    perceptron.mostrar_resultados_entrenamiento(resultado, entradas, salidas_esperadas, nombre_compuerta, es_entero=es_entero)

    return perceptron
