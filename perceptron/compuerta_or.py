import numpy as np
from activation_functions import sigmoid, sigmoid_derivative, step, step_derivative
from gate_trainer import entrenar_compuerta_base


def _entrenar_compuerta_or_base(funcion_activacion, funcion_derivada, tipo_descripcion, descripcion_entrenamiento, es_entero, config=None):
    salidas_or = np.array([0, 1, 1, 1])
    
    return entrenar_compuerta_base(
        funcion_activacion=funcion_activacion,
        funcion_derivada=funcion_derivada,
        tipo_descripcion=tipo_descripcion,
        descripcion_entrenamiento=descripcion_entrenamiento,
        es_entero=es_entero,
        salidas_esperadas=salidas_or,
        nombre_compuerta="OR",
        config=config
    )


def entrenar_compuerta_or_lineal(config=None):
    return _entrenar_compuerta_or_base(
        step, 
        step_derivative, 
        "Lineal (Escalón)", 
        "LINEAL", 
        es_entero=True,
        config=config
    )


def entrenar_compuerta_or_no_lineal(config=None):
    return _entrenar_compuerta_or_base(
        sigmoid, 
        sigmoid_derivative, 
        "No Lineal (Sigmoide)", 
        "NO LINEAL", 
        es_entero=False,
        config=config
    )
