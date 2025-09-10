import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-2 * x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return 2 * s * (1 - s)


def step(x):
    return 1 if x >= 0 else 0


def step_derivative(x):
    return 1


def linear(x):
    return x


def linear_derivative(x):
    return 1


ACTIVATION_FUNCTIONS = {
    'sigmoid': {
        'function': sigmoid,
        'derivative': sigmoid_derivative,
        'name': 'NO LINEAL (Sigmoide)'
    },
    'step': {
        'function': step,
        'derivative': step_derivative,
        'name': 'LINEAL (Escalón)'
    },
    'linear': {
        'function': linear,
        'derivative': linear_derivative,
        'name': 'REGRESIÓN (Lineal)'
    }
}


def get_activation_function(name):
    if name not in ACTIVATION_FUNCTIONS:
        available = ', '.join(ACTIVATION_FUNCTIONS.keys())
        raise ValueError(f"Función de activación '{name}' no disponible. Disponibles: {available}")
    
    func_info = ACTIVATION_FUNCTIONS[name]
    return func_info['function'], func_info['derivative'], func_info['name']
