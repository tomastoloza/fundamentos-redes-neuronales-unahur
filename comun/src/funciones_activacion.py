import numpy as np
from typing import Callable

class FuncionesActivacion:

    @staticmethod
    def escalon(x: np.ndarray) -> np.ndarray:

        return np.where(x >= 0, 1, -1)
    
    @staticmethod
    def escalon_derivada(x: np.ndarray) -> np.ndarray:

        return np.zeros_like(x)
    
    @staticmethod
    def sigmoide(x: np.ndarray) -> np.ndarray:

        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def sigmoide_derivada(x: np.ndarray) -> np.ndarray:

        return x * (1 - x)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:

        return np.tanh(x)
    
    @staticmethod
    def tanh_derivada(x: np.ndarray) -> np.ndarray:

        return 1 - x * x
    
    @staticmethod
    def lineal(x: np.ndarray) -> np.ndarray:

        return x
    
    @staticmethod
    def lineal_derivada(x: np.ndarray) -> np.ndarray:

        return np.ones_like(x)
    
    @classmethod
    def obtener_funcion_y_derivada(cls, nombre_funcion: str) -> tuple[Callable, Callable]:

        funciones_disponibles = {
            'escalon': (cls.escalon, cls.escalon_derivada),
            'sigmoide': (cls.sigmoide, cls.sigmoide_derivada),
            'tanh': (cls.tanh, cls.tanh_derivada),
            'lineal': (cls.lineal, cls.lineal_derivada)
        }
        
        if nombre_funcion not in funciones_disponibles:
            raise ValueError(f"Función de activación '{nombre_funcion}' no disponible. "
                           f"Opciones: {list(funciones_disponibles.keys())}")
        
        return funciones_disponibles[nombre_funcion]
