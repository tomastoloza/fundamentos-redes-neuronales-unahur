import numpy as np
from typing import Callable, Tuple, Optional
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from comun.src.funciones_activacion import FuncionesActivacion
from comun.src.utilidades_matematicas import UtilidadesMatematicas
from comun.src.evaluador_rendimiento import EvaluadorRendimiento
from comun.constantes.constantes_redes_neuronales import (
    TASA_APRENDIZAJE_DEFECTO, EPOCAS_MAXIMAS_DEFECTO, ERROR_OBJETIVO_DEFECTO,
    RANGO_PESO_MINIMO, RANGO_PESO_MAXIMO, MENSAJE_CONVERGENCIA, MENSAJE_NO_CONVERGENCIA
)

class PerceptronSimple:

    def __init__(self, num_entradas: int, funcion_activacion: str = 'escalon'):

        self.num_entradas = num_entradas
        self.pesos = UtilidadesMatematicas.inicializar_pesos_aleatorios(
            num_entradas + 1, 1, RANGO_PESO_MINIMO, RANGO_PESO_MAXIMO
        ).flatten()
        
        self.funcion_activacion, self.derivada_activacion = (
            FuncionesActivacion.obtener_funcion_y_derivada(funcion_activacion)
        )
        
        self.nombre_funcion = funcion_activacion
        self.evaluador = EvaluadorRendimiento()
        self.convergencia_alcanzada = False
        self.epoca_convergencia = 0
    
    def _agregar_sesgo(self, entradas: np.ndarray) -> np.ndarray:

        if entradas.ndim == 1:
            return np.append(entradas, 1)
        else:
            sesgos = np.ones((entradas.shape[0], 1))
            return np.hstack([entradas, sesgos])
    
    def _calcular_salida_neta(self, entradas_con_sesgo: np.ndarray) -> float:

        return np.dot(entradas_con_sesgo, self.pesos)
    
    def predecir(self, entradas: np.ndarray) -> np.ndarray:

        entradas_con_sesgo = self._agregar_sesgo(entradas)
        
        if entradas.ndim == 1:
            salida_neta = self._calcular_salida_neta(entradas_con_sesgo)
            return self.funcion_activacion(np.array([salida_neta]))[0]
        else:
            predicciones = []
            for entrada in entradas_con_sesgo:
                salida_neta = self._calcular_salida_neta(entrada)
                prediccion = self.funcion_activacion(np.array([salida_neta]))[0]
                predicciones.append(prediccion)
            return np.array(predicciones)
    
    def _actualizar_pesos(self, entrada_con_sesgo: np.ndarray, error: float, 
                         tasa_aprendizaje: float) -> None:

        self.pesos += tasa_aprendizaje * error * entrada_con_sesgo
    
    def entrenar(self, entradas: np.ndarray, salidas_esperadas: np.ndarray,
                tasa_aprendizaje: float = TASA_APRENDIZAJE_DEFECTO,
                max_epocas: int = EPOCAS_MAXIMAS_DEFECTO,
                error_objetivo: float = ERROR_OBJETIVO_DEFECTO,
                mostrar_progreso: bool = True) -> Tuple[bool, int]:

        self.evaluador.limpiar_historial()
        
        for epoca in range(max_epocas):
            error_total = 0.0
            
            for i, entrada in enumerate(entradas):
                entrada_con_sesgo = self._agregar_sesgo(entrada)
                prediccion = self.predecir(entrada)
                error = salidas_esperadas[i] - prediccion
                
                if abs(error) > 1e-10:
                    self._actualizar_pesos(entrada_con_sesgo, error, tasa_aprendizaje)
                
                error_total += abs(error)
            
            error_promedio = error_total / len(entradas)
            self.evaluador.registrar_error(error_promedio)
            
            if error_promedio <= error_objetivo:
                self.convergencia_alcanzada = True
                self.epoca_convergencia = epoca + 1
                if mostrar_progreso:
                    print(f"{MENSAJE_CONVERGENCIA} Época: {self.epoca_convergencia}")
                return True, self.epoca_convergencia
            
            if mostrar_progreso and epoca % 1000 == 0:
                print(f"Época {epoca}: Error promedio = {error_promedio:.6f}")
        
        if mostrar_progreso:
            print(MENSAJE_NO_CONVERGENCIA)
        return False, max_epocas
    
    def evaluar(self, entradas: np.ndarray, salidas_esperadas: np.ndarray) -> dict:

        predicciones = np.array([self.predecir(entrada) for entrada in entradas])
        
        if self.nombre_funcion == 'escalon':
            return self.evaluador.evaluar_clasificacion_binaria(predicciones, salidas_esperadas)
        else:
            error_cuadratico = UtilidadesMatematicas.calcular_error_cuadratico_medio(
                salidas_esperadas, predicciones
            )
            precision = UtilidadesMatematicas.calcular_precision(predicciones, salidas_esperadas)
            
            return {
                'error_cuadratico_medio': error_cuadratico,
                'precision': precision,
                'predicciones': predicciones
            }
    
    def obtener_informacion_entrenamiento(self) -> dict:

        info = {
            'convergencia_alcanzada': self.convergencia_alcanzada,
            'epoca_convergencia': self.epoca_convergencia,
            'funcion_activacion': self.nombre_funcion,
            'num_entradas': self.num_entradas,
            'pesos_finales': self.pesos.copy()
        }
        
        estadisticas = self.evaluador.obtener_estadisticas_entrenamiento()
        info.update(estadisticas)
        
        return info
