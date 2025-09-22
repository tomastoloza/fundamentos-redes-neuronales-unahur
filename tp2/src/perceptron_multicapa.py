import numpy as np
from typing import List, Tuple, Optional, Callable
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from comun.src.funciones_activacion import FuncionesActivacion
from comun.src.utilidades_matematicas import UtilidadesMatematicas
from comun.src.evaluador_rendimiento import EvaluadorRendimiento
from comun.constantes.constantes_redes_neuronales import (
    TASA_APRENDIZAJE_DEFECTO, EPOCAS_MAXIMAS_DEFECTO, ERROR_OBJETIVO_DEFECTO,
    INTERVALO_IMPRESION_DEFECTO, MENSAJE_CONVERGENCIA, MENSAJE_NO_CONVERGENCIA
)

class CapaRed:
    
    def __init__(self, num_entradas: int, num_neuronas: int, 
                 funcion_activacion: str = 'sigmoide'):
        self.num_entradas = num_entradas
        self.num_neuronas = num_neuronas
        
        self.pesos = UtilidadesMatematicas.inicializar_pesos_xavier(
            num_entradas, num_neuronas
        )
        self.sesgos = np.zeros((1, num_neuronas))
        
        self.funcion_activacion, self.derivada_activacion = (
            FuncionesActivacion.obtener_funcion_y_derivada(funcion_activacion)
        )
        
        self.ultima_entrada = None
        self.ultima_salida_neta = None
        self.ultima_salida_activada = None
    
    def propagacion_adelante(self, entradas: np.ndarray) -> np.ndarray:
        self.ultima_entrada = entradas
        
        self.ultima_salida_neta = np.dot(entradas, self.pesos) + self.sesgos
        
        self.ultima_salida_activada = self.funcion_activacion(self.ultima_salida_neta)
        
        return self.ultima_salida_activada
    
    def calcular_gradientes(self, error_siguiente_capa: np.ndarray) -> np.ndarray:
        derivada = self.derivada_activacion(self.ultima_salida_activada)
        delta = error_siguiente_capa * derivada
        
        self.gradientes_pesos = np.dot(self.ultima_entrada.T, delta)
        self.gradientes_sesgos = np.sum(delta, axis=0, keepdims=True)
        
        error_anterior = np.dot(delta, self.pesos.T)
        
        return error_anterior
    
    def actualizar_pesos(self, tasa_aprendizaje: float) -> None:
        self.pesos += tasa_aprendizaje * self.gradientes_pesos
        self.sesgos += tasa_aprendizaje * self.gradientes_sesgos

class PerceptronMulticapa:
    
    def __init__(self, arquitectura: List[int], 
                 funciones_activacion: List[str] = None):
        if len(arquitectura) < 2:
            raise ValueError("La arquitectura debe tener al menos 2 capas (entrada y salida)")
        
        self.arquitectura = arquitectura
        self.num_capas = len(arquitectura) - 1
        
        if funciones_activacion is None:
            funciones_activacion = ['sigmoide'] * self.num_capas
        
        if len(funciones_activacion) != self.num_capas:
            raise ValueError("Número de funciones de activación debe coincidir con número de capas")
        
        self.capas = []
        for i in range(self.num_capas):
            capa = CapaRed(
                num_entradas=arquitectura[i],
                num_neuronas=arquitectura[i + 1],
                funcion_activacion=funciones_activacion[i]
            )
            self.capas.append(capa)
        
        self.evaluador = EvaluadorRendimiento()
        self.convergencia_alcanzada = False
        self.epoca_convergencia = 0
    
    def _propagacion_adelante(self, entradas: np.ndarray) -> np.ndarray:
        salida_actual = entradas
        
        for capa in self.capas:
            salida_actual = capa.propagacion_adelante(salida_actual)
        
        return salida_actual
    
    def _retropropagacion(self, salidas_esperadas: np.ndarray, 
                         salidas_obtenidas: np.ndarray) -> None:
        error_actual = salidas_esperadas - salidas_obtenidas
        
        for i in reversed(range(self.num_capas)):
            error_actual = self.capas[i].calcular_gradientes(error_actual)
    
    def _actualizar_pesos(self, tasa_aprendizaje: float) -> None:
        for capa in self.capas:
            capa.actualizar_pesos(tasa_aprendizaje)
    
    def predecir(self, entradas: np.ndarray) -> np.ndarray:
        if entradas.ndim == 1:
            entradas = entradas.reshape(1, -1)
        
        return self._propagacion_adelante(entradas)
    
    def entrenar(self, entradas: np.ndarray, salidas_esperadas: np.ndarray,
                tasa_aprendizaje: float = TASA_APRENDIZAJE_DEFECTO,
                max_epocas: int = EPOCAS_MAXIMAS_DEFECTO,
                error_objetivo: float = ERROR_OBJETIVO_DEFECTO,
                mostrar_progreso: bool = True,
                intervalo_impresion: int = INTERVALO_IMPRESION_DEFECTO) -> Tuple[bool, int]:
        self.evaluador.limpiar_historial()
        
        if entradas.ndim == 1:
            entradas = entradas.reshape(1, -1)
        if salidas_esperadas.ndim == 1:
            salidas_esperadas = salidas_esperadas.reshape(-1, 1)
        
        if mostrar_progreso:
            print(f"Iniciando entrenamiento...")
            print(f"Arquitectura: {self.arquitectura}")
            print(f"Tasa de aprendizaje: {tasa_aprendizaje}")
            print(f"Épocas máximas: {max_epocas}")
        
        for epoca in range(max_epocas):
            salidas_obtenidas = self._propagacion_adelante(entradas)
            
            error_cuadratico = UtilidadesMatematicas.calcular_error_cuadratico_medio(
                salidas_esperadas, salidas_obtenidas
            )
            
            self.evaluador.registrar_error(error_cuadratico)
            
            if error_cuadratico <= error_objetivo:
                self.convergencia_alcanzada = True
                self.epoca_convergencia = epoca + 1
                if mostrar_progreso:
                    print(f"{MENSAJE_CONVERGENCIA} Época: {self.epoca_convergencia}")
                    print(f"Error final: {error_cuadratico:.6f}")
                return True, self.epoca_convergencia
            
            self._retropropagacion(salidas_esperadas, salidas_obtenidas)
            
            self._actualizar_pesos(tasa_aprendizaje)
            
            if mostrar_progreso and epoca % intervalo_impresion == 0:
                print(f"Época {epoca:>6}: Error = {error_cuadratico:.6f}")
        
        if mostrar_progreso:
            print(MENSAJE_NO_CONVERGENCIA)
            print(f"Error final: {error_cuadratico:.6f}")
        
        return False, max_epocas
    
    def evaluar(self, entradas: np.ndarray, salidas_esperadas: np.ndarray,
               tipo_problema: str = 'regresion') -> dict:
        predicciones = self.predecir(entradas)
        
        if tipo_problema == 'clasificacion_binaria':
            return self.evaluador.evaluar_clasificacion_binaria(predicciones, salidas_esperadas)
        elif tipo_problema == 'clasificacion_multiclase':
            return self.evaluador.evaluar_clasificacion_multiclase(predicciones, salidas_esperadas)
        else:
            error_cuadratico = UtilidadesMatematicas.calcular_error_cuadratico_medio(
                salidas_esperadas, predicciones
            )
            return {
                'error_cuadratico_medio': error_cuadratico,
                'predicciones': predicciones
            }
    
    def obtener_informacion_red(self) -> dict:
        total_parametros = 0
        for capa in self.capas:
            total_parametros += capa.pesos.size + capa.sesgos.size
        
        info = {
            'arquitectura': self.arquitectura,
            'num_capas': self.num_capas,
            'total_parametros': total_parametros,
            'convergencia_alcanzada': self.convergencia_alcanzada,
            'epoca_convergencia': self.epoca_convergencia
        }
        
        estadisticas = self.evaluador.obtener_estadisticas_entrenamiento()
        info.update(estadisticas)
        
        return info
    
    def obtener_pesos_por_capa(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(capa.pesos.copy(), capa.sesgos.copy()) for capa in self.capas]
