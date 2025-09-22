import numpy as np
from typing import Tuple, List, Dict
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from comun.constantes.constantes_redes_neuronales import (
    PATRONES_XOR_ENTRADA, PATRONES_XOR_SALIDA,
    PATRONES_AND_ENTRADA, PATRONES_AND_SALIDA,
    PATRONES_OR_ENTRADA, PATRONES_OR_SALIDA,
    ARCHIVO_ENTRENAMIENTO_TP1, ARCHIVO_SALIDA_TP1
)

class CargadorDatos:

    @staticmethod
    def cargar_datos_compuerta_logica(tipo_compuerta: str) -> Tuple[np.ndarray, np.ndarray]:

        compuertas_disponibles = {
            'and': (PATRONES_AND_ENTRADA, PATRONES_AND_SALIDA),
            'or': (PATRONES_OR_ENTRADA, PATRONES_OR_SALIDA),
            'xor': (PATRONES_XOR_ENTRADA, PATRONES_XOR_SALIDA)
        }
        
        if tipo_compuerta.lower() not in compuertas_disponibles:
            raise ValueError(f"Tipo de compuerta '{tipo_compuerta}' no válido. "
                           f"Opciones: {list(compuertas_disponibles.keys())}")
        
        entradas, salidas = compuertas_disponibles[tipo_compuerta.lower()]
        return np.array(entradas), np.array(salidas)
    
    @staticmethod
    def cargar_datos_desde_archivo(ruta_entradas: str, ruta_salidas: str) -> Tuple[np.ndarray, np.ndarray]:

        try:
            entradas = np.loadtxt(ruta_entradas)
            
            salidas = np.loadtxt(ruta_salidas)
            
            if len(entradas) != len(salidas):
                raise ValueError(f"Número de entradas ({len(entradas)}) no coincide "
                               f"con número de salidas ({len(salidas)})")
            
            return entradas, salidas
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No se pudo encontrar el archivo: {e.filename}")
        except Exception as e:
            raise ValueError(f"Error al cargar datos desde archivo: {str(e)}")
    
    @staticmethod
    def cargar_datos_tp1_ejercicio2(directorio_datos: str = None) -> Tuple[np.ndarray, np.ndarray]:

        if directorio_datos is None:
            directorio_actual = os.path.dirname(__file__)
            directorio_datos = os.path.join(directorio_actual, '..', 'datos')
        
        ruta_entradas = os.path.join(directorio_datos, ARCHIVO_ENTRENAMIENTO_TP1)
        ruta_salidas = os.path.join(directorio_datos, ARCHIVO_SALIDA_TP1)
        
        return CargadorDatos.cargar_datos_desde_archivo(ruta_entradas, ruta_salidas)
    
    @staticmethod
    def preparar_datos_entrenamiento(entradas: np.ndarray, salidas: np.ndarray,
                                   normalizar: bool = False) -> Tuple[np.ndarray, np.ndarray]:

        entradas_preparadas = entradas.copy()
        salidas_preparadas = salidas.copy()
        
        if normalizar:
            from comun.src.utilidades_matematicas import UtilidadesMatematicas
            entradas_preparadas, _, _ = UtilidadesMatematicas.normalizar_datos(entradas_preparadas)
        
        if salidas_preparadas.ndim > 1:
            salidas_preparadas = salidas_preparadas.flatten()
        
        return entradas_preparadas, salidas_preparadas
    
    @staticmethod
    def dividir_datos_entrenamiento_validacion(entradas: np.ndarray, salidas: np.ndarray,
                                             porcentaje_entrenamiento: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        from comun.src.utilidades_matematicas import UtilidadesMatematicas
        
        return UtilidadesMatematicas.dividir_datos_entrenamiento_prueba(
            entradas, salidas, porcentaje_entrenamiento
        )
    
    @staticmethod
    def obtener_informacion_datos(entradas: np.ndarray, salidas: np.ndarray) -> Dict[str, any]:

        return {
            'num_muestras': len(entradas),
            'num_caracteristicas': entradas.shape[1] if entradas.ndim > 1 else 1,
            'rango_entradas': (np.min(entradas), np.max(entradas)),
            'rango_salidas': (np.min(salidas), np.max(salidas)),
            'media_entradas': np.mean(entradas, axis=0),
            'desviacion_entradas': np.std(entradas, axis=0),
            'valores_unicos_salidas': np.unique(salidas),
            'forma_entradas': entradas.shape,
            'forma_salidas': salidas.shape
        }
