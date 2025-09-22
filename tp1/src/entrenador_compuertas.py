import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .perceptron_simple import PerceptronSimple
from .cargador_datos import CargadorDatos
from .visualizador_resultados import VisualizadorResultados
from comun.constantes.constantes_redes_neuronales import (
    TASA_APRENDIZAJE_DEFECTO, EPOCAS_MAXIMAS_DEFECTO, ERROR_OBJETIVO_DEFECTO
)

class EntrenadorCompuertas:

    def __init__(self, visualizador: Optional[VisualizadorResultados] = None):

        self.visualizador = visualizador or VisualizadorResultados()
        self.resultados_entrenamiento: Dict[str, Dict] = {}
    
    def entrenar_compuerta_individual(self, tipo_compuerta: str,
                                    funcion_activacion: str = 'escalon',
                                    tasa_aprendizaje: float = TASA_APRENDIZAJE_DEFECTO,
                                    max_epocas: int = EPOCAS_MAXIMAS_DEFECTO,
                                    error_objetivo: float = ERROR_OBJETIVO_DEFECTO,
                                    mostrar_progreso: bool = True) -> Dict:

        try:
            entradas, salidas_esperadas = CargadorDatos.cargar_datos_compuerta_logica(tipo_compuerta)
            
            perceptron = PerceptronSimple(
                num_entradas=entradas.shape[1],
                funcion_activacion=funcion_activacion
            )
            
            if mostrar_progreso:
                print(f"\n=== Entrenando Compuerta {tipo_compuerta.upper()} ===")
                print(f"Función de activación: {funcion_activacion}")
                print(f"Tasa de aprendizaje: {tasa_aprendizaje}")
            
            convergencia, epoca_convergencia = perceptron.entrenar(
                entradas=entradas,
                salidas_esperadas=salidas_esperadas,
                tasa_aprendizaje=tasa_aprendizaje,
                max_epocas=max_epocas,
                error_objetivo=error_objetivo,
                mostrar_progreso=mostrar_progreso
            )
            
            metricas = perceptron.evaluar(entradas, salidas_esperadas)
            info_entrenamiento = perceptron.obtener_informacion_entrenamiento()
            
            resultados = {
                'tipo_compuerta': tipo_compuerta,
                'funcion_activacion': funcion_activacion,
                'convergencia': convergencia,
                'epoca_convergencia': epoca_convergencia,
                'metricas': metricas,
                'info_entrenamiento': info_entrenamiento,
                'entradas': entradas,
                'salidas_esperadas': salidas_esperadas,
                'predicciones': metricas.get('predicciones', []),
                'perceptron': perceptron
            }
            
            self.resultados_entrenamiento[tipo_compuerta] = resultados
            
            if mostrar_progreso:
                self.visualizador.mostrar_resultados_entrenamiento(resultados)
            
            return resultados
            
        except Exception as e:
            error_msg = f"Error al entrenar compuerta {tipo_compuerta}: {str(e)}"
            print(error_msg)
            return {'error': error_msg}
    
    def entrenar_todas_las_compuertas(self, funcion_activacion: str = 'escalon',
                                    tasa_aprendizaje: float = TASA_APRENDIZAJE_DEFECTO,
                                    max_epocas: int = EPOCAS_MAXIMAS_DEFECTO,
                                    mostrar_progreso: bool = True) -> Dict[str, Dict]:

        compuertas = ['and', 'or', 'xor']
        resultados_completos = {}
        
        if mostrar_progreso:
            print(f"\n{'='*60}")
            print(f"ENTRENAMIENTO DE COMPUERTAS LÓGICAS")
            print(f"Función de activación: {funcion_activacion}")
            print(f"{'='*60}")
        
        for compuerta in compuertas:
            resultado = self.entrenar_compuerta_individual(
                tipo_compuerta=compuerta,
                funcion_activacion=funcion_activacion,
                tasa_aprendizaje=tasa_aprendizaje,
                max_epocas=max_epocas,
                mostrar_progreso=mostrar_progreso
            )
            resultados_completos[compuerta] = resultado
        
        if mostrar_progreso:
            self._mostrar_resumen_comparativo(resultados_completos)
        
        return resultados_completos
    
    def comparar_funciones_activacion(self, tipo_compuerta: str,
                                    funciones: list = None,
                                    mostrar_progreso: bool = True) -> Dict[str, Dict]:

        if funciones is None:
            funciones = ['escalon', 'sigmoide', 'lineal']
        
        resultados_comparacion = {}
        
        if mostrar_progreso:
            print(f"\n{'='*60}")
            print(f"COMPARACIÓN DE FUNCIONES DE ACTIVACIÓN - {tipo_compuerta.upper()}")
            print(f"{'='*60}")
        
        for funcion in funciones:
            resultado = self.entrenar_compuerta_individual(
                tipo_compuerta=tipo_compuerta,
                funcion_activacion=funcion,
                mostrar_progreso=mostrar_progreso
            )
            resultados_comparacion[funcion] = resultado
        
        if mostrar_progreso:
            self.visualizador.mostrar_comparacion_funciones(resultados_comparacion, tipo_compuerta)
        
        return resultados_comparacion
    
    def _mostrar_resumen_comparativo(self, resultados: Dict[str, Dict]) -> None:

        print(f"\n{'='*60}")
        print("RESUMEN COMPARATIVO DE COMPUERTAS")
        print(f"{'='*60}")
        
        for compuerta, resultado in resultados.items():
            if 'error' not in resultado:
                convergencia = "SÍ" if resultado['convergencia'] else "NO"
                epoca = resultado.get('epoca_convergencia', 'N/A')
                precision = resultado['metricas'].get('precision', 0) * 100
                
                print(f"{compuerta.upper():>8}: Convergencia: {convergencia:>3} | "
                      f"Época: {epoca:>6} | Precisión: {precision:>6.1f}%")
            else:
                print(f"{compuerta.upper():>8}: ERROR - {resultado['error']}")
    
    def obtener_resultados(self, tipo_compuerta: str = None) -> Dict:

        if tipo_compuerta is None:
            return self.resultados_entrenamiento.copy()
        else:
            return self.resultados_entrenamiento.get(tipo_compuerta, {})
    
    def limpiar_resultados(self) -> None:

        self.resultados_entrenamiento.clear()
