import numpy as np
from typing import Dict, List, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from comun.constantes.constantes_redes_neuronales import (
    PRECISION_DECIMALES, ANCHO_COLUMNA_MATRIZ
)

class VisualizadorResultados:

    def __init__(self, precision_decimales: int = PRECISION_DECIMALES):

        self.precision_decimales = precision_decimales
    
    def mostrar_resultados_entrenamiento(self, resultados: Dict[str, Any]) -> None:

        if 'error' in resultados:
            print(f"❌ Error: {resultados['error']}")
            return
        
        print(f"\n{'='*50}")
        print(f"RESULTADOS DE ENTRENAMIENTO - {resultados['tipo_compuerta'].upper()}")
        print(f"{'='*50}")
        
        self._mostrar_informacion_basica(resultados)
        
        self._mostrar_convergencia(resultados)
        
        self._mostrar_metricas(resultados)
        
        self._mostrar_tabla_predicciones(resultados)
        
        self._mostrar_pesos_finales(resultados)
    
    def _mostrar_informacion_basica(self, resultados: Dict[str, Any]) -> None:

        info = resultados['info_entrenamiento']
        
        print(f"Compuerta: {resultados['tipo_compuerta'].upper()}")
        print(f"Función de activación: {resultados['funcion_activacion']}")
        print(f"Número de entradas: {info['num_entradas']}")
    
    def _mostrar_convergencia(self, resultados: Dict[str, Any]) -> None:

        convergencia = "✅ SÍ" if resultados['convergencia'] else "❌ NO"
        epoca = resultados.get('epoca_convergencia', 'N/A')
        
        print(f"\nCONVERGENCIA:")
        print(f"  Alcanzada: {convergencia}")
        print(f"  Época: {epoca}")
        
        info = resultados['info_entrenamiento']
        if 'error_inicial' in info:
            print(f"  Error inicial: {info['error_inicial']:.{self.precision_decimales}f}")
            print(f"  Error final: {info['error_final']:.{self.precision_decimales}f}")
            if 'reduccion_error' in info:
                print(f"  Reducción de error: {info['reduccion_error']:.2f}%")
    
    def _mostrar_metricas(self, resultados: Dict[str, Any]) -> None:

        metricas = resultados['metricas']
        
        print(f"\nMÉTRICAS DE EVALUACIÓN:")
        
        if 'precision' in metricas:
            precision_porcentaje = metricas['precision'] * 100
            print(f"  Precisión: {precision_porcentaje:.2f}%")
        
        if 'error_cuadratico_medio' in metricas:
            print(f"  Error cuadrático medio: {metricas['error_cuadratico_medio']:.{self.precision_decimales}f}")
        
        if 'sensibilidad' in metricas:
            print(f"  Sensibilidad: {metricas['sensibilidad']:.{self.precision_decimales}f}")
        
        if 'especificidad' in metricas:
            print(f"  Especificidad: {metricas['especificidad']:.{self.precision_decimales}f}")
    
    def _mostrar_tabla_predicciones(self, resultados: Dict[str, Any]) -> None:

        entradas = resultados['entradas']
        salidas_esperadas = resultados['salidas_esperadas']
        predicciones = resultados['metricas'].get('predicciones', [])
        
        if len(predicciones) == 0:
            return
        
        print(f"\nTABLA DE PREDICCIONES:")
        print(f"{'Entrada':>15} | {'Esperado':>10} | {'Predicho':>10} | {'Correcto':>10}")
        print(f"{'-'*15} | {'-'*10} | {'-'*10} | {'-'*10}")
        
        for i, (entrada, esperado, predicho) in enumerate(zip(entradas, salidas_esperadas, predicciones)):
            entrada_str = str(entrada.tolist()) if hasattr(entrada, 'tolist') else str(entrada)
            correcto = "✅" if abs(esperado - predicho) < 0.5 else "❌"
            
            print(f"{entrada_str:>15} | {esperado:>10.{self.precision_decimales}f} | "
                  f"{predicho:>10.{self.precision_decimales}f} | {correcto:>10}")
    
    def _mostrar_pesos_finales(self, resultados: Dict[str, Any]) -> None:

        info = resultados['info_entrenamiento']
        pesos = info.get('pesos_finales', [])
        
        if len(pesos) > 0:
            print(f"\nPESOS FINALES:")
            for i, peso in enumerate(pesos):
                if i == len(pesos) - 1:
                    print(f"  Sesgo (w{i}): {peso:.{self.precision_decimales}f}")
                else:
                    print(f"  Peso {i+1} (w{i}): {peso:.{self.precision_decimales}f}")
    
    def mostrar_comparacion_funciones(self, resultados_comparacion: Dict[str, Dict],
                                    tipo_compuerta: str) -> None:

        print(f"\n{'='*70}")
        print(f"COMPARACIÓN DE FUNCIONES DE ACTIVACIÓN - {tipo_compuerta.upper()}")
        print(f"{'='*70}")
        
        print(f"{'Función':>12} | {'Convergencia':>12} | {'Época':>8} | {'Precisión':>10} | {'Error MSE':>12}")
        print(f"{'-'*12} | {'-'*12} | {'-'*8} | {'-'*10} | {'-'*12}")
        
        for funcion, resultado in resultados_comparacion.items():
            if 'error' not in resultado:
                convergencia = "SÍ" if resultado['convergencia'] else "NO"
                epoca = resultado.get('epoca_convergencia', 'N/A')
                precision = resultado['metricas'].get('precision', 0) * 100
                error_mse = resultado['metricas'].get('error_cuadratico_medio', 0)
                
                print(f"{funcion:>12} | {convergencia:>12} | {str(epoca):>8} | "
                      f"{precision:>9.1f}% | {error_mse:>12.{self.precision_decimales}f}")
            else:
                print(f"{funcion:>12} | {'ERROR':>12} | {'N/A':>8} | {'N/A':>10} | {'N/A':>12}")
    
    def mostrar_matriz_confusion(self, matriz: np.ndarray, etiquetas: List[str] = None) -> None:

        if etiquetas is None:
            etiquetas = [f"Clase {i}" for i in range(matriz.shape[0])]
        
        print(f"\nMATRIZ DE CONFUSIÓN:")
        print(f"{'':>12}", end="")
        for etiqueta in etiquetas:
            print(f"{etiqueta:>{ANCHO_COLUMNA_MATRIZ}}", end="")
        print()
        
        for i, fila in enumerate(matriz):
            print(f"{etiquetas[i]:>12}", end="")
            for valor in fila:
                print(f"{valor:>{ANCHO_COLUMNA_MATRIZ}}", end="")
            print()
    
    def mostrar_progreso_entrenamiento(self, epoca: int, error: float, 
                                     intervalo: int = 1000) -> None:

        if epoca % intervalo == 0:
            print(f"Época {epoca:>6}: Error = {error:.{self.precision_decimales}f}")
    
    def generar_reporte_completo(self, resultados_multiples: Dict[str, Dict],
                               titulo: str = "REPORTE COMPLETO") -> str:

        reporte = []
        reporte.append(f"{'='*80}")
        reporte.append(f"{titulo:^80}")
        reporte.append(f"{'='*80}")
        
        for nombre, resultados in resultados_multiples.items():
            reporte.append(f"\n{'-'*40}")
            reporte.append(f"EXPERIMENTO: {nombre.upper()}")
            reporte.append(f"{'-'*40}")
            
            if 'error' not in resultados:
                info = resultados['info_entrenamiento']
                reporte.append(f"Función de activación: {resultados['funcion_activacion']}")
                reporte.append(f"Convergencia: {'SÍ' if resultados['convergencia'] else 'NO'}")
                reporte.append(f"Época de convergencia: {resultados.get('epoca_convergencia', 'N/A')}")
                
                metricas = resultados['metricas']
                if 'precision' in metricas:
                    reporte.append(f"Precisión: {metricas['precision']*100:.2f}%")
                if 'error_cuadratico_medio' in metricas:
                    reporte.append(f"Error MSE: {metricas['error_cuadratico_medio']:.{self.precision_decimales}f}")
            else:
                reporte.append(f"ERROR: {resultados['error']}")
        
        return "\n".join(reporte)
