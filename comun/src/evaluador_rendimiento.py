import numpy as np
from typing import Dict, List, Tuple, Optional
from .utilidades_matematicas import UtilidadesMatematicas

class EvaluadorRendimiento:

    def __init__(self):
        self.historial_errores: List[float] = []
        self.historial_precision: List[float] = []
    
    def evaluar_clasificacion_binaria(self, predicciones: np.ndarray, 
                                    valores_reales: np.ndarray,
                                    umbral: float = 0.0) -> Dict[str, float]:

        pred_binarias = (predicciones > umbral).astype(int)
        real_binarias = (valores_reales > umbral).astype(int)
        
        verdaderos_positivos = np.sum((pred_binarias == 1) & (real_binarias == 1))
        verdaderos_negativos = np.sum((pred_binarias == 0) & (real_binarias == 0))
        falsos_positivos = np.sum((pred_binarias == 1) & (real_binarias == 0))
        falsos_negativos = np.sum((pred_binarias == 0) & (real_binarias == 1))
        
        total = len(valores_reales)
        precision = (verdaderos_positivos + verdaderos_negativos) / total
        
        sensibilidad = verdaderos_positivos / (verdaderos_positivos + falsos_negativos) if (verdaderos_positivos + falsos_negativos) > 0 else 0
        especificidad = verdaderos_negativos / (verdaderos_negativos + falsos_positivos) if (verdaderos_negativos + falsos_positivos) > 0 else 0
        
        error_cuadratico = UtilidadesMatematicas.calcular_error_cuadratico_medio(valores_reales, predicciones)
        
        return {
            'precision': precision,
            'error_cuadratico_medio': error_cuadratico,
            'sensibilidad': sensibilidad,
            'especificidad': especificidad,
            'verdaderos_positivos': verdaderos_positivos,
            'verdaderos_negativos': verdaderos_negativos,
            'falsos_positivos': falsos_positivos,
            'falsos_negativos': falsos_negativos
        }
    
    def evaluar_clasificacion_multiclase(self, predicciones: np.ndarray,
                                       valores_reales: np.ndarray) -> Dict[str, float]:

        clases_predichas = np.argmax(predicciones, axis=1)
        clases_reales = np.argmax(valores_reales, axis=1)
        
        precision = np.mean(clases_predichas == clases_reales)
        
        error_cuadratico = UtilidadesMatematicas.calcular_error_cuadratico_medio(valores_reales, predicciones)
        
        return {
            'precision': precision,
            'error_cuadratico_medio': error_cuadratico,
            'clases_predichas': clases_predichas,
            'clases_reales': clases_reales
        }
    
    def generar_matriz_confusion(self, clases_predichas: np.ndarray,
                               clases_reales: np.ndarray,
                               num_clases: int) -> np.ndarray:

        matriz = np.zeros((num_clases, num_clases), dtype=int)
        
        for real, predicha in zip(clases_reales, clases_predichas):
            matriz[real, predicha] += 1
        
        return matriz
    
    def registrar_error(self, error: float) -> None:

        self.historial_errores.append(error)
    
    def registrar_precision(self, precision: float) -> None:

        self.historial_precision.append(precision)
    
    def obtener_estadisticas_entrenamiento(self) -> Dict[str, float]:

        if not self.historial_errores:
            return {}
        
        return {
            'error_inicial': self.historial_errores[0],
            'error_final': self.historial_errores[-1],
            'error_minimo': min(self.historial_errores),
            'error_maximo': max(self.historial_errores),
            'reduccion_error': (self.historial_errores[0] - self.historial_errores[-1]) / self.historial_errores[0] * 100,
            'epocas_entrenamiento': len(self.historial_errores)
        }
    
    def limpiar_historial(self) -> None:

        self.historial_errores.clear()
        self.historial_precision.clear()
    
    def evaluar_robustez_ruido(self, red_neuronal, datos_limpios: np.ndarray,
                             datos_con_ruido: np.ndarray, 
                             etiquetas: np.ndarray) -> Dict[str, float]:

        predicciones_limpias = red_neuronal.predecir(datos_limpios)
        metricas_limpias = self.evaluar_clasificacion_binaria(predicciones_limpias, etiquetas)
        
        predicciones_ruido = red_neuronal.predecir(datos_con_ruido)
        metricas_ruido = self.evaluar_clasificacion_binaria(predicciones_ruido, etiquetas)
        
        degradacion_precision = metricas_limpias['precision'] - metricas_ruido['precision']
        
        return {
            'precision_sin_ruido': metricas_limpias['precision'],
            'precision_con_ruido': metricas_ruido['precision'],
            'degradacion_precision': degradacion_precision,
            'robustez_porcentaje': (1 - degradacion_precision) * 100
        }
