import numpy as np
from typing import Tuple, List

class UtilidadesMatematicas:

    @staticmethod
    def inicializar_pesos_aleatorios(filas: int, columnas: int, 
                                   rango_min: float = -0.5, 
                                   rango_max: float = 0.5) -> np.ndarray:

        return np.random.uniform(rango_min, rango_max, (filas, columnas))
    
    @staticmethod
    def inicializar_pesos_xavier(filas: int, columnas: int) -> np.ndarray:

        limite = np.sqrt(6.0 / (filas + columnas))
        return np.random.uniform(-limite, limite, (filas, columnas))
    
    @staticmethod
    def calcular_error_cuadratico_medio(salida_esperada: np.ndarray, 
                                      salida_obtenida: np.ndarray) -> float:

        return np.mean(np.square(salida_esperada - salida_obtenida))
    
    @staticmethod
    def normalizar_datos(datos: np.ndarray) -> Tuple[np.ndarray, float, float]:

        valor_min = np.min(datos)
        valor_max = np.max(datos)
        
        if valor_max == valor_min:
            return datos, valor_min, valor_max
        
        datos_normalizados = (datos - valor_min) / (valor_max - valor_min)
        return datos_normalizados, valor_min, valor_max
    
    @staticmethod
    def desnormalizar_datos(datos_normalizados: np.ndarray, 
                          valor_min: float, valor_max: float) -> np.ndarray:

        return datos_normalizados * (valor_max - valor_min) + valor_min
    
    @staticmethod
    def dividir_datos_entrenamiento_prueba(datos_entrada: np.ndarray, 
                                         datos_salida: np.ndarray,
                                         porcentaje_entrenamiento: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        num_muestras = len(datos_entrada)
        indices = np.random.permutation(num_muestras)
        
        num_entrenamiento = int(num_muestras * porcentaje_entrenamiento)
        
        indices_entrenamiento = indices[:num_entrenamiento]
        indices_prueba = indices[num_entrenamiento:]
        
        entrada_entrenamiento = datos_entrada[indices_entrenamiento]
        salida_entrenamiento = datos_salida[indices_entrenamiento]
        entrada_prueba = datos_entrada[indices_prueba]
        salida_prueba = datos_salida[indices_prueba]
        
        return entrada_entrenamiento, salida_entrenamiento, entrada_prueba, salida_prueba
    
    @staticmethod
    def calcular_precision(predicciones: np.ndarray, valores_reales: np.ndarray, 
                         umbral: float = 0.5) -> float:

        predicciones_binarias = (predicciones > umbral).astype(int)
        valores_reales_binarios = (valores_reales > umbral).astype(int)
        
        aciertos = np.sum(predicciones_binarias == valores_reales_binarios)
        total = len(valores_reales)
        
        return aciertos / total if total > 0 else 0.0
    
    @staticmethod
    def generar_ruido_binario(datos: np.ndarray, probabilidad: float = 0.02) -> np.ndarray:

        datos_con_ruido = datos.copy()
        mascara_ruido = np.random.random(datos.shape) < probabilidad
        datos_con_ruido[mascara_ruido] = 1 - datos_con_ruido[mascara_ruido]
        return datos_con_ruido
    
    @staticmethod
    def convertir_a_one_hot(etiquetas: List[int], num_clases: int) -> np.ndarray:

        one_hot = np.zeros((len(etiquetas), num_clases))
        for i, etiqueta in enumerate(etiquetas):
            one_hot[i, etiqueta] = 1
        return one_hot
    
    @staticmethod
    def obtener_clase_predicha(salida_one_hot: np.ndarray) -> int:

        return np.argmax(salida_one_hot)
