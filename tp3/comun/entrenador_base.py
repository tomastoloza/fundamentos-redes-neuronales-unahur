import os
import numpy as np
from abc import ABC, abstractmethod
from tensorflow import keras

from .constructor_modelos import ConstructorModelos
from .procesador_datos import ProcesadorDatos
from .visualizador_resultados import VisualizadorResultados


class EntrenadorBase(ABC):
    def __init__(self, conjunto_datos=1):
        self.procesador = ProcesadorDatos(conjunto_datos)
        self.constructor = ConstructorModelos()
        self.visualizador = VisualizadorResultados()
        self.datos = self.procesador.obtener_datos_procesados()
    
    def validar_datos(self):
        valido, errores = self.procesador.validar_datos()
        if not valido:
            print("Errores en los datos:")
            for error in errores:
                print(f"  - {error}")
            return False
        return True
    
    def crear_callbacks(self, config_entrenamiento):
        callbacks = []
        if config_entrenamiento.get('early_stopping', False):
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=config_entrenamiento.get('monitor', 'loss'),
                patience=config_entrenamiento.get('patience', 50),
                restore_best_weights=True,
                verbose=config_entrenamiento.get('verbose', 1)
            )
            callbacks.append(early_stopping)
        return callbacks
    
    def evaluar_modelo(self, modelo, datos_entrada, datos_objetivo):
        evaluacion = modelo.evaluate(datos_entrada, datos_objetivo, verbose=0)
        loss_final = evaluacion[0] if isinstance(evaluacion, list) else evaluacion
        
        predicciones = modelo.predict(datos_entrada, verbose=0)
        mse = np.mean((datos_objetivo - predicciones) ** 2)
        precision = np.mean((predicciones > 0.5) == (datos_objetivo > 0.5))
        
        return {
            'loss_final': float(loss_final),
            'mse': float(mse),
            'precision': float(precision),
            'predicciones': predicciones
        }
    
    def generar_nombre_modelo_base(self, config_modelo, config_entrenamiento, prefijo="tp3"):
        dimension_latente = config_modelo['dimension_latente']
        epochs = config_entrenamiento['epochs']
        learning_rate = config_modelo['learning_rate']
        
        lr_str = str(learning_rate).replace('.', '_')
        nombre = f"{prefijo}_lat{dimension_latente}_ep{epochs}_lr{lr_str}"
        return nombre
    
    def guardar_modelo(self, modelo, nombre_modelo, directorio="tp3/modelos"):
        if not os.path.exists(directorio):
            os.makedirs(directorio)
        
        ruta_completa = os.path.join(directorio, f"{nombre_modelo}.keras")
        modelo.save(ruta_completa)
        return ruta_completa
    
    @abstractmethod
    def entrenar_modelo(self, config_modelo, config_entrenamiento, **kwargs):
        pass
    
    @abstractmethod
    def generar_nombre_modelo(self, config_modelo, config_entrenamiento, **kwargs):
        pass
