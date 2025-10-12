import numpy as np
from tensorflow import keras

from .configuraciones import CONFIGURACIONES_AUTOCODIFICADOR, CONFIGURACIONES_ENTRENAMIENTO
from .constructor_modelos import ConstructorModelos
from .procesador_datos import ProcesadorDatos
from .generador_ruido import GeneradorRuido
from .entrenador import generar_nombre_modelo, guardar_modelo


class EntrenadorEliminadorRuido:
    def __init__(self):
        self.procesador = ProcesadorDatos()
        self.constructor = ConstructorModelos()
        self.generador_ruido = GeneradorRuido()
        self.datos_limpios = self.procesador.obtener_datos_procesados()
    
    def entrenar_con_ruido(self, config_modelo, config_entrenamiento, tipo_ruido, nivel_ruido):
        datos_ruidosos = self.generador_ruido.generar_conjunto_ruidoso(
            self.datos_limpios, tipo_ruido, nivel_ruido
        )
        
        modelo = self.constructor.crear_autocodificador_desde_config(config_modelo)
        
        callbacks = []
        if config_entrenamiento.get('early_stopping', False):
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config_entrenamiento.get('patience', 100),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        historial = modelo.fit(
            datos_ruidosos, self.datos_limpios,
            epochs=config_entrenamiento['epochs'],
            batch_size=config_modelo.get('batch_size', 32),
            verbose=0,
            validation_split=config_entrenamiento.get('validation_split', 0.0),
            callbacks=callbacks
        )
        
        return modelo, historial
    
    def evaluar_eliminacion_ruido(self, modelo, tipo_ruido, nivel_ruido):
        datos_ruidosos = self.generador_ruido.generar_conjunto_ruidoso(
            self.datos_limpios, tipo_ruido, nivel_ruido
        )
        
        datos_reconstruidos = modelo.predict(datos_ruidosos, verbose=0)
        
        mse_limpio = np.mean((self.datos_limpios - datos_reconstruidos) ** 2)
        mse_ruidoso = np.mean((datos_ruidosos - datos_reconstruidos) ** 2)
        
        precision_limpieza = np.mean(
            (datos_reconstruidos > 0.5) == (self.datos_limpios > 0.5)
        )
        
        mejora_snr = self.generador_ruido.calcular_mejora_snr(
            self.datos_limpios, datos_ruidosos, datos_reconstruidos
        )
        
        mejora_mse = ((mse_ruidoso - mse_limpio) / mse_ruidoso) * 100 if mse_ruidoso > 0 else 0
        
        return {
            'mse_limpio': float(mse_limpio),
            'mse_ruidoso': float(mse_ruidoso),
            'precision_limpieza': float(precision_limpieza),
            'mejora_snr': float(mejora_snr),
            'mejora_mse_porcentaje': float(mejora_mse),
            'efectivo': mejora_mse > 0
        }
    
    def generar_nombre_modelo_ruido(self, config_modelo, config_entrenamiento, tipo_ruido, nivel_ruido):
        nombre_base = generar_nombre_modelo(config_modelo, config_entrenamiento)
        nivel_str = str(nivel_ruido).replace('.', '_')
        return f"{nombre_base}_{tipo_ruido}_{nivel_str}"
    
    def entrenar_modelo_completo(self, config_modelo_nombre, config_entrenamiento_nombre, 
                               tipo_ruido, nivel_ruido):
        config_modelo = CONFIGURACIONES_AUTOCODIFICADOR[config_modelo_nombre].copy()
        config_entrenamiento = CONFIGURACIONES_ENTRENAMIENTO[config_entrenamiento_nombre].copy()
        
        modelo, historial = self.entrenar_con_ruido(
            config_modelo, config_entrenamiento, tipo_ruido, nivel_ruido
        )
        
        metricas = self.evaluar_eliminacion_ruido(modelo, tipo_ruido, nivel_ruido)
        
        nombre_modelo = self.generar_nombre_modelo_ruido(
            config_modelo, config_entrenamiento, tipo_ruido, nivel_ruido
        )
        
        ruta_modelo = guardar_modelo(modelo, nombre_modelo)
        
        return modelo, historial, metricas, nombre_modelo
