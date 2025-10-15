import numpy as np
from tensorflow import keras

from tp3.comun.entrenador_base import EntrenadorBase
from tp3.comun.generador_ruido import GeneradorRuido
from tp3.simbolos.configuraciones import CONFIGURACIONES_AUTOCODIFICADOR, CONFIGURACIONES_ENTRENAMIENTO


class EntrenadorEliminadorRuidoRefactorizado(EntrenadorBase):
    def __init__(self, conjunto_datos=1, num_versiones_ruido=10):
        super().__init__(conjunto_datos)
        self.generador_ruido = GeneradorRuido()
        self.datos_limpios = self.datos
        self.num_versiones_ruido = num_versiones_ruido
    
    def entrenar_modelo(self, config_modelo, config_entrenamiento, tipo_ruido, nivel_ruido, **kwargs):
        if not self.validar_datos():
            return None, None, None, None
        
        datos_ruidosos, datos_limpios_expandidos = self.generador_ruido.generar_multiples_versiones_ruidosas(
            self.datos_limpios, tipo_ruido, nivel_ruido, self.num_versiones_ruido
        )
        
        modelo = self.constructor.crear_autocodificador_desde_config(config_modelo)
        
        config_entrenamiento_modificado = config_entrenamiento.copy()
        config_entrenamiento_modificado['monitor'] = 'val_loss'
        config_entrenamiento_modificado['verbose'] = 0
        
        callbacks = self.crear_callbacks(config_entrenamiento_modificado)
        
        historial = modelo.fit(
            datos_ruidosos, datos_limpios_expandidos,
            epochs=config_entrenamiento['epochs'],
            batch_size=config_modelo.get('batch_size', 32),
            verbose=0,
            validation_split=config_entrenamiento.get('validation_split', 0.0),
            callbacks=callbacks
        )
        
        metricas = self.evaluar_eliminacion_ruido(modelo, tipo_ruido, nivel_ruido)
        
        return modelo, historial, metricas
    
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
    
    def generar_nombre_modelo(self, config_modelo, config_entrenamiento, tipo_ruido, nivel_ruido, **kwargs):
        nombre_base = self.generar_nombre_modelo_base(config_modelo, config_entrenamiento, "tp3_eliminador")
        nivel_str = str(nivel_ruido).replace('.', '_')
        return f"{nombre_base}_{tipo_ruido}_{nivel_str}_x{self.num_versiones_ruido}"
    
    def entrenar_modelo_completo(self, config_modelo_nombre, config_entrenamiento_nombre, 
                               tipo_ruido, nivel_ruido):
        config_modelo = CONFIGURACIONES_AUTOCODIFICADOR[config_modelo_nombre].copy()
        config_entrenamiento = CONFIGURACIONES_ENTRENAMIENTO[config_entrenamiento_nombre].copy()
        
        modelo, historial, metricas = self.entrenar_modelo(
            config_modelo, config_entrenamiento, tipo_ruido, nivel_ruido
        )
        
        if modelo is None:
            return None, None, None, None
        
        nombre_modelo = self.generar_nombre_modelo(
            config_modelo, config_entrenamiento, tipo_ruido, nivel_ruido
        )
        
        ruta_modelo = self.guardar_modelo(modelo, nombre_modelo)
        
        return modelo, historial, metricas, nombre_modelo
