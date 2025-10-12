import numpy as np


class GeneradorRuido:
    def __init__(self):
        pass
    
    def aplicar_ruido_binario(self, datos, probabilidad):
        datos_ruidosos = datos.copy()
        mascara = np.random.random(datos.shape) < probabilidad
        datos_ruidosos[mascara] = 1 - datos_ruidosos[mascara]
        return datos_ruidosos
    
    def aplicar_ruido_gaussiano(self, datos, desviacion_std):
        ruido = np.random.normal(0, desviacion_std, datos.shape)
        datos_ruidosos = datos + ruido
        datos_ruidosos = (datos_ruidosos > 0.5).astype(float)
        return datos_ruidosos
    
    def aplicar_ruido_dropout(self, datos, probabilidad):
        datos_ruidosos = datos.copy()
        mascara = np.random.random(datos.shape) < probabilidad
        datos_ruidosos[mascara] = 0
        return datos_ruidosos
    
    def generar_conjunto_ruidoso(self, datos, tipo_ruido, nivel_ruido):
        if tipo_ruido == 'binario':
            return self.aplicar_ruido_binario(datos, nivel_ruido)
        elif tipo_ruido == 'gaussiano':
            return self.aplicar_ruido_gaussiano(datos, nivel_ruido)
        elif tipo_ruido == 'dropout':
            return self.aplicar_ruido_dropout(datos, nivel_ruido)
        else:
            raise ValueError(f"Tipo de ruido no soportado: {tipo_ruido}")
    
    def calcular_snr(self, datos_limpios, datos_ruidosos):
        potencia_senal = np.mean(datos_limpios ** 2)
        potencia_ruido = np.mean((datos_limpios - datos_ruidosos) ** 2)
        
        if potencia_ruido == 0:
            return float('inf')
        
        snr = 10 * np.log10(potencia_senal / potencia_ruido)
        return snr
    
    def calcular_mejora_snr(self, datos_limpios, datos_ruidosos, datos_reconstruidos):
        snr_original = self.calcular_snr(datos_limpios, datos_ruidosos)
        snr_reconstruido = self.calcular_snr(datos_limpios, datos_reconstruidos)
        return snr_reconstruido - snr_original
