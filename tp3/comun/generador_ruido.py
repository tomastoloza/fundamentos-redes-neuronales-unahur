import numpy as np


class GeneradorRuido:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
    
    def aplicar_ruido_binario(self, datos, probabilidad):
        datos_ruidosos = datos.copy()
        
        num_pixeles_total = datos.size
        num_pixeles_esperado = int(np.round(num_pixeles_total * probabilidad))
        num_pixeles_ruido = self.rng.integers(0, num_pixeles_esperado + 1)
        
        if num_pixeles_ruido > 0:
            indices_planos = self.rng.choice(num_pixeles_total, size=num_pixeles_ruido, replace=False)
            datos_planos = datos_ruidosos.flatten()
            datos_planos[indices_planos] = 1 - datos_planos[indices_planos]
            datos_ruidosos = datos_planos.reshape(datos.shape)
        
        return datos_ruidosos
    
    def aplicar_ruido_gaussiano(self, datos, intensidad):
        ruido = self.rng.normal(0, intensidad * 2, datos.shape)
        datos_ruidosos = np.clip(datos + ruido, 0, 1)
        datos_ruidosos = (datos_ruidosos > 0.5).astype(float)
        return datos_ruidosos
    
    def aplicar_ruido_dropout(self, datos, probabilidad):
        datos_ruidosos = datos.copy()
        mascara = self.rng.random(datos.shape) < probabilidad
        datos_ruidosos[mascara] = 0
        return datos_ruidosos
    
    def aplicar_ruido_salt_pepper(self, datos, probabilidad):
        datos_ruidosos = datos.copy()
        ruido_aleatorio = self.rng.random(datos.shape)
        mascara_salt = ruido_aleatorio < (probabilidad / 2)
        mascara_pepper = (ruido_aleatorio >= (probabilidad / 2)) & (ruido_aleatorio < probabilidad)
        datos_ruidosos[mascara_salt] = 1.0
        datos_ruidosos[mascara_pepper] = 0.0
        return datos_ruidosos
    
    def generar_conjunto_ruidoso(self, datos, tipo_ruido, nivel_ruido):
        if tipo_ruido == 'binario':
            return self.aplicar_ruido_binario(datos, nivel_ruido)
        elif tipo_ruido == 'gaussiano':
            return self.aplicar_ruido_gaussiano(datos, nivel_ruido)
        elif tipo_ruido == 'dropout':
            return self.aplicar_ruido_dropout(datos, nivel_ruido)
        elif tipo_ruido == 'salt_pepper':
            return self.aplicar_ruido_salt_pepper(datos, nivel_ruido)
        else:
            raise ValueError(f"Tipo de ruido no soportado: {tipo_ruido}")
    
    def generar_multiples_versiones_ruidosas(self, datos, tipo_ruido, nivel_ruido, num_versiones):
        num_patrones = datos.shape[0]
        dim_patron = datos.shape[1]
        
        datos_ruidosos_expandidos = np.zeros((num_patrones * num_versiones, dim_patron))
        datos_limpios_expandidos = np.zeros((num_patrones * num_versiones, dim_patron))
        
        for i in range(num_patrones):
            patron_limpio = datos[i:i+1]
            
            for j in range(num_versiones):
                idx = i * num_versiones + j
                
                patron_ruidoso = self.generar_conjunto_ruidoso(
                    patron_limpio, tipo_ruido, nivel_ruido
                )
                
                datos_ruidosos_expandidos[idx] = patron_ruidoso[0]
                datos_limpios_expandidos[idx] = patron_limpio[0]
        
        return datos_ruidosos_expandidos, datos_limpios_expandidos
    
    def calcular_snr(self, datos_limpios, datos_ruidosos):
        potencia_senal = np.mean(datos_limpios ** 2)
        potencia_ruido = np.mean((datos_limpios - datos_ruidosos) ** 2)
        
        if potencia_ruido == 0 or potencia_senal == 0:
            return 0.0  # Retornar 0 en lugar de inf para evitar problemas
        
        try:
            snr = 10 * np.log10(potencia_senal / potencia_ruido)
            return snr if np.isfinite(snr) else 0.0
        except:
            return 0.0
    
    def calcular_mejora_snr(self, datos_limpios, datos_ruidosos, datos_reconstruidos):
        snr_original = self.calcular_snr(datos_limpios, datos_ruidosos)
        snr_reconstruido = self.calcular_snr(datos_limpios, datos_reconstruidos)
        return snr_reconstruido - snr_original
