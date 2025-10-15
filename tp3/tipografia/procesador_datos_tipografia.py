import numpy as np
from .generador_bitmap_tipografia import obtener_dataset_entrenamiento


class ProcesadorDatosTipografia:
    def __init__(self, tamaño_imagen=32):
        self.tamaño_imagen = tamaño_imagen
        self.datos_binarios = None
        self.simbolos = None
        
    def obtener_datos_procesados(self):
        if self.datos_binarios is None:
            X_flat, simbolos = obtener_dataset_entrenamiento(self.tamaño_imagen)
            self.datos_binarios = X_flat.astype(np.float32)
            self.simbolos = simbolos
        return self.datos_binarios
    
    def obtener_patron_como_matriz(self, indice):
        if self.datos_binarios is None:
            self.obtener_datos_procesados()
        return self.datos_binarios[indice].reshape(self.tamaño_imagen, self.tamaño_imagen)
    
    def mostrar_patron_ascii(self, indice, titulo=""):
        patron_2d = self.obtener_patron_como_matriz(indice)
        if titulo:
            print(f"\n{titulo}")
        for fila in patron_2d:
            print(''.join(['██' if pixel else '  ' for pixel in fila]))
    
    def obtener_estadisticas(self):
        if self.datos_binarios is None:
            self.obtener_datos_procesados()
            
        num_patrones = len(self.datos_binarios)
        pixels_por_patron = self.datos_binarios.shape[1]
        densidad_promedio = np.mean(self.datos_binarios) * 100
        
        return {
            'num_patrones': num_patrones,
            'pixels_por_patron': pixels_por_patron,
            'densidad_promedio': densidad_promedio,
            'tamaño_imagen': self.tamaño_imagen
        }
    
    def validar_datos(self):
        if self.datos_binarios is None:
            self.obtener_datos_procesados()
            
        errores = []
        
        pixels_esperados = self.tamaño_imagen * self.tamaño_imagen
        if self.datos_binarios.shape[1] != pixels_esperados:
            errores.append(f"Esperados {pixels_esperados} píxeles por patrón, encontrados {self.datos_binarios.shape[1]}")
            
        valores_unicos = np.unique(self.datos_binarios)
        if not np.array_equal(valores_unicos, [0., 1.]) and not np.array_equal(valores_unicos, [0.]) and not np.array_equal(valores_unicos, [1.]):
            errores.append(f"Datos no binarios encontrados: {valores_unicos}")
            
        if len(self.datos_binarios) != 30:
            errores.append(f"Esperados 30 patrones, encontrados {len(self.datos_binarios)}")
            
        return len(errores) == 0, errores
