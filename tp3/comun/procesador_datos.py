import numpy as np
from tp3.datos.tp3_input import entradas_1, entradas_2, entradas_3


class ProcesadorDatos:
    def __init__(self, conjunto_datos=1):
        self.conjunto_datos = conjunto_datos
        self.datos_hex = self._obtener_conjunto_datos(conjunto_datos)
        self.datos_binarios = None
        
    def _obtener_conjunto_datos(self, conjunto):
        conjuntos = {
            1: entradas_1,
            2: entradas_2, 
            3: entradas_3
        }
        
        if conjunto not in conjuntos:
            raise ValueError(f"Conjunto de datos {conjunto} no válido. Use 1, 2 o 3.")
            
        return conjuntos[conjunto]
    
    def convertir_hex_a_binario(self):
        datos_binarios = []
        for patron in self.datos_hex:
            patron_binario = []
            for fila_hex in patron:
                for bit in range(5):
                    patron_binario.append((fila_hex >> (4-bit)) & 1)
            datos_binarios.append(patron_binario)
        
        self.datos_binarios = np.array(datos_binarios, dtype=np.float32)
        return self.datos_binarios
    
    def obtener_datos_procesados(self):
        if self.datos_binarios is None:
            self.convertir_hex_a_binario()
        return self.datos_binarios
    
    def obtener_patron_como_matriz(self, indice):
        if self.datos_binarios is None:
            self.convertir_hex_a_binario()
        return self.datos_binarios[indice].reshape(7, 5)
    
    def mostrar_patron_ascii(self, indice, titulo=""):
        patron_2d = self.obtener_patron_como_matriz(indice)
        if titulo:
            print(f"\n{titulo}")
        for fila in patron_2d:
            print(''.join(['██' if pixel else '  ' for pixel in fila]))
    
    def obtener_estadisticas(self):
        if self.datos_binarios is None:
            self.convertir_hex_a_binario()
            
        num_patrones = len(self.datos_binarios)
        pixels_por_patron = self.datos_binarios.shape[1]
        densidad_promedio = np.mean(self.datos_binarios) * 100
        
        return {
            'num_patrones': num_patrones,
            'pixels_por_patron': pixels_por_patron,
            'densidad_promedio': densidad_promedio,
            'conjunto_usado': self.conjunto_datos
        }
    
    def validar_datos(self):
        if self.datos_binarios is None:
            self.convertir_hex_a_binario()
            
        errores = []
        
        if self.datos_binarios.shape[1] != 35:
            errores.append(f"Esperados 35 píxeles por patrón, encontrados {self.datos_binarios.shape[1]}")
            
        valores_unicos = np.unique(self.datos_binarios)
        if not np.array_equal(valores_unicos, [0., 1.]) and not np.array_equal(valores_unicos, [0.]) and not np.array_equal(valores_unicos, [1.]):
            errores.append(f"Datos no binarios encontrados: {valores_unicos}")
            
        if len(self.datos_binarios) != 32:
            errores.append(f"Esperados 32 patrones, encontrados {len(self.datos_binarios)}")
            
        return len(errores) == 0, errores
