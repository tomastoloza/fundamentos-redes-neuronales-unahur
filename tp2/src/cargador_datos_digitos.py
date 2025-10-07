import os
import sys
from typing import Tuple, List, Dict, Optional

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from comun.src.utilidades_matematicas import UtilidadesMatematicas
from comun.constantes.constantes_redes_neuronales import (
    TAMAÑO_ENTRADA_DIGITOS, ARCHIVO_DATOS_DIGITOS
)

class CargadorDatosDigitos:

    def __init__(self):

        self.datos_cargados = False
        self.patrones_digitos = {}
        self.etiquetas_digitos = []
        self.datos_entrada = None
        self.datos_salida = None
    
    def cargar_datos_desde_archivo(self, ruta_archivo: str) -> Tuple[np.ndarray, np.ndarray]:

        try:
            with open(ruta_archivo, 'r') as archivo:
                lineas = archivo.readlines()
            
            lineas = [linea.strip() for linea in lineas if linea.strip()]
            
            patrones = []
            etiquetas = []
            
            lineas_por_digito = 7
            total_digitos = 10
            
            if len(lineas) != lineas_por_digito * total_digitos:
                return self._cargar_formato_con_etiquetas(lineas)
            
            for digito in range(total_digitos):
                inicio = digito * lineas_por_digito
                fin = inicio + lineas_por_digito
                
                patron_actual = []
                for i in range(inicio, fin):
                    if i < len(lineas):
                        linea = lineas[i]
                        fila = [int(c) for c in linea if c in '01']
                        patron_actual.extend(fila)
                
                if len(patron_actual) == TAMAÑO_ENTRADA_DIGITOS:
                    patrones.append(np.array(patron_actual))
                    etiquetas.append(digito)
            
            datos_entrada = np.array(patrones)
            datos_salida = np.array(etiquetas)
            
            if len(patrones) == 0:
                raise ValueError("No se pudieron extraer patrones válidos del archivo")
            
            if datos_entrada.shape[1] != TAMAÑO_ENTRADA_DIGITOS:
                raise ValueError(f"Los patrones deben tener {TAMAÑO_ENTRADA_DIGITOS} elementos "
                               f"(5x7 píxeles), pero tienen {datos_entrada.shape[1]}")
            
            self.datos_entrada = datos_entrada
            self.datos_salida = datos_salida
            self._organizar_patrones_por_digito()
            self.datos_cargados = True
            
            return datos_entrada, datos_salida
            
        except FileNotFoundError:
            raise FileNotFoundError(f"No se pudo encontrar el archivo: {ruta_archivo}")
        except Exception as e:
            raise ValueError(f"Error al procesar el archivo de dígitos: {str(e)}")
    
    def _cargar_formato_con_etiquetas(self, lineas: List[str]) -> Tuple[np.ndarray, np.ndarray]:

        patrones = []
        etiquetas = []
        patron_actual = []
        digito_actual = None
        
        for linea in lineas:
            if linea.startswith('Dígito') or linea.startswith('Digito'):
                if patron_actual and digito_actual is not None:
                    patrones.append(np.array(patron_actual))
                    etiquetas.append(digito_actual)
                
                digito_actual = int(linea.split()[-1])
                patron_actual = []
            
            elif linea and all(c in '01 ' for c in linea):
                fila = [int(c) for c in linea if c in '01']
                patron_actual.extend(fila)
        
        if patron_actual and digito_actual is not None:
            patrones.append(np.array(patron_actual))
            etiquetas.append(digito_actual)
        
        return np.array(patrones), np.array(etiquetas)
    
    def _organizar_patrones_por_digito(self) -> None:

        self.patrones_digitos = {}
        
        for i, etiqueta in enumerate(self.datos_salida):
            if etiqueta not in self.patrones_digitos:
                self.patrones_digitos[etiqueta] = []
            self.patrones_digitos[etiqueta].append(self.datos_entrada[i])
        
        for digito in self.patrones_digitos:
            self.patrones_digitos[digito] = np.array(self.patrones_digitos[digito])
    
    def cargar_datos_tp2(self, directorio_datos: str = None) -> Tuple[np.ndarray, np.ndarray]:

        if directorio_datos is None:
            directorio_actual = os.path.dirname(__file__)
            directorio_datos = os.path.join(directorio_actual, '..', 'datos')
        
        ruta_archivo = os.path.join(directorio_datos, ARCHIVO_DATOS_DIGITOS)
        return self.cargar_datos_desde_archivo(ruta_archivo)
    
    def dividir_datos_por_digitos(self, digitos_entrenamiento: List[int],
                                digitos_prueba: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if not self.datos_cargados:
            raise ValueError("Primero debe cargar los datos usando cargar_datos_desde_archivo()")
        
        entradas_train = []
        salidas_train = []
        
        for digito in digitos_entrenamiento:
            if digito in self.patrones_digitos:
                patrones = self.patrones_digitos[digito]
                entradas_train.extend(patrones)
                salidas_train.extend([digito] * len(patrones))
        
        entradas_test = []
        salidas_test = []
        
        for digito in digitos_prueba:
            if digito in self.patrones_digitos:
                patrones = self.patrones_digitos[digito]
                entradas_test.extend(patrones)
                salidas_test.extend([digito] * len(patrones))
        
        return (np.array(entradas_train), np.array(salidas_train),
                np.array(entradas_test), np.array(salidas_test))
    
    def preparar_datos_clasificacion_binaria(self, digitos_positivos: List[int],
                                           digitos_negativos: List[int]) -> Tuple[np.ndarray, np.ndarray]:

        if not self.datos_cargados:
            raise ValueError("Primero debe cargar los datos usando cargar_datos_desde_archivo()")
        
        entradas = []
        salidas = []
        
        for digito in digitos_positivos:
            if digito in self.patrones_digitos:
                patrones = self.patrones_digitos[digito]
                entradas.extend(patrones)
                salidas.extend([1] * len(patrones))
        
        for digito in digitos_negativos:
            if digito in self.patrones_digitos:
                patrones = self.patrones_digitos[digito]
                entradas.extend(patrones)
                salidas.extend([-1] * len(patrones))
        
        return np.array(entradas), np.array(salidas)
    
    def preparar_datos_clasificacion_multiclase(self) -> Tuple[np.ndarray, np.ndarray]:

        if not self.datos_cargados:
            raise ValueError("Primero debe cargar los datos usando cargar_datos_desde_archivo()")
        
        entradas = self.datos_entrada
        salidas = self.datos_salida
        
        return entradas, salidas
    
    def generar_datos_con_ruido(self, entradas: np.ndarray, 
                              probabilidad_ruido: float = 0.02) -> np.ndarray:

        return UtilidadesMatematicas.generar_ruido_binario(entradas, probabilidad_ruido)
    
    def obtener_patron_digito(self, digito: int, indice: int = 0) -> Optional[np.ndarray]:

        if not self.datos_cargados:
            return None
        
        if digito not in self.patrones_digitos:
            return None
        
        patrones = self.patrones_digitos[digito]
        if indice >= len(patrones):
            return None
        
        return patrones[indice]
    
    def visualizar_patron(self, patron: np.ndarray, ancho: int = 5, alto: int = 7) -> str:

        if len(patron) != ancho * alto:
            raise ValueError(f"El patrón debe tener {ancho * alto} elementos")
        
        matriz = patron.reshape(alto, ancho)
        
        lineas = []
        for fila in matriz:
            linea = ''.join(['█' if pixel == 1 else '·' for pixel in fila])
            lineas.append(linea)
        
        return '\n'.join(lineas)
    
    def obtener_estadisticas_datos(self) -> Dict[str, any]:

        if not self.datos_cargados:
            return {}
        
        estadisticas = {
            'total_patrones': len(self.datos_entrada),
            'digitos_disponibles': sorted(list(self.patrones_digitos.keys())),
            'patrones_por_digito': {digito: len(patrones) 
                                  for digito, patrones in self.patrones_digitos.items()},
            'forma_entrada': self.datos_entrada.shape,
            'rango_valores': (np.min(self.datos_entrada), np.max(self.datos_entrada)),
            'densidad_promedio': np.mean(self.datos_entrada)
        }
        
        return estadisticas
    
    def crear_division_entrenamiento_prueba_estandar(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        digitos_entrenamiento = [0, 1, 2, 3, 4, 5, 6]
        digitos_prueba = [7, 8, 9]
        
        return self.dividir_datos_por_digitos(digitos_entrenamiento, digitos_prueba)
