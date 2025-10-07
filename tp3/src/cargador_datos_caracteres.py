import os
import sys
from typing import Tuple, List, Dict, Optional

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from comun.src.utilidades_matematicas import UtilidadesMatematicas


class CargadorDatosCaracteres:

    def __init__(self):
        self.datos_cargados = False
        self.patrones_caracteres = {}
        self.etiquetas_caracteres = []
        self.datos_entrada = None
        self.datos_salida = None
        self.ancho_caracter = 5
        self.alto_caracter = 7
        self.tamaño_entrada = self.ancho_caracter * self.alto_caracter

    def hex_a_binario(self, valor_hex: int) -> List[int]:
        binario = format(valor_hex, '05b')
        return [int(bit) for bit in binario]

    def caracter_hex_a_matriz(self, patron_hex: List[int]) -> np.ndarray:
        if len(patron_hex) != self.alto_caracter:
            raise ValueError(f"El patrón debe tener {self.alto_caracter} filas, "
                           f"pero tiene {len(patron_hex)}")
        
        matriz_binaria = []
        for fila_hex in patron_hex:
            fila_binaria = self.hex_a_binario(fila_hex)
            matriz_binaria.extend(fila_binaria)
        
        return np.array(matriz_binaria)

    def cargar_datos_desde_modulo(self, conjunto_datos: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datos'))
            import tp3_input
            
            if conjunto_datos == 1:
                datos_hex = tp3_input.entradas_1
            elif conjunto_datos == 2:
                datos_hex = tp3_input.entradas_2
            elif conjunto_datos == 3:
                datos_hex = tp3_input.entradas_3
            else:
                raise ValueError("conjunto_datos debe ser 1, 2, o 3")
            
            patrones = []
            etiquetas = []
            
            for i, patron_hex in enumerate(datos_hex):
                patron_lista = list(patron_hex)
                matriz_binaria = self.caracter_hex_a_matriz(patron_lista)
                patrones.append(matriz_binaria)
                etiquetas.append(i)
            
            datos_entrada = np.array(patrones)
            datos_salida = np.array(etiquetas)
            
            if datos_entrada.shape[1] != self.tamaño_entrada:
                raise ValueError(f"Los patrones deben tener {self.tamaño_entrada} elementos "
                               f"(7x5 píxeles), pero tienen {datos_entrada.shape[1]}")
            
            self.datos_entrada = datos_entrada
            self.datos_salida = datos_salida
            self._organizar_patrones_por_caracter()
            self.datos_cargados = True
            
            print(f"✅ Datos cargados exitosamente:")
            print(f"   - Conjunto: {conjunto_datos}")
            print(f"   - Total patrones: {len(patrones)}")
            print(f"   - Dimensiones: {datos_entrada.shape}")
            print(f"   - Tamaño por patrón: {self.tamaño_entrada} píxeles (7x5)")
            
            return datos_entrada, datos_salida
            
        except ImportError:
            raise ImportError("No se pudo importar el módulo tp3_input.py")
        except Exception as e:
            raise ValueError(f"Error al procesar los datos de caracteres: {str(e)}")

    def _organizar_patrones_por_caracter(self) -> None:
        self.patrones_caracteres = {}
        
        for i, etiqueta in enumerate(self.datos_salida):
            if etiqueta not in self.patrones_caracteres:
                self.patrones_caracteres[etiqueta] = []
            self.patrones_caracteres[etiqueta].append(self.datos_entrada[i])
        
        for caracter in self.patrones_caracteres:
            self.patrones_caracteres[caracter] = np.array(self.patrones_caracteres[caracter])

    def visualizar_patron(self, patron: np.ndarray) -> str:
        if len(patron) != self.tamaño_entrada:
            raise ValueError(f"El patrón debe tener {self.tamaño_entrada} elementos")
        
        matriz = patron.reshape(self.alto_caracter, self.ancho_caracter)
        
        lineas = []
        for fila in matriz:
            linea = ''.join(['█' if pixel == 1 else '·' for pixel in fila])
            lineas.append(linea)
        
        return '\n'.join(lineas)

    def mostrar_todos_los_patrones(self) -> None:
        if not self.datos_cargados:
            print("❌ No hay datos cargados. Use cargar_datos_desde_modulo() primero.")
            return
        
        print(f"\n📋 VISUALIZACIÓN DE TODOS LOS PATRONES")
        print(f"{'='*50}")
        
        for i, patron in enumerate(self.datos_entrada):
            print(f"\nPatrón {i:2d}:")
            print(self.visualizar_patron(patron))
            
            matriz = patron.reshape(self.alto_caracter, self.ancho_caracter)
            hex_values = []
            for fila in matriz:
                valor_hex = 0
                for j, bit in enumerate(fila):
                    valor_hex += bit * (2 ** (4-j))
                hex_values.append(f"0x{valor_hex:02x}")
            
            print(f"Hex: {', '.join(hex_values)}")

    def obtener_estadisticas_datos(self) -> Dict[str, any]:
        if not self.datos_cargados:
            return {}
        
        estadisticas = {
            'total_patrones': len(self.datos_entrada),
            'caracteres_disponibles': sorted(list(self.patrones_caracteres.keys())),
            'forma_entrada': self.datos_entrada.shape,
            'dimensiones_caracter': f"{self.alto_caracter}x{self.ancho_caracter}",
            'tamaño_entrada': self.tamaño_entrada,
            'rango_valores': (np.min(self.datos_entrada), np.max(self.datos_entrada)),
            'densidad_promedio': np.mean(self.datos_entrada)
        }
        
        return estadisticas

    def preparar_datos_autocodificador(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.datos_cargados:
            raise ValueError("Primero debe cargar los datos usando cargar_datos_desde_modulo()")
        
        return self.datos_entrada, self.datos_entrada

    def generar_datos_con_ruido(self, entradas: np.ndarray, 
                              probabilidad_ruido: float = 0.1) -> np.ndarray:
        return UtilidadesMatematicas.generar_ruido_binario(entradas, probabilidad_ruido)


def main():
    print("🚀 PROBANDO CARGADOR DE DATOS DE CARACTERES TP3")
    print("="*60)
    
    cargador = CargadorDatosCaracteres()
    
    try:
        datos_entrada, datos_salida = cargador.cargar_datos_desde_modulo(conjunto_datos=1)
        
        print(f"\n📊 ESTADÍSTICAS:")
        estadisticas = cargador.obtener_estadisticas_datos()
        for clave, valor in estadisticas.items():
            print(f"   {clave}: {valor}")
        
        print(f"\n🎨 EJEMPLOS DE PATRONES:")
        print("-" * 30)
        
        for i in range(min(5, len(datos_entrada))):
            print(f"\nPatrón {i}:")
            print(cargador.visualizar_patron(datos_entrada[i]))
        
        respuesta = input(f"\n¿Desea ver todos los {len(datos_entrada)} patrones? (s/n): ")
        if respuesta.lower() == 's':
            cargador.mostrar_todos_los_patrones()
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
