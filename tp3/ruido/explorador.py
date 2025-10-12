import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from tp3.comun.explorador_base import ExploradorBase
from tp3.comun.cargador_modelos import CargadorModelos
from tp3.comun.procesador_datos import ProcesadorDatos
from tp3.comun.generador_ruido import GeneradorRuido


class ExploradorEliminadorRuido(ExploradorBase):
    def __init__(self, modelo_path=None):
        super().__init__()
        self.procesador = ProcesadorDatos()
        self.datos_limpios = self.procesador.obtener_datos_procesados()
        self.cargador = CargadorModelos()
        self.generador_ruido = GeneradorRuido()
        
        self.modelo = None
        self.tipos_ruido = ['binario', 'gaussiano', 'salt_pepper']
        self.niveles_ruido = [0.05, 0.1, 0.15, 0.2]
        self.tipo_ruido_actual = 0
        self.nivel_ruido_actual = 0
        
        if modelo_path:
            self.cargar_modelo(modelo_path)
    
    def cargar_modelo(self, modelo_path):
        try:
            self.modelo = self.cargador.cargar_modelo(modelo_path)
            if self.modelo:
                print(f"✓ Modelo cargado: {modelo_path}")
                return True
        except Exception as e:
            print(f"✗ Error cargando modelo: {e}")
        return False
    
    def obtener_maximo_indice(self):
        return len(self.datos_limpios)
    
    def cambiar_parametro_arriba(self):
        if self.nivel_ruido_actual < len(self.niveles_ruido) - 1:
            self.nivel_ruido_actual += 1
            print(f"Nivel de ruido: {self.niveles_ruido[self.nivel_ruido_actual]}")
    
    def cambiar_parametro_abajo(self):
        if self.nivel_ruido_actual > 0:
            self.nivel_ruido_actual -= 1
            print(f"Nivel de ruido: {self.niveles_ruido[self.nivel_ruido_actual]}")
    
    def regenerar(self):
        if self.tipo_ruido_actual < len(self.tipos_ruido) - 1:
            self.tipo_ruido_actual += 1
        else:
            self.tipo_ruido_actual = 0
        print(f"Tipo de ruido: {self.tipos_ruido[self.tipo_ruido_actual]}")
    
    def generar_nuevo(self):
        print("Generando nueva muestra de ruido...")
    
    def crear_visualizacion_inicial(self):
        if not self.modelo:
            print("Error: No hay modelo cargado")
            return None, None
        
        fig, axes = self.crear_figura_base(
            "Explorador Eliminador de Ruido - TP3 Punto 2", 
            filas=2, columnas=2, figsize=(12, 8)
        )
        
        self.fig = fig
        self.axes = axes
        
        self.actualizar_visualizacion()
        return fig, axes
    
    def actualizar_visualizacion(self):
        if not self.modelo or not hasattr(self, 'axes'):
            return
        
        # Limpiar axes
        for ax in self.axes:
            ax.clear()
        
        # Obtener datos actuales
        patron_limpio = self.datos_limpios[self.indice_actual]
        tipo_ruido = self.tipos_ruido[self.tipo_ruido_actual]
        nivel_ruido = self.niveles_ruido[self.nivel_ruido_actual]
        
        # Generar ruido
        patron_ruidoso = self.generador_ruido.generar_conjunto_ruidoso(
            patron_limpio.reshape(1, -1), tipo_ruido, nivel_ruido
        )[0]
        
        # Reconstruir
        patron_reconstruido = self.modelo.predict(
            patron_ruidoso.reshape(1, -1), verbose=0
        )[0]
        
        # Visualizar patrones
        self.configurar_subplot_patron(
            self.axes[0], patron_limpio, f"Original (Carácter {self.indice_actual})", 'Greens'
        )
        
        self.configurar_subplot_patron(
            self.axes[1], patron_ruidoso, f"Ruidoso ({tipo_ruido} {nivel_ruido})", 'Reds'
        )
        
        self.configurar_subplot_patron(
            self.axes[2], patron_reconstruido, "Reconstruido", 'Blues'
        )
        
        # Métricas y controles
        self.mostrar_metricas_y_controles(
            self.axes[3], patron_limpio, patron_ruidoso, patron_reconstruido,
            tipo_ruido, nivel_ruido
        )
        
        # Mostrar patrones ASCII
        self.mostrar_patron_ascii(patron_limpio, f"Original (Carácter {self.indice_actual})")
        self.mostrar_patron_ascii(patron_ruidoso, f"Ruidoso ({tipo_ruido} {nivel_ruido})")
        self.mostrar_patron_ascii(patron_reconstruido, "Reconstruido")
        
        plt.draw()
    
    def mostrar_metricas_y_controles(self, ax, limpio, ruidoso, reconstruido, tipo_ruido, nivel_ruido):
        ax.axis('off')
        
        # Calcular métricas
        mse_limpio = np.mean((limpio - reconstruido) ** 2)
        mse_ruidoso = np.mean((ruidoso - reconstruido) ** 2)
        mejora_mse = ((mse_ruidoso - mse_limpio) / mse_ruidoso) * 100 if mse_ruidoso > 0 else 0
        
        precision = np.mean((reconstruido > 0.5) == (limpio > 0.5))
        
        snr_mejora = self.generador_ruido.calcular_mejora_snr(
            limpio.reshape(1, -1), 
            ruidoso.reshape(1, -1), 
            reconstruido.reshape(1, -1)
        )
        
        # Mostrar información
        info_text = f"""MÉTRICAS DE ELIMINACIÓN DE RUIDO
        
Carácter: {self.indice_actual + 1}/{len(self.datos_limpios)}
Tipo de ruido: {tipo_ruido}
Nivel de ruido: {nivel_ruido}

MSE (limpio vs reconstruido): {mse_limpio:.4f}
MSE (ruidoso vs reconstruido): {mse_ruidoso:.4f}
Mejora MSE: {mejora_mse:.1f}%

Precisión binaria: {precision:.1%}
Mejora SNR: {snr_mejora:.2f} dB

Efectivo: {'✓ Sí' if mejora_mse > 0 else '✗ No'}

CONTROLES:
← → : Navegar caracteres
↑ ↓ : Cambiar nivel de ruido
R   : Cambiar tipo de ruido
G   : Generar nueva muestra
H   : Mostrar ayuda
Q   : Salir"""
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=9)
    
    def explorar_interactivo(self):
        if not self.modelo:
            print("Error: Debe cargar un modelo primero")
            return
        
        print("=== EXPLORADOR ELIMINADOR DE RUIDO ===")
        print("Navegue con las teclas para explorar la eliminación de ruido")
        self.iniciar_exploracion_interactiva()


def main():
    parser = argparse.ArgumentParser(description='Explorador interactivo de eliminación de ruido')
    parser.add_argument('modelo', type=str, nargs='?',
                       help='Nombre o ruta del modelo a cargar')
    parser.add_argument('--listar', action='store_true',
                       help='Listar modelos disponibles')
    parser.add_argument('--mejor', action='store_true',
                       help='Usar el mejor modelo disponible')
    
    args = parser.parse_args()
    
    if args.listar:
        cargador = CargadorModelos()
        modelos = cargador.listar_modelos()
        print("Modelos disponibles:")
        for modelo in modelos:
            print(f"  - {modelo}")
        return
    
    modelo_path = None
    if args.mejor:
        # Buscar el mejor modelo (esto requeriría lógica adicional)
        print("Buscando el mejor modelo...")
        modelo_path = "tp3_lat2_ep300_lr0_001"  # Placeholder
    elif args.modelo:
        modelo_path = args.modelo
    
    print("=== EXPLORADOR ELIMINADOR DE RUIDO ===")
    print("TP3 - Punto 2: Eliminación de ruido en caracteres")
    print()
    
    explorador = ExploradorEliminadorRuido(modelo_path)
    if explorador.modelo is not None:
        explorador.explorar_interactivo()
    else:
        print("No se pudo cargar el modelo especificado.")
        print("Ejemplo de uso:")
        print("python -m tp3.punto2.explorador tp3_lat2_ep300_lr0_001")


if __name__ == "__main__":
    main()
