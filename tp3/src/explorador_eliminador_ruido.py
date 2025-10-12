import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from tensorflow import keras

from .cargador_modelos import CargadorModelos
from .procesador_datos import ProcesadorDatos
from .generador_ruido import GeneradorRuido


class ExploradorEliminadorRuido:
    def __init__(self, modelo_path=None):
        self.procesador = ProcesadorDatos()
        self.datos_limpios = self.procesador.obtener_datos_procesados()
        self.cargador = CargadorModelos()
        self.generador_ruido = GeneradorRuido()
        
        self.modelo = None
        self.tipo_ruido_actual = 'binario'
        self.nivel_ruido_actual = 0.1
        self.caracter_actual = 0
        
        if modelo_path:
            self.cargar_modelo(modelo_path)
    
    def cargar_modelo(self, modelo_path):
        try:
            self.modelo = self.cargador.cargar_modelo(modelo_path)
            print(f"Modelo cargado: {modelo_path}")
            
            # Extraer información del nombre del modelo
            if '_binario_' in modelo_path:
                self.tipo_ruido_actual = 'binario'
            elif '_gaussiano_' in modelo_path:
                self.tipo_ruido_actual = 'gaussiano'
            elif '_dropout_' in modelo_path:
                self.tipo_ruido_actual = 'dropout'
            
            print(f"Tipo de ruido detectado: {self.tipo_ruido_actual}")
            
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            self.modelo = None
            return False
        return True
    
    def generar_datos_ruidosos(self, caracter_idx, tipo_ruido, nivel_ruido):
        datos_caracter = self.datos_limpios[caracter_idx:caracter_idx+1]
        return self.generador_ruido.generar_conjunto_ruidoso(datos_caracter, tipo_ruido, nivel_ruido)
    
    def limpiar_ruido(self, datos_ruidosos):
        if self.modelo is None:
            return datos_ruidosos
        return self.modelo.predict(datos_ruidosos, verbose=0)
    
    def calcular_metricas(self, datos_limpios, datos_ruidosos, datos_reconstruidos):
        mse_limpio = np.mean((datos_limpios - datos_reconstruidos) ** 2)
        mse_ruidoso = np.mean((datos_ruidosos - datos_reconstruidos) ** 2)
        precision_limpieza = np.mean((datos_reconstruidos > 0.5) == (datos_limpios > 0.5))
        mejora_snr = self.generador_ruido.calcular_mejora_snr(datos_limpios, datos_ruidosos, datos_reconstruidos)
        
        return {
            'mse_limpio': mse_limpio,
            'mse_ruidoso': mse_ruidoso,
            'precision_limpieza': precision_limpieza,
            'mejora_snr': mejora_snr
        }
    
    def mostrar_patron_ascii(self, patron_binario, titulo="Patrón"):
        print(f"\n{titulo}:")
        for fila in patron_binario:
            print(''.join(['██' if pixel > 0.5 else '  ' for pixel in fila]))
        print()
    
    def explorar_interactivo(self):
        if self.modelo is None:
            print("Error: No hay modelo cargado.")
            return
        
        fig = plt.figure(figsize=(18, 8))
        
        # Layout: 3 filas, 6 columnas
        ax_original = plt.subplot2grid((3, 6), (0, 0))
        ax_ruidoso = plt.subplot2grid((3, 6), (0, 1))
        ax_limpio = plt.subplot2grid((3, 6), (0, 2))
        ax_diferencia = plt.subplot2grid((3, 6), (0, 3))
        
        # Controles
        ax_caracter = plt.subplot2grid((3, 6), (1, 0))
        ax_tipo_ruido = plt.subplot2grid((3, 6), (1, 1), colspan=2)
        ax_nivel_ruido = plt.subplot2grid((3, 6), (1, 3))
        
        # Métricas
        ax_metricas = plt.subplot2grid((3, 6), (0, 4), colspan=2, rowspan=2)
        
        # Botones
        ax_btn_binario = plt.subplot2grid((3, 6), (2, 0))
        ax_btn_gaussiano = plt.subplot2grid((3, 6), (2, 1))
        ax_btn_dropout = plt.subplot2grid((3, 6), (2, 2))
        ax_btn_regenerar = plt.subplot2grid((3, 6), (2, 3))
        
        # Inicializar imágenes
        im_original = ax_original.imshow(np.zeros((7, 5)), cmap='gray_r', interpolation='nearest', vmin=0, vmax=1)
        ax_original.set_title('Original')
        ax_original.axis('off')
        
        im_ruidoso = ax_ruidoso.imshow(np.zeros((7, 5)), cmap='gray_r', interpolation='nearest', vmin=0, vmax=1)
        ax_ruidoso.set_title('Con Ruido')
        ax_ruidoso.axis('off')
        
        im_limpio = ax_limpio.imshow(np.zeros((7, 5)), cmap='gray_r', interpolation='nearest', vmin=0, vmax=1)
        ax_limpio.set_title('Reconstruido')
        ax_limpio.axis('off')
        
        im_diferencia = ax_diferencia.imshow(np.zeros((7, 5)), cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
        ax_diferencia.set_title('Diferencia')
        ax_diferencia.axis('off')
        
        # Sliders
        slider_caracter = Slider(ax_caracter, 'Carácter', 0, len(self.datos_limpios)-1, 
                               valinit=self.caracter_actual, valfmt='%d')
        
        # Rangos de ruido según el tipo
        rangos_ruido = {
            'binario': (0.01, 0.5),
            'gaussiano': (0.05, 0.8),
            'dropout': (0.05, 0.7)
        }
        
        rango_min, rango_max = rangos_ruido[self.tipo_ruido_actual]
        slider_nivel = Slider(ax_nivel_ruido, 'Nivel Ruido', rango_min, rango_max,
                            valinit=self.nivel_ruido_actual, valfmt='%.2f')
        
        # Texto de métricas
        texto_metricas = ax_metricas.text(0.05, 0.95, '', transform=ax_metricas.transAxes,
                                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax_metricas.set_xlim(0, 1)
        ax_metricas.set_ylim(0, 1)
        ax_metricas.axis('off')
        ax_metricas.set_title('Métricas de Limpieza')
        
        # Botones
        btn_binario = Button(ax_btn_binario, 'Binario')
        btn_gaussiano = Button(ax_btn_gaussiano, 'Gaussiano')
        btn_dropout = Button(ax_btn_dropout, 'Dropout')
        btn_regenerar = Button(ax_btn_regenerar, 'Regenerar')
        
        def actualizar_visualizacion():
            caracter_idx = int(slider_caracter.val)
            nivel_ruido = slider_nivel.val
            
            # Datos originales
            datos_original = self.datos_limpios[caracter_idx:caracter_idx+1]
            patron_original = datos_original.reshape(7, 5)
            
            # Generar ruido
            datos_ruidosos = self.generar_datos_ruidosos(caracter_idx, self.tipo_ruido_actual, nivel_ruido)
            patron_ruidoso = datos_ruidosos.reshape(7, 5)
            
            # Limpiar ruido
            datos_reconstruidos = self.limpiar_ruido(datos_ruidosos)
            patron_reconstruido = datos_reconstruidos.reshape(7, 5)
            
            # Diferencia
            diferencia = patron_reconstruido - patron_original
            
            # Actualizar imágenes
            im_original.set_array(patron_original)
            im_ruidoso.set_array(patron_ruidoso)
            im_limpio.set_array(patron_reconstruido)
            im_diferencia.set_array(diferencia)
            
            # Calcular métricas
            metricas = self.calcular_metricas(datos_original, datos_ruidosos, datos_reconstruidos)
            
            # Actualizar títulos
            ax_original.set_title(f'Original (Carácter {caracter_idx})')
            ax_ruidoso.set_title(f'Con Ruido ({self.tipo_ruido_actual.title()})')
            ax_limpio.set_title('Reconstruido')
            ax_diferencia.set_title('Diferencia (Recons - Orig)')
            
            # Actualizar métricas
            texto_metricas_str = f"""MÉTRICAS DE LIMPIEZA
            
Tipo de ruido: {self.tipo_ruido_actual.upper()}
Nivel de ruido: {nivel_ruido:.3f}

MSE vs Original: {metricas['mse_limpio']:.6f}
MSE vs Ruidoso:  {metricas['mse_ruidoso']:.6f}
Precisión:       {metricas['precision_limpieza']:.1%}
Mejora SNR:      {metricas['mejora_snr']:.2f} dB

Píxeles originales:     {int(patron_original.sum())}
Píxeles con ruido:      {int(patron_ruidoso.sum())}
Píxeles reconstruidos:  {int((patron_reconstruido > 0.5).sum())}

Efectividad: {'✓ BUENA' if metricas['mejora_snr'] > -10 else '⚠ REGULAR' if metricas['mejora_snr'] > -20 else '✗ MALA'}
"""
            texto_metricas.set_text(texto_metricas_str)
            
            # Mostrar en consola si es interesante
            if abs(metricas['mejora_snr']) < 5 or metricas['precision_limpieza'] > 0.95:
                print(f"\n=== Resultado destacado ===")
                print(f"Carácter {caracter_idx}, {self.tipo_ruido_actual} {nivel_ruido:.3f}")
                print(f"Precisión: {metricas['precision_limpieza']:.1%}, SNR: {metricas['mejora_snr']:.2f}dB")
                self.mostrar_patron_ascii(patron_original, "Original")
                self.mostrar_patron_ascii(patron_ruidoso, "Con ruido")
                self.mostrar_patron_ascii(patron_reconstruido > 0.5, "Reconstruido")
            
            fig.canvas.draw()
        
        def cambiar_tipo_ruido(nuevo_tipo):
            self.tipo_ruido_actual = nuevo_tipo
            # Actualizar rango del slider
            rango_min, rango_max = rangos_ruido[nuevo_tipo]
            slider_nivel.valmin = rango_min
            slider_nivel.valmax = rango_max
            slider_nivel.ax.set_xlim(rango_min, rango_max)
            
            # Ajustar valor actual si está fuera del rango
            if slider_nivel.val < rango_min:
                slider_nivel.set_val(rango_min)
            elif slider_nivel.val > rango_max:
                slider_nivel.set_val(rango_max)
            
            actualizar_visualizacion()
        
        # Conectar eventos
        slider_caracter.on_changed(lambda val: actualizar_visualizacion())
        slider_nivel.on_changed(lambda val: actualizar_visualizacion())
        
        btn_binario.on_clicked(lambda event: cambiar_tipo_ruido('binario'))
        btn_gaussiano.on_clicked(lambda event: cambiar_tipo_ruido('gaussiano'))
        btn_dropout.on_clicked(lambda event: cambiar_tipo_ruido('dropout'))
        btn_regenerar.on_clicked(lambda event: actualizar_visualizacion())
        
        # Visualización inicial
        actualizar_visualizacion()
        
        plt.suptitle(f'Explorador Eliminador de Ruido - Modelo: {self.modelo.name if hasattr(self.modelo, "name") else "Cargado"}', 
                    fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Explorador interactivo del eliminador de ruido')
    parser.add_argument('modelo', type=str,
                       help='Nombre del modelo de eliminación de ruido a cargar (ej: tp3_lat2_ep1500_lr0_001_binario_0_1)')
    
    args = parser.parse_args()
    
    print("=== EXPLORADOR ELIMINADOR DE RUIDO ===")
    print("TP3 - Punto 1.2: Eliminación de ruido en caracteres")
    print()
    
    explorador = ExploradorEliminadorRuido(args.modelo)
    if explorador.modelo is not None:
        explorador.explorar_interactivo()
    else:
        print("No se pudo cargar el modelo especificado.")
        print("Ejemplo de uso:")
        print("python -m tp3.src.explorador_eliminador_ruido tp3_lat2_ep1500_lr0_001_binario_0_1")


if __name__ == "__main__":
    main()
