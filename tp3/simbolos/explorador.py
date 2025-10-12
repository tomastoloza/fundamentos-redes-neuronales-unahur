import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from tensorflow import keras
from tensorflow.keras import layers

from tp3.comun.cargador_modelos import CargadorModelos
from tp3.comun.procesador_datos import ProcesadorDatos


class ExploradorEspacioLatente:
    def __init__(self, modelo_path=None):
        self.procesador = ProcesadorDatos()
        self.datos = self.procesador.obtener_datos_procesados()
        self.cargador = CargadorModelos()
        
        self.encoder_func = None
        self.latentes = None
        self.dimension_latente = None
        
        if modelo_path:
            self.cargar_modelo(modelo_path)
        else:
            self.modelo = None
        

    def cargar_modelo(self, modelo_path):
        try:
            self.modelo = self.cargador.cargar_modelo(modelo_path)
            print(f"Modelo cargado: {modelo_path}")
            self.preparar_datos_latentes()
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            self.modelo = None
            return False
        return True
    
    def preparar_datos_latentes(self):
        entrada_encoder = self.modelo.input
        salida_encoder = self.modelo.get_layer('latente').output
        self.encoder_func = keras.Model(entrada_encoder, salida_encoder)
        
        self.latentes = self.encoder_func.predict(self.datos, verbose=0)
        self.dimension_latente = self.latentes.shape[1]
        
        print(f"Datos latentes preparados. Dimensión latente: {self.dimension_latente}")
    
    def generar_desde_latente(self, coordenadas_latentes):
        entrada_decoder = layers.Input(shape=(self.dimension_latente,))
        
        capas_decoder = []
        encontrado_latente = False
        for layer in self.modelo.layers:
            if layer.name == 'latente':
                encontrado_latente = True
                continue
            if encontrado_latente:
                capas_decoder.append(layer)
        
        x = entrada_decoder
        for layer in capas_decoder:
            x = layer(x)
        
        decoder_func = keras.Model(entrada_decoder, x)
        return decoder_func.predict(coordenadas_latentes, verbose=0)
    
    def mostrar_patron_ascii(self, patron_binario):
        print("Patrón generado (ASCII):")
        for fila in patron_binario:
            print(''.join(['██' if pixel else '  ' for pixel in fila]))
        print()
    
    def explorar_interactivo(self):
        if self.modelo is None:
            print("Error: No hay modelo cargado.")
            return
        
        if self.latentes is None:
            print("Error: No se pudieron preparar los datos latentes.")
            return
        
        if self.dimension_latente != 2:
            print("Error: El explorador solo funciona con modelos 2D.")
            print(f"Modelo actual tiene dimensión latente: {self.dimension_latente}D")
            print("Use un modelo con dimension_latente=2 (ej: simple_2d, profundo_2d)")
            return
            
        fig = plt.figure(figsize=(15, 6))
        
        ax_latente = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
        ax_patron = plt.subplot2grid((2, 4), (0, 2))
        ax_reconstruido = plt.subplot2grid((2, 4), (0, 3))
        ax_slider_x = plt.subplot2grid((2, 4), (1, 2))
        ax_slider_y = plt.subplot2grid((2, 4), (1, 3))

        ax_latente.scatter(self.latentes[:, 0], self.latentes[:, 1],
                                   c=range(len(self.datos)), cmap='tab20', s=100, alpha=0.7)
        
        for i, (x, y) in enumerate(self.latentes):
            ax_latente.annotate(f'{i}', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        ax_latente.set_title('Espacio Latente 2D\n(Click para generar)')
        ax_latente.set_xlabel('Dimensión 1')
        ax_latente.set_ylabel('Dimensión 2')
        ax_latente.grid(True, alpha=0.3)
        
        punto_actual, = ax_latente.plot([], [], 'ro', markersize=10, label='Punto actual')
        ax_latente.legend()
        
        im_patron = ax_patron.imshow(np.zeros((7, 5)), cmap='gray_r', interpolation='nearest', vmin=0, vmax=1)
        ax_patron.set_title('Patrón Original')
        ax_patron.axis('off')
        
        im_reconstruido = ax_reconstruido.imshow(np.zeros((7, 5)), cmap='gray_r', interpolation='nearest', vmin=0, vmax=1)
        ax_reconstruido.set_title('Patrón Generado')
        ax_reconstruido.axis('off')
        
        x_min, x_max = self.latentes[:, 0].min() - 5, self.latentes[:, 0].max() + 5
        y_min, y_max = self.latentes[:, 1].min() - 5, self.latentes[:, 1].max() + 5
        
        slider_x = Slider(ax_slider_x, 'X', x_min, x_max, valinit=0.0)
        slider_y = Slider(ax_slider_y, 'Y', y_min, y_max, valinit=0.0)
        
        def actualizar_patron(x, y):
            punto_actual.set_data([x], [y])
            
            distancias = np.sqrt((self.latentes[:, 0] - x)**2 + (self.latentes[:, 1] - y)**2)
            idx_cercano = np.argmin(distancias)
            
            if distancias[idx_cercano] < 2.0:
                patron_original = self.datos[idx_cercano].reshape(7, 5)
                im_patron.set_array(patron_original)
                ax_patron.set_title(f'Patrón Original {idx_cercano}')
            else:
                im_patron.set_array(np.zeros((7, 5)))
                ax_patron.set_title('Patrón Original (ninguno)')
            
            nuevo_patron = self.generar_desde_latente(np.array([[x, y]]))
            nuevo_patron_bin = (nuevo_patron > 0.5).astype(int).reshape(7, 5)
            im_reconstruido.set_array(nuevo_patron_bin)
            
            ax_reconstruido.set_title(f'Generado (x={x:.2f}, y={y:.2f})\nBits activos: {nuevo_patron_bin.sum()}')
            
            if abs(x) < 0.1 and abs(y) < 0.1:
                print(f"Coordenadas: ({x:.2f}, {y:.2f})")
                print(f"Valores continuos (min={nuevo_patron.min():.3f}, max={nuevo_patron.max():.3f})")
                self.mostrar_patron_ascii(nuevo_patron_bin)
            
            fig.canvas.draw()
        
        def on_click(event):
            if event.inaxes == ax_latente:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    slider_x.set_val(x)
                    slider_y.set_val(y)
                    actualizar_patron(x, y)
        
        def on_slider_change(val):
            actualizar_patron(slider_x.val, slider_y.val)
        
        slider_x.on_changed(on_slider_change)
        slider_y.on_changed(on_slider_change)
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        actualizar_patron(0.0, 0.0)
        
        plt.suptitle('Explorador Interactivo del Espacio Latente\n'
                    'Click en el espacio latente o usa los sliders para explorar', fontsize=14)
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Explorador interactivo del espacio latente')
    parser.add_argument('modelo', type=str,
                       help='Nombre del modelo a cargar (ej: tp3_lat2_ep300_lr0_001)')
    
    args = parser.parse_args()

    explorador = ExploradorEspacioLatente(args.modelo)
    if explorador.modelo is not None:
        explorador.explorar_interactivo()
    else:
        print("No se pudo cargar el modelo especificado.")

if __name__ == "__main__":
    main()
