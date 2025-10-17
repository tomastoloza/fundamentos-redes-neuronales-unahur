import argparse
import numpy as np
import matplotlib.pyplot as plt
from tp3.comun.explorador_base import ExploradorBase
from tp3.comun.cargador_modelos import CargadorModelos
from .procesador_imagenes import ProcesadorImagenes


class ExploradorEspacioLatenteImagenes(ExploradorBase):
    def __init__(self, modelo_path=None):
        super().__init__()
        self.procesador = ProcesadorImagenes()
        self.cargador = CargadorModelos()
        self.modelo = None
        self.encoder = None
        self.decoder = None
        self.datos_imagenes = None
        self.representaciones_latentes = None
        self.forma_imagen = None
        
        if modelo_path:
            self.cargar_modelo(modelo_path)
    
    def cargar_modelo(self, modelo_path):
        try:
            self.modelo = self.cargador.cargar_modelo(modelo_path)
            if self.modelo:
                print(f"✓ Modelo cargado: {modelo_path}")
                self.extraer_encoder_decoder()
                return True
        except Exception as e:
            print(f"✗ Error cargando modelo: {e}")
        return False
    
    def extraer_encoder_decoder(self):
        if not self.modelo:
            return
        
        try:
            capas = self.modelo.layers
            punto_medio = len(capas) // 2
            
            from tensorflow.keras.models import Model
            
            self.encoder = Model(
                inputs=self.modelo.input,
                outputs=capas[punto_medio - 1].output
            )
            
            entrada_decoder = self.encoder.output
            salida_decoder = entrada_decoder
            
            for capa in capas[punto_medio:]:
                salida_decoder = capa(salida_decoder)
            
            self.decoder = Model(
                inputs=entrada_decoder,
                outputs=salida_decoder
            )
            
            print(f"✓ Encoder y decoder extraídos")
            print(f"  Dimensión latente: {self.encoder.output.shape[1]}")
            
        except Exception as e:
            print(f"⚠️  Error extrayendo encoder/decoder: {e}")
    
    def cargar_datos(self, tamaño_imagen=(64, 64), max_imagenes=100):
        print("Cargando datos para exploración...")
        imagenes = self.procesador.cargar_imagenes(tamaño_imagen, max_imagenes)
        self.datos_imagenes = self.procesador.obtener_datos_aplanados()
        self.forma_imagen = self.procesador.obtener_forma_original()
        
        if self.encoder:
            print("Calculando representaciones latentes...")
            self.representaciones_latentes = self.encoder.predict(self.datos_imagenes, verbose=0)
            print(f"✓ {len(self.representaciones_latentes)} representaciones calculadas")
        
        return self.datos_imagenes
    
    def obtener_maximo_indice(self):
        return len(self.datos_imagenes) if self.datos_imagenes is not None else 0
    
    def crear_visualizacion_inicial(self):
        if not self.modelo or self.datos_imagenes is None:
            print("Error: Modelo o datos no cargados")
            return None, None
        
        fig = plt.figure(figsize=(15, 10))
        
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        ax_original = fig.add_subplot(gs[0, 0])
        ax_reconstruida = fig.add_subplot(gs[0, 1])
        ax_latente = fig.add_subplot(gs[0, 2])
        ax_controles = fig.add_subplot(gs[1, :])
        
        self.axes = [ax_original, ax_reconstruida, ax_latente, ax_controles]
        self.fig = fig
        
        fig.suptitle("Explorador de Espacio Latente - Imágenes", fontsize=16)
        
        self.actualizar_visualizacion()
        return fig, self.axes
    
    def actualizar_visualizacion(self):
        if not self.modelo or self.datos_imagenes is None or not hasattr(self, 'axes'):
            return
        
        for ax in self.axes:
            ax.clear()
        
        imagen_actual = self.datos_imagenes[self.indice_actual]
        imagen_forma = self.procesador.reconstruir_forma_imagen(imagen_actual.reshape(1, -1))[0]
        
        reconstruccion = self.modelo.predict(imagen_actual.reshape(1, -1), verbose=0)[0]
        reconstruccion_forma = self.procesador.reconstruir_forma_imagen(reconstruccion.reshape(1, -1))[0]
        
        self.axes[0].imshow(imagen_forma)
        self.axes[0].set_title(f"Original (Imagen {self.indice_actual + 1})")
        self.axes[0].axis('off')
        
        self.axes[1].imshow(reconstruccion_forma)
        self.axes[1].set_title("Reconstruida")
        self.axes[1].axis('off')
        
        if self.representaciones_latentes is not None:
            self.visualizar_espacio_latente()
        
        self.mostrar_metricas_y_controles(imagen_actual, reconstruccion)
        
        plt.draw()
    
    def visualizar_espacio_latente(self):
        if self.representaciones_latentes is None:
            return
        
        dim_latente = self.representaciones_latentes.shape[1]
        
        if dim_latente >= 2:
            scatter = self.axes[2].scatter(
                self.representaciones_latentes[:, 0],
                self.representaciones_latentes[:, 1],
                c=range(len(self.representaciones_latentes)),
                cmap='viridis',
                alpha=0.6,
                s=30
            )
            
            punto_actual = self.representaciones_latentes[self.indice_actual]
            self.axes[2].scatter(
                punto_actual[0], punto_actual[1],
                c='red', s=100, marker='x', linewidths=3
            )
            
            self.axes[2].set_title("Espacio Latente (2D)")
            self.axes[2].set_xlabel("Dimensión 1")
            self.axes[2].set_ylabel("Dimensión 2")
            self.axes[2].grid(True, alpha=0.3)
            
        else:
            self.axes[2].hist(self.representaciones_latentes.flatten(), bins=50, alpha=0.7)
            valor_actual = self.representaciones_latentes[self.indice_actual, 0]
            self.axes[2].axvline(valor_actual, color='red', linestyle='--', linewidth=2)
            self.axes[2].set_title("Distribución Espacio Latente (1D)")
            self.axes[2].set_xlabel("Valor latente")
            self.axes[2].set_ylabel("Frecuencia")
    
    def mostrar_metricas_y_controles(self, original, reconstruccion):
        self.axes[3].axis('off')
        
        mse = np.mean((original - reconstruccion) ** 2)
        mae = np.mean(np.abs(original - reconstruccion))
        
        if self.representaciones_latentes is not None:
            rep_actual = self.representaciones_latentes[self.indice_actual]
            info_latente = f"Representación latente: {rep_actual[:5]}{'...' if len(rep_actual) > 5 else ''}"
        else:
            info_latente = "Representación latente: No disponible"
        
        nombre_archivo = self.procesador.nombres_archivos[self.indice_actual] if self.indice_actual < len(self.procesador.nombres_archivos) else "N/A"
        
        info_text = f"""MÉTRICAS DE RECONSTRUCCIÓN
        
Imagen: {self.indice_actual + 1}/{len(self.datos_imagenes)}
Archivo: {nombre_archivo}
Forma: {self.forma_imagen}

MSE: {mse:.6f}
MAE: {mae:.6f}

{info_latente}

CONTROLES:
← → : Navegar imágenes
H   : Mostrar ayuda
Q   : Salir"""
        
        self.axes[3].text(0.05, 0.95, info_text, transform=self.axes[3].transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    def explorar_interactivo(self):
        if not self.modelo:
            print("Error: Debe cargar un modelo primero")
            return
        
        if self.datos_imagenes is None:
            print("Cargando datos...")
            self.cargar_datos()
        
        print("=== EXPLORADOR DE ESPACIO LATENTE - IMÁGENES ===")
        print("Use las teclas para navegar por las imágenes y explorar el espacio latente")
        self.iniciar_exploracion_interactiva()
    
    def generar_imagen_desde_latente(self, vector_latente):
        if not self.decoder:
            print("Error: Decoder no disponible")
            return None
        
        imagen_generada = self.decoder.predict(vector_latente.reshape(1, -1), verbose=0)[0]
        return self.procesador.reconstruir_forma_imagen(imagen_generada.reshape(1, -1))[0]
    
    def interpolar_en_espacio_latente(self, indice1, indice2, num_pasos=10):
        if self.representaciones_latentes is None:
            print("Error: Representaciones latentes no disponibles")
            return
        
        rep1 = self.representaciones_latentes[indice1]
        rep2 = self.representaciones_latentes[indice2]
        
        alphas = np.linspace(0, 1, num_pasos)
        interpolaciones = []
        
        for alpha in alphas:
            rep_interpolada = (1 - alpha) * rep1 + alpha * rep2
            imagen_interpolada = self.generar_imagen_desde_latente(rep_interpolada)
            if imagen_interpolada is not None:
                interpolaciones.append(imagen_interpolada)
        
        if interpolaciones:
            fig, axes = plt.subplots(1, len(interpolaciones), figsize=(len(interpolaciones) * 2, 2))
            if len(interpolaciones) == 1:
                axes = [axes]
            
            for i, img in enumerate(interpolaciones):
                axes[i].imshow(img)
                axes[i].set_title(f"α={alphas[i]:.2f}")
                axes[i].axis('off')
            
            plt.suptitle(f"Interpolación: Imagen {indice1+1} → Imagen {indice2+1}")
            plt.tight_layout()
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Explorador interactivo de espacio latente para imágenes')
    parser.add_argument('modelo', type=str, nargs='?',
                       help='Nombre o ruta del modelo a cargar')
    parser.add_argument('--listar', action='store_true',
                       help='Listar modelos disponibles')
    parser.add_argument('--tamaño', type=str, default='64x64',
                       help='Tamaño de imagen (ej: 64x64)')
    parser.add_argument('--max-imagenes', type=int, default=100,
                       help='Número máximo de imágenes a cargar')
    
    args = parser.parse_args()
    
    if args.listar:
        cargador = CargadorModelos()
        modelos = cargador.listar_modelos()
        print("Modelos disponibles:")
        for modelo in modelos:
            if 'tp4' in modelo or 'imagen' in modelo:
                print(f"  - {modelo}")
        return
    
    try:
        ancho, alto = map(int, args.tamaño.split('x'))
        tamaño_imagen = (ancho, alto)
    except:
        print(f"Error: Formato de tamaño inválido '{args.tamaño}'")
        return
    
    print("=== EXPLORADOR DE ESPACIO LATENTE - IMÁGENES ===")
    print("TP4: Autocodificadores para imágenes de suricatas")
    print()
    
    explorador = ExploradorEspacioLatenteImagenes(args.modelo)
    if explorador.modelo is not None:
        explorador.cargar_datos(tamaño_imagen, args.max_imagenes)
        explorador.explorar_interactivo()
    else:
        print("No se pudo cargar el modelo especificado.")


if __name__ == "__main__":
    main()
