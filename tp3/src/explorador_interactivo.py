import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tensorflow import keras
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tp3.src.autocodificador_caracteres import AutocodificadorCaracteres
from tp3.src.cargador_datos_caracteres import CargadorDatosCaracteres


class ExploradorInteractivoAvanzado:
    
    def __init__(self, autocodificador, dimension_latente, nombre_modelo):
        self.autocodificador = autocodificador
        self.dimension_latente = dimension_latente
        self.nombre_modelo = nombre_modelo
        
        self.fig = None
        self.ax_mapa = None
        self.ax_caracter = None
        
        self.punto_actual = None
        self.circulo_click = None
        self.caracteres_nombres = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
                                   ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                                   '8', '9', ':', ';', '<', '=', '>', '?']
        self.historial_clicks = []
    
    def inicializar_visualizacion(self):
        self.fig = plt.figure(figsize=(16, 8))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1])
        
        self.ax_mapa = self.fig.add_subplot(gs[0])
        self.ax_caracter = self.fig.add_subplot(gs[1])
        
        representaciones = self.autocodificador.representaciones_latentes
        
        if self.dimension_latente > 2:
            representaciones_2d = representaciones[:, :2]
            titulo_mapa = f"Espacio Latente (Dims 1-2 de {self.dimension_latente})"
            nota = f"⚠️ Proyección 2D"
        else:
            representaciones_2d = representaciones
            titulo_mapa = "Espacio Latente 2D Completo"
            nota = "✅ Visualización completa"
        
        scatter = self.ax_mapa.scatter(
            representaciones_2d[:, 0], 
            representaciones_2d[:, 1],
            c=range(len(representaciones_2d)), 
            cmap='tab20', 
            s=150, 
            alpha=0.7,
            edgecolors='black', 
            linewidth=1.5,
            zorder=3
        )
        
        for i, (x, y) in enumerate(representaciones_2d):
            self.ax_mapa.annotate(
                self.caracteres_nombres[i], 
                (x, y),
                xytext=(0, 0), 
                textcoords='offset points',
                fontsize=9, 
                fontweight='bold',
                ha='center', 
                va='center',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                         edgecolor='black', alpha=0.8),
                zorder=4
            )
        
        self.ax_mapa.set_xlabel('Dimensión Latente 1', fontsize=12, fontweight='bold')
        self.ax_mapa.set_ylabel('Dimensión Latente 2', fontsize=12, fontweight='bold')
        self.ax_mapa.set_title(f'{titulo_mapa}\n{nota}\nModelo: {self.nombre_modelo}', 
                              fontsize=11, fontweight='bold')
        self.ax_mapa.grid(True, alpha=0.3, linestyle='--')
        self.ax_mapa.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax_mapa.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        self.ax_caracter.axis('off')
        self.ax_caracter.text(0.5, 0.5, 'Haz click en el mapa\npara generar un carácter', 
                             ha='center', va='center', fontsize=14,
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        if self.dimension_latente > 2:
            instrucciones = (
                f"INSTRUCCIONES (L={self.dimension_latente}):\n"
                "• Click CERCA de carácter: usa latente completo ✅\n"
                "• Click LEJOS: genera nuevo (dims 3+ = 0)\n"
                "• Click derecho: Limpiar historial\n"
                "• Verde: carácter entrenamiento\n"
                "• Naranja: nuevo generado"
            )
        else:
            instrucciones = (
                f"INSTRUCCIONES (L={self.dimension_latente}):\n"
                "• Click izquierdo: Generar carácter\n"
                "• Click derecho: Limpiar historial\n"
                "• Puntos rojos: último click\n"
                "• Puntos verdes: entrenamiento\n"
                "• Puntos naranjas: nuevos"
            )
        self.fig.text(0.02, 0.02, instrucciones, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
    
    def on_click(self, event):
        if event.inaxes != self.ax_mapa:
            return
        
        if event.button == 3:
            self.limpiar_historial()
            return
        
        if event.button != 1:
            return
        
        punto_click_2d = np.array([event.xdata, event.ydata])
        
        representaciones = self.autocodificador.representaciones_latentes
        if self.dimension_latente > 2:
            representaciones_2d = representaciones[:, :2]
        else:
            representaciones_2d = representaciones
        
        distancias = np.sqrt(np.sum((representaciones_2d - punto_click_2d) ** 2, axis=1))
        idx_cercano = np.argmin(distancias)
        distancia_minima = distancias[idx_cercano]
        
        if distancia_minima < 0.15:
            punto_completo = representaciones[idx_cercano]
            es_caracter_entrenamiento = True
            caracter_nombre = self.caracteres_nombres[idx_cercano]
        else:
            if self.dimension_latente > 2:
                punto_completo = np.zeros(self.dimension_latente)
                punto_completo[:2] = punto_click_2d
            else:
                punto_completo = punto_click_2d
            es_caracter_entrenamiento = False
            caracter_nombre = None
        
        self.punto_actual = punto_completo
        self.historial_clicks.append((punto_click_2d, es_caracter_entrenamiento, caracter_nombre))
        
        if self.circulo_click is not None:
            self.circulo_click.remove()
        
        self.circulo_click = Circle(
            (punto_click_2d[0], punto_click_2d[1]), 
            0.05, 
            color='red', 
            fill=True,
            alpha=0.7,
            zorder=5
        )
        self.ax_mapa.add_patch(self.circulo_click)
        
        for hist_item in self.historial_clicks[:-1]:
            punto_hist = hist_item[0]
            es_entrenamiento = hist_item[1]
            color_hist = 'green' if es_entrenamiento else 'orange'
            circle = Circle(
                (punto_hist[0], punto_hist[1]), 
                0.03, 
                color=color_hist, 
                fill=True,
                alpha=0.5,
                zorder=4
            )
            self.ax_mapa.add_patch(circle)
        
        caracter_generado = self.autocodificador.decodificar_desde_latente(
            punto_completo.reshape(1, -1)
        )
        
        self.mostrar_caracter(caracter_generado[0], punto_completo, es_caracter_entrenamiento, caracter_nombre)
        
        self.fig.canvas.draw_idle()
    
    def mostrar_caracter(self, patron, punto_latente, es_entrenamiento=False, nombre_caracter=None):
        self.ax_caracter.clear()
        
        patron_2d = patron.reshape(7, 5)
        patron_binario = (patron_2d > 0.5).astype(float)
        
        self.ax_caracter.imshow(patron_binario, cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
        
        if es_entrenamiento and nombre_caracter:
            tipo = f"'{nombre_caracter}' ✅"
            color_titulo = 'green'
        else:
            tipo = "Generado"
            color_titulo = 'blue'
        
        if self.dimension_latente <= 3:
            valores = ', '.join([f'{v:.3f}' for v in punto_latente])
            vector_str = f'L={self.dimension_latente}: [{valores}]'
        else:
            primeros = ', '.join([f'{v:.3f}' for v in punto_latente[:2]])
            vector_str = f'L={self.dimension_latente}: [{primeros}, ...]'
        
        titulo = f'{tipo}\n{vector_str}'
        
        self.ax_caracter.set_title(titulo, fontsize=10, fontweight='bold', color=color_titulo)
        
        for i in range(7):
            for j in range(5):
                valor_bin = patron_binario[i, j]
                simbolo = '█' if valor_bin > 0.5 else '·'
                color = 'white' if valor_bin > 0.5 else 'gray'
                self.ax_caracter.text(j, i, simbolo, ha='center', va='center',
                                     color=color, fontsize=14, fontweight='bold')
        
        self.ax_caracter.axis('off')
        
        stats_text = (
            f"Píxeles: {np.sum(patron > 0.5)}/35\n"
            f"Promedio: {patron.mean():.3f}\n"
            f"Clicks: {len(self.historial_clicks)}"
        )
        self.ax_caracter.text(0.5, -0.12, stats_text, 
                             ha='center', va='top',
                             transform=self.ax_caracter.transAxes,
                             fontsize=9,
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    def limpiar_historial(self):
        self.historial_clicks = []
        
        for patch in list(self.ax_mapa.patches):
            if isinstance(patch, Circle):
                patch.remove()
        
        self.circulo_click = None
        
        self.ax_caracter.clear()
        self.ax_caracter.axis('off')
        self.ax_caracter.text(0.5, 0.5, 'Historial limpiado\nHaz click para continuar', 
                             ha='center', va='center', fontsize=14,
                             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        self.fig.canvas.draw_idle()
    
    def mostrar(self):
        plt.show()


def cargar_modelo(info_modelo):
    nombre_base = info_modelo['nombre_base']
    dim_latente = info_modelo['dim_latente']
    arquitectura = info_modelo['arquitectura']
    
    modelos_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'modelos')
    
    ruta_autocodificador = os.path.join(modelos_dir, f"{nombre_base}_autocodificador.keras")
    ruta_codificador = os.path.join(modelos_dir, f"{nombre_base}_codificador.keras")
    ruta_decodificador = os.path.join(modelos_dir, f"{nombre_base}_decodificador.keras")
    
    if not all(os.path.exists(r) for r in [ruta_autocodificador, ruta_codificador, ruta_decodificador]):
        return None
    
    autocodificador_obj = AutocodificadorCaracteres(
        dimension_latente=dim_latente,
        arquitectura=arquitectura
    )
    
    autocodificador_obj.autocodificador = keras.models.load_model(ruta_autocodificador)
    autocodificador_obj.codificador = keras.models.load_model(ruta_codificador)
    autocodificador_obj.decodificador = keras.models.load_model(ruta_decodificador)
    autocodificador_obj.entrenado = True
    
    return autocodificador_obj


def main():
    parser = argparse.ArgumentParser(description='Explorador interactivo del espacio latente')
    parser.add_argument('--modelo', type=str, required=True, help='Nombre del modelo a cargar')
    parser.add_argument('--listar', action='store_true', help='Listar modelos disponibles')
    
    args = parser.parse_args()
    
    if args.listar:
        from tp3.src.verificar_reconstruccion_multidimensional import escanear_modelos
        modelos = escanear_modelos()
        
        print("="*60)
        print("MODELOS DISPONIBLES")
        print("="*60)
        print(f"\nTotal: {sum(len(v) for v in modelos.values())} modelos encontrados\n")
        
        for dim in sorted(modelos.keys()):
            print(f"\n{'='*60}")
            print(f"Dimensión Latente = {dim} ({len(modelos[dim])} modelos)")
            print(f"{'='*60}")
            for info in modelos[dim]:
                epocas_str = f"ep{info['epocas']}" if info['epocas'] else ""
                lr_str = f"lr{info['lr']}" if info['lr'] else ""
                print(f"  {info['arquitectura']:15s} {epocas_str:8s} {lr_str:12s} | {info['nombre_base']}")
        return
    
    print("="*60)
    print("EXPLORADOR INTERACTIVO AVANZADO")
    print("="*60)
    
    from tp3.src.verificar_reconstruccion_multidimensional import escanear_modelos
    modelos = escanear_modelos()
    
    info_modelo = None
    for dim in modelos:
        for info in modelos[dim]:
            if info['nombre_base'] == args.modelo:
                info_modelo = info
                break
        if info_modelo:
            break
    
    if not info_modelo:
        print(f"\n❌ No se encontró el modelo: {args.modelo}")
        print("\nUsa --listar para ver modelos disponibles")
        return
    
    print(f"\n📊 Cargando modelo: {info_modelo['nombre_base']}")
    print(f"   Arquitectura: {info_modelo['arquitectura']}")
    print(f"   Dimensión Latente: {info_modelo['dim_latente']}")
    if info_modelo['epocas']:
        print(f"   Épocas: {info_modelo['epocas']}")
    if info_modelo['lr']:
        print(f"   Learning Rate: {info_modelo['lr']}")
    
    autocodificador = cargar_modelo(info_modelo)
    if autocodificador is None:
        print(f"\n❌ No se pudo cargar el modelo")
        return
    
    cargador = CargadorDatosCaracteres()
    datos, _ = cargador.cargar_datos_desde_modulo(1)
    
    autocodificador.datos_entrenamiento = datos
    autocodificador.datos_cargados = True
    
    print(f"\n📊 Generando representaciones latentes...")
    autocodificador.obtener_representacion_latente()
    
    print(f"\n🎨 Iniciando explorador interactivo...")
    
    explorador = ExploradorInteractivoAvanzado(
        autocodificador, 
        info_modelo['dim_latente'],
        info_modelo['nombre_base']
    )
    explorador.inicializar_visualizacion()
    
    print(f"\n✅ Explorador listo. Ventana interactiva abierta.")
    print("\nINSTRUCCIONES:")
    print("  • Click izquierdo: Generar carácter en ese punto")
    print("  • Click CERCA de carácter: usa vector latente completo")
    print("  • Click LEJOS: genera nuevo (dims 3+ = 0)")
    print("  • Click derecho: Limpiar historial")
    
    explorador.mostrar()


if __name__ == "__main__":
    main()
