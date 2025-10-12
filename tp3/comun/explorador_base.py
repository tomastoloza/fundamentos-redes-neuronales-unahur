import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod


class ExploradorBase(ABC):
    def __init__(self):
        self.indice_actual = 0
        self.configurar_matplotlib()
    
    def configurar_matplotlib(self):
        plt.ion()
        plt.style.use('default')
    
    def mostrar_ayuda(self):
        print("\n=== CONTROLES DE NAVEGACIÓN ===")
        print("← →  : Navegar elementos")
        print("↑ ↓  : Cambiar parámetros")
        print("R    : Regenerar/Actualizar")
        print("G    : Generar nuevo")
        print("H    : Mostrar ayuda")
        print("Q    : Salir")
        print("===============================\n")
    
    def procesar_tecla(self, event):
        if event.key == 'left':
            self.navegar_anterior()
        elif event.key == 'right':
            self.navegar_siguiente()
        elif event.key == 'up':
            self.cambiar_parametro_arriba()
        elif event.key == 'down':
            self.cambiar_parametro_abajo()
        elif event.key == 'r':
            self.regenerar()
        elif event.key == 'g':
            self.generar_nuevo()
        elif event.key == 'h':
            self.mostrar_ayuda()
        elif event.key == 'q':
            self.salir()
        
        self.actualizar_visualizacion()
    
    def navegar_anterior(self):
        if self.indice_actual > 0:
            self.indice_actual -= 1
            print(f"Navegando a elemento {self.indice_actual}")
    
    def navegar_siguiente(self):
        max_indice = self.obtener_maximo_indice()
        if self.indice_actual < max_indice - 1:
            self.indice_actual += 1
            print(f"Navegando a elemento {self.indice_actual}")
    
    def crear_figura_base(self, titulo, filas=2, columnas=2, figsize=(12, 8)):
        fig, axes = plt.subplots(filas, columnas, figsize=figsize)
        fig.suptitle(titulo, fontsize=14, fontweight='bold')
        
        if filas == 1 and columnas == 1:
            axes = [axes]
        elif filas == 1 or columnas == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        return fig, axes
    
    def mostrar_patron_ascii(self, patron, titulo="Patrón"):
        patron_2d = patron.reshape(7, 5)
        print(f"\n{titulo}:")
        for fila in patron_2d:
            linea = ''.join(['█' if pixel > 0.5 else '·' for pixel in fila])
            print(f"  {linea}")
    
    def mostrar_metricas(self, metricas, titulo="Métricas"):
        print(f"\n=== {titulo} ===")
        for clave, valor in metricas.items():
            if isinstance(valor, float):
                print(f"{clave}: {valor:.4f}")
            else:
                print(f"{clave}: {valor}")
    
    def configurar_subplot_patron(self, ax, patron, titulo, cmap='gray'):
        patron_2d = patron.reshape(7, 5)
        im = ax.imshow(patron_2d, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(titulo)
        ax.set_xticks([])
        ax.set_yticks([])
        return im
    
    def iniciar_exploracion_interactiva(self):
        self.mostrar_ayuda()
        fig, axes = self.crear_visualizacion_inicial()
        
        fig.canvas.mpl_connect('key_press_event', self.procesar_tecla)
        
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nExploración interrumpida por el usuario")
        finally:
            plt.close('all')
    
    @abstractmethod
    def obtener_maximo_indice(self):
        pass
    
    @abstractmethod
    def cambiar_parametro_arriba(self):
        pass
    
    @abstractmethod
    def cambiar_parametro_abajo(self):
        pass
    
    @abstractmethod
    def regenerar(self):
        pass
    
    @abstractmethod
    def generar_nuevo(self):
        pass
    
    @abstractmethod
    def actualizar_visualizacion(self):
        pass
    
    @abstractmethod
    def crear_visualizacion_inicial(self):
        pass
    
    def salir(self):
        print("Cerrando explorador...")
        plt.close('all')
        exit(0)
