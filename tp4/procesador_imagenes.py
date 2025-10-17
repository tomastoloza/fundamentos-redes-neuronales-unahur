import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class ProcesadorImagenes:
    def __init__(self, directorio_datos='datos/suricatas_200x200'):
        self.directorio_datos = directorio_datos
        self.imagenes = []
        self.nombres_archivos = []
        self.tamaño_objetivo = (200, 200)
        
    def redimensionar_con_padding(self, imagen_pil, tamaño_objetivo, color_relleno=(0, 0, 0)):
        if imagen_pil.mode != 'RGB':
            imagen_pil = imagen_pil.convert('RGB')
        
        imagen_pil.thumbnail(tamaño_objetivo, Image.Resampling.LANCZOS)
        
        fondo = Image.new('RGB', tamaño_objetivo, color_relleno)
        
        posicion = (
            (tamaño_objetivo[0] - imagen_pil.width) // 2,
            (tamaño_objetivo[1] - imagen_pil.height) // 2
        )
        
        fondo.paste(imagen_pil, posicion)
        return fondo
    
    def cargar_imagenes(self, tamaño=(64, 64), max_imagenes=None, mantener_aspecto=True, color_relleno=(0, 0, 0)):
        self.tamaño_objetivo = tamaño
        archivos_imagen = [f for f in os.listdir(self.directorio_datos) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if max_imagenes:
            archivos_imagen = archivos_imagen[:max_imagenes]
            
        print(f"Cargando {len(archivos_imagen)} imágenes...")
        if mantener_aspecto:
            print(f"Modo: Mantener aspecto con padding RGB{color_relleno}")
        else:
            print(f"Modo: Redimensionado directo (puede deformar)")
        
        imagenes_procesadas = []
        nombres_validos = []
        
        for i, archivo in enumerate(archivos_imagen):
            try:
                ruta_completa = os.path.join(self.directorio_datos, archivo)
                imagen = Image.open(ruta_completa)
                
                if mantener_aspecto:
                    imagen_procesada = self.redimensionar_con_padding(imagen, tamaño, color_relleno)
                else:
                    if imagen.mode != 'RGB':
                        imagen = imagen.convert('RGB')
                    imagen_procesada = imagen.resize(tamaño, Image.Resampling.LANCZOS)
                
                array_imagen = np.array(imagen_procesada, dtype=np.float32) / 255.0
                
                imagenes_procesadas.append(array_imagen)
                nombres_validos.append(archivo)
                
                if (i + 1) % 50 == 0:
                    print(f"  Procesadas {i + 1}/{len(archivos_imagen)} imágenes")
                    
            except Exception as e:
                print(f"Error procesando {archivo}: {e}")
                continue
        
        self.imagenes = np.array(imagenes_procesadas)
        self.nombres_archivos = nombres_validos
        
        print(f"✓ Cargadas {len(self.imagenes)} imágenes de {tamaño[0]}x{tamaño[1]} píxeles")
        return self.imagenes
    
    def obtener_datos_aplanados(self):
        if len(self.imagenes) == 0:
            raise ValueError("No hay imágenes cargadas. Ejecute cargar_imagenes() primero.")
        
        return self.imagenes.reshape(len(self.imagenes), -1)
    
    def obtener_forma_original(self):
        if len(self.imagenes) == 0:
            return self.tamaño_objetivo + (3,)
        return self.imagenes.shape[1:]
    
    def reconstruir_forma_imagen(self, datos_aplanados):
        forma_original = self.obtener_forma_original()
        return datos_aplanados.reshape(-1, *forma_original)
    
    def mostrar_imagen(self, indice, titulo="Imagen"):
        if indice >= len(self.imagenes):
            raise ValueError(f"Índice {indice} fuera de rango. Hay {len(self.imagenes)} imágenes.")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(self.imagenes[indice])
        plt.title(f"{titulo} - {self.nombres_archivos[indice]}")
        plt.axis('off')
        plt.show()
    
    def mostrar_comparacion(self, indices, titulos=None):
        if titulos is None:
            titulos = [f"Imagen {i}" for i in indices]
        
        num_imagenes = len(indices)
        fig, axes = plt.subplots(1, num_imagenes, figsize=(4 * num_imagenes, 4))
        
        if num_imagenes == 1:
            axes = [axes]
        
        for i, (idx, titulo) in enumerate(zip(indices, titulos)):
            axes[i].imshow(self.imagenes[idx])
            axes[i].set_title(f"{titulo}\n{self.nombres_archivos[idx]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def obtener_estadisticas(self):
        if len(self.imagenes) == 0:
            return {}
        
        datos_aplanados = self.obtener_datos_aplanados()
        
        return {
            'num_imagenes': len(self.imagenes),
            'forma_imagen': self.imagenes.shape[1:],
            'tamaño_total': datos_aplanados.shape[1],
            'valor_min': float(np.min(datos_aplanados)),
            'valor_max': float(np.max(datos_aplanados)),
            'valor_promedio': float(np.mean(datos_aplanados)),
            'desviacion_estandar': float(np.std(datos_aplanados))
        }
    
    def mostrar_estadisticas(self):
        stats = self.obtener_estadisticas()
        if not stats:
            print("No hay imágenes cargadas.")
            return
        
        print("=== ESTADÍSTICAS DEL DATASET ===")
        print(f"Número de imágenes: {stats['num_imagenes']}")
        print(f"Forma de cada imagen: {stats['forma_imagen']}")
        print(f"Tamaño aplanado: {stats['tamaño_total']} píxeles")
        print(f"Rango de valores: [{stats['valor_min']:.3f}, {stats['valor_max']:.3f}]")
        print(f"Valor promedio: {stats['valor_promedio']:.3f}")
        print(f"Desviación estándar: {stats['desviacion_estandar']:.3f}")
    
    def obtener_muestra_aleatoria(self, num_muestras=5):
        if num_muestras > len(self.imagenes):
            num_muestras = len(self.imagenes)
        
        indices = np.random.choice(len(self.imagenes), num_muestras, replace=False)
        return indices
    
    def guardar_imagen_reconstruida(self, imagen_array, nombre_archivo):
        imagen_array = np.clip(imagen_array, 0, 1)
        imagen_uint8 = (imagen_array * 255).astype(np.uint8)
        
        imagen_pil = Image.fromarray(imagen_uint8)
        ruta_salida = os.path.join('tp4/resultados', nombre_archivo)
        
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        imagen_pil.save(ruta_salida)
        print(f"Imagen guardada: {ruta_salida}")
        
    def crear_grid_imagenes(self, imagenes_array, filas=2, columnas=5, titulo="Grid de Imágenes"):
        num_imagenes = min(len(imagenes_array), filas * columnas)
        
        fig, axes = plt.subplots(filas, columnas, figsize=(columnas * 3, filas * 3))
        axes = axes.flatten() if filas * columnas > 1 else [axes]
        
        for i in range(num_imagenes):
            axes[i].imshow(imagenes_array[i])
            axes[i].axis('off')
        
        for i in range(num_imagenes, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(titulo, fontsize=16)
        plt.tight_layout()
        plt.show()
