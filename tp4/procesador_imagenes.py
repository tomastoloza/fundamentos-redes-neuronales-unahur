import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class ProcesadorImagenes:
    def __init__(self, directorio_datos='tp4/datos'):
        self.directorio_datos = directorio_datos
        self.imagenes = []
        self.labels = []
        self.nombres_archivos = []
        self.clases = []
        self.mapeo_clases = {}
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
    
    def cargar_imagenes_multiclase(self, tamaño=(200, 200), max_imagenes_por_clase=None, mantener_aspecto=True, color_relleno=(0, 0, 0)):
        self.tamaño_objetivo = tamaño
        
        carpetas_animales = [d for d in os.listdir(self.directorio_datos) 
                           if os.path.isdir(os.path.join(self.directorio_datos, d)) and not d.startswith('.')]
        
        if not carpetas_animales:
            raise ValueError(f"No se encontraron carpetas de animales en {self.directorio_datos}")
        
        self.clases = sorted(carpetas_animales)
        self.mapeo_clases = {clase: i for i, clase in enumerate(self.clases)}
        
        print(f"Clases encontradas: {self.clases}")
        print(f"Cargando imágenes de tamaño {tamaño}...")
        if mantener_aspecto:
            print(f"Modo: Mantener aspecto con padding RGB{color_relleno}")
        else:
            print(f"Modo: Redimensionado directo (puede deformar)")
        
        imagenes_procesadas = []
        labels_procesados = []
        nombres_validos = []
        
        for clase in self.clases:
            ruta_clase = os.path.join(self.directorio_datos, clase)
            archivos_imagen = [f for f in os.listdir(ruta_clase) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if max_imagenes_por_clase:
                archivos_imagen = archivos_imagen[:max_imagenes_por_clase]
            
            print(f"  - {clase}: {len(archivos_imagen)} imágenes")
            
            for archivo in archivos_imagen:
            try:
                    ruta_completa = os.path.join(ruta_clase, archivo)
                imagen = Image.open(ruta_completa)
                
                if mantener_aspecto:
                    imagen_procesada = self.redimensionar_con_padding(imagen, tamaño, color_relleno)
                else:
                    imagen_procesada = imagen.resize(tamaño, Image.Resampling.LANCZOS)
                
                    if imagen_procesada.mode != 'RGB':
                        imagen_procesada = imagen_procesada.convert('RGB')
                    
                array_imagen = np.array(imagen_procesada, dtype=np.float32) / 255.0
                imagenes_procesadas.append(array_imagen)
                    labels_procesados.append(self.mapeo_clases[clase])
                    nombres_validos.append(f"{clase}/{archivo}")
                    
            except Exception as e:
                    print(f"Error procesando {clase}/{archivo}: {e}")
        
        self.imagenes = np.array(imagenes_procesadas)
        self.labels = np.array(labels_procesados)
        self.nombres_archivos = nombres_validos
        
        print(f"\nCarga completada:")
        print(f"  - {len(self.imagenes)} imágenes cargadas")
        print(f"  - Forma del array: {self.imagenes.shape}")
        print(f"  - Rango de valores: [{self.imagenes.min():.3f}, {self.imagenes.max():.3f}]")
        print(f"  - Distribución por clase:")
        
        for i, clase in enumerate(self.clases):
            count = np.sum(self.labels == i)
            print(f"    {clase}: {count} imágenes")
        
        return self.imagenes, self.labels
    
    def preparar_datos_entrenamiento(self, test_size=0.2, validation_split=0.2, random_state=42):
        if len(self.imagenes) == 0 or len(self.labels) == 0:
            raise ValueError("Primero debe cargar las imágenes usando cargar_imagenes_multiclase()")
        
        labels_categoricos = to_categorical(self.labels, num_classes=len(self.clases))
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.imagenes, labels_categoricos, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=validation_split, 
            random_state=random_state, 
            stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"División de datos:")
        print(f"  - Entrenamiento: {len(X_train)} imágenes")
        print(f"  - Validación: {len(X_val)} imágenes")
        print(f"  - Prueba: {len(X_test)} imágenes")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def preparar_datos_con_augmentation(self, config_augmentation=None, test_size=0.2, 
                                       validation_split=0.2, random_state=42):
        """
        Preparar datos con data augmentation para entrenamiento.
        
        Args:
            config_augmentation (str o dict): Configuración de augmentation
            test_size (float): Proporción para conjunto de prueba
            validation_split (float): Proporción para validación
            random_state (int): Semilla aleatoria
            
        Returns:
            dict: Diccionario con generadores y datos preparados
        """
        if len(self.imagenes) == 0 or len(self.labels) == 0:
            raise ValueError("Primero debe cargar las imágenes usando cargar_imagenes_multiclase()")
        
        # División normal de datos
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.preparar_datos_entrenamiento(
            test_size, validation_split, random_state
        )
        
        # Si no hay configuración de augmentation, devolver datos normales
        if not config_augmentation or config_augmentation == 'ninguno':
            return {
                'train_data': (X_train, y_train),
                'validation_data': (X_val, y_val),
                'test_data': (X_test, y_test),
                'use_generator': False
            }
        
        # Crear generador de augmentation
        from .generador_augmentation import GeneradorAugmentation
        
        if isinstance(config_augmentation, str):
            generador = GeneradorAugmentation.crear_desde_nombre(config_augmentation)
        else:
            generador = GeneradorAugmentation(config_augmentation)
        
        # Ajustar generador a datos de entrenamiento
        generador.ajustar_a_datos(X_train)
        
        print(f"✓ Data Augmentation configurado:")
        generador.mostrar_configuracion()
        
        return {
            'generador': generador,
            'train_data': (X_train, y_train),
            'validation_data': (X_val, y_val),
            'test_data': (X_test, y_test),
            'use_generator': True
        }
    
    def obtener_nombres_clases(self):
        return self.clases
    
    def obtener_mapeo_clases(self):
        return self.mapeo_clases
    
    def mostrar_imagen(self, indice, titulo="Imagen"):
        if indice >= len(self.imagenes):
            raise ValueError(f"Índice {indice} fuera de rango. Hay {len(self.imagenes)} imágenes.")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(self.imagenes[indice])
        clase_nombre = self.clases[self.labels[indice]] if len(self.labels) > 0 else "Sin clase"
        plt.title(f"{titulo} - {self.nombres_archivos[indice]} ({clase_nombre})")
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
            clase_nombre = self.clases[self.labels[idx]] if len(self.labels) > 0 else "Sin clase"
            axes[i].set_title(f"{titulo}\n{self.nombres_archivos[idx]}\n({clase_nombre})")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def mostrar_muestras_por_clase(self, num_muestras=2):
        if len(self.imagenes) == 0:
            print("No hay imágenes cargadas.")
            return
        
        fig, axes = plt.subplots(len(self.clases), num_muestras, figsize=(num_muestras * 3, len(self.clases) * 3))
        
        if len(self.clases) == 1:
            axes = axes.reshape(1, -1)
        
        for i, clase in enumerate(self.clases):
            indices_clase = np.where(self.labels == i)[0]
            muestras_indices = np.random.choice(indices_clase, min(num_muestras, len(indices_clase)), replace=False)
            
            for j, idx in enumerate(muestras_indices):
                if j < num_muestras:
                    axes[i, j].imshow(self.imagenes[idx])
                    axes[i, j].set_title(f"{clase}")
                    axes[i, j].axis('off')
            
            for j in range(len(muestras_indices), num_muestras):
                axes[i, j].axis('off')
        
        plt.suptitle("Muestras por Clase", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def obtener_estadisticas(self):
        if len(self.imagenes) == 0:
            return {}
        
        return {
            'num_imagenes': len(self.imagenes),
            'num_clases': len(self.clases),
            'forma_imagen': self.imagenes.shape[1:],
            'clases': self.clases,
            'distribucion_clases': {clase: np.sum(self.labels == i) for i, clase in enumerate(self.clases)},
            'valor_min': float(np.min(self.imagenes)),
            'valor_max': float(np.max(self.imagenes)),
            'valor_promedio': float(np.mean(self.imagenes)),
            'desviacion_estandar': float(np.std(self.imagenes))
        }
    
    def mostrar_estadisticas(self):
        stats = self.obtener_estadisticas()
        if not stats:
            print("No hay imágenes cargadas.")
            return
        
        print("=== ESTADÍSTICAS DEL DATASET ===")
        print(f"Número de imágenes: {stats['num_imagenes']}")
        print(f"Número de clases: {stats['num_clases']}")
        print(f"Forma de cada imagen: {stats['forma_imagen']}")
        print(f"Clases: {stats['clases']}")
        print(f"Distribución por clase:")
        for clase, count in stats['distribucion_clases'].items():
            porcentaje = (count / stats['num_imagenes']) * 100
            print(f"  - {clase}: {count} imágenes ({porcentaje:.1f}%)")
        print(f"Rango de valores: [{stats['valor_min']:.3f}, {stats['valor_max']:.3f}]")
        print(f"Valor promedio: {stats['valor_promedio']:.3f}")
        print(f"Desviación estándar: {stats['desviacion_estandar']:.3f}")