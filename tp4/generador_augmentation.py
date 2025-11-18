from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


class GeneradorAugmentation:
    def __init__(self, config_augmentation=None):
        self.config = config_augmentation or self._get_default_config()
        self.generator = None
        self._crear_generador()
    
    def _get_default_config(self):
        """Configuración por defecto moderada"""
        return {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'horizontal_flip': True,
            'zoom_range': 0.2,
            'brightness_range': [0.8, 1.2],
            'fill_mode': 'nearest',
            'rescale': None 
        }
    
    def _crear_generador(self):
        self.generator = ImageDataGenerator(**self.config)
    
    def ajustar_a_datos(self, X_train):
        if self.generator is None:
            raise ValueError("Generador no inicializado")
        
        self.generator.fit(X_train)
        print(f"✓ Generador ajustado a {len(X_train)} imágenes de entrenamiento")
    
    def crear_flujo_entrenamiento(self, X_train, y_train, batch_size=32, shuffle=True):
        if self.generator is None:
            raise ValueError("Generador no inicializado")
        
        return self.generator.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    def calcular_steps_per_epoch(self, num_samples, batch_size):
        return max(1, num_samples // batch_size)
    
    def generar_ejemplos_augmentation(self, imagen, num_ejemplos=9):
        if self.generator is None:
            raise ValueError("Generador no inicializado")
        
        imagen_batch = np.expand_dims(imagen, axis=0)
        
        ejemplos = []
        generador_temp = self.generator.flow(
            imagen_batch, 
            batch_size=1,
            shuffle=False
        )
        
        for i in range(num_ejemplos):
            imagen_augmented = next(generador_temp)[0]
            ejemplos.append(imagen_augmented)
        
        return ejemplos
    
    def obtener_configuracion(self):
        return self.config.copy()
    
    def actualizar_configuracion(self, nueva_config):
        self.config.update(nueva_config)
        self._crear_generador()
        print(f"✓ Configuración actualizada: {list(nueva_config.keys())}")
    
    def mostrar_configuracion(self):
        print("=== CONFIGURACIÓN DATA AUGMENTATION ===")
        for key, value in self.config.items():
            if value is not None:
                print(f"  {key}: {value}")
        print("=" * 40)
    
    @staticmethod
    def crear_desde_nombre(nombre_config):
        from .configuraciones import obtener_configuracion_augmentation
        
        try:
            config = obtener_configuracion_augmentation(nombre_config)
            return GeneradorAugmentation(config)
        except Exception as e:
            print(f"⚠️ Error cargando configuración '{nombre_config}': {e}")
            print("Usando configuración por defecto")
            return GeneradorAugmentation()



