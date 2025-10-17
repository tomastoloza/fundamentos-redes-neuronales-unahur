import argparse
import numpy as np
from datetime import datetime
from tp3.comun.entrenador_base import EntrenadorBase
from tp3.comun.constructor_modelos import ConstructorModelos
from tp3.comun.cargador_modelos import CargadorModelos
from .procesador_imagenes import ProcesadorImagenes
from .configuraciones import obtener_configuracion, obtener_configuracion_entrenamiento, listar_configuraciones


class EntrenadorAutocodificadorImagenes(EntrenadorBase):
    def __init__(self):
        super().__init__()
        self.procesador = ProcesadorImagenes()
        self.constructor = ConstructorModelos()
        self.cargador = CargadorModelos()
        self.datos_entrenamiento = None
        self.forma_imagen = None
        
    def cargar_datos(self, tamaño_imagen=(64, 64), max_imagenes=None, mantener_aspecto=True):
        print("Cargando dataset de imágenes...")
        imagenes = self.procesador.cargar_imagenes(tamaño_imagen, max_imagenes, mantener_aspecto)
        self.datos_entrenamiento = self.procesador.obtener_datos_aplanados()
        self.forma_imagen = self.procesador.obtener_forma_original()
        
        print(f"✓ Dataset cargado: {len(imagenes)} imágenes de {tamaño_imagen}")
        self.procesador.mostrar_estadisticas()
        return self.datos_entrenamiento
    
    def crear_modelo(self, config_autocodificador):
        if self.datos_entrenamiento is None:
            raise ValueError("Debe cargar los datos primero con cargar_datos()")
        
        tamaño_entrada = self.datos_entrenamiento.shape[1]
        
        modelo = self.constructor.crear_autocodificador_denso(
            tamaño_entrada=tamaño_entrada,
            capas_encoder=config_autocodificador['capas_encoder'],
            dimension_latente=config_autocodificador['dimension_latente'],
            capas_decoder=config_autocodificador['capas_decoder'],
            activacion=config_autocodificador['activacion'],
            activacion_salida=config_autocodificador['activacion_salida']
        )
        
        modelo.compile(
            optimizer=self.obtener_optimizador(config_autocodificador['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return modelo
    
    def entrenar(self, nombre_config_autocodificador, nombre_config_entrenamiento, 
                 tamaño_imagen=(64, 64), max_imagenes=None, mantener_aspecto=True):
        
        config_autocodificador = obtener_configuracion(nombre_config_autocodificador)
        config_entrenamiento = obtener_configuracion_entrenamiento(nombre_config_entrenamiento)
        
        if tamaño_imagen != config_autocodificador['tamaño_imagen']:
            print(f"⚠️  Advertencia: Tamaño solicitado {tamaño_imagen} difiere del configurado {config_autocodificador['tamaño_imagen']}")
        
        self.cargar_datos(tamaño_imagen, max_imagenes, mantener_aspecto)
        
        modelo = self.crear_modelo(config_autocodificador)
        
        print(f"\n=== ENTRENANDO AUTOCODIFICADOR PARA IMÁGENES ===")
        print(f"Configuración: {nombre_config_autocodificador}")
        print(f"Entrenamiento: {nombre_config_entrenamiento}")
        print(f"Dimensión latente: {config_autocodificador['dimension_latente']}")
        print(f"Tamaño imagen: {tamaño_imagen}")
        
        callbacks = self.crear_callbacks(
            patience=config_entrenamiento['patience']
        )
        
        historia = modelo.fit(
            self.datos_entrenamiento,
            self.datos_entrenamiento,
            epochs=config_entrenamiento['epochs'],
            batch_size=config_autocodificador['batch_size'],
            validation_split=config_entrenamiento['validation_split'],
            callbacks=callbacks,
            verbose=1
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tamaño_str = f"{tamaño_imagen[0]}x{tamaño_imagen[1]}"
        nombre_modelo = f"tp4_imagenes_{nombre_config_autocodificador}_{tamaño_str}_{timestamp}"
        
        ruta_modelo = self.cargador.guardar_modelo(modelo, nombre_modelo)
        
        print(f"\n✓ Entrenamiento completado")
        print(f"✓ Modelo guardado: {ruta_modelo}")
        
        return modelo, historia, nombre_modelo
    
    def evaluar_reconstruccion(self, modelo, num_muestras=5):
        if self.datos_entrenamiento is None:
            raise ValueError("No hay datos cargados")
        
        indices = self.procesador.obtener_muestra_aleatoria(num_muestras)
        muestras = self.datos_entrenamiento[indices]
        
        reconstrucciones = modelo.predict(muestras, verbose=0)
        
        imagenes_originales = self.procesador.reconstruir_forma_imagen(muestras)
        imagenes_reconstruidas = self.procesador.reconstruir_forma_imagen(reconstrucciones)
        
        mse_promedio = np.mean((muestras - reconstrucciones) ** 2)
        mae_promedio = np.mean(np.abs(muestras - reconstrucciones))
        
        print(f"\n=== EVALUACIÓN DE RECONSTRUCCIÓN ===")
        print(f"MSE promedio: {mse_promedio:.6f}")
        print(f"MAE promedio: {mae_promedio:.6f}")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, num_muestras, figsize=(num_muestras * 3, 6))
        
        for i in range(num_muestras):
            axes[0, i].imshow(imagenes_originales[i])
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(imagenes_reconstruidas[i])
            axes[1, i].set_title(f"Reconstruida {i+1}")
            axes[1, i].axis('off')
        
        plt.suptitle("Comparación Original vs Reconstruida", fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return mse_promedio, mae_promedio


def entrenar_modelo():
    parser = argparse.ArgumentParser(description='Entrenador de autocodificadores para imágenes')
    parser.add_argument('--config', type=str, required=True,
                       help='Nombre de la configuración del autocodificador')
    parser.add_argument('--entrenamiento', type=str, default='normal',
                       help='Tipo de entrenamiento (rapido, normal, exhaustivo)')
    parser.add_argument('--tamaño', type=str, default='64x64',
                       help='Tamaño de imagen (ej: 64x64, 128x128)')
    parser.add_argument('--max-imagenes', type=int, default=None,
                       help='Número máximo de imágenes a cargar')
    parser.add_argument('--listar', action='store_true',
                       help='Listar configuraciones disponibles')
    parser.add_argument('--evaluar', action='store_true',
                       help='Evaluar reconstrucción después del entrenamiento')
    parser.add_argument('--no-mantener-aspecto', action='store_true',
                       help='No mantener aspecto (puede deformar imágenes)')
    
    args = parser.parse_args()
    
    if args.listar:
        listar_configuraciones()
        return
    
    try:
        ancho, alto = map(int, args.tamaño.split('x'))
        tamaño_imagen = (ancho, alto)
    except:
        print(f"Error: Formato de tamaño inválido '{args.tamaño}'. Use formato 'anchoxalto' (ej: 64x64)")
        return
    
    entrenador = EntrenadorAutocodificadorImagenes()
    
    try:
        mantener_aspecto = not args.no_mantener_aspecto
        modelo, historia, nombre_modelo = entrenador.entrenar(
            args.config,
            args.entrenamiento,
            tamaño_imagen,
            args.max_imagenes,
            mantener_aspecto
        )
        
        if args.evaluar:
            entrenador.evaluar_reconstruccion(modelo)
        
        print(f"\n✓ Proceso completado exitosamente")
        print(f"Modelo: {nombre_modelo}")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")


if __name__ == "__main__":
    entrenar_modelo()
