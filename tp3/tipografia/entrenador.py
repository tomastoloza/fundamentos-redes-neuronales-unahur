import argparse
import os
import numpy as np
from tensorflow import keras

from tp3.comun.visualizador_resultados import VisualizadorResultados
from .procesador_datos_tipografia import ProcesadorDatosTipografia
from .constructor_modelos_tipografia import ConstructorModelosTipografia
from .configuraciones import obtener_configuracion, obtener_configuracion_entrenamiento, listar_configuraciones


class EntrenadorTipografia:
    def __init__(self, tamaño_imagen=32):
        self.procesador = ProcesadorDatosTipografia(tamaño_imagen)
        self.constructor = ConstructorModelosTipografia(tamaño_imagen)
        self.visualizador = VisualizadorResultados()
        self.datos = self.procesador.obtener_datos_procesados()
        self.tamaño_imagen = tamaño_imagen
    
    def validar_datos(self):
        valido, errores = self.procesador.validar_datos()
        if not valido:
            print("Errores en los datos:")
            for error in errores:
                print(f"  - {error}")
            return False
        return True
    
    def crear_callbacks(self, config_entrenamiento):
        callbacks = []
        if config_entrenamiento.get('early_stopping', False):
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=config_entrenamiento.get('monitor', 'loss'),
                patience=config_entrenamiento.get('patience', 50),
                restore_best_weights=True,
                verbose=config_entrenamiento.get('verbose', 1)
            )
            callbacks.append(early_stopping)
        return callbacks
    
    def evaluar_modelo(self, modelo, datos_entrada, datos_objetivo):
        evaluacion = modelo.evaluate(datos_entrada, datos_objetivo, verbose=0)
        loss_final = evaluacion[0] if isinstance(evaluacion, list) else evaluacion
        
        predicciones = modelo.predict(datos_entrada, verbose=0)
        mse = np.mean((datos_objetivo - predicciones) ** 2)
        precision = np.mean((predicciones > 0.5) == (datos_objetivo > 0.5))
        
        return {
            'loss_final': float(loss_final),
            'mse': float(mse),
            'precision': float(precision),
            'predicciones': predicciones
        }
    
    def entrenar_modelo(self, config_modelo, config_entrenamiento, mostrar_graficos=True, **kwargs):
        if not self.validar_datos():
            return None, None, None, None
        
        modelo = self.constructor.crear_autocodificador_desde_config(config_modelo)
        callbacks = self.crear_callbacks(config_entrenamiento)
        
        historial = modelo.fit(
            self.datos, self.datos,
            epochs=config_entrenamiento['epochs'],
            batch_size=config_entrenamiento.get('batch_size', 32),
            validation_split=config_entrenamiento.get('validation_split', 0.15),
            verbose=1,
            callbacks=callbacks
        )
        
        print("\n=== EVALUACIÓN FINAL ===")
        metricas = self.evaluar_modelo(modelo, self.datos, self.datos)
        
        print(f"Loss final: {metricas['loss_final']:.6f}")
        print(f"MSE: {metricas['mse']:.6f}")
        print(f"Precisión: {metricas['precision']:.4f}")
        
        nombre_modelo = self.generar_nombre_modelo(config_modelo, config_entrenamiento)
        ruta_modelo = self.guardar_modelo(modelo, nombre_modelo)
        
        print(f"\nModelo guardado en: {ruta_modelo}")
        
        if mostrar_graficos:
            self.visualizador.mostrar_resultados_completos(modelo, self.datos, historial, config_modelo)
        
        return modelo, self.datos, historial, metricas
    
    def generar_nombre_modelo(self, config_modelo, config_entrenamiento, **kwargs):
        dimension_latente = config_modelo['dimension_latente']
        epochs = config_entrenamiento['epochs']
        learning_rate = config_modelo['learning_rate']
        
        lr_str = str(learning_rate).replace('.', '_')
        nombre = f"tp3_tipografia_lat{dimension_latente}_ep{epochs}_lr{lr_str}"
        return nombre
    
    def guardar_modelo(self, modelo, nombre_modelo, directorio="tp3/modelos"):
        if not os.path.exists(directorio):
            os.makedirs(directorio)
        
        ruta_completa = os.path.join(directorio, f"{nombre_modelo}.keras")
        modelo.save(ruta_completa)
        return ruta_completa


def entrenar_modelo(config_modelo, config_entrenamiento, mostrar_graficos=True, tamaño_imagen=32):
    entrenador = EntrenadorTipografia(tamaño_imagen)
    return entrenador.entrenar_modelo(config_modelo, config_entrenamiento, mostrar_graficos)


def generar_nombre_modelo(config_modelo, config_entrenamiento):
    entrenador = EntrenadorTipografia()
    return entrenador.generar_nombre_modelo(config_modelo, config_entrenamiento)


def guardar_modelo(modelo, nombre_modelo):
    entrenador = EntrenadorTipografia()
    return entrenador.guardar_modelo(modelo, nombre_modelo)


def main():
    parser = argparse.ArgumentParser(description='Entrenador de autocodificadores para tipografía')
    parser.add_argument('config_modelo', type=str, nargs='?',
                       help='Configuración del modelo (simple_2d, profundo_2d, ultra_profundo_2d, etc.)')
    parser.add_argument('config_entrenamiento', type=str, nargs='?',
                       help='Configuración de entrenamiento (rapido, normal, exhaustivo)')
    parser.add_argument('--config-modelo', '--config_modelo', type=str, dest='config_modelo_flag',
                       help='Configuración del modelo (alternativa con flag)')
    parser.add_argument('--config-entrenamiento', '--config_entrenamiento', type=str, dest='config_entrenamiento_flag',
                       help='Configuración de entrenamiento (alternativa con flag)')
    parser.add_argument('--sin-graficos', action='store_true',
                       help='No mostrar gráficos')
    parser.add_argument('--tamaño', type=int, default=32,
                       help='Tamaño de imagen (default: 32)')
    parser.add_argument('--listar', action='store_true',
                       help='Listar configuraciones disponibles')
    
    args = parser.parse_args()
    
    if args.listar:
        listar_configuraciones()
        return
    
    config_modelo_nombre = args.config_modelo_flag or args.config_modelo
    config_entrenamiento_nombre = args.config_entrenamiento_flag or args.config_entrenamiento
    
    if not config_modelo_nombre or not config_entrenamiento_nombre:
        print("Error: Debe especificar tanto la configuración del modelo como la de entrenamiento")
        return
    
    try:
        config_modelo = obtener_configuracion(config_modelo_nombre)
        config_entrenamiento = obtener_configuracion_entrenamiento(config_entrenamiento_nombre)
        
        modelo, datos, historial, metricas = entrenar_modelo(
            config_modelo, 
            config_entrenamiento, 
            mostrar_graficos=not args.sin_graficos,
            tamaño_imagen=args.tamaño
        )
        
        if modelo is None:
            print("Error en el procesamiento de datos. Entrenamiento cancelado.")
            return
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Configuraciones disponibles:")
        listar_configuraciones()


if __name__ == "__main__":
    main()
