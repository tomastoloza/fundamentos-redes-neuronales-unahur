import argparse
import os

from tensorflow import keras

from .configuraciones import obtener_configuracion, obtener_configuracion_entrenamiento, listar_configuraciones
from .constructor_modelos import ConstructorModelos
from .procesador_datos import ProcesadorDatos
from .visualizador_resultados import VisualizadorResultados


def entrenar_modelo(config_modelo, config_entrenamiento, mostrar_graficos=True, conjunto_datos=1):
    procesador = ProcesadorDatos(conjunto_datos)
    datos = procesador.obtener_datos_procesados()
    constructor = ConstructorModelos()
    visualizador = VisualizadorResultados()
    
    estadisticas = procesador.obtener_estadisticas()
    valido, errores = procesador.validar_datos()
    
    if not valido:
        print("Errores en los datos:")
        for error in errores:
            print(f"  - {error}")
        return None, None, None, None

    modelo = constructor.crear_autocodificador_desde_config(config_modelo)

    callbacks = []
    if config_entrenamiento.get('early_stopping', False):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=config_entrenamiento.get('patience', 50),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    historial = modelo.fit(
        datos, datos,
        epochs=config_entrenamiento['epochs'],
        batch_size=config_entrenamiento.get('batch_size', 32),
        verbose=1,
        callbacks=callbacks
    )
    
    print("\n=== EVALUACIÓN FINAL ===")
    evaluacion = modelo.evaluate(datos, datos, verbose=0)
    loss_final = evaluacion[0] if isinstance(evaluacion, list) else evaluacion
    print(f"Loss final: {loss_final:.6f}")
    
    predicciones = modelo.predict(datos, verbose=0)
    mse = ((datos - predicciones) ** 2).mean()
    precision = ((predicciones > 0.5) == (datos > 0.5)).mean()
    
    metricas = {
        'loss_final': loss_final,
        'mse': mse,
        'precision': precision
    }
    
    nombre_modelo = generar_nombre_modelo(config_modelo, config_entrenamiento)
    ruta_modelo = guardar_modelo(modelo, nombre_modelo)
    
    print(f"\nModelo guardado en: {ruta_modelo}")

    if mostrar_graficos:
        visualizador.mostrar_resultados_completos(modelo, datos, historial, config_modelo)
    
    return modelo, datos, historial, metricas


def generar_nombre_modelo(config_modelo, config_entrenamiento):
    dimension_latente = config_modelo['dimension_latente']
    epochs = config_entrenamiento['epochs']
    learning_rate = config_modelo['learning_rate']
    
    lr_str = str(learning_rate).replace('.', '_')
    
    nombre = f"tp3_lat{dimension_latente}_ep{epochs}_lr{lr_str}"
    return nombre

def guardar_modelo(modelo, nombre_modelo):
    directorio_modelos = "tp3/modelos"
    
    if not os.path.exists(directorio_modelos):
        os.makedirs(directorio_modelos)
    
    ruta_completa = os.path.join(directorio_modelos, f"{nombre_modelo}.keras")
    modelo.save(ruta_completa)
    
    return ruta_completa

def main():
    parser = argparse.ArgumentParser(description='Entrenador de autocodificadores')
    parser.add_argument('config_modelo', type=str,
                       help='Configuración del modelo (simple_2d, profundo_2d, ancho_3d, compacto_5d)')
    parser.add_argument('config_entrenamiento', type=str,
                       help='Configuración de entrenamiento (rapido, normal, exhaustivo)')
    parser.add_argument('--sin-graficos', action='store_true',
                       help='No mostrar gráficos')
    parser.add_argument('--conjunto', type=int, default=1, choices=[1, 2, 3],
                       help='Conjunto de datos a usar (1, 2 o 3)')
    
    args = parser.parse_args()
    
    try:
        config_modelo = obtener_configuracion(args.config_modelo)
        config_entrenamiento = obtener_configuracion_entrenamiento(args.config_entrenamiento)
        
        modelo, datos, historial, metricas = entrenar_modelo(
            config_modelo, 
            config_entrenamiento, 
            mostrar_graficos=not args.sin_graficos,
            conjunto_datos=args.conjunto
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
