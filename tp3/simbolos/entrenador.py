import argparse

from tp3.comun.entrenador_base import EntrenadorBase
from .configuraciones import obtener_configuracion, obtener_configuracion_entrenamiento, listar_configuraciones


class EntrenadorAutocodificador(EntrenadorBase):
    def __init__(self, conjunto_datos=1):
        super().__init__(conjunto_datos)
    
    def entrenar_modelo(self, config_modelo, config_entrenamiento, mostrar_graficos=True, **kwargs):
        if not self.validar_datos():
            return None, None, None, None
        
        modelo = self.constructor.crear_autocodificador_desde_config(config_modelo)
        callbacks = self.crear_callbacks(config_entrenamiento)
        
        historial = modelo.fit(
            self.datos, self.datos,
            epochs=config_entrenamiento['epochs'],
            batch_size=config_entrenamiento.get('batch_size', 32),
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
        return self.generar_nombre_modelo_base(config_modelo, config_entrenamiento, "tp3")


def entrenar_modelo(config_modelo, config_entrenamiento, mostrar_graficos=True, conjunto_datos=1):
    entrenador = EntrenadorAutocodificador(conjunto_datos)
    return entrenador.entrenar_modelo(config_modelo, config_entrenamiento, mostrar_graficos)


def generar_nombre_modelo(config_modelo, config_entrenamiento):
    entrenador = EntrenadorAutocodificador()
    return entrenador.generar_nombre_modelo(config_modelo, config_entrenamiento)


def guardar_modelo(modelo, nombre_modelo):
    entrenador = EntrenadorAutocodificador()
    return entrenador.guardar_modelo(modelo, nombre_modelo)


def main():
    parser = argparse.ArgumentParser(description='Entrenador de autocodificadores')
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
    parser.add_argument('--conjunto', type=int, default=1, choices=[1, 2, 3],
                       help='Conjunto de datos a usar (1, 2 o 3)')
    parser.add_argument('--listar', action='store_true',
                       help='Listar configuraciones disponibles')
    
    args = parser.parse_args()
    
    if args.listar:
        listar_configuraciones()
        return
    
    # Usar flags si están disponibles, sino usar argumentos posicionales
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
