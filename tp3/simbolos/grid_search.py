import time

from tp3.comun.grid_search_base import GridSearchBase
from .configuraciones import CONFIGURACIONES_AUTOCODIFICADOR, CONFIGURACIONES_ENTRENAMIENTO
from .entrenador import EntrenadorAutocodificador


class GridSearchAutocodificador(GridSearchBase):
    def __init__(self, directorio_resultados="tp3/resultados", max_workers=None):
        super().__init__(directorio_resultados, max_workers)
        self.entrenador = EntrenadorAutocodificador()
    
    def generar_configuraciones(self):
        configuraciones = []
        
        for config_modelo_nombre in CONFIGURACIONES_AUTOCODIFICADOR.keys():
            for config_entrenamiento_nombre in CONFIGURACIONES_ENTRENAMIENTO.keys():
                configuraciones.append({
                    'experimento': len(configuraciones) + 1,
                    'config_modelo_nombre': config_modelo_nombre,
                    'config_entrenamiento_nombre': config_entrenamiento_nombre
                })
        
        return configuraciones
    
    def ejecutar_experimento_individual(self, configuracion):
        experimento = configuracion['experimento']
        config_modelo_nombre = configuracion['config_modelo_nombre']
        config_entrenamiento_nombre = configuracion['config_entrenamiento_nombre']
        
        inicio_tiempo = time.time()
        
        try:
            config_modelo = CONFIGURACIONES_AUTOCODIFICADOR[config_modelo_nombre].copy()
            config_entrenamiento = CONFIGURACIONES_ENTRENAMIENTO[config_entrenamiento_nombre].copy()
            
            entrenador = EntrenadorAutocodificador()
            nombre_modelo = entrenador.generar_nombre_modelo(config_modelo, config_entrenamiento)
            
            modelo, datos, historial, metricas = entrenador.entrenar_modelo(
                config_modelo, 
                config_entrenamiento, 
                mostrar_graficos=False
            )
            
            tiempo_entrenamiento = time.time() - inicio_tiempo
            
            if modelo is None:
                return None
            
            epochs_ejecutadas = len(historial.history['loss'])
            convergio = epochs_ejecutadas < config_entrenamiento['epochs']
            
            resultado = {
                'experimento': experimento,
                'arquitectura': config_modelo_nombre,
                'entrenamiento': config_entrenamiento_nombre,
                'dimension_latente': config_modelo['dimension_latente'],
                'epochs_configuradas': config_entrenamiento['epochs'],
                'epochs_ejecutadas': epochs_ejecutadas,
                'learning_rate': config_modelo['learning_rate'],
                'batch_size': config_entrenamiento.get('batch_size', 32),
                'loss_final': metricas['loss_final'],
                'mse': metricas['mse'],
                'precision': metricas['precision'],
                'convergio': convergio,
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'nombre_modelo': nombre_modelo
            }
            
            return resultado
            
        except Exception as e:
            print(f"Error en experimento {experimento}: {e}")
            return None
    
    def mostrar_resumen_configuracion(self, resultado):
        print(f"   Arquitectura: {resultado['arquitectura']}")
        print(f"   Dimensión latente: {resultado['dimension_latente']}")
        print(f"   Learning rate: {resultado['learning_rate']}")
        print(f"   Epochs: {resultado['epochs_ejecutadas']}/{resultado['epochs_configuradas']}")
        print(f"   MSE: {resultado['mse']:.6f}")
        print(f"   Convergió: {'Sí' if resultado['convergio'] else 'No'}")
    
    def generar_nombre_archivo_resultados(self, prefijo="grid_search"):
        return super().generar_nombre_archivo_resultados("grid_search_autocodificador")


def ejecutar_experimento_paralelo(args):
    grid_search = GridSearchAutocodificador()
    return grid_search.ejecutar_experimento_individual(args)


def main():
    print("=== GRID SEARCH AUTOCODIFICADORES ===")
    
    grid_search = GridSearchAutocodificador()
    resultados, mejores = grid_search.ejecutar_grid_search_completo(
        metrica_principal="precision", 
        top_n=10
    )
    
    if mejores:
        print(f"\n=== MEJOR CONFIGURACIÓN ===")
        mejor = mejores[0]
        print(f"Precisión: {mejor['precision']:.4f}")
        grid_search.mostrar_resumen_configuracion(mejor)


if __name__ == "__main__":
    main()
