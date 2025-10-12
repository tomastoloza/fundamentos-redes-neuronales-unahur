import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from .configuraciones import CONFIGURACIONES_AUTOCODIFICADOR, CONFIGURACIONES_ENTRENAMIENTO
from .entrenador import entrenar_modelo, generar_nombre_modelo


def ejecutar_experimento_paralelo(args):
    experimento, config_modelo_nombre, config_entrenamiento_nombre = args
    
    inicio_tiempo = time.time()
    
    try:
        config_modelo = CONFIGURACIONES_AUTOCODIFICADOR[config_modelo_nombre].copy()
        config_entrenamiento = CONFIGURACIONES_ENTRENAMIENTO[config_entrenamiento_nombre].copy()
        
        nombre_modelo = generar_nombre_modelo(config_modelo, config_entrenamiento)
        
        modelo, datos, historial, metricas = entrenar_modelo(
            config_modelo, 
            config_entrenamiento, 
            mostrar_graficos=False,
            conjunto_datos=1
        )
        
        tiempo_entrenamiento = time.time() - inicio_tiempo
        epochs_ejecutadas = len(historial.history['loss'])
        convergencia = epochs_ejecutadas < config_entrenamiento['epochs']
        
        resultado = {
            'nombre_modelo': nombre_modelo,
            'experimento': experimento,
            'config_modelo': config_modelo_nombre,
            'config_entrenamiento': config_entrenamiento_nombre,
            'dimension_latente': config_modelo['dimension_latente'],
            'capas_encoder': str(config_modelo['capas_encoder']),
            'capas_decoder': str(config_modelo['capas_decoder']),
            'learning_rate': config_modelo['learning_rate'],
            'epochs_config': config_entrenamiento['epochs'],
            'epochs_ejecutadas': epochs_ejecutadas,
            'loss_final': round(metricas['loss_final'], 6),
            'mse_final': round(float(metricas['mse']), 6),
            'precision_final': round(float(metricas['precision']), 4),
            'tiempo_entrenamiento': round(tiempo_entrenamiento, 2),
            'convergencia': convergencia,
            'exito': True
        }
        
        return resultado
        
    except Exception as e:
        tiempo_entrenamiento = time.time() - inicio_tiempo
        return {
            'nombre_modelo': f"{config_modelo_nombre}_{config_entrenamiento_nombre}",
            'experimento': experimento,
            'config_modelo': config_modelo_nombre,
            'config_entrenamiento': config_entrenamiento_nombre,
            'tiempo_entrenamiento': round(tiempo_entrenamiento, 2),
            'error': str(e),
            'exito': False
        }


class GridSearchAutocodificador:
    def __init__(self, directorio_resultados="tp3/resultados", max_workers=None):
        self.directorio_resultados = directorio_resultados
        self.max_workers = max_workers
        
        if not os.path.exists(self.directorio_resultados):
            os.makedirs(self.directorio_resultados)
    
    def ejecutar_grid_search(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_csv = os.path.join(self.directorio_resultados, f"grid_search_{timestamp}.csv")
        
        configuraciones_modelo = list(CONFIGURACIONES_AUTOCODIFICADOR.keys())
        configuraciones_entrenamiento = list(CONFIGURACIONES_ENTRENAMIENTO.keys())
        
        experimentos = []
        experimento = 1
        for config_modelo_nombre, config_entrenamiento_nombre in itertools.product(
            configuraciones_modelo, configuraciones_entrenamiento
        ):
            experimentos.append((experimento, config_modelo_nombre, config_entrenamiento_nombre))
            experimento += 1
        
        total_experimentos = len(experimentos)
        
        resultados = []
        completados = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(ejecutar_experimento_paralelo, exp): exp for exp in experimentos}
            
            for future in as_completed(futures):
                exp_args = futures[future]
                experimento_num, config_modelo, config_entrenamiento = exp_args
                
                try:
                    resultado = future.result()
                    resultados.append(resultado)
                    completados += 1
                    
                    if resultado['exito']:
                        print(f"✓ [{completados}/{total_experimentos}] {config_modelo}+{config_entrenamiento}: "
                              f"Loss={resultado['loss_final']:.6f}, MSE={resultado['mse_final']:.6f}, "
                              f"Precisión={resultado['precision_final']:.1%}, Tiempo={resultado['tiempo_entrenamiento']:.1f}s")
                    else:
                        print(f"✗ [{completados}/{total_experimentos}] {config_modelo}+{config_entrenamiento}: "
                              f"Error - {resultado.get('error', 'Desconocido')}")
                        
                except Exception as e:
                    print(f"✗ [{completados}/{total_experimentos}] {config_modelo}+{config_entrenamiento}: "
                          f"Error crítico - {e}")
                    completados += 1
        
        resultados.sort(key=lambda x: x['experimento'])
        
        with open(archivo_csv, 'w', newline='') as csvfile:
            fieldnames = [
                'nombre_modelo', 'experimento', 'config_modelo', 'config_entrenamiento',
                'dimension_latente', 'capas_encoder', 'capas_decoder',
                'learning_rate', 'epochs_config', 'epochs_ejecutadas',
                'loss_final', 'mse_final', 'precision_final',
                'tiempo_entrenamiento', 'convergencia', 'exito', 'error'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for resultado in resultados:
                writer.writerow(resultado)
        
        print(f"\nGrid search completado. Resultados guardados en: {archivo_csv}")
        return archivo_csv


def main():
    import multiprocessing
    
    max_workers = min(4, multiprocessing.cpu_count())
    grid_search = GridSearchAutocodificador(max_workers=max_workers)
    archivo_resultados = grid_search.ejecutar_grid_search()
    print(f"\nResultados disponibles en: {archivo_resultados}")


if __name__ == "__main__":
    main()
