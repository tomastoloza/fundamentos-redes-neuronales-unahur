import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from .configuraciones import CONFIGURACIONES_AUTOCODIFICADOR, CONFIGURACIONES_ENTRENAMIENTO
from .entrenador_eliminador_ruido import EntrenadorEliminadorRuido


TIPOS_RUIDO = ['binario', 'gaussiano', 'dropout']
NIVELES_RUIDO = {
    'binario': [0.05, 0.10, 0.15, 0.20],
    'gaussiano': [0.1, 0.2, 0.3, 0.4],
    'dropout': [0.10, 0.20, 0.30, 0.40]
}


def ejecutar_experimento_ruido_paralelo(args):
    experimento, config_modelo_nombre, config_entrenamiento_nombre, tipo_ruido, nivel_ruido = args
    
    inicio_tiempo = time.time()
    
    try:
        entrenador = EntrenadorEliminadorRuido()
        
        modelo, historial, metricas, nombre_modelo = entrenador.entrenar_modelo_completo(
            config_modelo_nombre, config_entrenamiento_nombre, tipo_ruido, nivel_ruido
        )
        
        tiempo_entrenamiento = time.time() - inicio_tiempo
        epochs_ejecutadas = len(historial.history['loss'])
        
        config_modelo = CONFIGURACIONES_AUTOCODIFICADOR[config_modelo_nombre]
        config_entrenamiento = CONFIGURACIONES_ENTRENAMIENTO[config_entrenamiento_nombre]
        convergencia = epochs_ejecutadas < config_entrenamiento['epochs']
        
        resultado = {
            'nombre_modelo': nombre_modelo,
            'experimento': experimento,
            'config_modelo': config_modelo_nombre,
            'config_entrenamiento': config_entrenamiento_nombre,
            'tipo_ruido': tipo_ruido,
            'nivel_ruido': nivel_ruido,
            'dimension_latente': config_modelo['dimension_latente'],
            'capas_encoder': str(config_modelo['capas_encoder']),
            'capas_decoder': str(config_modelo['capas_decoder']),
            'learning_rate': config_modelo['learning_rate'],
            'epochs_config': config_entrenamiento['epochs'],
            'epochs_ejecutadas': epochs_ejecutadas,
            'mse_limpio': round(metricas['mse_limpio'], 6),
            'mse_ruidoso': round(metricas['mse_ruidoso'], 6),
            'precision_limpieza': round(metricas['precision_limpieza'], 4),
            'mejora_snr': round(metricas['mejora_snr'], 2),
            'mejora_mse_porcentaje': round(metricas['mejora_mse_porcentaje'], 2),
            'efectivo': metricas['efectivo'],
            'tiempo_entrenamiento': round(tiempo_entrenamiento, 2),
            'convergencia': convergencia,
            'exito': True
        }
        
        return resultado
        
    except Exception as e:
        tiempo_entrenamiento = time.time() - inicio_tiempo
        return {
            'nombre_modelo': f"{config_modelo_nombre}_{config_entrenamiento_nombre}_{tipo_ruido}_{nivel_ruido}",
            'experimento': experimento,
            'config_modelo': config_modelo_nombre,
            'config_entrenamiento': config_entrenamiento_nombre,
            'tipo_ruido': tipo_ruido,
            'nivel_ruido': nivel_ruido,
            'tiempo_entrenamiento': round(tiempo_entrenamiento, 2),
            'error': str(e),
            'exito': False
        }


class GridSearchEliminadorRuido:
    def __init__(self, directorio_resultados="tp3/resultados", max_workers=None):
        self.directorio_resultados = directorio_resultados
        self.max_workers = max_workers
        
        if not os.path.exists(self.directorio_resultados):
            os.makedirs(self.directorio_resultados)
    
    def ejecutar_grid_search_ruido(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_csv = os.path.join(self.directorio_resultados, f"grid_search_ruido_{timestamp}.csv")
        
        configuraciones_modelo = [k for k in CONFIGURACIONES_AUTOCODIFICADOR.keys() 
                                if CONFIGURACIONES_AUTOCODIFICADOR[k]['dimension_latente'] == 2]
        configuraciones_entrenamiento = list(CONFIGURACIONES_ENTRENAMIENTO.keys())
        
        experimentos = []
        experimento = 1
        
        for config_modelo, config_entrenamiento, tipo_ruido in itertools.product(
            configuraciones_modelo, configuraciones_entrenamiento, TIPOS_RUIDO
        ):
            for nivel_ruido in NIVELES_RUIDO[tipo_ruido]:
                experimentos.append((experimento, config_modelo, config_entrenamiento, tipo_ruido, nivel_ruido))
                experimento += 1
        
        total_experimentos = len(experimentos)
        print(f"=== GRID SEARCH ELIMINADOR DE RUIDO ===")
        print(f"Configuraciones 2D: {len(configuraciones_modelo)}")
        print(f"Tipos de ruido: {len(TIPOS_RUIDO)}")
        print(f"Total experimentos: {total_experimentos}")
        print(f"Workers: {self.max_workers or 'auto'}")
        print(f"Archivo resultados: {archivo_csv}")
        print()
        
        resultados = []
        completados = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(ejecutar_experimento_ruido_paralelo, exp): exp for exp in experimentos}
            
            for future in as_completed(futures):
                exp_args = futures[future]
                experimento_num, config_modelo, config_entrenamiento, tipo_ruido, nivel_ruido = exp_args
                
                try:
                    resultado = future.result()
                    resultados.append(resultado)
                    completados += 1
                    
                    if resultado['exito']:
                        print(f"✓ [{completados}/{total_experimentos}] {config_modelo}+{tipo_ruido}({nivel_ruido}): "
                              f"MSE_limpio={resultado['mse_limpio']:.6f}, "
                              f"Mejora_SNR={resultado['mejora_snr']:.1f}dB, "
                              f"Efectivo={resultado['efectivo']}, "
                              f"Tiempo={resultado['tiempo_entrenamiento']:.1f}s")
                    else:
                        print(f"✗ [{completados}/{total_experimentos}] {config_modelo}+{tipo_ruido}({nivel_ruido}): "
                              f"Error - {resultado.get('error', 'Desconocido')}")
                        
                except Exception as e:
                    print(f"✗ [{completados}/{total_experimentos}] {config_modelo}+{tipo_ruido}({nivel_ruido}): "
                          f"Error crítico - {e}")
                    completados += 1
        
        resultados.sort(key=lambda x: x['experimento'])
        
        with open(archivo_csv, 'w', newline='') as csvfile:
            fieldnames = [
                'nombre_modelo', 'experimento', 'config_modelo', 'config_entrenamiento',
                'tipo_ruido', 'nivel_ruido', 'dimension_latente', 'capas_encoder', 'capas_decoder',
                'learning_rate', 'epochs_config', 'epochs_ejecutadas',
                'mse_limpio', 'mse_ruidoso', 'precision_limpieza', 'mejora_snr', 
                'mejora_mse_porcentaje', 'efectivo', 'tiempo_entrenamiento', 'convergencia', 
                'exito', 'error'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for resultado in resultados:
                writer.writerow(resultado)
        
        print(f"\nGrid search de eliminación de ruido completado.")
        print(f"Resultados guardados en: {archivo_csv}")
        return archivo_csv


def main():
    import multiprocessing
    
    max_workers = min(3, multiprocessing.cpu_count())
    grid_search = GridSearchEliminadorRuido(max_workers=max_workers)
    archivo_resultados = grid_search.ejecutar_grid_search_ruido()
    print(f"\nResultados disponibles en: {archivo_resultados}")


if __name__ == "__main__":
    main()
