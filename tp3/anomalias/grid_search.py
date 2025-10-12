import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from .configuraciones import (
    CONFIGURACIONES_ARQUITECTURA, 
    CONFIGURACIONES_ENTRENAMIENTO,
    CONFIGURACIONES_DATOS,
    CONFIGURACIONES_EVALUACION
)
from .entrenador import EntrenadorAnomalias


def ejecutar_experimento_anomalias_paralelo(args):
    """
    Ejecuta un experimento individual de detección de anomalías en paralelo.
    """
    (experimento, config_arquitectura_nombre, config_entrenamiento_nombre, 
     config_datos_nombre, config_evaluacion_nombre) = args
    
    inicio_tiempo = time.time()
    
    try:
        # Crear entrenador con configuración específica
        config_arquitectura = CONFIGURACIONES_ARQUITECTURA[config_arquitectura_nombre]
        entrenador = EntrenadorAnomalias(
            longitud_serie=config_arquitectura['longitud_serie'],
            dimension_latente=config_arquitectura['dimension_latente']
        )
        
        # Obtener configuraciones
        config_datos = CONFIGURACIONES_DATOS[config_datos_nombre]
        config_entrenamiento = CONFIGURACIONES_ENTRENAMIENTO[config_entrenamiento_nombre]
        config_evaluacion = CONFIGURACIONES_EVALUACION[config_evaluacion_nombre]
        
        # Ejecutar experimento completo
        resultados = entrenador.ejecutar_experimento_completo(
            config_datos=config_datos,
            config_entrenamiento=config_entrenamiento,
            config_evaluacion=config_evaluacion
        )
        
        tiempo_total = time.time() - inicio_tiempo
        
        # Extraer métricas
        metricas = resultados['metricas_evaluacion']
        
        resultado = {
            'experimento': experimento,
            'config_arquitectura': config_arquitectura_nombre,
            'config_entrenamiento': config_entrenamiento_nombre,
            'config_datos': config_datos_nombre,
            'config_evaluacion': config_evaluacion_nombre,
            'dimension_latente': config_arquitectura['dimension_latente'],
            'longitud_serie': config_arquitectura['longitud_serie'],
            'epochs': config_entrenamiento['epochs'],
            'learning_rate': config_entrenamiento['learning_rate'],
            'batch_size': config_entrenamiento['batch_size'],
            'num_entrenamiento': config_datos['num_entrenamiento'],
            'num_prueba_normal': config_datos['num_prueba_normal'],
            'num_prueba_anomala': config_datos['num_prueba_anomala'],
            'percentil_umbral': config_evaluacion['percentil_umbral'],
            'umbral_anomalia': resultados['umbral_anomalia'],
            'precision': metricas['precision'],
            'recall': metricas['recall'],
            'f1_score': metricas['f1_score'],
            'accuracy': metricas['accuracy'],
            'tiempo_entrenamiento': tiempo_total,
            'nombre_modelo': resultados['nombre_modelo'],
            'convergio': len(resultados['historial_entrenamiento'].history['loss']) < config_entrenamiento['epochs'],
            'loss_final': min(resultados['historial_entrenamiento'].history['loss']),
            'val_loss_final': min(resultados['historial_entrenamiento'].history.get('val_loss', [float('inf')])),
            'estado': 'exitoso'
        }
        
        print(f"✓ Experimento {experimento} completado - F1: {metricas['f1_score']:.3f}, "
              f"Precision: {metricas['precision']:.3f}, Recall: {metricas['recall']:.3f}")
        
        return resultado
        
    except Exception as e:
        tiempo_total = time.time() - inicio_tiempo
        
        print(f"✗ Experimento {experimento} falló: {str(e)}")
        
        return {
            'experimento': experimento,
            'config_arquitectura': config_arquitectura_nombre,
            'config_entrenamiento': config_entrenamiento_nombre,
            'config_datos': config_datos_nombre,
            'config_evaluacion': config_evaluacion_nombre,
            'dimension_latente': CONFIGURACIONES_ARQUITECTURA[config_arquitectura_nombre]['dimension_latente'],
            'longitud_serie': CONFIGURACIONES_ARQUITECTURA[config_arquitectura_nombre]['longitud_serie'],
            'epochs': CONFIGURACIONES_ENTRENAMIENTO[config_entrenamiento_nombre]['epochs'],
            'learning_rate': CONFIGURACIONES_ENTRENAMIENTO[config_entrenamiento_nombre]['learning_rate'],
            'batch_size': CONFIGURACIONES_ENTRENAMIENTO[config_entrenamiento_nombre]['batch_size'],
            'num_entrenamiento': CONFIGURACIONES_DATOS[config_datos_nombre]['num_entrenamiento'],
            'num_prueba_normal': CONFIGURACIONES_DATOS[config_datos_nombre]['num_prueba_normal'],
            'num_prueba_anomala': CONFIGURACIONES_DATOS[config_datos_nombre]['num_prueba_anomala'],
            'percentil_umbral': CONFIGURACIONES_EVALUACION[config_evaluacion_nombre]['percentil_umbral'],
            'umbral_anomalia': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'tiempo_entrenamiento': tiempo_total,
            'nombre_modelo': '',
            'convergio': False,
            'loss_final': float('inf'),
            'val_loss_final': float('inf'),
            'estado': f'error: {str(e)}'
        }


class GridSearchAnomalias:
    def __init__(self, directorio_resultados="tp3/resultados", max_workers=None):
        self.directorio_resultados = directorio_resultados
        self.max_workers = max_workers
        os.makedirs(directorio_resultados, exist_ok=True)
    
    def generar_configuraciones(self):
        """
        Genera todas las combinaciones de configuraciones para el grid search.
        """
        configuraciones = []
        
        for config_arquitectura in CONFIGURACIONES_ARQUITECTURA.keys():
            for config_entrenamiento in CONFIGURACIONES_ENTRENAMIENTO.keys():
                for config_datos in CONFIGURACIONES_DATOS.keys():
                    for config_evaluacion in CONFIGURACIONES_EVALUACION.keys():
                        configuraciones.append((
                            len(configuraciones) + 1,
                            config_arquitectura,
                            config_entrenamiento,
                            config_datos,
                            config_evaluacion
                        ))
        
        return configuraciones
    
    def ejecutar_grid_search_completo(self):
        """
        Ejecuta el grid search completo para detección de anomalías.
        """
        print("=== GRID SEARCH DETECCIÓN DE ANOMALÍAS ===")
        
        configuraciones = self.generar_configuraciones()
        total_experimentos = len(configuraciones)
        
        print(f"Total de experimentos: {total_experimentos}")
        print(f"Configuraciones de arquitectura: {len(CONFIGURACIONES_ARQUITECTURA)}")
        print(f"Configuraciones de entrenamiento: {len(CONFIGURACIONES_ENTRENAMIENTO)}")
        print(f"Configuraciones de datos: {len(CONFIGURACIONES_DATOS)}")
        print(f"Configuraciones de evaluación: {len(CONFIGURACIONES_EVALUACION)}")
        print(f"Workers paralelos: {self.max_workers or 'auto'}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_resultados = os.path.join(
            self.directorio_resultados, 
            f"grid_search_anomalias_{timestamp}.csv"
        )
        
        inicio_total = time.time()
        resultados = []
        
        # Ejecutar experimentos en paralelo
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(ejecutar_experimento_anomalias_paralelo, config): config 
                for config in configuraciones
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    resultado = future.result()
                    resultados.append(resultado)
                    
                    progreso = (i / total_experimentos) * 100
                    tiempo_transcurrido = time.time() - inicio_total
                    tiempo_estimado = (tiempo_transcurrido / i) * total_experimentos
                    tiempo_restante = tiempo_estimado - tiempo_transcurrido
                    
                    print(f"Progreso: {i}/{total_experimentos} ({progreso:.1f}%) - "
                          f"Tiempo restante: {tiempo_restante/60:.1f}min")
                    
                except Exception as e:
                    print(f"Error en experimento: {e}")
        
        # Guardar resultados en CSV
        self._guardar_resultados_csv(resultados, archivo_resultados)
        
        # Mostrar resumen
        tiempo_total = time.time() - inicio_total
        self._mostrar_resumen(resultados, tiempo_total)
        
        return resultados, archivo_resultados
    
    def _guardar_resultados_csv(self, resultados, archivo):
        """
        Guarda los resultados en formato CSV.
        """
        if not resultados:
            print("No hay resultados para guardar")
            return
        
        fieldnames = resultados[0].keys()
        
        with open(archivo, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(resultados)
        
        print(f"Resultados guardados en: {archivo}")
    
    def _mostrar_resumen(self, resultados, tiempo_total):
        """
        Muestra un resumen de los resultados del grid search.
        """
        if not resultados:
            print("No hay resultados para mostrar")
            return
        
        exitosos = [r for r in resultados if r['estado'] == 'exitoso']
        fallidos = len(resultados) - len(exitosos)
        
        print(f"\n=== RESUMEN GRID SEARCH ANOMALÍAS ===")
        print(f"Tiempo total: {tiempo_total/60:.1f} minutos")
        print(f"Experimentos exitosos: {len(exitosos)}")
        print(f"Experimentos fallidos: {fallidos}")
        
        if exitosos:
            # Ordenar por F1-score
            exitosos.sort(key=lambda x: x['f1_score'], reverse=True)
            
            print(f"\n=== TOP 5 MEJORES CONFIGURACIONES (F1-Score) ===")
            for i, resultado in enumerate(exitosos[:5], 1):
                print(f"{i}. F1: {resultado['f1_score']:.3f} | "
                      f"Precision: {resultado['precision']:.3f} | "
                      f"Recall: {resultado['recall']:.3f} | "
                      f"Arch: {resultado['config_arquitectura']} | "
                      f"Lat: {resultado['dimension_latente']} | "
                      f"LR: {resultado['learning_rate']} | "
                      f"Epochs: {resultado['epochs']}")
            
            # Estadísticas generales
            f1_scores = [r['f1_score'] for r in exitosos]
            precision_scores = [r['precision'] for r in exitosos]
            recall_scores = [r['recall'] for r in exitosos]
            
            print(f"\n=== ESTADÍSTICAS GENERALES ===")
            print(f"F1-Score promedio: {sum(f1_scores)/len(f1_scores):.3f}")
            print(f"F1-Score máximo: {max(f1_scores):.3f}")
            print(f"F1-Score mínimo: {min(f1_scores):.3f}")
            print(f"Precision promedio: {sum(precision_scores)/len(precision_scores):.3f}")
            print(f"Recall promedio: {sum(recall_scores)/len(recall_scores):.3f}")


def main():
    """
    Función principal para ejecutar el grid search desde línea de comandos.
    """
    print("Iniciando Grid Search para Detección de Anomalías...")
    
    grid_search = GridSearchAnomalias(max_workers=2)  # Limitar workers para evitar sobrecarga
    resultados, archivo = grid_search.ejecutar_grid_search_completo()
    
    print(f"\nGrid Search completado. Resultados guardados en: {archivo}")


if __name__ == "__main__":
    main()
