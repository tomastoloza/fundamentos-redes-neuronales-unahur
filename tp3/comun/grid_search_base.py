import os
import csv
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


class GridSearchBase(ABC):
    def __init__(self, directorio_resultados="tp3/resultados", max_workers=None):
        self.directorio_resultados = directorio_resultados
        self.max_workers = max_workers
        self.crear_directorio_resultados()
    
    def crear_directorio_resultados(self):
        if not os.path.exists(self.directorio_resultados):
            os.makedirs(self.directorio_resultados)
    
    def generar_nombre_archivo_resultados(self, prefijo="grid_search"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefijo}_{timestamp}.csv"
    
    def guardar_resultados_csv(self, resultados, nombre_archivo):
        ruta_archivo = os.path.join(self.directorio_resultados, nombre_archivo)
        
        if not resultados:
            print("No hay resultados para guardar")
            return ruta_archivo
        
        fieldnames = resultados[0].keys()
        
        with open(ruta_archivo, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(resultados)
        
        return ruta_archivo
    
    def mostrar_progreso(self, completados, total, tiempo_inicio):
        tiempo_transcurrido = time.time() - tiempo_inicio
        porcentaje = (completados / total) * 100
        tiempo_estimado = (tiempo_transcurrido / completados) * total if completados > 0 else 0
        tiempo_restante = tiempo_estimado - tiempo_transcurrido
        
        print(f"Progreso: {completados}/{total} ({porcentaje:.1f}%) - "
              f"Tiempo transcurrido: {tiempo_transcurrido:.1f}s - "
              f"Tiempo estimado restante: {tiempo_restante:.1f}s")
    
    def ejecutar_experimentos_paralelos(self, configuraciones, funcion_experimento):
        resultados = []
        tiempo_inicio = time.time()
        total_experimentos = len(configuraciones)
        
        print(f"Iniciando grid search con {total_experimentos} experimentos...")
        
        if self.max_workers == 1:
            for i, config in enumerate(configuraciones):
                try:
                    resultado = funcion_experimento(config)
                    if resultado:
                        resultados.append(resultado)
                    self.mostrar_progreso(i + 1, total_experimentos, tiempo_inicio)
                except Exception as e:
                    print(f"Error en experimento {i + 1}: {e}")
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(funcion_experimento, config): i 
                          for i, config in enumerate(configuraciones)}
                
                completados = 0
                for future in as_completed(futures):
                    try:
                        resultado = future.result()
                        if resultado:
                            resultados.append(resultado)
                    except Exception as e:
                        config_idx = futures[future]
                        print(f"Error en experimento {config_idx + 1}: {e}")
                    
                    completados += 1
                    if completados % 10 == 0 or completados == total_experimentos:
                        self.mostrar_progreso(completados, total_experimentos, tiempo_inicio)
        
        tiempo_total = time.time() - tiempo_inicio
        print(f"\nGrid search completado en {tiempo_total:.1f} segundos")
        print(f"Experimentos exitosos: {len(resultados)}/{total_experimentos}")
        
        return resultados
    
    def analizar_mejores_resultados(self, resultados, metrica_principal, top_n=10):
        if not resultados:
            print("No hay resultados para analizar")
            return []
        
        resultados_ordenados = sorted(
            resultados, 
            key=lambda x: x.get(metrica_principal, 0), 
            reverse=True
        )
        
        print(f"\n=== TOP {top_n} MEJORES RESULTADOS ({metrica_principal}) ===")
        for i, resultado in enumerate(resultados_ordenados[:top_n]):
            print(f"{i+1}. {metrica_principal}: {resultado.get(metrica_principal, 'N/A'):.4f}")
            self.mostrar_resumen_configuracion(resultado)
            print()
        
        return resultados_ordenados[:top_n]
    
    @abstractmethod
    def generar_configuraciones(self):
        pass
    
    @abstractmethod
    def ejecutar_experimento_individual(self, configuracion):
        pass
    
    @abstractmethod
    def mostrar_resumen_configuracion(self, resultado):
        pass
    
    def ejecutar_grid_search_completo(self, metrica_principal="precision", top_n=10):
        configuraciones = self.generar_configuraciones()
        resultados = self.ejecutar_experimentos_paralelos(
            configuraciones, 
            self.ejecutar_experimento_individual
        )
        
        if resultados:
            nombre_archivo = self.generar_nombre_archivo_resultados()
            ruta_archivo = self.guardar_resultados_csv(resultados, nombre_archivo)
            print(f"Resultados guardados en: {ruta_archivo}")
            
            mejores = self.analizar_mejores_resultados(resultados, metrica_principal, top_n)
            return resultados, mejores
        
        return [], []
