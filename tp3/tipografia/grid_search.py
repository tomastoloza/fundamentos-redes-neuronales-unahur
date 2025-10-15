import time
import os
import csv
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from .configuraciones import CONFIGURACIONES_AUTOCODIFICADOR, CONFIGURACIONES_ENTRENAMIENTO
from .entrenador import EntrenadorTipografia


class GridSearchTipografia:
    def __init__(self, directorio_resultados="tp3/resultados", max_workers=None, tamaño_imagen=32):
        self.directorio_resultados = directorio_resultados
        self.max_workers = max_workers
        self.tamaño_imagen = tamaño_imagen
        self.entrenador = EntrenadorTipografia(tamaño_imagen)
        
        if not os.path.exists(directorio_resultados):
            os.makedirs(directorio_resultados)
    
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
            
            entrenador = EntrenadorTipografia(self.tamaño_imagen)
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
                'nombre_modelo': nombre_modelo,
                'tamaño_imagen': self.tamaño_imagen
            }
            
            return resultado
            
        except Exception as e:
            print(f"Error en experimento {experimento}: {e}")
            return None
    
    def ejecutar_grid_search_completo(self, metrica_principal="precision", top_n=10):
        configuraciones = self.generar_configuraciones()
        total_experimentos = len(configuraciones)
        
        print(f"\n=== GRID SEARCH TIPOGRAFÍA ===")
        print(f"Total de experimentos: {total_experimentos}")
        print(f"Tamaño de imagen: {self.tamaño_imagen}x{self.tamaño_imagen}")
        print(f"Configuraciones de modelo: {len(CONFIGURACIONES_AUTOCODIFICADOR)}")
        print(f"Configuraciones de entrenamiento: {len(CONFIGURACIONES_ENTRENAMIENTO)}")
        print(f"Métrica principal: {metrica_principal}")
        print(f"Workers: {self.max_workers if self.max_workers else 'auto'}\n")
        
        resultados = []
        completados = 0
        
        if self.max_workers and self.max_workers > 1:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futuros = {executor.submit(ejecutar_experimento_paralelo, config): config 
                          for config in configuraciones}
                
                for futuro in as_completed(futuros):
                    resultado = futuro.result()
                    if resultado:
                        resultados.append(resultado)
                        completados += 1
                        print(f"\n[{completados}/{total_experimentos}] Experimento {resultado['experimento']} completado")
                        self.mostrar_resumen_configuracion(resultado)
        else:
            for config in configuraciones:
                resultado = self.ejecutar_experimento_individual(config)
                if resultado:
                    resultados.append(resultado)
                    completados += 1
                    print(f"\n[{completados}/{total_experimentos}] Experimento {resultado['experimento']} completado")
                    self.mostrar_resumen_configuracion(resultado)
        
        if not resultados:
            print("No se completaron experimentos exitosamente.")
            return [], []
        
        archivo_resultados = self.guardar_resultados_csv(resultados)
        print(f"\n=== RESULTADOS GUARDADOS ===")
        print(f"Archivo: {archivo_resultados}")
        
        mejores = self.analizar_mejores_resultados(resultados, metrica_principal, top_n)
        
        return resultados, mejores
    
    def guardar_resultados_csv(self, resultados):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"grid_search_tipografia_{timestamp}.csv"
        ruta_completa = os.path.join(self.directorio_resultados, nombre_archivo)
        
        campos = ['experimento', 'arquitectura', 'entrenamiento', 'dimension_latente', 
                 'epochs_configuradas', 'epochs_ejecutadas', 'learning_rate', 'batch_size',
                 'loss_final', 'mse', 'precision', 'convergio', 'tiempo_entrenamiento', 
                 'nombre_modelo', 'tamaño_imagen']
        
        with open(ruta_completa, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados)
        
        return ruta_completa
    
    def analizar_mejores_resultados(self, resultados, metrica_principal, top_n):
        resultados_ordenados = sorted(resultados, key=lambda x: x[metrica_principal], reverse=True)
        mejores = resultados_ordenados[:top_n]
        
        print(f"\n=== TOP {top_n} MEJORES CONFIGURACIONES (por {metrica_principal}) ===")
        for i, resultado in enumerate(mejores, 1):
            print(f"\n{i}. {resultado['arquitectura']} - {resultado['entrenamiento']}")
            print(f"   {metrica_principal.capitalize()}: {resultado[metrica_principal]:.4f}")
            self.mostrar_resumen_configuracion(resultado)
        
        return mejores
    
    def mostrar_resumen_configuracion(self, resultado):
        print(f"   Arquitectura: {resultado['arquitectura']}")
        print(f"   Dimensión latente: {resultado['dimension_latente']}")
        print(f"   Learning rate: {resultado['learning_rate']}")
        print(f"   Epochs: {resultado['epochs_ejecutadas']}/{resultado['epochs_configuradas']}")
        print(f"   MSE: {resultado['mse']:.6f}")
        print(f"   Convergió: {'Sí' if resultado['convergio'] else 'No'}")


def ejecutar_experimento_paralelo(args):
    grid_search = GridSearchTipografia()
    return grid_search.ejecutar_experimento_individual(args)


def main():
    print("=== GRID SEARCH AUTOCODIFICADORES TIPOGRAFÍA ===")
    
    grid_search = GridSearchTipografia(tamaño_imagen=32)
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
