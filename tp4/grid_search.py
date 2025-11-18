import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from .entrenador import EntrenadorCNN
from .configuraciones import CONFIGURACIONES_CNN, CONFIGURACIONES_ENTRENAMIENTO, CONFIGURACIONES_AUGMENTATION


def ejecutar_experimento_cnn_grid_paralelo(args):
    config_cnn, config_entrenamiento, config_augmentation, tamaño_imagen, max_imagenes_por_clase, experimento_id = args
    
    inicio_tiempo = time.time()
    
    try:
        entrenador = EntrenadorCNN()
        modelo, historia, nombre_modelo = entrenador.entrenar(
            config_cnn['nombre'], 
            config_entrenamiento['nombre'], 
            tamaño_imagen, 
            max_imagenes_por_clase,
            config_augmentation=config_augmentation['nombre'] if config_augmentation else None
        )
        
        tiempo_total = time.time() - inicio_tiempo
        
        mejor_val_acc = max(historia.history.get('val_accuracy', [0]))
        mejor_val_loss = min(historia.history.get('val_loss', [float('inf')]))
        mejor_epoch = historia.history['val_accuracy'].index(mejor_val_acc) + 1 if 'val_accuracy' in historia.history else 1
        
        resultado = {
            'experimento': experimento_id,
            'config_cnn': config_cnn['nombre'],
            'config_entrenamiento': config_entrenamiento['nombre'],
            'config_augmentation': config_augmentation['nombre'] if config_augmentation else 'ninguno',
            'nombre_modelo': nombre_modelo,
            'val_accuracy_final': mejor_val_acc,
            'val_loss_final': mejor_val_loss,
            'mejor_epoch': mejor_epoch,
            'tiempo_entrenamiento': tiempo_total,
            'parametros_modelo': modelo.count_params() if hasattr(modelo, 'count_params') else 0
        }
        
        return resultado
        
    except Exception as e:
        tiempo_total = time.time() - inicio_tiempo
        return {
            'experimento': experimento_id,
            'config_cnn': config_cnn['nombre'],
            'config_entrenamiento': config_entrenamiento['nombre'],
            'config_augmentation': config_augmentation['nombre'] if config_augmentation else 'ninguno',
            'error': str(e),
            'tiempo_entrenamiento': tiempo_total
        }

class GridSearchCNN:
    def __init__(self, max_workers=None):
        self.entrenador = EntrenadorCNN()
        self.tamaño_imagen = (200, 200)
        self.max_imagenes_por_clase = None
        self.max_workers = max_workers
        
    def configurar_experimento(self, tamaño_imagen=(200, 200), max_imagenes_por_clase=None):
        self.tamaño_imagen = tamaño_imagen
        self.max_imagenes_por_clase = max_imagenes_por_clase
        
        print(f"Configurando experimentos CNN para imágenes {tamaño_imagen[0]}x{tamaño_imagen[1]}")
        if max_imagenes_por_clase:
            print(f"Limitando a {max_imagenes_por_clase} imágenes por clase")
    
    def ejecutar_experimento_individual(self, config_cnn, config_entrenamiento, experimento_id):
        nombre_cnn = config_cnn['nombre']
        nombre_entrenamiento = config_entrenamiento['nombre']
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENTO CNN {experimento_id}")
        print(f"Arquitectura CNN: {nombre_cnn}")
        print(f"Entrenamiento: {nombre_entrenamiento}")
        print(f"Tamaño imagen: {self.tamaño_imagen}")
        print(f"{'='*60}")
        
        try:
            inicio = datetime.now()
            
            modelo, historia, nombre_modelo = self.entrenador.entrenar(
                nombre_cnn,
                nombre_entrenamiento,
                self.tamaño_imagen,
                self.max_imagenes_por_clase
            )
            
            fin = datetime.now()
            tiempo_entrenamiento = (fin - inicio).total_seconds()
            
            loss_final = min(historia.history['loss'])
            val_loss_final = min(historia.history.get('val_loss', [float('inf')]))
            accuracy_final = max(historia.history.get('accuracy', [0]))
            val_accuracy_final = max(historia.history.get('val_accuracy', [0]))
            
            epochs_ejecutadas = len(historia.history['loss'])
            epochs_configuradas = config_entrenamiento['epochs']
            convergio = epochs_ejecutadas < epochs_configuradas
            
            mejor_epoch = np.argmax(historia.history.get('val_accuracy', [0])) + 1
            
            resultado = {
                'experimento_id': experimento_id,
                'config_cnn': nombre_cnn,
                'config_entrenamiento': nombre_entrenamiento,
                'tamaño_imagen': f"{self.tamaño_imagen[0]}x{self.tamaño_imagen[1]}",
                'max_imagenes_por_clase': self.max_imagenes_por_clase,
                'nombre_modelo': nombre_modelo,
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'epochs_ejecutadas': epochs_ejecutadas,
                'epochs_configuradas': epochs_configuradas,
                'convergio': convergio,
                'loss_final': loss_final,
                'val_loss_final': val_loss_final,
                'accuracy_final': accuracy_final,
                'val_accuracy_final': val_accuracy_final,
                'mejor_epoch': mejor_epoch,
                'learning_rate': config_cnn['learning_rate'],
                'batch_size': config_cnn['batch_size'],
                'dropout_rate': config_cnn['dropout_rate'],
                'num_capas_conv': len(config_cnn['capas_conv']),
                'num_capas_dense': len(config_cnn['capas_dense']),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ Experimento CNN {experimento_id} completado:")
            print(f"  - Loss final: {loss_final:.4f}")
            print(f"  - Val Loss: {val_loss_final:.4f}")
            print(f"  - Accuracy: {accuracy_final:.4f}")
            print(f"  - Val Accuracy: {val_accuracy_final:.4f}")
            print(f"  - Tiempo: {tiempo_entrenamiento:.1f}s")
            print(f"  - Épocas: {epochs_ejecutadas}")
            
            return resultado
            
        except Exception as e:
            print(f"❌ Error en experimento CNN {experimento_id}: {e}")
            return {
                'experimento_id': experimento_id,
                'config_cnn': nombre_cnn,
                'config_entrenamiento': nombre_entrenamiento,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def ejecutar_grid_search_arquitecturas(self, nombres_arquitecturas, nombre_entrenamiento, tamaño_imagen=(200, 200), max_imagenes_por_clase=None):
        self.configurar_experimento(tamaño_imagen, max_imagenes_por_clase)
        
        configuraciones_cnn = []
        for nombre in nombres_arquitecturas:
            if nombre in CONFIGURACIONES_CNN:
                config = CONFIGURACIONES_CNN[nombre].copy()
                config['nombre'] = nombre
                configuraciones_cnn.append(config)
            else:
                print(f"⚠ Configuración CNN '{nombre}' no encontrada, omitiendo...")
        
        if not configuraciones_cnn:
            raise ValueError("No se encontraron configuraciones CNN válidas")
        
        config_entrenamiento = CONFIGURACIONES_ENTRENAMIENTO[nombre_entrenamiento].copy()
        config_entrenamiento['nombre'] = nombre_entrenamiento
        configuraciones_entrenamiento = [config_entrenamiento]
        
        return self.ejecutar_grid_search(configuraciones_cnn, configuraciones_entrenamiento)
    
    def ejecutar_grid_search(self, configuraciones_cnn, configuraciones_entrenamiento, configuraciones_augmentation=None):
        if configuraciones_augmentation is None:
            configuraciones_augmentation = [None]
            
        experimentos = []
        experimento_id = 1
        
        for config_cnn in configuraciones_cnn:
            for config_entrenamiento in configuraciones_entrenamiento:
                for config_augmentation in configuraciones_augmentation:
                    experimentos.append((
                        config_cnn, config_entrenamiento, config_augmentation,
                        self.tamaño_imagen, self.max_imagenes_por_clase, 
                        experimento_id
                    ))
                    experimento_id += 1
        
        total_experimentos = len(experimentos)
        print(f"🚀 Iniciando Grid Search CNN PARALELO: {total_experimentos} experimentos")
        print(f"Workers: {self.max_workers if self.max_workers else 'auto'}")
        
        archivo_resultados = self.generar_nombre_archivo_resultados()
        self._inicializar_csv(archivo_resultados)
        print(f"📊 Resultados se guardarán en: {archivo_resultados}")
        
        resultados = []
        inicio_total = time.time()
        
        if self.max_workers == 1:
            for i, exp in enumerate(experimentos):
                resultado = ejecutar_experimento_cnn_grid_paralelo(exp)
                resultados.append(resultado)
                self._actualizar_csv_incremental(archivo_resultados, resultado)
                self._mostrar_progreso(i + 1, total_experimentos, inicio_total)
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(ejecutar_experimento_cnn_grid_paralelo, exp): exp 
                          for exp in experimentos}
                
                completados = 0
                for future in as_completed(futures):
                    try:
                        resultado = future.result()
                        resultados.append(resultado)
                        self._actualizar_csv_incremental(archivo_resultados, resultado)
                        
                        if 'error' not in resultado:
                            print(f"✓ Experimento {resultado['experimento']}: {resultado['config_cnn']} + "
                                  f"{resultado['config_entrenamiento']} - "
                                  f"Val Acc: {resultado['val_accuracy_final']:.4f}")
                        else:
                            print(f"✗ Experimento {resultado['experimento']}: Error - {resultado['error']}")
                            
                    except Exception as e:
                        exp_args = futures[future]
                        print(f"✗ Experimento {exp_args[4]}: Error crítico - {e}")
                    
                    completados += 1
                    if completados % 3 == 0 or completados == total_experimentos:
                        self._mostrar_progreso(completados, total_experimentos, inicio_total)
        
        tiempo_total = time.time() - inicio_total
        exitosos = sum(1 for r in resultados if 'error' not in r)
        
        print(f"\n🎯 Grid Search paralelo completado en {tiempo_total:.1f} segundos")
        print(f"Experimentos exitosos: {exitosos}/{total_experimentos}")
        
        self._finalizar_csv(archivo_resultados, resultados, tiempo_total)
        return archivo_resultados
    
    def _mostrar_progreso(self, completados, total, tiempo_inicio):
        progreso = (completados / total) * 100
        tiempo_transcurrido = time.time() - tiempo_inicio
        
        if completados > 0:
            tiempo_por_experimento = tiempo_transcurrido / completados
            tiempo_restante = tiempo_por_experimento * (total - completados)
            print(f"Progreso: {completados}/{total} ({progreso:.1f}%) - "
                  f"Tiempo restante: {tiempo_restante/60:.1f}min")
    
    def _inicializar_csv(self, archivo_resultados):
        import os
        os.makedirs(os.path.dirname(archivo_resultados), exist_ok=True)
        
        columnas = [
            'experimento', 'config_cnn', 'config_entrenamiento', 'config_augmentation', 'nombre_modelo',
            'val_accuracy_final', 'val_loss_final', 'mejor_epoch', 
            'tiempo_entrenamiento', 'parametros_modelo', 'timestamp', 'error'
        ]
        
        df_vacio = pd.DataFrame(columns=columnas)
        df_vacio.to_csv(archivo_resultados, index=False)
        print(f"📝 CSV inicializado con columnas: {', '.join(columnas)}")
    
    def _actualizar_csv_incremental(self, archivo_resultados, resultado):
        import os
        from threading import Lock
        
        if not hasattr(self, '_csv_lock'):
            self._csv_lock = Lock()
        
        with self._csv_lock:
            try:
                resultado_copia = resultado.copy()
                resultado_copia['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if os.path.exists(archivo_resultados):
                    df_existente = pd.read_csv(archivo_resultados)
                    df_nuevo = pd.DataFrame([resultado_copia])
                    df_actualizado = pd.concat([df_existente, df_nuevo], ignore_index=True)
                else:
                    df_actualizado = pd.DataFrame([resultado_copia])
                
                df_actualizado.to_csv(archivo_resultados, index=False)
                
                if 'error' not in resultado:
                    print(f"💾 Resultado guardado: Exp {resultado['experimento']} - "
                          f"Val Acc: {resultado['val_accuracy_final']:.4f}")
                else:
                    print(f"💾 Error guardado: Exp {resultado['experimento']}")
                    
            except Exception as e:
                print(f"⚠️ Error al actualizar CSV: {e}")
    
    def _finalizar_csv(self, archivo_resultados, resultados, tiempo_total):
        try:
            df_final = pd.read_csv(archivo_resultados)
            
            if len(df_final) > 0:
                df_final = df_final.sort_values('val_accuracy_final', ascending=False)
                df_final.to_csv(archivo_resultados, index=False)
                
                resultados_validos = df_final[df_final['error'].isna()]
                
                if len(resultados_validos) > 0:
                    print(f"\n{'='*80}")
                    print(f"RESUMEN GRID SEARCH CNN - TOP 5 MODELOS")
                    print(f"{'='*80}")
                    
                    top_5 = resultados_validos.head()
                    for i, (_, row) in enumerate(top_5.iterrows(), 1):
                        print(f"{i}. {row['config_cnn']} + {row['config_entrenamiento']}")
                        print(f"   Val Accuracy: {row['val_accuracy_final']:.4f} | "
                              f"Val Loss: {row['val_loss_final']:.4f} | "
                              f"Tiempo: {row['tiempo_entrenamiento']:.1f}s")
                    
                    print(f"\n📊 Estadísticas finales:")
                    print(f"   - Experimentos totales: {len(df_final)}")
                    print(f"   - Experimentos exitosos: {len(resultados_validos)}")
                    print(f"   - Mejor accuracy: {resultados_validos['val_accuracy_final'].max():.4f}")
                    print(f"   - Tiempo total: {tiempo_total:.1f}s")
                    print(f"   - Tiempo promedio por experimento: {resultados_validos['tiempo_entrenamiento'].mean():.1f}s")
                
                print(f"   - Resultados finales en: {archivo_resultados}")
                
        except Exception as e:
            print(f"⚠️ Error al finalizar CSV: {e}")
    
    def ejecutar_grid_search_completo(self, tamaño_imagen=(200, 200), max_imagenes_por_clase=None):
        self.configurar_experimento(tamaño_imagen, max_imagenes_por_clase)
        
        configuraciones_cnn = []
        for nombre, config in CONFIGURACIONES_CNN.items():
            if config['tamaño_imagen'] == tamaño_imagen:
                config_copia = config.copy()
                config_copia['nombre'] = nombre
                configuraciones_cnn.append(config_copia)
        
        configuraciones_entrenamiento = []
        for nombre, config in CONFIGURACIONES_ENTRENAMIENTO.items():
            config_copia = config.copy()
            config_copia['nombre'] = nombre
            configuraciones_entrenamiento.append(config_copia)
        
        configuraciones_augmentation = []
        for nombre, config in CONFIGURACIONES_AUGMENTATION.items():
            config_copia = config.copy()
            config_copia['nombre'] = nombre
            configuraciones_augmentation.append(config_copia)
        
        return self.ejecutar_grid_search(configuraciones_cnn, configuraciones_entrenamiento, configuraciones_augmentation)
    
    def ejecutar_grid_search_optimizado(self, tamaño_imagen=(200, 200), max_imagenes_por_clase=None):
        """Grid search optimizado con las mejores configuraciones basado en resultados anteriores"""
        self.configurar_experimento(tamaño_imagen, max_imagenes_por_clase)
        
        # Mejores arquitecturas CNN basadas en resultados
        mejores_cnn = ['ligera_200x200', 'simple_200x200', 'profunda_200x200']
        configuraciones_cnn = []
        for nombre in mejores_cnn:
            if nombre in CONFIGURACIONES_CNN and CONFIGURACIONES_CNN[nombre]['tamaño_imagen'] == tamaño_imagen:
                config_copia = CONFIGURACIONES_CNN[nombre].copy()
                config_copia['nombre'] = nombre
                configuraciones_cnn.append(config_copia)
        
        # Mejores configuraciones de entrenamiento
        mejores_entrenamiento = ['normal', 'exhaustivo']
        configuraciones_entrenamiento = []
        for nombre in mejores_entrenamiento:
            if nombre in CONFIGURACIONES_ENTRENAMIENTO:
                config_copia = CONFIGURACIONES_ENTRENAMIENTO[nombre].copy()
                config_copia['nombre'] = nombre
                configuraciones_entrenamiento.append(config_copia)
        
        # Configuraciones de data augmentation seleccionadas
        augmentation_seleccionadas = ['ninguno', 'moderado', 'animales_optimizado']
        configuraciones_augmentation = []
        for nombre in augmentation_seleccionadas:
            if nombre in CONFIGURACIONES_AUGMENTATION:
                config_copia = CONFIGURACIONES_AUGMENTATION[nombre].copy()
                config_copia['nombre'] = nombre
                configuraciones_augmentation.append(config_copia)
        
        print(f"🎯 Grid Search OPTIMIZADO:")
        print(f"   - CNN: {len(configuraciones_cnn)} arquitecturas")
        print(f"   - Entrenamiento: {len(configuraciones_entrenamiento)} configuraciones")
        print(f"   - Data Augmentation: {len(configuraciones_augmentation)} configuraciones")
        print(f"   - Total experimentos: {len(configuraciones_cnn) * len(configuraciones_entrenamiento) * len(configuraciones_augmentation)}")
        
        return self.ejecutar_grid_search(configuraciones_cnn, configuraciones_entrenamiento, configuraciones_augmentation)
    
    def generar_nombre_archivo_resultados(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"tp4/resultados/grid_search_cnn_{timestamp}.csv"
    
    
    def analizar_resultados(self, archivo_csv):
        df = pd.read_csv(archivo_csv)
        
        if 'error' in df.columns:
            df_exitosos = df[df['error'].isna()]
        else:
            df_exitosos = df
        
        if len(df_exitosos) == 0:
            print("No hay experimentos exitosos para analizar")
            return
        
        print("=== ANÁLISIS DE RESULTADOS CNN ===\n")
        
        print("📊 MEJORES MODELOS POR MÉTRICA:")
        
        mejor_accuracy = df_exitosos.loc[df_exitosos['val_accuracy_final'].idxmax()]
        print(f"🏆 Mejor Accuracy: {mejor_accuracy['config_cnn']} (Accuracy: {mejor_accuracy['val_accuracy_final']:.4f})")
        
        mejor_loss = df_exitosos.loc[df_exitosos['val_loss_final'].idxmin()]
        print(f"🏆 Mejor Loss: {mejor_loss['config_cnn']} (Loss: {mejor_loss['val_loss_final']:.4f})")
        
        mas_rapido = df_exitosos.loc[df_exitosos['tiempo_entrenamiento'].idxmin()]
        print(f"⚡ Más Rápido: {mas_rapido['config_cnn']} (Tiempo: {mas_rapido['tiempo_entrenamiento']:.1f}s)")
        
        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"Experimentos totales: {len(df)}")
        print(f"Experimentos exitosos: {len(df_exitosos)}")
        print(f"Tasa de éxito: {len(df_exitosos)/len(df)*100:.1f}%")
        
        if 'convergio' in df_exitosos.columns:
            convergencias = df_exitosos['convergio'].sum()
            print(f"Modelos que convergieron: {convergencias}/{len(df_exitosos)} ({convergencias/len(df_exitosos)*100:.1f}%)")
        
        print(f"\n⏱️  TIEMPOS DE ENTRENAMIENTO:")
        tiempo_promedio = df_exitosos['tiempo_entrenamiento'].mean()
        tiempo_min = df_exitosos['tiempo_entrenamiento'].min()
        tiempo_max = df_exitosos['tiempo_entrenamiento'].max()
        print(f"Promedio: {tiempo_promedio:.1f}s")
        print(f"Mínimo: {tiempo_min:.1f}s")
        print(f"Máximo: {tiempo_max:.1f}s")
        
        print(f"\n🏗️  ANÁLISIS POR ARQUITECTURA:")
        analisis_arquitectura = df_exitosos.groupby('config_cnn').agg({
            'val_accuracy_final': ['mean', 'max'],
            'val_loss_final': ['mean', 'min'],
            'tiempo_entrenamiento': 'mean'
        }).round(4)
        
        for arquitectura in analisis_arquitectura.index:
            stats = analisis_arquitectura.loc[arquitectura]
            print(f"  {arquitectura}:")
            print(f"    Accuracy promedio: {stats[('val_accuracy_final', 'mean')]:.4f}")
            print(f"    Loss promedio: {stats[('val_loss_final', 'mean')]:.4f}")
            print(f"    Tiempo promedio: {stats[('tiempo_entrenamiento', 'mean')]:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Grid Search PARALELO para CNN de clasificación de animales')
    parser.add_argument('--arquitecturas', nargs='+', 
                       help='Lista de arquitecturas CNN a probar')
    parser.add_argument('--entrenamiento', type=str, default='normal',
                       help='Tipo de entrenamiento')
    parser.add_argument('--tamaño', type=str, default='200x200',
                       help='Tamaño de imagen (ej: 200x200)')
    parser.add_argument('--max-imagenes-por-clase', type=int, default=None,
                       help='Número máximo de imágenes por clase')
    parser.add_argument('--completo', action='store_true',
                       help='Ejecutar grid search completo')
    parser.add_argument('--optimizado', action='store_true',
                       help='Ejecutar grid search optimizado con mejores configuraciones + data augmentation')
    parser.add_argument('--analizar', type=str,
                       help='Analizar resultados de archivo CSV')
    parser.add_argument('--workers', type=int, default=None,
                       help='Número de workers para paralelización (auto si no se especifica)')
    
    args = parser.parse_args()
    
    if args.analizar:
        grid_search = GridSearchCNN()
        grid_search.analizar_resultados(args.analizar)
        return
    
    try:
        ancho, alto = map(int, args.tamaño.split('x'))
        tamaño_imagen = (ancho, alto)
    except:
        print(f"Error: Formato de tamaño inválido '{args.tamaño}'")
        return
    
    grid_search = GridSearchCNN(max_workers=args.workers)
    
    if args.completo:
        print("Ejecutando grid search CNN completo PARALELO...")
        resultados = grid_search.ejecutar_grid_search_completo(tamaño_imagen, args.max_imagenes_por_clase)
    elif args.optimizado:
        print("Ejecutando grid search CNN OPTIMIZADO con Data Augmentation...")
        resultados = grid_search.ejecutar_grid_search_optimizado(tamaño_imagen, args.max_imagenes_por_clase)
    elif args.arquitecturas:
        print(f"Ejecutando grid search CNN PARALELO para arquitecturas: {args.arquitecturas}")
        resultados = grid_search.ejecutar_grid_search_arquitecturas(
            args.arquitecturas, args.entrenamiento, tamaño_imagen, args.max_imagenes_por_clase
        )
    else:
        print("Debe especificar --completo, --optimizado o --arquitecturas")
        return
    
    if resultados:
        print(f"\n✓ Grid search CNN PARALELO completado. Resultados guardados en: {resultados}")


if __name__ == "__main__":
    main()