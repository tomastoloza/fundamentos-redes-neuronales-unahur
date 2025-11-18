import argparse
import numpy as np
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import classification_report, confusion_matrix
from tp3.comun.cargador_modelos import CargadorModelos
from .procesador_imagenes import ProcesadorImagenes
from .constructor_modelos_cnn import ConstructorModelosCNN
from .configuraciones import obtener_configuracion, obtener_configuracion_entrenamiento, listar_configuraciones


class EntrenadorCNN:
    def __init__(self):
        self.procesador = ProcesadorImagenes()
        self.constructor = ConstructorModelosCNN()
        self.cargador = CargadorModelos()
        self.datos_entrenamiento = None
        self.datos_validacion = None
        self.datos_prueba = None
        self.clases = None
        
    def cargar_datos(self, tamaño_imagen=(200, 200), max_imagenes_por_clase=None, mantener_aspecto=True):
        print("Cargando dataset multiclase de imágenes...")
        imagenes, labels = self.procesador.cargar_imagenes_multiclase(
            tamaño_imagen, max_imagenes_por_clase, mantener_aspecto
        )
        
        self.datos_entrenamiento, self.datos_validacion, self.datos_prueba = self.procesador.preparar_datos_entrenamiento()
        self.clases = self.procesador.obtener_nombres_clases()
        
        print(f"✓ Dataset cargado: {len(imagenes)} imágenes de {len(self.clases)} clases")
        self.procesador.mostrar_estadisticas()
        return self.datos_entrenamiento, self.datos_validacion, self.datos_prueba
    
    def crear_modelo(self, config_cnn):
        if self.datos_entrenamiento is None and self.clases is None:
            raise ValueError("Debe cargar los datos primero con cargar_datos()")
        
        modelo = self.constructor.construir_modelo(config_cnn)
        
        print(f"✓ Modelo CNN creado:")
        print(f"  - Arquitectura: {len(config_cnn['capas_conv'])} capas conv + {len(config_cnn['capas_dense'])} capas dense")
        print(f"  - Parámetros aproximados: {self.constructor.calcular_parametros_aproximados(config_cnn):,}")
        print(f"  - Clases de salida: {config_cnn['num_clases']}")
        
        return modelo
    
    def entrenar_con_augmentation(self, nombre_config_cnn, nombre_config_entrenamiento, 
                                 config_augmentation='animales_optimizado', tamaño_imagen=(200, 200), 
                                 max_imagenes_por_clase=None):
        """
        Entrenar modelo CNN con data augmentation.
        
        Args:
            nombre_config_cnn (str): Nombre de configuración CNN
            nombre_config_entrenamiento (str): Nombre de configuración de entrenamiento
            config_augmentation (str): Nombre de configuración de augmentation
            tamaño_imagen (tuple): Tamaño de las imágenes
            max_imagenes_por_clase (int): Máximo de imágenes por clase
            
        Returns:
            tuple: (modelo, historia, nombre_modelo)
        """
        config_cnn = obtener_configuracion(nombre_config_cnn)
        config_entrenamiento = obtener_configuracion_entrenamiento(nombre_config_entrenamiento)
        
        # Cargar datos con augmentation
        print("Cargando dataset multiclase de imágenes...")
        imagenes, labels = self.procesador.cargar_imagenes_multiclase(
            tamaño_imagen, max_imagenes_por_clase
        )
        
        datos_preparados = self.procesador.preparar_datos_con_augmentation(
            config_augmentation=config_augmentation
        )
        
        self.clases = self.procesador.obtener_nombres_clases()
        print(f"✓ Dataset cargado: {len(imagenes)} imágenes de {len(self.clases)} clases")
        self.procesador.mostrar_estadisticas()
        
        # Crear modelo
        modelo = self.crear_modelo(config_cnn)
        
        # Configurar callbacks
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config_entrenamiento['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config_entrenamiento['patience']//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\n{'='*60}")
        print(f"INICIANDO ENTRENAMIENTO CNN CON DATA AUGMENTATION")
        print(f"Configuración CNN: {nombre_config_cnn}")
        print(f"Entrenamiento: {nombre_config_entrenamiento}")
        print(f"Data Augmentation: {config_augmentation}")
        print(f"{'='*60}")
        
        inicio = datetime.now()
        
        # Entrenar con o sin generador según configuración
        if datos_preparados['use_generator']:
            # Entrenamiento con data augmentation
            X_train, y_train = datos_preparados['train_data']
            X_val, y_val = datos_preparados['validation_data']
            
            generador = datos_preparados['generador']
            train_generator = generador.crear_flujo_entrenamiento(
                X_train, y_train, 
                batch_size=config_cnn['batch_size']
            )
            
            steps_per_epoch = generador.calcular_steps_per_epoch(
                len(X_train), config_cnn['batch_size']
        )
        
        historia = modelo.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=config_entrenamiento['epochs'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Entrenamiento sin augmentation
            X_train, y_train = datos_preparados['train_data']
            X_val, y_val = datos_preparados['validation_data']
            
            historia = modelo.fit(
                X_train, y_train,
                batch_size=config_cnn['batch_size'],
            epochs=config_entrenamiento['epochs'],
                validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        fin = datetime.now()
        tiempo_entrenamiento = (fin - inicio).total_seconds()
        
        # Guardar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        aug_suffix = f"_{config_augmentation}" if config_augmentation != 'ninguno' else "_sin_aug"
        nombre_modelo = f"cnn_{nombre_config_cnn}_{nombre_config_entrenamiento}{aug_suffix}_{timestamp}"
        ruta_modelo = f"tp4/modelos/{nombre_modelo}.keras"
        
        modelo.save(ruta_modelo)
        print(f"✓ Modelo guardado: {ruta_modelo}")
        
        self.datos_prueba = datos_preparados['test_data']
        self.evaluar_modelo_final(modelo, nombre_modelo, tiempo_entrenamiento, historia)
        
        return modelo, historia, nombre_modelo
    
    def entrenar(self, nombre_config_cnn, nombre_config_entrenamiento, tamaño_imagen=(200, 200), 
                max_imagenes_por_clase=None, config_augmentation=None):
        
        if config_augmentation and config_augmentation != 'ninguno':
            return self.entrenar_con_augmentation(
                nombre_config_cnn, nombre_config_entrenamiento, 
                config_augmentation, tamaño_imagen, max_imagenes_por_clase
            )
        
        # Entrenamiento tradicional sin augmentation
        config_cnn = obtener_configuracion(nombre_config_cnn)
        config_entrenamiento = obtener_configuracion_entrenamiento(nombre_config_entrenamiento)
        
        if self.datos_entrenamiento is None:
            self.cargar_datos(tamaño_imagen, max_imagenes_por_clase)
        
        modelo = self.crear_modelo(config_cnn)
        
        X_train, y_train = self.datos_entrenamiento
        X_val, y_val = self.datos_validacion
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config_entrenamiento['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config_entrenamiento['patience']//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\n{'='*60}")
        print(f"INICIANDO ENTRENAMIENTO CNN")
        print(f"Configuración: {nombre_config_cnn}")
        print(f"Entrenamiento: {nombre_config_entrenamiento}")
        print(f"{'='*60}")
        
        inicio = datetime.now()
        
        historia = modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config_entrenamiento['epochs'],
            batch_size=config_cnn['batch_size'],
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        fin = datetime.now()
        tiempo_entrenamiento = (fin - inicio).total_seconds()
        
        nombre_modelo = self.generar_nombre_modelo(nombre_config_cnn, nombre_config_entrenamiento)
        ruta_modelo = f"tp4/modelos/{nombre_modelo}"
        import os
        os.makedirs("tp4/modelos", exist_ok=True)
        modelo.save(ruta_modelo)
        print(f"✓ Modelo guardado: {ruta_modelo}")
        
        self.evaluar_modelo_final(modelo, nombre_modelo, tiempo_entrenamiento, historia)
        
        return modelo, historia, nombre_modelo
    
    def evaluar_modelo_final(self, modelo, nombre_modelo, tiempo_entrenamiento, historia):
        X_test, y_test = self.datos_prueba
        
        print(f"\n{'='*60}")
        print(f"EVALUACIÓN FINAL - {nombre_modelo}")
        print(f"{'='*60}")
        
        loss_test, accuracy_test, precision_test, recall_test = modelo.evaluate(X_test, y_test, verbose=0)
        
        y_pred = modelo.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        print(f"Métricas en conjunto de prueba:")
        print(f"  - Loss: {loss_test:.4f}")
        print(f"  - Accuracy: {accuracy_test:.4f}")
        print(f"  - Precision: {precision_test:.4f}")
        print(f"  - Recall: {recall_test:.4f}")
        print(f"  - Tiempo de entrenamiento: {tiempo_entrenamiento:.1f}s")
        
        print(f"\nReporte de clasificación:")
        print(classification_report(y_test_classes, y_pred_classes, target_names=self.clases))
        
        print(f"\nMatriz de confusión:")
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        print(cm)
        
        mejor_epoch = np.argmax(historia.history['val_accuracy'])
        mejor_val_acc = max(historia.history['val_accuracy'])
        
        print(f"\nMejor época: {mejor_epoch + 1}")
        print(f"Mejor accuracy de validación: {mejor_val_acc:.4f}")
        
        return {
            'loss_test': loss_test,
            'accuracy_test': accuracy_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'tiempo_entrenamiento': tiempo_entrenamiento,
            'mejor_epoch': mejor_epoch,
            'mejor_val_accuracy': mejor_val_acc,
            'y_pred_classes': y_pred_classes,
            'y_test_classes': y_test_classes
        }
    
    def generar_nombre_modelo(self, config_cnn, config_entrenamiento):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cnn_{config_cnn}_{config_entrenamiento}_{timestamp}.keras"
    
    def mostrar_resumen_arquitectura(self, nombre_config):
        config = obtener_configuracion(nombre_config)
        resumen = self.constructor.obtener_resumen_arquitectura(config)
        
        print(f"\nArquitectura CNN - {nombre_config}:")
        for i, linea in enumerate(resumen):
            print(f"  {i+1:2d}. {linea}")


def entrenar_modelo():
    parser = argparse.ArgumentParser(description='Entrenador CNN para clasificación de animales')
    parser.add_argument('--config', help='Configuración de CNN a usar')
    parser.add_argument('--entrenamiento', default='normal', help='Configuración de entrenamiento')
    parser.add_argument('--augmentation', help='Configuración de data augmentation')
    parser.add_argument('--tamaño', default='200x200', help='Tamaño de imagen')
    parser.add_argument('--max-imagenes', type=int, help='Máximo de imágenes por clase')
    parser.add_argument('--evaluar', action='store_true', help='Evaluar modelo después del entrenamiento')
    parser.add_argument('--listar', action='store_true', help='Listar configuraciones disponibles')
    parser.add_argument('--paralelo', action='store_true', help='Ejecutar múltiples configuraciones en paralelo')
    parser.add_argument('--workers', type=int, help='Número de workers para ejecución paralela')
    parser.add_argument('--configs-cnn', nargs='+', help='Múltiples configuraciones CNN para paralelo')
    parser.add_argument('--configs-entrenamiento', nargs='+', help='Múltiples configuraciones entrenamiento para paralelo')
    parser.add_argument('--configs-augmentation', nargs='+', help='Múltiples configuraciones augmentation para paralelo')
    
    args = parser.parse_args()
    
    if args.listar:
        listar_configuraciones()
        return
    
    tamaño_partes = args.tamaño.split('x')
    tamaño_imagen = (int(tamaño_partes[0]), int(tamaño_partes[1]))
    
    if args.paralelo:
        if not args.configs_cnn:
            print("Error: --configs-cnn requerido para modo paralelo")
            return
        
        configs_entrenamiento = args.configs_entrenamiento or [args.entrenamiento]
        configs_augmentation = args.configs_augmentation or [args.augmentation] if args.augmentation else ['ninguno']
        
        entrenador_paralelo = EntrenadorCNNParalelo(max_workers=args.workers)
        
        try:
            resultados = entrenador_paralelo.entrenar_multiples_configuraciones(
                args.configs_cnn,
                configs_entrenamiento,
                configs_augmentation,
                tamaño_imagen,
                args.max_imagenes
            )
            
            exitosos = [r for r in resultados if r['exito']]
            if exitosos:
                mejor = max(exitosos, key=lambda x: x['mejor_val_accuracy'])
                print(f"\n🏆 Mejor resultado: {mejor['config_cnn']} + {mejor['config_entrenamiento']} + "
                      f"{mejor['config_augmentation']} - Val Acc: {mejor['mejor_val_accuracy']:.4f}")
            
        except KeyboardInterrupt:
            print("\n⚠ Entrenamiento paralelo interrumpido")
        except Exception as e:
            print(f"\n❌ Error en entrenamiento paralelo: {e}")
    else:
        if not args.config:
            print("Error: --config requerido para modo individual")
        return
    
        entrenador = EntrenadorCNN()
        
        if args.evaluar:
            entrenador.mostrar_resumen_arquitectura(args.config)
        
        try:
            modelo, historia, nombre = entrenador.entrenar(
                args.config, 
                args.entrenamiento, 
                tamaño_imagen, 
                args.max_imagenes,
                args.augmentation
            )
            print(f"\n✓ Entrenamiento completado: {nombre}")
            
        except KeyboardInterrupt:
            print("\n⚠ Entrenamiento interrumpido por el usuario")
        except Exception as e:
            print(f"\n❌ Error durante el entrenamiento: {e}")


def ejecutar_experimento_cnn_paralelo(args):
    nombre_config_cnn, nombre_config_entrenamiento, config_augmentation, tamaño_imagen, max_imagenes_por_clase, experimento_id = args
    
    inicio_tiempo = time.time()
    
    try:
        entrenador = EntrenadorCNN()
        modelo, historia, nombre_modelo = entrenador.entrenar(
            nombre_config_cnn, 
            nombre_config_entrenamiento, 
            tamaño_imagen,
            max_imagenes_por_clase,
            config_augmentation
        )
        
        tiempo_total = time.time() - inicio_tiempo
        
        mejor_val_acc = max(historia.history.get('val_accuracy', [0]))
        mejor_epoch = historia.history['val_accuracy'].index(mejor_val_acc) + 1 if 'val_accuracy' in historia.history else 1
        
        resultado = {
            'experimento': experimento_id,
            'config_cnn': nombre_config_cnn,
            'config_entrenamiento': nombre_config_entrenamiento,
            'config_augmentation': config_augmentation or 'ninguno',
            'nombre_modelo': nombre_modelo,
            'mejor_val_accuracy': mejor_val_acc,
            'mejor_epoch': mejor_epoch,
            'tiempo_entrenamiento': tiempo_total,
            'exito': True
        }
        
        return resultado
        
    except Exception as e:
        tiempo_total = time.time() - inicio_tiempo
        return {
            'experimento': experimento_id,
            'config_cnn': nombre_config_cnn,
            'config_entrenamiento': nombre_config_entrenamiento,
            'config_augmentation': config_augmentation or 'ninguno',
            'error': str(e),
            'tiempo_entrenamiento': tiempo_total,
            'exito': False
        }

class EntrenadorCNNParalelo:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
    
    def entrenar_multiples_configuraciones(self, configuraciones_cnn, configuraciones_entrenamiento, 
                                         configuraciones_augmentation=None, tamaño_imagen=(200, 200), 
                                         max_imagenes_por_clase=None):
        
        if configuraciones_augmentation is None:
            configuraciones_augmentation = ['ninguno']
        
        experimentos = []
        experimento_id = 1
        
        for config_cnn in configuraciones_cnn:
            for config_entrenamiento in configuraciones_entrenamiento:
                for config_aug in configuraciones_augmentation:
                    experimentos.append((
                        config_cnn, config_entrenamiento, config_aug, 
                        tamaño_imagen, max_imagenes_por_clase, experimento_id
                    ))
                    experimento_id += 1
        
        total_experimentos = len(experimentos)
        print(f"Iniciando entrenamiento paralelo con {total_experimentos} experimentos...")
        
        resultados = []
        inicio_total = time.time()
        
        if self.max_workers == 1:
            for i, exp in enumerate(experimentos):
                resultado = ejecutar_experimento_cnn_paralelo(exp)
                resultados.append(resultado)
                self._mostrar_progreso(i + 1, total_experimentos, inicio_total)
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(ejecutar_experimento_cnn_paralelo, exp): exp 
                          for exp in experimentos}
                
                completados = 0
                for future in as_completed(futures):
                    try:
                        resultado = future.result()
                        resultados.append(resultado)
                        
                        if resultado['exito']:
                            print(f"✓ Experimento {resultado['experimento']}: {resultado['config_cnn']} + "
                                  f"{resultado['config_entrenamiento']} + {resultado['config_augmentation']} - "
                                  f"Val Acc: {resultado['mejor_val_accuracy']:.4f}")
                        else:
                            print(f"✗ Experimento {resultado['experimento']}: Error - {resultado['error']}")
                            
                    except Exception as e:
                        exp_args = futures[future]
                        print(f"✗ Experimento {exp_args[5]}: Error crítico - {e}")
                    
                    completados += 1
                    if completados % 5 == 0 or completados == total_experimentos:
                        self._mostrar_progreso(completados, total_experimentos, inicio_total)
        
        tiempo_total = time.time() - inicio_total
        exitosos = sum(1 for r in resultados if r['exito'])
        
        print(f"\nEntrenamiento paralelo completado en {tiempo_total:.1f} segundos")
        print(f"Experimentos exitosos: {exitosos}/{total_experimentos}")
        
        return resultados
    
    def _mostrar_progreso(self, completados, total, tiempo_inicio):
        progreso = (completados / total) * 100
        tiempo_transcurrido = time.time() - tiempo_inicio
        
        if completados > 0:
            tiempo_por_experimento = tiempo_transcurrido / completados
            tiempo_restante = tiempo_por_experimento * (total - completados)
            print(f"Progreso: {completados}/{total} ({progreso:.1f}%) - "
                  f"Tiempo restante: {tiempo_restante/60:.1f}min")

if __name__ == "__main__":
    entrenar_modelo()