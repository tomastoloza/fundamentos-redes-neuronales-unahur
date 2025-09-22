"""
Implementación de clasificación de 10 clases de dígitos utilizando TensorFlow/Keras.
Incluye evaluación con ruido y comparación con la implementación personalizada del TP2.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import os
import sys

# Configurar path para acceder a módulos compartidos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tp2.src.cargador_datos_digitos import CargadorDatosDigitos

# Configurar TensorFlow para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

class Clasificacion10ClasesKeras:
    """
    Implementación de clasificación de 10 clases usando TensorFlow/Keras.
    Arquitectura: [35, 20, 15, 10] - 35 entradas, 2 capas ocultas, 10 salidas
    """
    
    def __init__(self, arquitectura=[35, 20, 15, 10], learning_rate=0.01):
        """
        Inicializa el modelo de clasificación de 10 clases.
        
        Args:
            arquitectura: Lista con el número de neuronas por capa
            learning_rate: Tasa de aprendizaje
        """
        self.arquitectura = arquitectura
        self.learning_rate = learning_rate
        self.modelo = None
        self.historia_entrenamiento = None
        self.cargador_datos = CargadorDatosDigitos()
        
        # Cargar y preparar datos
        self._cargar_datos()
        self._crear_modelo()
    
    def _cargar_datos(self):
        """Carga y prepara los datos de dígitos para clasificación multiclase."""
        # Cargar datos desde archivo
        self.cargador_datos.cargar_datos_tp2()
        
        # División estándar: 0-6 para entrenamiento, 7-9 para prueba
        X_train, y_train, X_test, y_test = self.cargador_datos.crear_division_entrenamiento_prueba_estandar()
        
        # Convertir a arrays numpy con tipo float32
        self.X_train = X_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        
        # Convertir etiquetas a one-hot encoding
        self.y_train = keras.utils.to_categorical(y_train, num_classes=10)
        self.y_test = keras.utils.to_categorical(y_test, num_classes=10)
        
        # Guardar etiquetas originales para análisis
        self.y_train_original = y_train
        self.y_test_original = y_test
        
        # Información de los conjuntos
        self.digitos_train = [0, 1, 2, 3, 4, 5, 6]
        self.digitos_test = [7, 8, 9]
    
    def _crear_modelo(self):
        """Crea el modelo de red neuronal con Keras."""
        modelo_layers = []
        
        # Capa de entrada y primera capa oculta
        modelo_layers.append(
            layers.Dense(self.arquitectura[1], 
                        activation='sigmoid', 
                        input_shape=(self.arquitectura[0],),
                        name='capa_oculta_1')
        )
        
        # Capas ocultas adicionales
        for i in range(2, len(self.arquitectura) - 1):
            modelo_layers.append(
                layers.Dense(self.arquitectura[i], 
                            activation='sigmoid',
                            name=f'capa_oculta_{i}')
            )
        
        # Capa de salida (10 clases con softmax)
        modelo_layers.append(
            layers.Dense(self.arquitectura[-1], 
                        activation='softmax',
                        name='capa_salida')
        )
        
        self.modelo = keras.Sequential(modelo_layers)
        
        # Compilar el modelo
        self.modelo.compile(
            optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def entrenar(self, max_epochs=1000, tolerancia_error=0.01, verbose=True):
        """
        Entrena el modelo de clasificación de 10 clases.
        
        Args:
            max_epochs: Número máximo de épocas
            tolerancia_error: Error mínimo para considerar convergencia
            verbose: Si mostrar información durante el entrenamiento
            
        Returns:
            Tuple: (convergio, epoca_final, error_final, tiempo_entrenamiento)
        """
        if verbose:
            print("=== ENTRENAMIENTO CLASIFICACIÓN 10 CLASES CON KERAS ===")
            print(f"Arquitectura: {self.arquitectura}")
            print(f"Tasa de aprendizaje: {self.learning_rate}")
            print(f"Datos entrenamiento: {self.X_train.shape}")
            print(f"Datos prueba: {self.X_test.shape}")
            print(f"Dígitos entrenamiento: {self.digitos_train}")
            print(f"Dígitos prueba: {self.digitos_test}")
            print()
        
        tiempo_inicio = time.time()
        
        # Callback personalizado para detener cuando se alcance la tolerancia
        class EarlyStoppingCallback(keras.callbacks.Callback):
            def __init__(self, tolerancia):
                self.tolerancia = tolerancia
                self.convergio = False
                self.epoca_convergencia = 0
                
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                if loss <= self.tolerancia:
                    self.convergio = True
                    self.epoca_convergencia = epoch + 1
                    self.model.stop_training = True
        
        callback_convergencia = EarlyStoppingCallback(tolerancia_error)
        
        # Entrenar el modelo
        self.historia_entrenamiento = self.modelo.fit(
            self.X_train, self.y_train,
            epochs=max_epochs,
            batch_size=len(self.X_train),  # Batch completo
            verbose=0,  # Silencioso para control personalizado
            callbacks=[callback_convergencia]
        )
        
        tiempo_entrenamiento = time.time() - tiempo_inicio
        
        # Obtener métricas finales
        error_final = self.historia_entrenamiento.history['loss'][-1]
        epoca_final = len(self.historia_entrenamiento.history['loss'])
        convergio = callback_convergencia.convergio
        
        if verbose:
            print(f"Entrenamiento completado:")
            print(f"  - Convergió: {'Sí' if convergio else 'No'}")
            print(f"  - Épocas: {epoca_final}")
            print(f"  - Error final: {error_final:.6f}")
            print(f"  - Tiempo: {tiempo_entrenamiento:.2f} segundos")
            print()
        
        return convergio, epoca_final, error_final, tiempo_entrenamiento
    
    def evaluar(self, verbose=True):
        """
        Evalúa el rendimiento del modelo en entrenamiento y prueba.
        
        Args:
            verbose: Si mostrar información detallada
            
        Returns:
            Dict con métricas de evaluación
        """
        # Predicciones en entrenamiento
        pred_train = self.modelo.predict(self.X_train, verbose=0)
        pred_train_classes = np.argmax(pred_train, axis=1)
        
        # Predicciones en prueba
        pred_test = self.modelo.predict(self.X_test, verbose=0)
        pred_test_classes = np.argmax(pred_test, axis=1)
        
        # Calcular métricas
        accuracy_train = np.mean(pred_train_classes == self.y_train_original)
        accuracy_test = np.mean(pred_test_classes == self.y_test_original)
        
        if verbose:
            print("=== EVALUACIÓN CLASIFICACIÓN 10 CLASES KERAS ===")
            print("\n--- CONJUNTO DE ENTRENAMIENTO ---")
            for i, digito in enumerate(self.digitos_train):
                esperada = self.y_train_original[i]
                prediccion = pred_train_classes[i]
                confianza = np.max(pred_train[i])
                correcto = "✓" if prediccion == esperada else "✗"
                
                print(f"Dígito {digito} | Pred: {prediccion} | Conf: {confianza:.4f} | {correcto}")
            
            print(f"\nPrecisión entrenamiento: {accuracy_train:.1%}")
            
            print("\n--- CONJUNTO DE PRUEBA ---")
            for i, digito in enumerate(self.digitos_test):
                esperada = self.y_test_original[i]
                prediccion = pred_test_classes[i]
                confianza = np.max(pred_test[i])
                correcto = "✓" if prediccion == esperada else "✗"
                
                print(f"Dígito {digito} | Pred: {prediccion} | Conf: {confianza:.4f} | {correcto}")
            
            print(f"\nPrecisión prueba: {accuracy_test:.1%}")
            
            # Análisis de generalización
            diferencia_accuracy = accuracy_train - accuracy_test
            print(f"\n--- ANÁLISIS DE GENERALIZACIÓN ---")
            print(f"Diferencia de precisión: {diferencia_accuracy:.1%}")
            
            if diferencia_accuracy > 0.5:
                print("⚠️  SOBREAJUSTE SEVERO detectado")
            elif diferencia_accuracy > 0.2:
                print("⚠️  Posible sobreajuste")
            else:
                print("✓ Buena generalización")
            print()
        
        return {
            'accuracy_train': accuracy_train,
            'accuracy_test': accuracy_test,
            'pred_train': pred_train,
            'pred_test': pred_test,
            'pred_train_classes': pred_train_classes,
            'pred_test_classes': pred_test_classes,
            'diferencia_accuracy': accuracy_train - accuracy_test
        }
    
    def evaluar_con_ruido(self, probabilidad_ruido=0.02, verbose=True):
        """
        Evalúa la robustez del modelo ante ruido en los datos.
        
        Args:
            probabilidad_ruido: Probabilidad de intercambio de bits
            verbose: Si mostrar información detallada
            
        Returns:
            Dict con métricas de evaluación con ruido
        """
        if verbose:
            print("=== EVALUACIÓN CON RUIDO ===")
            print(f"Probabilidad de ruido: {probabilidad_ruido}")
            print()
        
        # Generar datos con ruido
        X_train_ruido = self.cargador_datos.generar_datos_con_ruido(
            self.X_train, probabilidad_ruido
        )
        X_test_ruido = self.cargador_datos.generar_datos_con_ruido(
            self.X_test, probabilidad_ruido
        )
        
        # Predicciones con ruido
        pred_train_ruido = self.modelo.predict(X_train_ruido, verbose=0)
        pred_train_ruido_classes = np.argmax(pred_train_ruido, axis=1)
        
        pred_test_ruido = self.modelo.predict(X_test_ruido, verbose=0)
        pred_test_ruido_classes = np.argmax(pred_test_ruido, axis=1)
        
        # Calcular métricas con ruido
        accuracy_train_ruido = np.mean(pred_train_ruido_classes == self.y_train_original)
        accuracy_test_ruido = np.mean(pred_test_ruido_classes == self.y_test_original)
        
        # Calcular degradación
        # Primero necesitamos las métricas sin ruido
        pred_train_limpio = self.modelo.predict(self.X_train, verbose=0)
        pred_train_limpio_classes = np.argmax(pred_train_limpio, axis=1)
        accuracy_train_limpio = np.mean(pred_train_limpio_classes == self.y_train_original)
        
        pred_test_limpio = self.modelo.predict(self.X_test, verbose=0)
        pred_test_limpio_classes = np.argmax(pred_test_limpio, axis=1)
        accuracy_test_limpio = np.mean(pred_test_limpio_classes == self.y_test_original)
        
        degradacion_train = accuracy_train_limpio - accuracy_train_ruido
        degradacion_test = accuracy_test_limpio - accuracy_test_ruido
        
        if verbose:
            print("--- COMPARACIÓN LIMPIO vs RUIDO ---")
            print(f"Entrenamiento limpio: {accuracy_train_limpio:.1%}")
            print(f"Entrenamiento con ruido: {accuracy_train_ruido:.1%}")
            print(f"Degradación entrenamiento: {degradacion_train:.1%}")
            print()
            print(f"Prueba limpio: {accuracy_test_limpio:.1%}")
            print(f"Prueba con ruido: {accuracy_test_ruido:.1%}")
            print(f"Degradación prueba: {degradacion_test:.1%}")
            print()
            
            # Análisis de robustez
            print("--- ANÁLISIS DE ROBUSTEZ ---")
            if degradacion_train < 0.05:
                print("✓ Excelente robustez en datos de entrenamiento")
            elif degradacion_train < 0.15:
                print("⚠️  Robustez moderada en datos de entrenamiento")
            else:
                print("❌ Baja robustez en datos de entrenamiento")
            
            if degradacion_test < 0.05:
                print("✓ Excelente robustez en datos de prueba")
            elif degradacion_test < 0.15:
                print("⚠️  Robustez moderada en datos de prueba")
            else:
                print("❌ Baja robustez en datos de prueba")
            print()
            
            # Mostrar ejemplos con ruido
            print("--- EJEMPLOS CON RUIDO ---")
            for i in range(min(3, len(self.digitos_train))):
                digito = self.digitos_train[i]
                patron_limpio = self.X_train[i]
                patron_ruido = X_train_ruido[i]
                
                # Contar bits diferentes
                bits_diferentes = np.sum(patron_limpio != patron_ruido)
                
                print(f"Dígito {digito}: {bits_diferentes} bits modificados de 35 totales")
                
                # Mostrar visualización si es posible
                if hasattr(self.cargador_datos, 'visualizar_patron'):
                    print("Original:")
                    print(self.cargador_datos.visualizar_patron(patron_limpio))
                    print("Con ruido:")
                    print(self.cargador_datos.visualizar_patron(patron_ruido))
                    print()
        
        return {
            'accuracy_train_limpio': accuracy_train_limpio,
            'accuracy_train_ruido': accuracy_train_ruido,
            'accuracy_test_limpio': accuracy_test_limpio,
            'accuracy_test_ruido': accuracy_test_ruido,
            'degradacion_train': degradacion_train,
            'degradacion_test': degradacion_test,
            'X_train_ruido': X_train_ruido,
            'X_test_ruido': X_test_ruido
        }
    
    def mostrar_arquitectura(self):
        """Muestra la arquitectura del modelo."""
        print("=== ARQUITECTURA DEL MODELO KERAS ===")
        self.modelo.summary()
        print()


def ejecutar_experimento_clasificacion_10_clases_keras():
    """Ejecuta el experimento completo de clasificación de 10 clases con Keras."""
    print("🔥 EXPERIMENTO CLASIFICACIÓN 10 CLASES CON TENSORFLOW/KERAS")
    print("=" * 65)
    
    # Crear y entrenar el modelo
    modelo = Clasificacion10ClasesKeras(
        arquitectura=[35, 20, 15, 10], 
        learning_rate=0.01
    )
    
    # Mostrar arquitectura
    modelo.mostrar_arquitectura()
    
    # Entrenar
    convergio, epocas, error_final, tiempo = modelo.entrenar(
        max_epochs=1000, 
        tolerancia_error=0.01,
        verbose=True
    )
    
    # Evaluar sin ruido
    metricas = modelo.evaluar(verbose=True)
    
    # Evaluar con ruido
    metricas_ruido = modelo.evaluar_con_ruido(
        probabilidad_ruido=0.02,
        verbose=True
    )
    
    return {
        'convergio': convergio,
        'epocas': epocas,
        'error_final': error_final,
        'tiempo_entrenamiento': tiempo,
        'accuracy_train': metricas['accuracy_train'],
        'accuracy_test': metricas['accuracy_test'],
        'diferencia_accuracy': metricas['diferencia_accuracy'],
        'accuracy_train_ruido': metricas_ruido['accuracy_train_ruido'],
        'accuracy_test_ruido': metricas_ruido['accuracy_test_ruido'],
        'degradacion_train': metricas_ruido['degradacion_train'],
        'degradacion_test': metricas_ruido['degradacion_test']
    }


if __name__ == "__main__":
    # Ejecutar experimento
    resultados = ejecutar_experimento_clasificacion_10_clases_keras()
    
    print("🎯 RESUMEN DE RESULTADOS:")
    print(f"  - Convergencia: {'✓ Sí' if resultados['convergio'] else '✗ No'}")
    print(f"  - Épocas: {resultados['epocas']}")
    print(f"  - Error final: {resultados['error_final']:.6f}")
    print(f"  - Precisión entrenamiento: {resultados['accuracy_train']:.1%}")
    print(f"  - Precisión prueba: {resultados['accuracy_test']:.1%}")
    print(f"  - Diferencia (sobreajuste): {resultados['diferencia_accuracy']:.1%}")
    print(f"  - Degradación por ruido (train): {resultados['degradacion_train']:.1%}")
    print(f"  - Degradación por ruido (test): {resultados['degradacion_test']:.1%}")
    print(f"  - Tiempo: {resultados['tiempo_entrenamiento']:.2f}s")
