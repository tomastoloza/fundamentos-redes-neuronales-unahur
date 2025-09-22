"""
Implementación de discriminación de números pares utilizando TensorFlow/Keras.
Comparación directa con la implementación personalizada del TP2.
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

class DiscriminacionParesKeras:
    """
    Implementación de discriminación de números pares usando TensorFlow/Keras.
    Arquitectura configurable para comparar diferentes diseños.
    """
    
    def __init__(self, arquitectura=[35, 20, 1], learning_rate=0.01):
        """
        Inicializa el modelo de discriminación de pares.
        
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
        """Carga y prepara los datos de dígitos para clasificación binaria."""
        # Cargar datos desde archivo
        self.cargador_datos.cargar_datos_tp2()
        
        # Definir dígitos pares e impares para entrenamiento
        digitos_pares_train = [0, 2, 4, 6]
        digitos_impares_train = [1, 3]
        
        # Definir dígitos para prueba
        digitos_pares_test = [8]
        digitos_impares_test = [5, 7, 9]
        
        # Preparar datos de entrenamiento
        entradas_train = []
        salidas_train = []
        
        for digito in digitos_pares_train:
            patron = self.cargador_datos.obtener_patron_digito(digito)
            if patron is not None:
                entradas_train.append(patron)
                salidas_train.append(1)  # Par = 1
        
        for digito in digitos_impares_train:
            patron = self.cargador_datos.obtener_patron_digito(digito)
            if patron is not None:
                entradas_train.append(patron)
                salidas_train.append(0)  # Impar = 0
        
        # Preparar datos de prueba
        entradas_test = []
        salidas_test = []
        
        for digito in digitos_pares_test:
            patron = self.cargador_datos.obtener_patron_digito(digito)
            if patron is not None:
                entradas_test.append(patron)
                salidas_test.append(1)  # Par = 1
        
        for digito in digitos_impares_test:
            patron = self.cargador_datos.obtener_patron_digito(digito)
            if patron is not None:
                entradas_test.append(patron)
                salidas_test.append(0)  # Impar = 0
        
        # Convertir a arrays numpy
        self.X_train = np.array(entradas_train, dtype=np.float32)
        self.y_train = np.array(salidas_train, dtype=np.float32).reshape(-1, 1)
        self.X_test = np.array(entradas_test, dtype=np.float32)
        self.y_test = np.array(salidas_test, dtype=np.float32).reshape(-1, 1)
        
        # Información de los conjuntos
        self.digitos_train_pares = digitos_pares_train
        self.digitos_train_impares = digitos_impares_train
        self.digitos_test_pares = digitos_pares_test
        self.digitos_test_impares = digitos_impares_test
    
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
        
        # Capa de salida
        modelo_layers.append(
            layers.Dense(self.arquitectura[-1], 
                        activation='sigmoid',
                        name='capa_salida')
        )
        
        self.modelo = keras.Sequential(modelo_layers)
        
        # Compilar el modelo
        self.modelo.compile(
            optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def entrenar(self, max_epochs=1000, tolerancia_error=0.01, verbose=True):
        """
        Entrena el modelo de discriminación de pares.
        
        Args:
            max_epochs: Número máximo de épocas
            tolerancia_error: Error mínimo para considerar convergencia
            verbose: Si mostrar información durante el entrenamiento
            
        Returns:
            Tuple: (convergio, epoca_final, error_final, tiempo_entrenamiento)
        """
        if verbose:
            print("=== ENTRENAMIENTO DISCRIMINACIÓN PARES CON KERAS ===")
            print(f"Arquitectura: {self.arquitectura}")
            print(f"Tasa de aprendizaje: {self.learning_rate}")
            print(f"Datos entrenamiento: {self.X_train.shape}")
            print(f"Datos prueba: {self.X_test.shape}")
            print(f"Dígitos pares entrenamiento: {self.digitos_train_pares}")
            print(f"Dígitos impares entrenamiento: {self.digitos_train_impares}")
            print(f"Dígitos pares prueba: {self.digitos_test_pares}")
            print(f"Dígitos impares prueba: {self.digitos_test_impares}")
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
        pred_train_bin = (pred_train > 0.5).astype(int)
        
        # Predicciones en prueba
        pred_test = self.modelo.predict(self.X_test, verbose=0)
        pred_test_bin = (pred_test > 0.5).astype(int)
        
        # Calcular métricas
        accuracy_train = np.mean(pred_train_bin.flatten() == self.y_train.flatten())
        accuracy_test = np.mean(pred_test_bin.flatten() == self.y_test.flatten())
        
        mse_train = np.mean((pred_train - self.y_train) ** 2)
        mse_test = np.mean((pred_test - self.y_test) ** 2)
        
        if verbose:
            print("=== EVALUACIÓN DISCRIMINACIÓN PARES KERAS ===")
            print("\n--- CONJUNTO DE ENTRENAMIENTO ---")
            digitos_train = self.digitos_train_pares + self.digitos_train_impares
            for i, digito in enumerate(digitos_train):
                esperada = int(self.y_train[i, 0])
                prediccion = pred_train[i, 0]
                binaria = int(pred_train_bin[i, 0])
                correcto = "✓" if binaria == esperada else "✗"
                tipo = "Par" if esperada == 1 else "Impar"
                
                print(f"Dígito {digito} ({tipo}) | Pred: {prediccion:.4f} | Bin: {binaria} | {correcto}")
            
            print(f"\nPrecisión entrenamiento: {accuracy_train:.1%}")
            print(f"MSE entrenamiento: {mse_train:.6f}")
            
            print("\n--- CONJUNTO DE PRUEBA ---")
            digitos_test = self.digitos_test_pares + self.digitos_test_impares
            for i, digito in enumerate(digitos_test):
                esperada = int(self.y_test[i, 0])
                prediccion = pred_test[i, 0]
                binaria = int(pred_test_bin[i, 0])
                correcto = "✓" if binaria == esperada else "✗"
                tipo = "Par" if esperada == 1 else "Impar"
                
                print(f"Dígito {digito} ({tipo}) | Pred: {prediccion:.4f} | Bin: {binaria} | {correcto}")
            
            print(f"\nPrecisión prueba: {accuracy_test:.1%}")
            print(f"MSE prueba: {mse_test:.6f}")
            
            # Análisis de generalización
            diferencia_accuracy = accuracy_train - accuracy_test
            print(f"\n--- ANÁLISIS DE GENERALIZACIÓN ---")
            print(f"Diferencia de precisión: {diferencia_accuracy:.1%}")
            
            if diferencia_accuracy > 0.3:
                print("⚠️  SOBREAJUSTE SEVERO detectado")
            elif diferencia_accuracy > 0.1:
                print("⚠️  Posible sobreajuste")
            else:
                print("✓ Buena generalización")
            print()
        
        return {
            'accuracy_train': accuracy_train,
            'accuracy_test': accuracy_test,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'pred_train': pred_train,
            'pred_test': pred_test,
            'diferencia_accuracy': accuracy_train - accuracy_test
        }
    
    def mostrar_arquitectura(self):
        """Muestra la arquitectura del modelo."""
        print("=== ARQUITECTURA DEL MODELO KERAS ===")
        self.modelo.summary()
        print()


def ejecutar_experimento_discriminacion_keras():
    """Ejecuta el experimento completo de discriminación de pares con Keras."""
    print("🔥 EXPERIMENTO DISCRIMINACIÓN PARES CON TENSORFLOW/KERAS")
    print("=" * 60)
    
    # Arquitecturas a probar (basadas en las memorias del proyecto)
    arquitecturas = {
        'MINIMA': [35, 10, 1],
        'COMPACTA': [35, 15, 8, 1],
        'DIRECTA_ORIGINAL': [35, 20, 10, 1],
        'BALANCEADA': [35, 25, 12, 1]
    }
    
    resultados = {}
    
    for nombre, arquitectura in arquitecturas.items():
        print(f"\n🧠 PROBANDO ARQUITECTURA {nombre}: {arquitectura}")
        print("-" * 50)
        
        # Crear y entrenar el modelo
        modelo = DiscriminacionParesKeras(
            arquitectura=arquitectura, 
            learning_rate=0.01
        )
        
        # Entrenar
        convergio, epocas, error_final, tiempo = modelo.entrenar(
            max_epochs=1000, 
            tolerancia_error=0.01,
            verbose=True
        )
        
        # Evaluar
        metricas = modelo.evaluar(verbose=True)
        
        # Guardar resultados
        resultados[nombre] = {
            'arquitectura': arquitectura,
            'convergio': convergio,
            'epocas': epocas,
            'error_final': error_final,
            'tiempo_entrenamiento': tiempo,
            'accuracy_train': metricas['accuracy_train'],
            'accuracy_test': metricas['accuracy_test'],
            'diferencia_accuracy': metricas['diferencia_accuracy']
        }
    
    # Resumen comparativo
    print("\n🎯 RESUMEN COMPARATIVO DE ARQUITECTURAS:")
    print("=" * 70)
    print(f"{'Arquitectura':<15} {'Precisión Test':<12} {'Épocas':<8} {'Tiempo':<8} {'Generalización'}")
    print("-" * 70)
    
    for nombre, resultado in resultados.items():
        precision_test = f"{resultado['accuracy_test']:.1%}"
        epocas = str(resultado['epocas'])
        tiempo = f"{resultado['tiempo_entrenamiento']:.1f}s"
        
        if resultado['diferencia_accuracy'] > 0.3:
            generalizacion = "Sobreajuste severo"
        elif resultado['diferencia_accuracy'] > 0.1:
            generalizacion = "Posible sobreajuste"
        else:
            generalizacion = "Buena"
        
        print(f"{nombre:<15} {precision_test:<12} {epocas:<8} {tiempo:<8} {generalizacion}")
    
    return resultados


if __name__ == "__main__":
    # Ejecutar experimento
    resultados = ejecutar_experimento_discriminacion_keras()
    
    # Encontrar la mejor arquitectura
    mejor_arquitectura = max(resultados.items(), key=lambda x: x[1]['accuracy_test'])
    
    print(f"\n🏆 MEJOR ARQUITECTURA: {mejor_arquitectura[0]}")
    print(f"   Precisión en prueba: {mejor_arquitectura[1]['accuracy_test']:.1%}")
    print(f"   Épocas: {mejor_arquitectura[1]['epocas']}")
    print(f"   Tiempo: {mejor_arquitectura[1]['tiempo_entrenamiento']:.2f}s")
