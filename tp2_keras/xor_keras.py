"""
Implementación del problema XOR utilizando TensorFlow/Keras.
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

# Configurar TensorFlow para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

class XORKeras:
    """
    Implementación del problema XOR usando TensorFlow/Keras.
    Arquitectura: [2, 4, 1] - 2 entradas, 4 neuronas ocultas, 1 salida
    """
    
    def __init__(self, arquitectura=[2, 4, 1], learning_rate=0.5):
        """
        Inicializa el modelo XOR con Keras.
        
        Args:
            arquitectura: Lista con el número de neuronas por capa
            learning_rate: Tasa de aprendizaje
        """
        self.arquitectura = arquitectura
        self.learning_rate = learning_rate
        self.modelo = None
        self.historia_entrenamiento = None
        
        # Datos del problema XOR
        self.datos_entrada = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        self.datos_salida = np.array([[0], [1], [1], [0]], dtype=np.float32)
        
        self._crear_modelo()
    
    def _crear_modelo(self):
        """Crea el modelo de red neuronal con Keras."""
        self.modelo = keras.Sequential([
            layers.Dense(self.arquitectura[1], 
                        activation='sigmoid', 
                        input_shape=(self.arquitectura[0],),
                        name='capa_oculta'),
            layers.Dense(self.arquitectura[2], 
                        activation='sigmoid',
                        name='capa_salida')
        ])
        
        # Compilar el modelo
        self.modelo.compile(
            optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def entrenar(self, max_epochs=2000, tolerancia_error=0.01, verbose=True):
        """
        Entrena el modelo XOR.
        
        Args:
            max_epochs: Número máximo de épocas
            tolerancia_error: Error mínimo para considerar convergencia
            verbose: Si mostrar información durante el entrenamiento
            
        Returns:
            Tuple: (convergio, epoca_final, error_final, tiempo_entrenamiento)
        """
        if verbose:
            print("=== ENTRENAMIENTO XOR CON KERAS ===")
            print(f"Arquitectura: {self.arquitectura}")
            print(f"Tasa de aprendizaje: {self.learning_rate}")
            print(f"Datos de entrada: {self.datos_entrada.shape}")
            print(f"Datos de salida: {self.datos_salida.shape}")
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
            self.datos_entrada, self.datos_salida,
            epochs=max_epochs,
            batch_size=4,
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
    
    def predecir(self, entradas=None):
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            entradas: Datos de entrada. Si es None, usa los datos XOR estándar.
            
        Returns:
            Array con las predicciones
        """
        if entradas is None:
            entradas = self.datos_entrada
            
        return self.modelo.predict(entradas, verbose=0)
    
    def evaluar(self, verbose=True):
        """
        Evalúa el rendimiento del modelo en el problema XOR.
        
        Args:
            verbose: Si mostrar información detallada
            
        Returns:
            Dict con métricas de evaluación
        """
        predicciones = self.predecir()
        predicciones_binarias = (predicciones > 0.5).astype(int)
        
        # Calcular métricas
        accuracy = np.mean(predicciones_binarias.flatten() == self.datos_salida.flatten())
        mse = np.mean((predicciones - self.datos_salida) ** 2)
        
        if verbose:
            print("=== EVALUACIÓN XOR KERAS ===")
            print("Entrada | Salida Esperada | Predicción | Binaria | Correcto")
            print("-" * 60)
            
            for i in range(len(self.datos_entrada)):
                entrada = self.datos_entrada[i]
                esperada = int(self.datos_salida[i, 0])
                prediccion = predicciones[i, 0]
                binaria = int(predicciones_binarias[i, 0])
                correcto = "✓" if binaria == esperada else "✗"
                
                print(f"  {entrada}   |      {esperada}       |   {prediccion:.4f}   |    {binaria}    |    {correcto}")
            
            print("-" * 60)
            print(f"Precisión: {accuracy:.1%}")
            print(f"Error cuadrático medio: {mse:.6f}")
            print()
        
        return {
            'accuracy': accuracy,
            'mse': mse,
            'predicciones': predicciones,
            'predicciones_binarias': predicciones_binarias
        }
    
    def obtener_pesos(self):
        """Obtiene los pesos del modelo para análisis."""
        pesos = {}
        for i, capa in enumerate(self.modelo.layers):
            pesos[f'capa_{i}'] = {
                'pesos': capa.get_weights()[0],
                'sesgos': capa.get_weights()[1]
            }
        return pesos
    
    def mostrar_arquitectura(self):
        """Muestra la arquitectura del modelo."""
        print("=== ARQUITECTURA DEL MODELO KERAS ===")
        self.modelo.summary()
        print()


def ejecutar_experimento_xor_keras():
    """Ejecuta el experimento completo de XOR con Keras."""
    print("🔥 EXPERIMENTO XOR CON TENSORFLOW/KERAS")
    print("=" * 50)
    
    # Crear y entrenar el modelo
    xor_keras = XORKeras(arquitectura=[2, 4, 1], learning_rate=0.5)
    
    # Mostrar arquitectura
    xor_keras.mostrar_arquitectura()
    
    # Entrenar
    convergio, epocas, error_final, tiempo = xor_keras.entrenar(
        max_epochs=2000, 
        tolerancia_error=0.01,
        verbose=True
    )
    
    # Evaluar
    metricas = xor_keras.evaluar(verbose=True)
    
    # Mostrar pesos finales
    print("=== PESOS FINALES ===")
    pesos = xor_keras.obtener_pesos()
    for capa_nombre, capa_pesos in pesos.items():
        print(f"{capa_nombre}:")
        print(f"  Pesos: {capa_pesos['pesos']}")
        print(f"  Sesgos: {capa_pesos['sesgos']}")
        print()
    
    return {
        'convergio': convergio,
        'epocas': epocas,
        'error_final': error_final,
        'tiempo_entrenamiento': tiempo,
        'accuracy': metricas['accuracy'],
        'mse': metricas['mse']
    }


if __name__ == "__main__":
    # Ejecutar experimento
    resultados = ejecutar_experimento_xor_keras()
    
    print("🎯 RESUMEN DE RESULTADOS:")
    print(f"  - Convergencia: {'✓ Sí' if resultados['convergio'] else '✗ No'}")
    print(f"  - Épocas: {resultados['epocas']}")
    print(f"  - Error final: {resultados['error_final']:.6f}")
    print(f"  - Precisión: {resultados['accuracy']:.1%}")
    print(f"  - Tiempo: {resultados['tiempo_entrenamiento']:.2f}s")
