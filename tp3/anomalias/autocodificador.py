import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

try:
    from .generador_datos_energia import GeneradorDatosEnergia
except ImportError:
    from generador_datos_energia import GeneradorDatosEnergia


class AutocodificadorAnomalias:
    def __init__(self, longitud_serie=168, dimension_latente=16):
        self.longitud_serie = longitud_serie
        self.dimension_latente = dimension_latente
        self.modelo = None
        self.encoder = None
        self.decoder = None
        self.umbral_anomalia = None
        self.historial_entrenamiento = None
        self.datos_normalizacion = None
        
    def crear_arquitectura(self, activacion_salida='linear'):
        entrada = layers.Input(shape=(self.longitud_serie,))
        
        x = layers.Dense(self.longitud_serie // 2, activation='relu')(entrada)
        x = layers.Dense(self.longitud_serie // 4, activation='relu')(x)
        
        latente = layers.Dense(self.dimension_latente, activation='relu', name='latente')(x)
        
        x = layers.Dense(self.longitud_serie // 4, activation='relu')(latente)
        x = layers.Dense(self.longitud_serie // 2, activation='relu')(x)
        salida = layers.Dense(self.longitud_serie, activation=activacion_salida)(x)
        
        self.modelo = keras.Model(entrada, salida, name='autocodificador_anomalias')
        
        self.encoder = keras.Model(entrada, latente, name='encoder_anomalias')
        
        entrada_decoder = layers.Input(shape=(self.dimension_latente,))
        x_decoder = self.modelo.layers[-3](entrada_decoder)
        x_decoder = self.modelo.layers[-2](x_decoder)
        salida_decoder = self.modelo.layers[-1](x_decoder)
        self.decoder = keras.Model(entrada_decoder, salida_decoder, name='decoder_anomalias')
        
        return self.modelo
    
    def compilar_modelo(self, learning_rate=0.001):
        if self.modelo is None:
            self.crear_arquitectura()
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.modelo
    
    def entrenar(self, datos_entrenamiento, validation_split=0.2, epochs=100, batch_size=32, 
                 patience=10, verbose=1):
        
        if self.modelo is None:
            self.compilar_modelo()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
        self.historial_entrenamiento = self.modelo.fit(
            datos_entrenamiento, datos_entrenamiento,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.historial_entrenamiento
    
    def calcular_error_reconstruccion(self, datos):
        if self.modelo is None:
            raise ValueError("El modelo debe estar entrenado antes de calcular errores")
        
        reconstrucciones = self.modelo.predict(datos, verbose=0)
        errores = np.mean(np.square(datos - reconstrucciones), axis=1)
        return errores, reconstrucciones
    
    def establecer_umbral_anomalia(self, datos_validacion, percentil=95):
        errores_validacion, _ = self.calcular_error_reconstruccion(datos_validacion)
        self.umbral_anomalia = np.percentile(errores_validacion, percentil)
        
        print(f"Umbral de anomalía establecido en: {self.umbral_anomalia:.6f}")
        print(f"Basado en percentil {percentil}% de errores de validación")
        
        return self.umbral_anomalia
    
    def detectar_anomalias(self, datos_prueba):
        if self.umbral_anomalia is None:
            raise ValueError("Debe establecer el umbral de anomalía primero")
        
        errores, reconstrucciones = self.calcular_error_reconstruccion(datos_prueba)
        predicciones_anomalia = errores > self.umbral_anomalia
        
        return predicciones_anomalia, errores, reconstrucciones
    
    def evaluar_deteccion(self, datos_prueba, etiquetas_reales):
        predicciones, errores, _ = self.detectar_anomalias(datos_prueba)
        
        tp = np.sum((etiquetas_reales == True) & (predicciones == True))
        fp = np.sum((etiquetas_reales == False) & (predicciones == True))
        tn = np.sum((etiquetas_reales == False) & (predicciones == False))
        fn = np.sum((etiquetas_reales == True) & (predicciones == False))
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print("Reporte de Clasificación:")
        print(f"              precision    recall  f1-score   support")
        print(f"")
        print(f"      Normal       {tn/(tn+fn):.2f}      {tn/(tn+fp):.2f}      {2*tn/(2*tn+fn+fp):.2f}      {tn+fn}")
        print(f"    Anomalía       {precision:.2f}      {recall:.2f}      {f1:.2f}      {tp+fn}")
        print(f"")
        print(f"    accuracy                           {accuracy:.2f}      {tp+tn+fp+fn}")
        print(f"   macro avg       {(tn/(tn+fn)+precision)/2:.2f}      {(tn/(tn+fp)+recall)/2:.2f}      {(2*tn/(2*tn+fn+fp)+f1)/2:.2f}      {tp+tn+fp+fn}")
        
        print("\nMatriz de Confusión:")
        print("Predicción:  Normal  Anomalía")
        print(f"Real Normal:   {tn:4d}     {fp:4d}")
        print(f"Real Anomalía: {fn:4d}     {tp:4d}")
        
        metricas = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'matriz_confusion': cm,
            'errores_reconstruccion': errores,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
        
        return metricas
    
    def generar_muestra_sintetica(self, num_muestras=1):
        if self.decoder is None:
            raise ValueError("El decoder debe estar disponible para generar muestras")
        
        vectores_latentes_aleatorios = np.random.normal(0, 1, (num_muestras, self.dimension_latente))
        muestras_sinteticas = self.decoder.predict(vectores_latentes_aleatorios, verbose=0)
        
        return muestras_sinteticas, vectores_latentes_aleatorios
    
    def generar_desde_interpolacion(self, muestra1, muestra2, num_pasos=10):
        if self.encoder is None or self.decoder is None:
            raise ValueError("Encoder y decoder deben estar disponibles")
        
        latente1 = self.encoder.predict(muestra1.reshape(1, -1), verbose=0)
        latente2 = self.encoder.predict(muestra2.reshape(1, -1), verbose=0)
        
        alphas = np.linspace(0, 1, num_pasos)
        interpolaciones = []
        
        for alpha in alphas:
            latente_interpolado = (1 - alpha) * latente1 + alpha * latente2
            muestra_interpolada = self.decoder.predict(latente_interpolado, verbose=0)
            interpolaciones.append(muestra_interpolada[0])
        
        return np.array(interpolaciones)
    
    def visualizar_entrenamiento(self):
        if self.historial_entrenamiento is None:
            print("No hay historial de entrenamiento disponible")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.historial_entrenamiento.history['loss'], label='Entrenamiento')
        ax1.plot(self.historial_entrenamiento.history['val_loss'], label='Validación')
        ax1.set_title('Pérdida durante el entrenamiento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('MSE')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.historial_entrenamiento.history['mae'], label='Entrenamiento')
        ax2.plot(self.historial_entrenamiento.history['val_mae'], label='Validación')
        ax2.set_title('Error Absoluto Medio')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualizar_deteccion(self, datos_prueba, etiquetas_reales, num_ejemplos=5):
        predicciones, errores, reconstrucciones = self.detectar_anomalias(datos_prueba)
        
        indices_normales = np.where(etiquetas_reales == False)[0][:num_ejemplos]
        indices_anomalos = np.where(etiquetas_reales == True)[0][:num_ejemplos]
        
        fig, axes = plt.subplots(2, num_ejemplos, figsize=(15, 8))
        
        for i, idx in enumerate(indices_normales):
            axes[0, i].plot(datos_prueba[idx], label='Original', alpha=0.7)
            axes[0, i].plot(reconstrucciones[idx], label='Reconstruido', alpha=0.7)
            axes[0, i].set_title(f'Normal\nError: {errores[idx]:.4f}')
            axes[0, i].legend()
            axes[0, i].grid(True)
        
        for i, idx in enumerate(indices_anomalos):
            axes[1, i].plot(datos_prueba[idx], label='Original', alpha=0.7)
            axes[1, i].plot(reconstrucciones[idx], label='Reconstruido', alpha=0.7)
            axes[1, i].set_title(f'Anomalía\nError: {errores[idx]:.4f}')
            axes[1, i].legend()
            axes[1, i].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def guardar_modelo(self, ruta_base):
        if self.modelo is None:
            raise ValueError("No hay modelo para guardar")
        
        import os
        from datetime import datetime
        
        # Asegurar que el directorio base existe
        directorio = os.path.dirname(ruta_base)
        if directorio:
            os.makedirs(directorio, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_modelo = f"{ruta_base}_anomalias_lat{self.dimension_latente}_{timestamp}.keras"
        
        try:
            self.modelo.save(nombre_modelo)
            
            metadatos = {
                'longitud_serie': self.longitud_serie,
                'dimension_latente': self.dimension_latente,
                'umbral_anomalia': self.umbral_anomalia,
                'datos_normalizacion': self.datos_normalizacion
            }
            
            np.save(f"{nombre_modelo}_metadatos.npy", metadatos)
            
            return nombre_modelo
        except Exception as e:
            print(f"Error al guardar modelo: {e}")
            raise
    
    def cargar_modelo(self, ruta_modelo):
        self.modelo = keras.models.load_model(ruta_modelo)
        
        try:
            metadatos = np.load(f"{ruta_modelo}_metadatos.npy", allow_pickle=True).item()
            self.longitud_serie = metadatos['longitud_serie']
            self.dimension_latente = metadatos['dimension_latente']
            self.umbral_anomalia = metadatos['umbral_anomalia']
            self.datos_normalizacion = metadatos['datos_normalizacion']
        except FileNotFoundError:
            print("Metadatos no encontrados, usando valores por defecto")
        
        entrada = self.modelo.input
        latente = self.modelo.get_layer('latente').output
        self.encoder = keras.Model(entrada, latente)
        
        entrada_decoder = layers.Input(shape=(self.dimension_latente,))
        x = entrada_decoder
        for layer in self.modelo.layers[self.modelo.layers.index(self.modelo.get_layer('latente')) + 1:]:
            x = layer(x)
        self.decoder = keras.Model(entrada_decoder, x)
        
        return self.modelo
