import argparse
import os
import pickle
import sys
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tp3.src.cargador_datos_caracteres import CargadorDatosCaracteres
from tp3.src.configuraciones_arquitecturas import obtener_arquitecturas_disponibles


class AutocodificadorCaracteres:

    def __init__(self, dimension_entrada: int = 35, dimension_latente: int = 2, 
                 arquitectura: str = 'estandar'):
        self.dimension_entrada = dimension_entrada
        self.dimension_latente = dimension_latente
        self.arquitectura = arquitectura
        self.cargador_datos = CargadorDatosCaracteres()
        self.datos_cargados = False
        self.entrenado = False
        self.autocodificador = None
        self.codificador = None
        self.decodificador = None
        self.datos_entrenamiento = None
        self.representaciones_latentes = None
        self.historial_entrenamiento = None
        self.arquitecturas_disponibles = obtener_arquitecturas_disponibles()
        self.mapeo_caracteres = {}
        self.conjunto_datos_actual = None

    def crear_arquitectura_autocodificador(self):
        if self.arquitectura not in self.arquitecturas_disponibles:
            raise ValueError(f"Arquitectura '{self.arquitectura}' no disponible. "
                           f"Opciones: {list(self.arquitecturas_disponibles.keys())}")
        
        config = self.arquitecturas_disponibles[self.arquitectura]
        capas_codificador = config['codificador']
        activacion = config['activacion']
        usar_bn = bool(config.get('batch_norm', False))
        tasa_dropout = float(config.get('dropout', 0.0))
        l1_val = config.get('l1', None)
        l2_val = config.get('l2', None)
        if l1_val is not None and l2_val is not None:
            reg = regularizers.l1_l2(l1=l1_val, l2=l2_val)
        elif l1_val is not None:
            reg = regularizers.l1(l1_val)
        elif l2_val is not None:
            reg = regularizers.l2(l2_val)
        else:
            reg = None
        ruido = float(config.get('noise', 0.0))
        dim_latente = int(config.get('latente', self.dimension_latente))
        
        input_layer = keras.Input(shape=(self.dimension_entrada,))
        
        x = input_layer
        if ruido and ruido > 0:
            x = layers.GaussianNoise(ruido, name='input_noise')(x)
        for i, neuronas in enumerate(capas_codificador):
            x = layers.Dense(neuronas, activation=None,
                             kernel_regularizer=reg,
                             name=f'encoder_hidden_{i+1}')(x)
            if usar_bn:
                x = layers.BatchNormalization(name=f'encoder_bn_{i+1}')(x)
            x = layers.Activation(activacion, name=f'encoder_act_{i+1}')(x)
            if tasa_dropout and tasa_dropout > 0:
                x = layers.Dropout(tasa_dropout, name=f'encoder_do_{i+1}')(x)
        
        encoded = layers.Dense(dim_latente, activation=activacion,
                               kernel_regularizer=reg,
                               name='espacio_latente')(x)
        
        x = encoded
        capas_decodificador = capas_codificador[::-1]
        for i, neuronas in enumerate(capas_decodificador):
            x = layers.Dense(neuronas, activation=None,
                             kernel_regularizer=reg,
                             name=f'decoder_hidden_{i+1}')(x)
            if usar_bn:
                x = layers.BatchNormalization(name=f'decoder_bn_{i+1}')(x)
            x = layers.Activation(activacion, name=f'decoder_act_{i+1}')(x)
        
        decoded = layers.Dense(self.dimension_entrada, activation='sigmoid',
                             name='output_layer')(x)
        
        autocodificador = Model(input_layer, decoded, name='autocodificador')
        codificador = Model(input_layer, encoded, name='codificador')
        
        encoded_input = keras.Input(shape=(dim_latente,))
        
        decoder_layers = []
        for layer in autocodificador.layers:
            if 'decoder' in layer.name or 'output' in layer.name:
                decoder_layers.append(layer)
        
        x_dec = encoded_input
        for layer in decoder_layers:
            x_dec = layer(x_dec)
        
        decodificador = Model(encoded_input, x_dec, name='decodificador')
        
        return autocodificador, codificador, decodificador

    def inicializar_autocodificador(self):
        self.autocodificador, self.codificador, self.decodificador = self.crear_arquitectura_autocodificador()
        
        self.autocodificador.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        config = self.arquitecturas_disponibles[self.arquitectura]
        print(f"Autocodificador Keras inicializado:")
        print(f"Arquitectura: {self.arquitectura} - {config['descripcion']}")
        print(f"Entrada: {self.dimension_entrada} -> Codificador: {config['codificador']} -> Latente: {self.dimension_latente}")
        print(f"Activación: {config['activacion']}")
        print(f"Parámetros totales: {self.autocodificador.count_params()}")

    def cargar_datos(self, conjunto_datos: int = 1):
        datos_entrada, _ = self.cargador_datos.cargar_datos_desde_modulo(conjunto_datos)
        self.datos_entrenamiento = datos_entrada
        self.conjunto_datos = conjunto_datos
        self.conjunto_datos_actual = conjunto_datos
        self.datos_cargados = True
        self._crear_mapeo_caracteres(conjunto_datos)
        print(f"Datos cargados: {datos_entrada.shape}")
        print(f"Caracteres disponibles: {list(self.mapeo_caracteres.keys())}")
        return datos_entrada

    def entrenar_autocodificador(self, tasa_aprendizaje: float = 0.001,
                               max_epocas: int = 1000, error_objetivo: float = 0.01,
                               paciencia: int = 50, tipo_perdida: str = 'mse',
                               monitor: str = 'val_loss', usar_scheduler: bool = False,
                               batch_size: int = 32):
        if not self.datos_cargados:
            raise ValueError("Primero debe cargar los datos")
        
        if self.autocodificador is None:
            self.inicializar_autocodificador()

        print(f"\nEntrenando autocodificador con Keras...")
        print(f"Datos de entrada: {self.datos_entrenamiento.shape}")
        print(f"Datos de salida: {self.datos_entrenamiento.shape} (mismos que entrada)")
        
        if tipo_perdida.lower() == 'bce':
            loss_fn = BinaryCrossentropy()
        else:
            loss_fn = MeanSquaredError()
        self.autocodificador.compile(optimizer=Adam(learning_rate=tasa_aprendizaje),
                                     loss=loss_fn,
                                     metrics=['mae', 'mse'])

        early_stopping = EarlyStopping(monitor=monitor, patience=paciencia, restore_best_weights=True, verbose=0)
        plateau = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=max(5, paciencia // 5),
                                    min_lr=1e-5, verbose=0)
        callbacks = [early_stopping]
        if usar_scheduler:
            callbacks.append(plateau)

        self.historial_entrenamiento = self.autocodificador.fit(
            self.datos_entrenamiento,
            self.datos_entrenamiento,
            epochs=max_epocas,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )

        self.entrenado = True
        
        clave_mse = 'val_mse' if 'val_mse' in self.historial_entrenamiento.history else 'mse'
        error_mse_final = self.historial_entrenamiento.history[clave_mse][-1]
        epocas_entrenadas = len(self.historial_entrenamiento.history['loss'])
        convergencia = error_mse_final <= error_objetivo
        
        clave_loss = monitor if monitor in self.historial_entrenamiento.history else 'loss'
        error_loss_final = self.historial_entrenamiento.history[clave_loss][-1]
        
        print(f"MSE final ({clave_mse}): {error_mse_final:.6f} | Loss final ({clave_loss}): {error_loss_final:.6f} | Objetivo MSE: {error_objetivo} | Convergió: {convergencia}")
        
        return convergencia, epocas_entrenadas

    def obtener_representacion_latente(self, datos: np.ndarray = None) -> np.ndarray:
        if not self.entrenado:
            raise ValueError("El autocodificador debe estar entrenado")
        
        if datos is None:
            datos = self.datos_entrenamiento

        self.representaciones_latentes = self.codificador.predict(datos, verbose=0)
        return self.representaciones_latentes

    def decodificar_desde_latente(self, representacion_latente: np.ndarray) -> np.ndarray:
        if not self.entrenado:
            raise ValueError("El autocodificador debe estar entrenado")

        return self.decodificador.predict(representacion_latente, verbose=0)

    def reconstruir_caracteres(self, datos: np.ndarray = None) -> np.ndarray:
        if datos is None:
            datos = self.datos_entrenamiento
        
        return self.autocodificador.predict(datos, verbose=0)

    def visualizar_espacio_latente(self, mostrar_indices: bool = True):
        if not self.entrenado:
            raise ValueError("El autocodificador debe estar entrenado")
        
        if self.representaciones_latentes is None:
            self.obtener_representacion_latente()

        if self.dimension_latente != 2:
            print(f"Advertencia: Visualización 2D con dimensión latente {self.dimension_latente}")
            representaciones_2d = self.representaciones_latentes[:, :2]
        else:
            representaciones_2d = self.representaciones_latentes

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(representaciones_2d[:, 0], representaciones_2d[:, 1], 
                            c=range(len(representaciones_2d)), cmap='tab20', s=100)
        
        if mostrar_indices:
            for i, (x, y) in enumerate(representaciones_2d):
                plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

        plt.xlabel('Dimensión Latente 1')
        plt.ylabel('Dimensión Latente 2')
        plt.title('Representación en Espacio Latente 2D')
        plt.colorbar(scatter, label='Índice del Carácter')
        plt.grid(True, alpha=0.3)
        plt.show()

    def generar_caracter_nuevo(self, punto_latente: np.ndarray) -> np.ndarray:
        if punto_latente.ndim == 1:
            punto_latente = punto_latente.reshape(1, -1)
        
        caracter_generado = self.decodificar_desde_latente(punto_latente)
        return caracter_generado

    def interpolar_entre_caracteres(self, indice1: int, indice2: int, 
                                  num_pasos: int = 5) -> List[np.ndarray]:
        if self.representaciones_latentes is None:
            self.obtener_representacion_latente()

        punto1 = self.representaciones_latentes[indice1]
        punto2 = self.representaciones_latentes[indice2]
        
        caracteres_interpolados = []
        for i in range(num_pasos):
            alpha = i / (num_pasos - 1)
            punto_interpolado = (1 - alpha) * punto1 + alpha * punto2
            caracter_interpolado = self.generar_caracter_nuevo(punto_interpolado)
            caracteres_interpolados.append(caracter_interpolado[0])
        
        return caracteres_interpolados

    def mostrar_reconstrucciones(self, num_ejemplos: int = 8):
        if not self.entrenado:
            raise ValueError("El autocodificador debe estar entrenado")

        indices = np.random.choice(len(self.datos_entrenamiento), num_ejemplos, replace=False)
        originales = self.datos_entrenamiento[indices]
        reconstruidos = self.reconstruir_caracteres(originales)

        print(f"\n{'='*60}")
        print("COMPARACIÓN ORIGINAL vs RECONSTRUIDO")
        print(f"{'='*60}")

        for i, (original, reconstruido) in enumerate(zip(originales, reconstruidos)):
            print(f"\nCarácter {indices[i]}:")
            print("ORIGINAL:")
            print(self.cargador_datos.visualizar_patron(original))
            print("RECONSTRUIDO:")
            reconstruido_binario = (reconstruido > 0.5).astype(int)
            print(self.cargador_datos.visualizar_patron(reconstruido_binario))
            
            error = np.mean((original - reconstruido) ** 2)
            print(f"Error MSE: {error:.4f}")

    def mostrar_interpolacion(self, indice1: int, indice2: int, num_pasos: int = 5):
        caracteres_interpolados = self.interpolar_entre_caracteres(indice1, indice2, num_pasos)
        
        print(f"\n{'='*60}")
        print(f"INTERPOLACIÓN ENTRE CARÁCTER {indice1} Y CARÁCTER {indice2}")
        print(f"{'='*60}")

        for i, caracter in enumerate(caracteres_interpolados):
            alpha = i / (num_pasos - 1)
            print(f"\nPaso {i+1} (α={alpha:.2f}):")
            caracter_binario = (caracter > 0.5).astype(int)
            print(self.cargador_datos.visualizar_patron(caracter_binario))

    def evaluar_calidad_reconstruccion(self) -> Dict[str, float]:
        if not self.entrenado:
            raise ValueError("El autocodificador debe estar entrenado")

        reconstruidos = self.reconstruir_caracteres()
        
        mse = np.mean((self.datos_entrenamiento - reconstruidos) ** 2)
        mae = np.mean(np.abs(self.datos_entrenamiento - reconstruidos))
        
        reconstruidos_binarios = (reconstruidos > 0.5).astype(int)
        precision_binaria = np.mean(self.datos_entrenamiento == reconstruidos_binarios)

        return {
            'mse': mse,
            'mae': mae,
            'precision_binaria': precision_binaria,
            'num_patrones': len(self.datos_entrenamiento)
        }

    def generar_nuevos_caracteres(self, num_caracteres: int = 5, 
                                metodo: str = 'aleatorio') -> np.ndarray:
        if not self.entrenado:
            raise ValueError("El autocodificador debe estar entrenado")
        
        if self.representaciones_latentes is None:
            self.obtener_representacion_latente()

        if metodo == 'aleatorio':
            min_vals = np.min(self.representaciones_latentes, axis=0)
            max_vals = np.max(self.representaciones_latentes, axis=0)
            puntos_aleatorios = np.random.uniform(min_vals, max_vals, 
                                                (num_caracteres, self.dimension_latente))
        elif metodo == 'gaussiano':
            media = np.mean(self.representaciones_latentes, axis=0)
            covarianza = np.cov(self.representaciones_latentes.T)
            puntos_aleatorios = np.random.multivariate_normal(media, covarianza, num_caracteres)
        else:
            raise ValueError("Método debe ser 'aleatorio' o 'gaussiano'")
        
        caracteres_generados = self.decodificar_desde_latente(puntos_aleatorios)
        
        print(f"\nCaracteres generados usando método '{metodo}':")
        for i, caracter in enumerate(caracteres_generados):
            print(f"\nCarácter generado {i+1}:")
            caracter_binario = (caracter > 0.5).astype(int)
            print(self.cargador_datos.visualizar_patron(caracter_binario))
        
        return caracteres_generados

    def _crear_mapeo_caracteres(self, conjunto_datos: int):
        if conjunto_datos == 1:
            caracteres_base = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?']
        elif conjunto_datos == 2:
            caracteres_base = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']
        elif conjunto_datos == 3:
            caracteres_base = ['`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                             'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL']
        else:
            raise ValueError("conjunto_datos debe ser 1, 2, o 3")
        
        self.mapeo_caracteres = {char: idx for idx, char in enumerate(caracteres_base)}
        self.mapeo_inverso = {idx: char for char, idx in self.mapeo_caracteres.items()}

    def _validar_caracter(self, caracter: str) -> int:
        if not self.datos_cargados:
            raise ValueError("Primero debe cargar los datos usando cargar_datos()")
        
        if caracter not in self.mapeo_caracteres:
            caracteres_disponibles = list(self.mapeo_caracteres.keys())
            raise ValueError(f"El carácter '{caracter}' no está en el conjunto de entrenamiento {self.conjunto_datos_actual}. "
                           f"Caracteres disponibles: {caracteres_disponibles}")
        
        return self.mapeo_caracteres[caracter]

    def analizar_caracter_por_simbolo(self, caracter: str) -> Dict[str, any]:
        indice = self._validar_caracter(caracter)
        
        caracter_original = self.datos_entrenamiento[indice]
        
        print(f"\n📋 CARÁCTER '{caracter}' (índice {indice}):")
        print(self.cargador_datos.visualizar_patron(caracter_original))
        
        representacion_latente = self.obtener_representacion_latente(
            caracter_original.reshape(1, -1)
        )[0]
        
        print(f"\n🎯 REPRESENTACIÓN LATENTE:")
        print(f"   Coordenadas: ({representacion_latente[0]:.6f}, {representacion_latente[1]:.6f})")
        
        caracter_reconstruido = self.decodificar_desde_latente(
            representacion_latente.reshape(1, -1)
        )[0]
        
        print(f"\n🔄 CARÁCTER RECONSTRUIDO:")
        caracter_reconstruido_binario = (caracter_reconstruido > 0.5).astype(int)
        print(self.cargador_datos.visualizar_patron(caracter_reconstruido_binario))
        
        error_mse = np.mean((caracter_original - caracter_reconstruido) ** 2)
        precision_binaria = np.mean(caracter_original == caracter_reconstruido_binario)
        
        print(f"\n📊 MÉTRICAS DE CALIDAD:")
        print(f"   MSE: {error_mse:.6f}")
        print(f"   Precisión binaria: {precision_binaria:.3f} ({precision_binaria*100:.1f}%)")
        
        calidad = "Excelente" if precision_binaria > 0.9 else "Buena" if precision_binaria > 0.7 else "Regular"
        print(f"   Calidad: {calidad}")
        
        return {
            'caracter': caracter,
            'indice': indice,
            'representacion_latente': representacion_latente,
            'error_mse': error_mse,
            'precision_binaria': precision_binaria,
            'calidad': calidad
        }

    def interpolar_entre_caracteres_simbolos(self, caracter1: str, caracter2: str, 
                                           num_pasos: int = 5) -> List[np.ndarray]:
        indice1 = self._validar_caracter(caracter1)
        indice2 = self._validar_caracter(caracter2)
        
        caracteres_interpolados = self.interpolar_entre_caracteres(indice1, indice2, num_pasos)
        
        print(f"\n{'='*60}")
        print(f"INTERPOLACIÓN ENTRE '{caracter1}' Y '{caracter2}'")
        print(f"{'='*60}")

        for i, caracter in enumerate(caracteres_interpolados):
            alpha = i / (num_pasos - 1)
            print(f"\nPaso {i+1} (α={alpha:.2f}):")
            caracter_binario = (caracter > 0.5).astype(int)
            print(self.cargador_datos.visualizar_patron(caracter_binario))
        
        return caracteres_interpolados

    def obtener_caracteres_disponibles(self) -> List[str]:
        if not self.datos_cargados:
            raise ValueError("Primero debe cargar los datos usando cargar_datos()")
        return list(self.mapeo_caracteres.keys())

    def mostrar_todos_los_caracteres_con_simbolos(self):
        if not self.datos_cargados:
            print("❌ No hay datos cargados. Use cargar_datos() primero.")
            return
        
        print(f"\n📋 TODOS LOS CARACTERES DEL CONJUNTO {self.conjunto_datos_actual}")
        print(f"{'='*60}")
        
        for caracter, indice in self.mapeo_caracteres.items():
            patron = self.datos_entrenamiento[indice]
            print(f"\nCarácter '{caracter}' (índice {indice}):")
            print(self.cargador_datos.visualizar_patron(patron))

    def guardar_modelo(self, nombre_archivo: str, directorio: str = "modelos"):
        if not self.entrenado:
            raise ValueError("El autocodificador debe estar entrenado antes de guardarlo")
        
        if not os.path.exists(directorio):
            os.makedirs(directorio)
        
        ruta_base = os.path.join(directorio, nombre_archivo)
        
        self.autocodificador.save(f"{ruta_base}_autocodificador.keras")
        self.codificador.save(f"{ruta_base}_codificador.keras")
        self.decodificador.save(f"{ruta_base}_decodificador.keras")
        
        metadatos = {
            'dimension_entrada': self.dimension_entrada,
            'dimension_latente': self.dimension_latente,
            'arquitectura': self.arquitectura,
            'datos_entrenamiento': self.datos_entrenamiento,
            'representaciones_latentes': self.representaciones_latentes,
            'historial_entrenamiento': self.historial_entrenamiento.history if self.historial_entrenamiento else None,
            'conjunto_datos': getattr(self, 'conjunto_datos', None)
        }
        
        with open(f"{ruta_base}_metadatos.pkl", 'wb') as f:
            pickle.dump(metadatos, f)
        
        print(f"✅ Modelo guardado exitosamente:")
        print(f"   📁 Directorio: {directorio}")
        print(f"   📄 Archivos: {nombre_archivo}_*.keras, {nombre_archivo}_metadatos.pkl")

    def cargar_modelo(self, nombre_archivo: str, directorio: str = "modelos"):
        ruta_base = os.path.join(directorio, nombre_archivo)
        
        if not os.path.exists(f"{ruta_base}_metadatos.pkl"):
            raise FileNotFoundError(f"No se encontró el archivo de metadatos: {ruta_base}_metadatos.pkl")
        
        with open(f"{ruta_base}_metadatos.pkl", 'rb') as f:
            metadatos = pickle.load(f)
        
        self.dimension_entrada = metadatos['dimension_entrada']
        self.dimension_latente = metadatos['dimension_latente']
        self.arquitectura = metadatos['arquitectura']
        self.datos_entrenamiento = metadatos['datos_entrenamiento']
        self.representaciones_latentes = metadatos['representaciones_latentes']
        self.conjunto_datos = metadatos.get('conjunto_datos', None)
        self.conjunto_datos_actual = self.conjunto_datos
        if self.conjunto_datos_actual:
            self._crear_mapeo_caracteres(self.conjunto_datos_actual)
        
        # Intentar cargar formato .keras primero, luego .h5 para compatibilidad
        try:
            self.autocodificador = keras.models.load_model(f"{ruta_base}_autocodificador.keras")
            self.codificador = keras.models.load_model(f"{ruta_base}_codificador.keras")
            self.decodificador = keras.models.load_model(f"{ruta_base}_decodificador.keras")
        except FileNotFoundError:
            # Fallback a formato .h5 para modelos antiguos
            try:
                self.autocodificador = keras.models.load_model(f"{ruta_base}_autocodificador.h5")
                self.codificador = keras.models.load_model(f"{ruta_base}_codificador.h5")
                self.decodificador = keras.models.load_model(f"{ruta_base}_decodificador.h5")
                print("⚠️  Modelo cargado en formato legacy .h5")
            except FileNotFoundError:
                raise FileNotFoundError(f"No se encontraron archivos del modelo: {ruta_base}_autocodificador.[keras|h5]")
        
        self.datos_cargados = True
        self.entrenado = True
        
        print(f"✅ Modelo cargado exitosamente:")
        print(f"   🏗️  Arquitectura: {self.arquitectura}")
        print(f"   📊 Dimensiones: {self.dimension_entrada} -> {self.dimension_latente}")
        print(f"   🔢 Parámetros: {self.autocodificador.count_params()}")
        if self.conjunto_datos:
            print(f"   📁 Conjunto original: {self.conjunto_datos}")

    def listar_arquitecturas_disponibles(self):
        print(f"\n{'='*60}")
        print("ARQUITECTURAS DISPONIBLES")
        print(f"{'='*60}")
        for nombre, config in self.arquitecturas_disponibles.items():
            print(f"\n🏗️  {nombre.upper()}:")
            print(f"   Descripción: {config['descripcion']}")
            print(f"   Codificador: {self.dimension_entrada} -> {' -> '.join(map(str, config['codificador']))} -> {self.dimension_latente}")
            print(f"   Activación: {config['activacion']}")

    @staticmethod
    def comparar_arquitecturas(conjunto_datos: int = 1, dimension_latente: int = 2, 
                             max_epocas: int = 500, arquitecturas_a_probar: List[str] = None):
        if arquitecturas_a_probar is None:
            arquitecturas_a_probar = ['minima', 'estandar', 'compacta', 'profunda']
        
        resultados = {}
        
        print(f"\n{'='*80}")
        print("COMPARACIÓN DE ARQUITECTURAS DE AUTOCODIFICADORES")
        print(f"{'='*80}")
        
        for arquitectura in arquitecturas_a_probar:
            print(f"\n🔄 Probando arquitectura: {arquitectura.upper()}")
            print("-" * 50)
            
            try:
                autocodificador = AutocodificadorCaracteres(
                    dimension_latente=dimension_latente,
                    arquitectura=arquitectura
                )
                
                autocodificador.cargar_datos(conjunto_datos)
                convergencia, epocas = autocodificador.entrenar_autocodificador(
                    max_epocas=max_epocas,
                    error_objetivo=0.01
                )
                
                metricas = autocodificador.evaluar_calidad_reconstruccion()
                
                resultados[arquitectura] = {
                    'convergencia': convergencia,
                    'epocas': epocas,
                    'parametros': autocodificador.autocodificador.count_params(),
                    'mse': metricas['mse'],
                    'mae': metricas['mae'],
                    'precision_binaria': metricas['precision_binaria'],
                    'config': autocodificador.arquitecturas_disponibles[arquitectura]
                }
                
                print(f"✅ Completado - Épocas: {epocas}, MSE: {metricas['mse']:.6f}")
                
            except Exception as e:
                print(f"❌ Error con arquitectura {arquitectura}: {e}")
                resultados[arquitectura] = {'error': str(e)}
        
        print(f"\n{'='*80}")
        print("RESUMEN COMPARATIVO")
        print(f"{'='*80}")
        
        print(f"{'Arquitectura':<15} {'Épocas':<8} {'Parámetros':<12} {'MSE':<12} {'Precisión':<10} {'Estado'}")
        print("-" * 80)
        
        for arquitectura, resultado in resultados.items():
            if 'error' not in resultado:
                estado = "✅ OK" if resultado['convergencia'] else "⚠️ No conv."
                print(f"{arquitectura:<15} {resultado['epocas']:<8} {resultado['parametros']:<12} "
                      f"{resultado['mse']:<12.6f} {resultado['precision_binaria']:<10.3f} {estado}")
            else:
                print(f"{arquitectura:<15} {'ERROR':<8} {'-':<12} {'-':<12} {'-':<10} ❌")
        
        mejor_arquitectura = min(
            [k for k, v in resultados.items() if 'error' not in v],
            key=lambda k: resultados[k]['mse']
        )
        
        print(f"\n🏆 MEJOR ARQUITECTURA: {mejor_arquitectura.upper()}")
        print(f"   MSE: {resultados[mejor_arquitectura]['mse']:.6f}")
        print(f"   Épocas: {resultados[mejor_arquitectura]['epocas']}")
        print(f"   Parámetros: {resultados[mejor_arquitectura]['parametros']}")
        
        return resultados


def entrenar_modelo_cli(args):
    print(f"🚀 ENTRENANDO AUTOCODIFICADOR")
    print(f"{'='*60}")
    print(f"📁 Conjunto de datos: {args.conjunto}")
    print(f"🏗️  Arquitectura: {args.arquitectura}")
    print(f"💾 Nombre del modelo: {args.modelo}")
    
    try:
        autocodificador = AutocodificadorCaracteres(
            dimension_latente=2,
            arquitectura=args.arquitectura
        )
        
        autocodificador.cargar_datos(conjunto_datos=args.conjunto)
        
        convergencia, epocas = autocodificador.entrenar_autocodificador(
            tasa_aprendizaje=args.tasa_aprendizaje,
            max_epocas=args.epocas,
            error_objetivo=args.error_objetivo
        )
        
        metricas = autocodificador.evaluar_calidad_reconstruccion()
        
        print(f"\n📊 RESULTADOS DEL ENTRENAMIENTO:")
        print(f"   ✅ Convergencia: {'Sí' if convergencia else 'No'}")
        print(f"   📈 Épocas: {epocas}")
        print(f"   🔧 Parámetros: {autocodificador.autocodificador.count_params()}")
        print(f"   📊 MSE: {metricas['mse']:.6f}")
        print(f"   📊 Precisión binaria: {metricas['precision_binaria']:.3f}")
        
        autocodificador.guardar_modelo(args.modelo)
        
        print(f"\n🎉 Modelo entrenado y guardado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return False
    
    return True


def analizar_caracter_cli(args):
    print(f"🔍 ANALIZANDO CARÁCTER")
    print(f"{'='*60}")
    print(f"💾 Modelo: {args.modelo}")
    print(f"🔤 Carácter: {args.caracter}")
    
    try:
        autocodificador = AutocodificadorCaracteres()
        autocodificador.cargar_modelo(args.modelo)
        
        try:
            caracter_int = int(args.caracter)
            if caracter_int >= len(autocodificador.datos_entrenamiento):
                print(f"❌ Carácter {caracter_int} no existe. Máximo: {len(autocodificador.datos_entrenamiento)-1}")
                return False
            
            caracter_original = autocodificador.datos_entrenamiento[caracter_int]
            
            print(f"\n📋 CARÁCTER ORIGINAL {caracter_int}:")
            print(autocodificador.cargador_datos.visualizar_patron(caracter_original))
            
            representacion_latente = autocodificador.obtener_representacion_latente(
                caracter_original.reshape(1, -1)
            )[0]
            
            print(f"\n🎯 REPRESENTACIÓN LATENTE:")
            print(f"   Coordenadas: ({representacion_latente[0]:.6f}, {representacion_latente[1]:.6f})")
            
            caracter_reconstruido = autocodificador.decodificar_desde_latente(
                representacion_latente.reshape(1, -1)
            )[0]
            
            print(f"\n🔄 CARÁCTER RECONSTRUIDO:")
            caracter_reconstruido_binario = (caracter_reconstruido > 0.5).astype(int)
            print(autocodificador.cargador_datos.visualizar_patron(caracter_reconstruido_binario))
            
            error_mse = np.mean((caracter_original - caracter_reconstruido) ** 2)
            precision_binaria = np.mean(caracter_original == caracter_reconstruido_binario)
            
            print(f"\n📊 MÉTRICAS DE CALIDAD:")
            print(f"   MSE: {error_mse:.6f}")
            print(f"   Precisión binaria: {precision_binaria:.3f} ({precision_binaria*100:.1f}%)")
            
            calidad = "Excelente" if precision_binaria > 0.9 else "Buena" if precision_binaria > 0.7 else "Regular"
            print(f"   Calidad: {calidad}")
            
        except ValueError:
            autocodificador.analizar_caracter_por_simbolo(args.caracter)
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        return False
    
    return True


def generar_caracter_cli(args):
    print(f"🎨 GENERANDO NUEVO CARÁCTER")
    print(f"{'='*60}")
    print(f"💾 Modelo: {args.modelo}")
    print(f"🎲 Método: {args.metodo}")
    
    try:
        autocodificador = AutocodificadorCaracteres()
        autocodificador.cargar_modelo(args.modelo)
        
        if args.metodo == 'interpolacion':
            if args.caracter1 is None or args.caracter2 is None:
                print("❌ Para interpolación necesita especificar --caracter1 y --caracter2")
                return False
            
            try:
                caracter1_int = int(args.caracter1)
                caracter2_int = int(args.caracter2)
                
                if (caracter1_int >= len(autocodificador.datos_entrenamiento) or 
                    caracter2_int >= len(autocodificador.datos_entrenamiento)):
                    print(f"❌ Caracteres inválidos. Máximo: {len(autocodificador.datos_entrenamiento)-1}")
                    return False
                
                print(f"🔄 Interpolando entre carácter {caracter1_int} y {caracter2_int}:")
                autocodificador.mostrar_interpolacion(caracter1_int, caracter2_int, args.pasos)
                
            except ValueError:
                print(f"🔄 Interpolando entre carácter '{args.caracter1}' y '{args.caracter2}':")
                autocodificador.interpolar_entre_caracteres_simbolos(args.caracter1, args.caracter2, args.pasos)
        else:
            print(f"🎲 Generando {args.cantidad} caracteres con método '{args.metodo}':")
            autocodificador.generar_nuevos_caracteres(args.cantidad, args.metodo)
        
    except Exception as e:
        print(f"❌ Error durante la generación: {e}")
        return False
    
    return True


def listar_modelos_cli():
    directorio = "modelos"
    if not os.path.exists(directorio):
        print("📁 No hay modelos guardados")
        return
    
    archivos = [f for f in os.listdir(directorio) if f.endswith('_metadatos.pkl')]
    
    if not archivos:
        print("📁 No hay modelos guardados")
        return
    
    print(f"📁 MODELOS DISPONIBLES:")
    print(f"{'='*60}")
    
    for archivo in archivos:
        nombre_modelo = archivo.replace('_metadatos.pkl', '')
        ruta_metadatos = os.path.join(directorio, archivo)
        
        try:
            with open(ruta_metadatos, 'rb') as f:
                metadatos = pickle.load(f)
            
            print(f"\n💾 {nombre_modelo}")
            print(f"   🏗️  Arquitectura: {metadatos['arquitectura']}")
            print(f"   📁 Conjunto: {metadatos.get('conjunto_datos', 'N/A')}")
            print(f"   📊 Dimensiones: {metadatos['dimension_entrada']} -> {metadatos['dimension_latente']}")
            
        except Exception as e:
            print(f"\n💾 {nombre_modelo} (error leyendo metadatos)")


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Autocodificador de Caracteres TP3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

🔹 ENTRENAR UN MODELO:
  python3 autocodificador_caracteres.py entrenar --conjunto 1 --arquitectura estandar --modelo mi_modelo

🔹 ANALIZAR UN CARÁCTER:
  python3 autocodificador_caracteres.py analizar --modelo mi_modelo --caracter A
  python3 autocodificador_caracteres.py analizar --modelo mi_modelo --caracter 0

🔹 GENERAR NUEVOS CARACTERES:
  python3 autocodificador_caracteres.py generar --modelo mi_modelo --metodo aleatorio --cantidad 3
  python3 autocodificador_caracteres.py generar --modelo mi_modelo --metodo interpolacion --caracter1 A --caracter2 B
  python3 autocodificador_caracteres.py generar --modelo mi_modelo --metodo interpolacion --caracter1 0 --caracter2 1

🔹 LISTAR MODELOS:
  python3 autocodificador_caracteres.py listar
        """
    )
    
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponibles')
    
    # Comando entrenar
    parser_entrenar = subparsers.add_parser('entrenar', help='Entrenar un nuevo modelo')
    parser_entrenar.add_argument('--conjunto', type=int, choices=[1, 2, 3], required=True,
                                help='Conjunto de datos (1, 2, o 3)')
    parser_entrenar.add_argument('--arquitectura', choices=['minima', 'estandar', 'compacta', 'profunda', 'amplia', 'experimental'],
                                required=True, help='Arquitectura del autocodificador')
    parser_entrenar.add_argument('--modelo', required=True, help='Nombre del modelo a guardar')
    parser_entrenar.add_argument('--epocas', type=int, default=500, help='Número máximo de épocas (default: 500)')
    parser_entrenar.add_argument('--tasa-aprendizaje', type=float, default=0.001, help='Tasa de aprendizaje (default: 0.001)')
    parser_entrenar.add_argument('--error-objetivo', type=float, default=0.01, help='Error objetivo (default: 0.01)')
    
    # Comando analizar
    parser_analizar = subparsers.add_parser('analizar', help='Analizar un carácter específico')
    parser_analizar.add_argument('--modelo', required=True, help='Nombre del modelo a cargar')
    parser_analizar.add_argument('--caracter', required=True, help='Carácter a analizar (símbolo como "A" o índice como 0)')
    
    # Comando generar
    parser_generar = subparsers.add_parser('generar', help='Generar nuevos caracteres')
    parser_generar.add_argument('--modelo', required=True, help='Nombre del modelo a cargar')
    parser_generar.add_argument('--metodo', choices=['aleatorio', 'gaussiano', 'interpolacion'], 
                               default='aleatorio', help='Método de generación')
    parser_generar.add_argument('--cantidad', type=int, default=3, help='Cantidad de caracteres a generar')
    parser_generar.add_argument('--caracter1', help='Primer carácter para interpolación (símbolo como "A" o índice como 0)')
    parser_generar.add_argument('--caracter2', help='Segundo carácter para interpolación (símbolo como "B" o índice como 1)')
    parser_generar.add_argument('--pasos', type=int, default=5, help='Pasos de interpolación')
    
    # Comando listar
    parser_listar = subparsers.add_parser('listar', help='Listar modelos guardados')
    
    args = parser.parse_args()
    
    if args.comando == 'entrenar':
        return entrenar_modelo_cli(args)
    elif args.comando == 'analizar':
        return analizar_caracter_cli(args)
    elif args.comando == 'generar':
        return generar_caracter_cli(args)
    elif args.comando == 'listar':
        listar_modelos_cli()
        return True
    else:
        parser.print_help()
        return False


if __name__ == "__main__":
    main()
