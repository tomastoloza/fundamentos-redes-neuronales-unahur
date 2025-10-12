import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

from .generador_datos_energia import GeneradorDatosEnergia
from .autocodificador import AutocodificadorAnomalias


class EntrenadorAnomalias:
    def __init__(self, longitud_serie=168, dimension_latente=16, directorio_datos='tp3/datos', 
                 directorio_modelos='tp3/modelos'):
        self.longitud_serie = longitud_serie
        self.dimension_latente = dimension_latente
        
        # Ensure paths are relative to tp3 directory, not src
        if os.path.basename(os.getcwd()) == 'src':
            self.directorio_datos = os.path.join('..', directorio_datos)
            self.directorio_modelos = os.path.join('..', directorio_modelos)
        else:
            self.directorio_datos = directorio_datos
            self.directorio_modelos = directorio_modelos
        
        self.generador_datos = GeneradorDatosEnergia(longitud_serie=longitud_serie)
        self.autocodificador = AutocodificadorAnomalias(
            longitud_serie=longitud_serie, 
            dimension_latente=dimension_latente
        )
        
        self.datos_entrenamiento = None
        self.datos_validacion = None
        self.datos_prueba = None
        self.metadatos_prueba = None
        self.datos_normalizacion = None
        
        os.makedirs(self.directorio_datos, exist_ok=True)
        os.makedirs(self.directorio_modelos, exist_ok=True)
    
    def generar_y_preparar_datos(self, num_entrenamiento=1000, num_prueba_normal=200, 
                                num_prueba_anomala=50, guardar=True):
        
        datos_entrenamiento, metadatos_entrenamiento = self.generador_datos.generar_conjunto_entrenamiento(
            num_muestras=num_entrenamiento
        )
        
        datos_prueba, metadatos_prueba = self.generador_datos.generar_conjunto_prueba(
            num_normales=num_prueba_normal,
            num_anomalas=num_prueba_anomala
        )
        
        datos_entrenamiento_norm, media_train, std_train = self.generador_datos.normalizar_datos(datos_entrenamiento)
        datos_prueba_norm, _, _ = self.generador_datos.normalizar_datos(datos_prueba)
        
        self.datos_entrenamiento = datos_entrenamiento_norm
        self.datos_prueba = datos_prueba_norm
        self.metadatos_prueba = metadatos_prueba
        self.datos_normalizacion = {'media': media_train, 'std': std_train}
        
        if guardar:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            archivo_train = os.path.join(self.directorio_datos, f"energia_entrenamiento_{timestamp}")
            archivo_test = os.path.join(self.directorio_datos, f"energia_prueba_{timestamp}")
            
            self.generador_datos.guardar_datos(datos_entrenamiento, metadatos_entrenamiento, archivo_train)
            self.generador_datos.guardar_datos(datos_prueba, metadatos_prueba, archivo_test)
            
        return self.datos_entrenamiento, self.datos_prueba, self.metadatos_prueba
    
    def entrenar_modelo(self, validation_split=0.2, epochs=100, batch_size=32, 
                       learning_rate=0.001, patience=15, percentil_umbral=95):
        
        if self.datos_entrenamiento is None:
            raise ValueError("Debe generar los datos primero usando generar_y_preparar_datos()")
        
        self.autocodificador.crear_arquitectura()
        self.autocodificador.compilar_modelo(learning_rate=learning_rate)
        
        self.autocodificador.modelo.summary()
        
        historial = self.autocodificador.entrenar(
            self.datos_entrenamiento,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=1
        )
        
        num_validacion = int(len(self.datos_entrenamiento) * validation_split)
        datos_validacion = self.datos_entrenamiento[-num_validacion:]
        
        umbral = self.autocodificador.establecer_umbral_anomalia(datos_validacion, percentil_umbral)
        
        self.autocodificador.datos_normalizacion = self.datos_normalizacion
        
        return historial, umbral
    
    def evaluar_modelo(self, mostrar_visualizaciones=True):
        if self.datos_prueba is None or self.metadatos_prueba is None:
            raise ValueError("Debe tener datos de prueba disponibles")
        
        etiquetas_reales = np.array([meta['es_anomalo'] for meta in self.metadatos_prueba])
        
        metricas = self.autocodificador.evaluar_deteccion(self.datos_prueba, etiquetas_reales)
        
        if mostrar_visualizaciones:
            self.autocodificador.visualizar_entrenamiento()
            self.autocodificador.visualizar_deteccion(self.datos_prueba, etiquetas_reales)
        
        return metricas
    
    def generar_muestras_sinteticas(self, num_muestras=5, mostrar_visualizacion=True):
        muestras_sinteticas, vectores_latentes = self.autocodificador.generar_muestra_sintetica(num_muestras)
        
        if self.datos_normalizacion:
            muestras_desnormalizadas = self.generador_datos.desnormalizar_datos(
                muestras_sinteticas, 
                self.datos_normalizacion['media'][:num_muestras], 
                self.datos_normalizacion['std'][:num_muestras]
            )
        else:
            muestras_desnormalizadas = muestras_sinteticas
        
        if mostrar_visualizacion:
            self.visualizar_muestras_sinteticas(muestras_desnormalizadas)
        
        return muestras_sinteticas, muestras_desnormalizadas, vectores_latentes
    
    def generar_interpolaciones(self, indices_muestras=None, num_pasos=10, mostrar_visualizacion=True):
        if indices_muestras is None:
            indices_normales = [i for i, meta in enumerate(self.metadatos_prueba) if not meta['es_anomalo']]
            indices_muestras = np.random.choice(indices_normales, 2, replace=False)
        
        muestra1 = self.datos_prueba[indices_muestras[0]]
        muestra2 = self.datos_prueba[indices_muestras[1]]
        
        interpolaciones = self.autocodificador.generar_desde_interpolacion(muestra1, muestra2, num_pasos)
        
        if mostrar_visualizacion:
            self.visualizar_interpolaciones(interpolaciones, muestra1, muestra2)
        
        return interpolaciones
    
    def visualizar_muestras_sinteticas(self, muestras):
        num_muestras = len(muestras)
        fig, axes = plt.subplots(1, num_muestras, figsize=(15, 3))
        
        if num_muestras == 1:
            axes = [axes]
        
        for i, muestra in enumerate(muestras):
            axes[i].plot(muestra)
            axes[i].set_title(f'Muestra Sintética {i+1}')
            axes[i].set_xlabel('Tiempo (horas)')
            axes[i].set_ylabel('Consumo Energético')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualizar_interpolaciones(self, interpolaciones, muestra1, muestra2):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        for i, interpolacion in enumerate(interpolaciones):
            row = i // 5
            col = i % 5
            axes[row, col].plot(interpolacion)
            axes[row, col].set_title(f'Paso {i+1}')
            axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        plt.plot(muestra1, label='Muestra Original 1', linewidth=2, alpha=0.7)
        plt.plot(muestra2, label='Muestra Original 2', linewidth=2, alpha=0.7)
        
        for i, interpolacion in enumerate(interpolaciones[1:-1]):
            alpha = 0.3 + 0.4 * (i / len(interpolaciones))
            plt.plot(interpolacion, alpha=alpha, color='green', linewidth=1)
        
        plt.title('Interpolación entre Muestras')
        plt.xlabel('Tiempo (horas)')
        plt.ylabel('Consumo Energético')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def guardar_modelo_completo(self, nombre_base=None):
        if nombre_base is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_base = f"tp3_anomalias_{timestamp}"
        
        # Asegurar que el directorio existe
        os.makedirs(self.directorio_modelos, exist_ok=True)
        
        ruta_completa = os.path.join(self.directorio_modelos, nombre_base)
        
        try:
            nombre_modelo = self.autocodificador.guardar_modelo(ruta_completa)
            print(f"Modelo guardado como: {nombre_modelo}")
            return nombre_modelo
        except Exception as e:
            print(f"Error al guardar modelo: {e}")
            # Intentar guardar directamente con Keras
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_fallback = os.path.join(self.directorio_modelos, f"anomalias_lat{self.dimension_latente}_{timestamp}.keras")
            self.autocodificador.modelo.save(nombre_fallback)
            print(f"Modelo guardado (fallback) como: {nombre_fallback}")
            return nombre_fallback
    
    def ejecutar_experimento_completo(self, config_datos=None, config_entrenamiento=None, 
                                    config_evaluacion=None):
        
        if config_datos is None:
            config_datos = {
                'num_entrenamiento': 1000,
                'num_prueba_normal': 200,
                'num_prueba_anomala': 50
            }
        
        if config_entrenamiento is None:
            config_entrenamiento = {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'patience': 15,
                'percentil_umbral': 95
            }
        
        if config_evaluacion is None:
            config_evaluacion = {
                'num_muestras_sinteticas': 5,
                'num_interpolaciones': 10,
                'mostrar_visualizaciones': True
            }
        
        self.generar_y_preparar_datos(**config_datos)
        
        historial, umbral = self.entrenar_modelo(**config_entrenamiento)
        
        metricas = self.evaluar_modelo(config_evaluacion['mostrar_visualizaciones'])
        
        muestras_sint, muestras_desnorm, vectores = self.generar_muestras_sinteticas(
            config_evaluacion['num_muestras_sinteticas'],
            config_evaluacion['mostrar_visualizaciones']
        )
        
        interpolaciones = self.generar_interpolaciones(
            num_pasos=config_evaluacion['num_interpolaciones'],
            mostrar_visualizacion=config_evaluacion['mostrar_visualizaciones']
        )
        
        nombre_modelo = self.guardar_modelo_completo()
        
        resultados = {
            'historial_entrenamiento': historial,
            'umbral_anomalia': umbral,
            'metricas_evaluacion': metricas,
            'muestras_sinteticas': muestras_sint,
            'interpolaciones': interpolaciones,
            'nombre_modelo': nombre_modelo
        }
        
        return resultados


def main():
    print("=== ENTRENADOR DE DETECCIÓN DE ANOMALÍAS ===")
    print("Iniciando experimento completo de detección de anomalías en consumo energético...")
    
    entrenador = EntrenadorAnomalias(
        longitud_serie=168,
        dimension_latente=16
    )
    
    try:
        resultados = entrenador.ejecutar_experimento_completo()
        
        print("\n=== EXPERIMENTO COMPLETADO EXITOSAMENTE ===")
        print(f"Modelo guardado: {resultados['nombre_modelo']}")
        print(f"Umbral de anomalía: {resultados['umbral_anomalia']:.4f}")
        
        metricas = resultados['metricas_evaluacion']
        print(f"Precisión: {metricas['precision']:.3f}")
        print(f"Recall: {metricas['recall']:.3f}")
        print(f"F1-Score: {metricas['f1_score']:.3f}")
        
    except Exception as e:
        print(f"Error durante el experimento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
