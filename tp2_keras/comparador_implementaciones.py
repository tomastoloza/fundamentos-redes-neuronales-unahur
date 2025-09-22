"""
Módulo comparador entre implementaciones personalizadas y TensorFlow/Keras.
Ejecuta todos los experimentos del TP2 en ambas implementaciones y genera análisis comparativo.
"""

import numpy as np
import time
import os
import sys
import warnings

# Suprimir warnings de TensorFlow para output más limpio
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configurar path para acceder a módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importar implementaciones Keras
from tp2_keras.xor_keras import XORKeras, ejecutar_experimento_xor_keras
from tp2_keras.discriminacion_pares_keras import DiscriminacionParesKeras, ejecutar_experimento_discriminacion_keras
from tp2_keras.clasificacion_10_clases_keras import Clasificacion10ClasesKeras, ejecutar_experimento_clasificacion_10_clases_keras

# Importar implementaciones personalizadas
try:
    from tp2.src.main_tp2 import EjecutorTP2
    from tp2.src.perceptron_multicapa import PerceptronMulticapa
    from tp2.src.cargador_datos_digitos import CargadorDatosDigitos
    from comun.src.funciones_activacion import FuncionesActivacion
except ImportError as e:
    print(f"⚠️  Error importando módulos personalizados: {e}")
    print("Asegúrate de que el proyecto esté correctamente configurado.")
    sys.exit(1)


class ComparadorImplementaciones:
    """
    Comparador entre implementaciones personalizadas y TensorFlow/Keras.
    Ejecuta experimentos paralelos y genera análisis comparativo detallado.
    """
    
    def __init__(self):
        """Inicializa el comparador."""
        self.resultados_personalizados = {}
        self.resultados_keras = {}
        self.comparaciones = {}
    
    def ejecutar_comparacion_xor(self):
        """Compara implementaciones del problema XOR."""
        print("🔥 COMPARACIÓN: PROBLEMA XOR")
        print("=" * 60)
        
        # Ejecutar implementación Keras
        print("\n📊 EJECUTANDO IMPLEMENTACIÓN KERAS...")
        print("-" * 40)
        try:
            self.resultados_keras['xor'] = ejecutar_experimento_xor_keras()
        except Exception as e:
            print(f"❌ Error en implementación Keras: {e}")
            self.resultados_keras['xor'] = None
        
        # Ejecutar implementación personalizada
        print("\n🧠 EJECUTANDO IMPLEMENTACIÓN PERSONALIZADA...")
        print("-" * 40)
        try:
            # Crear datos XOR
            datos_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            salidas_xor = np.array([[-1], [1], [1], [-1]])  # Formato personalizado usa -1/1
            
            # Crear red personalizada
            red_personalizada = PerceptronMulticapa(
                arquitectura=[2, 4, 1],
                funciones_activacion=[FuncionesActivacion.sigmoide, FuncionesActivacion.sigmoide]
            )
            
            tiempo_inicio = time.time()
            convergio, epoca = red_personalizada.entrenar(
                datos_xor, salidas_xor,
                tasa_aprendizaje=0.1,
                max_epocas=2000,
                tolerancia_error=0.01
            )
            tiempo_entrenamiento = time.time() - tiempo_inicio
            
            # Evaluar
            predicciones = red_personalizada.predecir(datos_xor)
            predicciones_binarias = np.where(predicciones > 0, 1, -1)
            accuracy = np.mean(predicciones_binarias.flatten() == salidas_xor.flatten())
            mse = np.mean((predicciones - salidas_xor) ** 2)
            
            self.resultados_personalizados['xor'] = {
                'convergio': convergio,
                'epocas': epoca,
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'accuracy': accuracy,
                'mse': mse,
                'error_final': mse
            }
            
            print(f"✓ Implementación personalizada completada:")
            print(f"  - Convergencia: {'Sí' if convergio else 'No'}")
            print(f"  - Épocas: {epoca}")
            print(f"  - Precisión: {accuracy:.1%}")
            print(f"  - Tiempo: {tiempo_entrenamiento:.2f}s")
            
        except Exception as e:
            print(f"❌ Error en implementación personalizada: {e}")
            self.resultados_personalizados['xor'] = None
        
        # Generar comparación
        self._generar_comparacion_xor()
    
    def ejecutar_comparacion_discriminacion_pares(self):
        """Compara implementaciones de discriminación de números pares."""
        print("\n🔥 COMPARACIÓN: DISCRIMINACIÓN DE NÚMEROS PARES")
        print("=" * 60)
        
        # Ejecutar implementación Keras
        print("\n📊 EJECUTANDO IMPLEMENTACIÓN KERAS...")
        print("-" * 40)
        try:
            self.resultados_keras['pares'] = ejecutar_experimento_discriminacion_keras()
        except Exception as e:
            print(f"❌ Error en implementación Keras: {e}")
            self.resultados_keras['pares'] = None
        
        # Ejecutar implementación personalizada (arquitectura MINIMA que fue la mejor)
        print("\n🧠 EJECUTANDO IMPLEMENTACIÓN PERSONALIZADA (MINIMA)...")
        print("-" * 40)
        try:
            # Cargar datos
            cargador = CargadorDatosDigitos()
            cargador.cargar_datos_tp2()
            
            # Preparar datos para clasificación binaria
            entradas, salidas = cargador.preparar_datos_clasificacion_binaria(
                digitos_positivos=[0, 2, 4, 6],  # Pares
                digitos_negativos=[1, 3]         # Impares
            )
            
            # Datos de prueba
            entradas_test = []
            salidas_test = []
            for digito in [8]:  # Par
                patron = cargador.obtener_patron_digito(digito)
                if patron is not None:
                    entradas_test.append(patron)
                    salidas_test.append(1)
            
            for digito in [5, 7, 9]:  # Impares
                patron = cargador.obtener_patron_digito(digito)
                if patron is not None:
                    entradas_test.append(patron)
                    salidas_test.append(-1)
            
            entradas_test = np.array(entradas_test)
            salidas_test = np.array(salidas_test).reshape(-1, 1)
            
            # Crear red personalizada (arquitectura MINIMA)
            red_personalizada = PerceptronMulticapa(
                arquitectura=[35, 10, 1],
                funciones_activacion=[FuncionesActivacion.sigmoide, FuncionesActivacion.sigmoide]
            )
            
            tiempo_inicio = time.time()
            convergio, epoca = red_personalizada.entrenar(
                entradas, salidas,
                tasa_aprendizaje=0.01,
                max_epocas=1000,
                tolerancia_error=0.01
            )
            tiempo_entrenamiento = time.time() - tiempo_inicio
            
            # Evaluar en entrenamiento
            pred_train = red_personalizada.predecir(entradas)
            pred_train_bin = np.where(pred_train > 0, 1, -1)
            accuracy_train = np.mean(pred_train_bin.flatten() == salidas.flatten())
            
            # Evaluar en prueba
            pred_test = red_personalizada.predecir(entradas_test)
            pred_test_bin = np.where(pred_test > 0, 1, -1)
            accuracy_test = np.mean(pred_test_bin.flatten() == salidas_test.flatten())
            
            mse = np.mean((pred_train - salidas) ** 2)
            
            self.resultados_personalizados['pares'] = {
                'MINIMA': {
                    'arquitectura': [35, 10, 1],
                    'convergio': convergio,
                    'epocas': epoca,
                    'tiempo_entrenamiento': tiempo_entrenamiento,
                    'accuracy_train': accuracy_train,
                    'accuracy_test': accuracy_test,
                    'diferencia_accuracy': accuracy_train - accuracy_test,
                    'error_final': mse
                }
            }
            
            print(f"✓ Implementación personalizada completada:")
            print(f"  - Convergencia: {'Sí' if convergio else 'No'}")
            print(f"  - Épocas: {epoca}")
            print(f"  - Precisión entrenamiento: {accuracy_train:.1%}")
            print(f"  - Precisión prueba: {accuracy_test:.1%}")
            print(f"  - Tiempo: {tiempo_entrenamiento:.2f}s")
            
        except Exception as e:
            print(f"❌ Error en implementación personalizada: {e}")
            self.resultados_personalizados['pares'] = None
        
        # Generar comparación
        self._generar_comparacion_pares()
    
    def ejecutar_comparacion_clasificacion_10_clases(self):
        """Compara implementaciones de clasificación de 10 clases."""
        print("\n🔥 COMPARACIÓN: CLASIFICACIÓN 10 CLASES")
        print("=" * 60)
        
        # Ejecutar implementación Keras
        print("\n📊 EJECUTANDO IMPLEMENTACIÓN KERAS...")
        print("-" * 40)
        try:
            self.resultados_keras['10_clases'] = ejecutar_experimento_clasificacion_10_clases_keras()
        except Exception as e:
            print(f"❌ Error en implementación Keras: {e}")
            self.resultados_keras['10_clases'] = None
        
        # Ejecutar implementación personalizada
        print("\n🧠 EJECUTANDO IMPLEMENTACIÓN PERSONALIZADA...")
        print("-" * 40)
        try:
            # Cargar datos
            cargador = CargadorDatosDigitos()
            X_train, y_train, X_test, y_test = cargador.crear_division_entrenamiento_prueba_estandar()
            
            # Convertir etiquetas a one-hot para implementación personalizada
            def to_one_hot(labels, num_classes=10):
                one_hot = np.zeros((len(labels), num_classes))
                for i, label in enumerate(labels):
                    one_hot[i, label] = 1
                return one_hot
            
            y_train_onehot = to_one_hot(y_train)
            y_test_onehot = to_one_hot(y_test)
            
            # Crear red personalizada
            red_personalizada = PerceptronMulticapa(
                arquitectura=[35, 20, 15, 10],
                funciones_activacion=[
                    FuncionesActivacion.sigmoide, 
                    FuncionesActivacion.sigmoide, 
                    FuncionesActivacion.sigmoide
                ]
            )
            
            tiempo_inicio = time.time()
            convergio, epoca = red_personalizada.entrenar(
                X_train, y_train_onehot,
                tasa_aprendizaje=0.01,
                max_epocas=1000,
                tolerancia_error=0.01
            )
            tiempo_entrenamiento = time.time() - tiempo_inicio
            
            # Evaluar en entrenamiento
            pred_train = red_personalizada.predecir(X_train)
            pred_train_classes = np.argmax(pred_train, axis=1)
            accuracy_train = np.mean(pred_train_classes == y_train)
            
            # Evaluar en prueba
            pred_test = red_personalizada.predecir(X_test)
            pred_test_classes = np.argmax(pred_test, axis=1)
            accuracy_test = np.mean(pred_test_classes == y_test)
            
            # Evaluar con ruido
            X_train_ruido = cargador.generar_datos_con_ruido(X_train, 0.02)
            X_test_ruido = cargador.generar_datos_con_ruido(X_test, 0.02)
            
            pred_train_ruido = red_personalizada.predecir(X_train_ruido)
            pred_train_ruido_classes = np.argmax(pred_train_ruido, axis=1)
            accuracy_train_ruido = np.mean(pred_train_ruido_classes == y_train)
            
            pred_test_ruido = red_personalizada.predecir(X_test_ruido)
            pred_test_ruido_classes = np.argmax(pred_test_ruido, axis=1)
            accuracy_test_ruido = np.mean(pred_test_ruido_classes == y_test)
            
            mse = np.mean((pred_train - y_train_onehot) ** 2)
            
            self.resultados_personalizados['10_clases'] = {
                'convergio': convergio,
                'epocas': epoca,
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'diferencia_accuracy': accuracy_train - accuracy_test,
                'accuracy_train_ruido': accuracy_train_ruido,
                'accuracy_test_ruido': accuracy_test_ruido,
                'degradacion_train': accuracy_train - accuracy_train_ruido,
                'degradacion_test': accuracy_test - accuracy_test_ruido,
                'error_final': mse
            }
            
            print(f"✓ Implementación personalizada completada:")
            print(f"  - Convergencia: {'Sí' if convergio else 'No'}")
            print(f"  - Épocas: {epoca}")
            print(f"  - Precisión entrenamiento: {accuracy_train:.1%}")
            print(f"  - Precisión prueba: {accuracy_test:.1%}")
            print(f"  - Degradación por ruido (train): {accuracy_train - accuracy_train_ruido:.1%}")
            print(f"  - Tiempo: {tiempo_entrenamiento:.2f}s")
            
        except Exception as e:
            print(f"❌ Error en implementación personalizada: {e}")
            self.resultados_personalizados['10_clases'] = None
        
        # Generar comparación
        self._generar_comparacion_10_clases()
    
    def _generar_comparacion_xor(self):
        """Genera comparación detallada para el problema XOR."""
        print("\n🎯 COMPARACIÓN DETALLADA: XOR")
        print("=" * 50)
        
        if self.resultados_keras['xor'] and self.resultados_personalizados['xor']:
            keras_res = self.resultados_keras['xor']
            personal_res = self.resultados_personalizados['xor']
            
            print(f"{'Métrica':<20} {'Keras':<15} {'Personalizada':<15} {'Diferencia'}")
            print("-" * 65)
            print(f"{'Convergencia':<20} {'Sí' if keras_res['convergio'] else 'No':<15} {'Sí' if personal_res['convergio'] else 'No':<15} {'-'}")
            print(f"{'Épocas':<20} {keras_res['epocas']:<15} {personal_res['epocas']:<15} {keras_res['epocas'] - personal_res['epocas']:+d}")
            print(f"{'Precisión':<20} {keras_res['accuracy']:.1%:<15} {personal_res['accuracy']:.1%:<15} {keras_res['accuracy'] - personal_res['accuracy']:+.1%}")
            print(f"{'Error final':<20} {keras_res['error_final']:.6f:<15} {personal_res['error_final']:.6f:<15} {keras_res['error_final'] - personal_res['error_final']:+.6f}")
            print(f"{'Tiempo (s)':<20} {keras_res['tiempo_entrenamiento']:.2f:<15} {personal_res['tiempo_entrenamiento']:.2f:<15} {keras_res['tiempo_entrenamiento'] - personal_res['tiempo_entrenamiento']:+.2f}")
            
            # Análisis
            print("\n📋 ANÁLISIS:")
            if keras_res['accuracy'] >= 0.99 and personal_res['accuracy'] >= 0.99:
                print("✓ Ambas implementaciones resuelven correctamente el problema XOR")
            
            if abs(keras_res['epocas'] - personal_res['epocas']) < 100:
                print("✓ Convergencia similar en ambas implementaciones")
            elif keras_res['epocas'] < personal_res['epocas']:
                print("📊 Keras converge más rápido")
            else:
                print("🧠 Implementación personalizada converge más rápido")
        else:
            print("❌ No se pueden comparar los resultados debido a errores en la ejecución")
    
    def _generar_comparacion_pares(self):
        """Genera comparación detallada para discriminación de pares."""
        print("\n🎯 COMPARACIÓN DETALLADA: DISCRIMINACIÓN PARES")
        print("=" * 55)
        
        if self.resultados_keras['pares'] and self.resultados_personalizados['pares']:
            # Comparar con la mejor arquitectura de Keras (MINIMA)
            keras_minima = self.resultados_keras['pares']['MINIMA']
            personal_minima = self.resultados_personalizados['pares']['MINIMA']
            
            print(f"{'Métrica':<25} {'Keras':<15} {'Personalizada':<15} {'Diferencia'}")
            print("-" * 70)
            print(f"{'Épocas':<25} {keras_minima['epocas']:<15} {personal_minima['epocas']:<15} {keras_minima['epocas'] - personal_minima['epocas']:+d}")
            print(f"{'Precisión Train':<25} {keras_minima['accuracy_train']:.1%:<15} {personal_minima['accuracy_train']:.1%:<15} {keras_minima['accuracy_train'] - personal_minima['accuracy_train']:+.1%}")
            print(f"{'Precisión Test':<25} {keras_minima['accuracy_test']:.1%:<15} {personal_minima['accuracy_test']:.1%:<15} {keras_minima['accuracy_test'] - personal_minima['accuracy_test']:+.1%}")
            print(f"{'Sobreajuste':<25} {keras_minima['diferencia_accuracy']:.1%:<15} {personal_minima['diferencia_accuracy']:.1%:<15} {keras_minima['diferencia_accuracy'] - personal_minima['diferencia_accuracy']:+.1%}")
            print(f"{'Tiempo (s)':<25} {keras_minima['tiempo_entrenamiento']:.2f:<15} {personal_minima['tiempo_entrenamiento']:.2f:<15} {keras_minima['tiempo_entrenamiento'] - personal_minima['tiempo_entrenamiento']:+.2f}")
            
            # Análisis
            print("\n📋 ANÁLISIS:")
            if keras_minima['accuracy_test'] > 0.7 and personal_minima['accuracy_test'] > 0.7:
                print("✓ Ambas implementaciones logran buena generalización")
            
            mejor_keras = keras_minima['accuracy_test']
            mejor_personal = personal_minima['accuracy_test']
            
            if mejor_keras > mejor_personal:
                print("📊 Keras obtiene mejor precisión en prueba")
            elif mejor_personal > mejor_keras:
                print("🧠 Implementación personalizada obtiene mejor precisión en prueba")
            else:
                print("⚖️  Rendimiento similar en ambas implementaciones")
        else:
            print("❌ No se pueden comparar los resultados debido a errores en la ejecución")
    
    def _generar_comparacion_10_clases(self):
        """Genera comparación detallada para clasificación de 10 clases."""
        print("\n🎯 COMPARACIÓN DETALLADA: CLASIFICACIÓN 10 CLASES")
        print("=" * 60)
        
        if self.resultados_keras['10_clases'] and self.resultados_personalizados['10_clases']:
            keras_res = self.resultados_keras['10_clases']
            personal_res = self.resultados_personalizados['10_clases']
            
            print(f"{'Métrica':<25} {'Keras':<15} {'Personalizada':<15} {'Diferencia'}")
            print("-" * 70)
            print(f"{'Épocas':<25} {keras_res['epocas']:<15} {personal_res['epocas']:<15} {keras_res['epocas'] - personal_res['epocas']:+d}")
            print(f"{'Precisión Train':<25} {keras_res['accuracy_train']:.1%:<15} {personal_res['accuracy_train']:.1%:<15} {keras_res['accuracy_train'] - personal_res['accuracy_train']:+.1%}")
            print(f"{'Precisión Test':<25} {keras_res['accuracy_test']:.1%:<15} {personal_res['accuracy_test']:.1%:<15} {keras_res['accuracy_test'] - personal_res['accuracy_test']:+.1%}")
            print(f"{'Sobreajuste':<25} {keras_res['diferencia_accuracy']:.1%:<15} {personal_res['diferencia_accuracy']:.1%:<15} {keras_res['diferencia_accuracy'] - personal_res['diferencia_accuracy']:+.1%}")
            print(f"{'Degradación Ruido Train':<25} {keras_res['degradacion_train']:.1%:<15} {personal_res['degradacion_train']:.1%:<15} {keras_res['degradacion_train'] - personal_res['degradacion_train']:+.1%}")
            print(f"{'Degradación Ruido Test':<25} {keras_res['degradacion_test']:.1%:<15} {personal_res['degradacion_test']:.1%:<15} {keras_res['degradacion_test'] - personal_res['degradacion_test']:+.1%}")
            print(f"{'Tiempo (s)':<25} {keras_res['tiempo_entrenamiento']:.2f:<15} {personal_res['tiempo_entrenamiento']:.2f:<15} {keras_res['tiempo_entrenamiento'] - personal_res['tiempo_entrenamiento']:+.2f}")
            
            # Análisis
            print("\n📋 ANÁLISIS:")
            if keras_res['accuracy_train'] > 0.9 and personal_res['accuracy_train'] > 0.9:
                print("✓ Ambas implementaciones memorizan bien los datos de entrenamiento")
            
            if keras_res['accuracy_test'] < 0.5 and personal_res['accuracy_test'] < 0.5:
                print("⚠️  Ambas implementaciones sufren de sobreajuste severo")
            
            if keras_res['degradacion_train'] < 0.1 and personal_res['degradacion_train'] < 0.1:
                print("✓ Ambas implementaciones son robustas al ruido en datos conocidos")
            
            if keras_res['tiempo_entrenamiento'] < personal_res['tiempo_entrenamiento']:
                print("📊 Keras es más eficiente en tiempo de entrenamiento")
            else:
                print("🧠 Implementación personalizada es más eficiente en tiempo")
        else:
            print("❌ No se pueden comparar los resultados debido a errores en la ejecución")
    
    def generar_resumen_final(self):
        """Genera un resumen final de todas las comparaciones."""
        print("\n" + "="*80)
        print("🏆 RESUMEN FINAL DE COMPARACIONES")
        print("="*80)
        
        print("\n📊 KERAS vs 🧠 IMPLEMENTACIÓN PERSONALIZADA")
        print("-" * 50)
        
        # Análisis por experimento
        experimentos = ['xor', 'pares', '10_clases']
        nombres = ['XOR', 'Discriminación Pares', 'Clasificación 10 Clases']
        
        for exp, nombre in zip(experimentos, nombres):
            print(f"\n{nombre}:")
            
            if exp in self.resultados_keras and exp in self.resultados_personalizados:
                keras_ok = self.resultados_keras[exp] is not None
                personal_ok = self.resultados_personalizados[exp] is not None
                
                if keras_ok and personal_ok:
                    print("  ✓ Ambas implementaciones ejecutadas exitosamente")
                    
                    if exp == 'xor':
                        keras_acc = self.resultados_keras[exp]['accuracy']
                        personal_acc = self.resultados_personalizados[exp]['accuracy']
                        if keras_acc >= 0.99 and personal_acc >= 0.99:
                            print("  ✓ Ambas resuelven correctamente el problema")
                    
                    elif exp == 'pares':
                        keras_test = self.resultados_keras[exp]['MINIMA']['accuracy_test']
                        personal_test = self.resultados_personalizados[exp]['MINIMA']['accuracy_test']
                        if keras_test > personal_test:
                            print("  📊 Keras obtiene mejor generalización")
                        elif personal_test > keras_test:
                            print("  🧠 Implementación personalizada obtiene mejor generalización")
                        else:
                            print("  ⚖️  Generalización similar")
                    
                    elif exp == '10_clases':
                        keras_over = self.resultados_keras[exp]['diferencia_accuracy']
                        personal_over = self.resultados_personalizados[exp]['diferencia_accuracy']
                        if keras_over < personal_over:
                            print("  📊 Keras tiene menos sobreajuste")
                        elif personal_over < keras_over:
                            print("  🧠 Implementación personalizada tiene menos sobreajuste")
                        else:
                            print("  ⚖️  Sobreajuste similar")
                
                elif keras_ok:
                    print("  📊 Solo Keras ejecutado exitosamente")
                elif personal_ok:
                    print("  🧠 Solo implementación personalizada ejecutada exitosamente")
                else:
                    print("  ❌ Ambas implementaciones fallaron")
            else:
                print("  ❌ Experimento no ejecutado")
        
        # Conclusiones generales
        print("\n🎯 CONCLUSIONES GENERALES:")
        print("-" * 30)
        print("✓ TensorFlow/Keras ofrece:")
        print("  - API más simple y directa")
        print("  - Optimizaciones automáticas")
        print("  - Mejor manejo de memoria")
        print("  - Callbacks y herramientas integradas")
        
        print("\n✓ Implementación personalizada ofrece:")
        print("  - Control total sobre el algoritmo")
        print("  - Comprensión profunda del funcionamiento")
        print("  - Flexibilidad para modificaciones específicas")
        print("  - Valor educativo superior")
        
        print("\n⚖️  Ambas implementaciones:")
        print("  - Producen resultados comparables")
        print("  - Sufren de los mismos problemas fundamentales (sobreajuste)")
        print("  - Demuestran la importancia de la arquitectura y los datos")
        print("  - Confirman los principios teóricos de las redes neuronales")
    
    def ejecutar_comparacion_completa(self):
        """Ejecuta la comparación completa de todas las implementaciones."""
        print("🚀 INICIANDO COMPARACIÓN COMPLETA")
        print("🔥 KERAS vs IMPLEMENTACIÓN PERSONALIZADA")
        print("="*80)
        
        tiempo_inicio = time.time()
        
        # Ejecutar todas las comparaciones
        self.ejecutar_comparacion_xor()
        self.ejecutar_comparacion_discriminacion_pares()
        self.ejecutar_comparacion_clasificacion_10_clases()
        
        # Generar resumen final
        self.generar_resumen_final()
        
        tiempo_total = time.time() - tiempo_inicio
        print(f"\n⏱️  TIEMPO TOTAL DE COMPARACIÓN: {tiempo_total:.2f} segundos")
        print("🎉 COMPARACIÓN COMPLETADA")


def main():
    """Función principal para ejecutar la comparación completa."""
    comparador = ComparadorImplementaciones()
    comparador.ejecutar_comparacion_completa()


if __name__ == "__main__":
    main()
