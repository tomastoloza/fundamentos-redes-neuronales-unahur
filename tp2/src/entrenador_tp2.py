import numpy as np
from typing import Dict, Tuple, List, Optional
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .perceptron_multicapa import PerceptronMulticapa
from .cargador_datos_digitos import CargadorDatosDigitos
from comun.src.utilidades_matematicas import UtilidadesMatematicas
from comun.constantes.constantes_redes_neuronales import (
    ARQUITECTURAS_TP2, PATRONES_XOR_ENTRADA, PATRONES_XOR_SALIDA,
    TASA_APRENDIZAJE_DEFECTO, EPOCAS_MAXIMAS_DEFECTO, ERROR_OBJETIVO_DEFECTO,
    PROBABILIDAD_RUIDO_DEFECTO
)

class EntrenadorTP2:

    def __init__(self, cargador_datos: Optional[CargadorDatosDigitos] = None):

        self.cargador_datos = cargador_datos or CargadorDatosDigitos()
        self.resultados_experimentos = {}
        self.redes_entrenadas = {}
    
    def entrenar_problema_xor(self, arquitectura: List[int] = None,
                            tasa_aprendizaje: float = TASA_APRENDIZAJE_DEFECTO,
                            max_epocas: int = EPOCAS_MAXIMAS_DEFECTO,
                            mostrar_progreso: bool = True) -> Dict:

        if arquitectura is None:
            arquitectura = [2, 4, 1]
        
        if mostrar_progreso:
            print(f"\n🔹 Entrenando Red Neuronal para XOR")
            print(f"Arquitectura: {arquitectura}")
        
        entradas = np.array(PATRONES_XOR_ENTRADA)
        salidas = np.array(PATRONES_XOR_SALIDA).reshape(-1, 1)
        
        salidas_normalizadas = np.where(salidas == 1, 0.9, 0.1)
        
        red = PerceptronMulticapa(arquitectura, ['sigmoide'] * (len(arquitectura) - 1))
        
        convergencia, epoca = red.entrenar(
            entradas=entradas,
            salidas_esperadas=salidas_normalizadas,
            tasa_aprendizaje=tasa_aprendizaje,
            max_epocas=max_epocas,
            mostrar_progreso=mostrar_progreso
        )
        
        predicciones = red.predecir(entradas)
        predicciones_binarias = np.where(predicciones > 0.5, 1, -1)
        
        precision = np.mean(predicciones_binarias.flatten() == salidas.flatten())
        error_cuadratico = UtilidadesMatematicas.calcular_error_cuadratico_medio(
            salidas_normalizadas, predicciones
        )
        
        resultado = {
            'problema': 'XOR',
            'arquitectura': arquitectura,
            'convergencia': convergencia,
            'epoca_convergencia': epoca,
            'precision': precision,
            'error_cuadratico_medio': error_cuadratico,
            'predicciones': predicciones,
            'predicciones_binarias': predicciones_binarias,
            'entradas': entradas,
            'salidas_esperadas': salidas,
            'red': red,
            'info_red': red.obtener_informacion_red()
        }
        
        self.resultados_experimentos['xor'] = resultado
        self.redes_entrenadas['xor'] = red
        
        if mostrar_progreso:
            self._mostrar_resultados_xor(resultado)
        
        return resultado
    
    def entrenar_discriminacion_numeros_pares(self, arquitectura: List[int] = None,
                                            digitos_entrenamiento: List[int] = None,
                                            digitos_prueba: List[int] = None,
                                            tasa_aprendizaje: float = TASA_APRENDIZAJE_DEFECTO,
                                            max_epocas: int = EPOCAS_MAXIMAS_DEFECTO,
                                            mostrar_progreso: bool = True) -> Dict:

        if arquitectura is None:
            arquitectura = ARQUITECTURAS_TP2['MINIMA']
        
        if digitos_entrenamiento is None:
            digitos_entrenamiento = [0, 2, 4, 6, 1, 3]
        
        if digitos_prueba is None:
            digitos_prueba = [5, 7, 8, 9]
        
        if mostrar_progreso:
            print(f"\n🔹 Entrenando Discriminación de Números Pares")
            print(f"Arquitectura: {arquitectura}")
            print(f"Dígitos entrenamiento: {digitos_entrenamiento}")
            print(f"Dígitos prueba: {digitos_prueba}")
        
        try:
            self.cargador_datos.cargar_datos_tp2()
        except FileNotFoundError:
            print("⚠️ Archivo de datos no encontrado. Usando datos simulados.")
            return self._crear_resultado_error("Archivo de datos no encontrado")
        
        digitos_pares_train = [d for d in digitos_entrenamiento if d % 2 == 0]
        digitos_impares_train = [d for d in digitos_entrenamiento if d % 2 == 1]
        
        entradas_train, salidas_train = self.cargador_datos.preparar_datos_clasificacion_binaria(
            digitos_pares_train, digitos_impares_train
        )
        
        digitos_pares_test = [d for d in digitos_prueba if d % 2 == 0]
        digitos_impares_test = [d for d in digitos_prueba if d % 2 == 1]
        
        entradas_test, salidas_test = self.cargador_datos.preparar_datos_clasificacion_binaria(
            digitos_pares_test, digitos_impares_test
        )
        
        salidas_train_norm = np.where(salidas_train == 1, 0.9, 0.1).reshape(-1, 1)
        salidas_test_norm = np.where(salidas_test == 1, 0.9, 0.1).reshape(-1, 1)
        
        red = PerceptronMulticapa(arquitectura, ['sigmoide'] * (len(arquitectura) - 1))
        
        convergencia, epoca = red.entrenar(
            entradas=entradas_train,
            salidas_esperadas=salidas_train_norm,
            tasa_aprendizaje=tasa_aprendizaje,
            max_epocas=max_epocas,
            mostrar_progreso=mostrar_progreso
        )
        
        pred_train = red.predecir(entradas_train)
        precision_train = np.mean((pred_train > 0.5).flatten() == (salidas_train == 1))
        
        pred_test = red.predecir(entradas_test)
        precision_test = np.mean((pred_test > 0.5).flatten() == (salidas_test == 1))
        
        resultado = {
            'problema': 'discriminacion_pares',
            'arquitectura': arquitectura,
            'convergencia': convergencia,
            'epoca_convergencia': epoca,
            'precision_entrenamiento': precision_train,
            'precision_prueba': precision_test,
            'digitos_entrenamiento': digitos_entrenamiento,
            'digitos_prueba': digitos_prueba,
            'entradas_train': entradas_train,
            'salidas_train': salidas_train,
            'entradas_test': entradas_test,
            'salidas_test': salidas_test,
            'predicciones_train': pred_train,
            'predicciones_test': pred_test,
            'red': red,
            'info_red': red.obtener_informacion_red()
        }
        
        self.resultados_experimentos['discriminacion_pares'] = resultado
        self.redes_entrenadas['discriminacion_pares'] = red
        
        if mostrar_progreso:
            self._mostrar_resultados_discriminacion_pares(resultado)
        
        return resultado
    
    def entrenar_clasificacion_10_clases(self, arquitectura: List[int] = None,
                                       digitos_entrenamiento: List[int] = None,
                                       digitos_prueba: List[int] = None,
                                       tasa_aprendizaje: float = TASA_APRENDIZAJE_DEFECTO,
                                       max_epocas: int = EPOCAS_MAXIMAS_DEFECTO,
                                       mostrar_progreso: bool = True) -> Dict:

        if arquitectura is None:
            arquitectura = [35, 20, 15, 10]
        
        if digitos_entrenamiento is None:
            digitos_entrenamiento = [0, 1, 2, 3, 4, 5, 6]
        
        if digitos_prueba is None:
            digitos_prueba = [7, 8, 9]
        
        if mostrar_progreso:
            print(f"\n🔹 Entrenando Clasificación 10 Clases")
            print(f"Arquitectura: {arquitectura}")
            print(f"Dígitos entrenamiento: {digitos_entrenamiento}")
            print(f"Dígitos prueba: {digitos_prueba}")
        
        try:
            self.cargador_datos.cargar_datos_tp2()
        except FileNotFoundError:
            print("⚠️ Archivo de datos no encontrado. Usando datos simulados.")
            return self._crear_resultado_error("Archivo de datos no encontrado")
        
        entradas_train, salidas_train, entradas_test, salidas_test = (
            self.cargador_datos.dividir_datos_por_digitos(
                digitos_entrenamiento, digitos_prueba
            )
        )
        
        salidas_train_encoded = np.zeros((len(salidas_train), 10))
        for i, digito in enumerate(salidas_train):
            salidas_train_encoded[i, digito] = 1
        
        red = PerceptronMulticapa(arquitectura, ['sigmoide'] * (len(arquitectura) - 1))
        
        convergencia, epoca = red.entrenar(
            entradas=entradas_train,
            salidas_esperadas=salidas_train_encoded,
            tasa_aprendizaje=tasa_aprendizaje,
            max_epocas=max_epocas,
            mostrar_progreso=mostrar_progreso
        )
        
        pred_train = red.predecir(entradas_train)
        pred_test = red.predecir(entradas_test)
        
        clases_pred_train = np.argmax(pred_train, axis=1)
        clases_pred_test = np.argmax(pred_test, axis=1)
        
        precision_train = np.mean(clases_pred_train == salidas_train)
        precision_test = np.mean(clases_pred_test == salidas_test)
        
        resultado = {
            'problema': 'clasificacion_10_clases',
            'arquitectura': arquitectura,
            'convergencia': convergencia,
            'epoca_convergencia': epoca,
            'precision_entrenamiento': precision_train,
            'precision_prueba': precision_test,
            'digitos_entrenamiento': digitos_entrenamiento,
            'digitos_prueba': digitos_prueba,
            'entradas_train': entradas_train,
            'salidas_train': salidas_train,
            'entradas_test': entradas_test,
            'salidas_test': salidas_test,
            'predicciones_train': pred_train,
            'predicciones_test': pred_test,
            'clases_predichas_train': clases_pred_train,
            'clases_predichas_test': clases_pred_test,
            'red': red,
            'info_red': red.obtener_informacion_red()
        }
        
        self.resultados_experimentos['clasificacion_10_clases'] = resultado
        self.redes_entrenadas['clasificacion_10_clases'] = red
        
        if mostrar_progreso:
            self._mostrar_resultados_clasificacion_10_clases(resultado)
        
        return resultado
    
    def evaluar_robustez_ruido(self, nombre_experimento: str,
                             probabilidad_ruido: float = PROBABILIDAD_RUIDO_DEFECTO,
                             mostrar_progreso: bool = True) -> Dict:

        if nombre_experimento not in self.redes_entrenadas:
            raise ValueError(f"Experimento '{nombre_experimento}' no encontrado")
        
        red = self.redes_entrenadas[nombre_experimento]
        resultado_original = self.resultados_experimentos[nombre_experimento]
        
        if mostrar_progreso:
            print(f"\n🔹 Evaluando Robustez al Ruido - {nombre_experimento}")
            print(f"Probabilidad de ruido: {probabilidad_ruido}")
        
        if 'entradas_test' in resultado_original:
            entradas_limpias = resultado_original['entradas_test']
            salidas_esperadas = resultado_original['salidas_test']
        else:
            entradas_limpias = resultado_original['entradas']
            salidas_esperadas = resultado_original['salidas_esperadas']
        
        entradas_con_ruido = self.cargador_datos.generar_datos_con_ruido(
            entradas_limpias, probabilidad_ruido
        )
        
        pred_limpias = red.predecir(entradas_limpias)
        
        pred_ruido = red.predecir(entradas_con_ruido)
        
        if nombre_experimento == 'clasificacion_10_clases':
            precision_limpia = np.mean(np.argmax(pred_limpias, axis=1) == salidas_esperadas)
            precision_ruido = np.mean(np.argmax(pred_ruido, axis=1) == salidas_esperadas)
        else:
            precision_limpia = np.mean((pred_limpias > 0.5).flatten() == (salidas_esperadas.flatten() == 1))
            precision_ruido = np.mean((pred_ruido > 0.5).flatten() == (salidas_esperadas.flatten() == 1))
        
        degradacion = precision_limpia - precision_ruido
        robustez = (1 - degradacion) * 100 if degradacion >= 0 else 100
        
        resultado_ruido = {
            'experimento_base': nombre_experimento,
            'probabilidad_ruido': probabilidad_ruido,
            'precision_sin_ruido': precision_limpia,
            'precision_con_ruido': precision_ruido,
            'degradacion': degradacion,
            'robustez_porcentaje': robustez,
            'entradas_limpias': entradas_limpias,
            'entradas_con_ruido': entradas_con_ruido,
            'predicciones_limpias': pred_limpias,
            'predicciones_ruido': pred_ruido
        }
        
        if mostrar_progreso:
            self._mostrar_resultados_ruido(resultado_ruido)
        
        return resultado_ruido
    
    def _mostrar_resultados_xor(self, resultado: Dict) -> None:

        print(f"\n📊 RESULTADOS XOR:")
        print(f"  Convergencia: {'✅ SÍ' if resultado['convergencia'] else '❌ NO'}")
        print(f"  Época: {resultado['epoca_convergencia']}")
        print(f"  Precisión: {resultado['precision']*100:.1f}%")
        print(f"  Error MSE: {resultado['error_cuadratico_medio']:.6f}")
        
        print(f"\n  Tabla de Predicciones:")
        entradas = resultado['entradas']
        esperadas = resultado['salidas_esperadas'].flatten()
        predichas = resultado['predicciones_binarias'].flatten()
        
        for i, (entrada, esperada, predicha) in enumerate(zip(entradas, esperadas, predichas)):
            correcto = "✅" if esperada == predicha else "❌"
            print(f"    {entrada} → Esperado: {esperada:>2}, Predicho: {predicha:>2} {correcto}")
    
    def _mostrar_resultados_discriminacion_pares(self, resultado: Dict) -> None:

        print(f"\n📊 RESULTADOS DISCRIMINACIÓN PARES:")
        print(f"  Convergencia: {'✅ SÍ' if resultado['convergencia'] else '❌ NO'}")
        print(f"  Época: {resultado['epoca_convergencia']}")
        print(f"  Precisión Entrenamiento: {resultado['precision_entrenamiento']*100:.1f}%")
        print(f"  Precisión Prueba: {resultado['precision_prueba']*100:.1f}%")
        
        diferencia = resultado['precision_entrenamiento'] - resultado['precision_prueba']
        if diferencia > 0.1:
            print(f"  ⚠️ Posible sobreajuste (diferencia: {diferencia*100:.1f}%)")
        
        print(f"  Capacidad de generalización: {'Buena' if diferencia < 0.2 else 'Limitada'}")
    
    def _mostrar_resultados_clasificacion_10_clases(self, resultado: Dict) -> None:

        print(f"\n📊 RESULTADOS CLASIFICACIÓN 10 CLASES:")
        print(f"  Convergencia: {'✅ SÍ' if resultado['convergencia'] else '❌ NO'}")
        print(f"  Época: {resultado['epoca_convergencia']}")
        print(f"  Precisión Entrenamiento: {resultado['precision_entrenamiento']*100:.1f}%")
        print(f"  Precisión Prueba: {resultado['precision_prueba']*100:.1f}%")
        
        diferencia = resultado['precision_entrenamiento'] - resultado['precision_prueba']
        if diferencia > 0.5:
            print(f"  ⚠️ Sobreajuste severo detectado (diferencia: {diferencia*100:.1f}%)")
        
        print(f"\n  Predicciones por dígito de prueba:")
        for i, (real, pred) in enumerate(zip(resultado['salidas_test'], resultado['clases_predichas_test'])):
            correcto = "✅" if real == pred else "❌"
            print(f"    Dígito {real} → Predicho: {pred} {correcto}")
    
    def _mostrar_resultados_ruido(self, resultado: Dict) -> None:

        print(f"\n📊 RESULTADOS EVALUACIÓN CON RUIDO:")
        print(f"  Precisión sin ruido: {resultado['precision_sin_ruido']*100:.1f}%")
        print(f"  Precisión con ruido: {resultado['precision_con_ruido']*100:.1f}%")
        print(f"  Degradación: {resultado['degradacion']*100:.1f}%")
        print(f"  Robustez: {resultado['robustez_porcentaje']:.1f}%")
        
        if resultado['robustez_porcentaje'] > 90:
            print(f"  🏆 Excelente robustez al ruido")
        elif resultado['robustez_porcentaje'] > 70:
            print(f"  👍 Buena robustez al ruido")
        else:
            print(f"  ⚠️ Sensibilidad al ruido detectada")
    
    def _crear_resultado_error(self, mensaje_error: str) -> Dict:

        return {
            'error': True,
            'mensaje': mensaje_error,
            'convergencia': False,
            'precision_entrenamiento': 0.0,
            'precision_prueba': 0.0
        }
    
    def obtener_resultados(self, nombre_experimento: str = None) -> Dict:

        if nombre_experimento is None:
            return self.resultados_experimentos.copy()
        else:
            return self.resultados_experimentos.get(nombre_experimento, {})
    
    def limpiar_resultados(self) -> None:

        self.resultados_experimentos.clear()
        self.redes_entrenadas.clear()
