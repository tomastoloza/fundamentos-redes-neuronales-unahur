import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tp2.src.entrenador_tp2 import EntrenadorTP2
from tp2.src.cargador_datos_digitos import CargadorDatosDigitos
from comun.constantes.constantes_redes_neuronales import (
    ARQUITECTURAS_TP2, TASA_APRENDIZAJE_DEFECTO, PROBABILIDAD_RUIDO_DEFECTO
)

class EjecutorTP2:

    def __init__(self):

        self.cargador_datos = CargadorDatosDigitos()
        self.entrenador = EntrenadorTP2(self.cargador_datos)
        self.resultados_experimentos = {}
    
    def ejecutar_ejercicio_1_xor(self) -> None:

        print("\n" + "="*80)
        print("TP2 - EJERCICIO 1: FUNCIÓN LÓGICA XOR")
        print("="*80)
        
        resultado_xor = self.entrenador.entrenar_problema_xor(
            arquitectura=[2, 4, 1],
            tasa_aprendizaje=0.1,
            max_epocas=10000,
            mostrar_progreso=True
        )
        
        self.resultados_experimentos['xor'] = resultado_xor
        
        self._analizar_resultados_xor(resultado_xor)
    
    def ejecutar_ejercicio_2_discriminacion_pares(self) -> None:

        print("\n" + "="*80)
        print("TP2 - EJERCICIO 2: DISCRIMINACIÓN DE NÚMEROS PARES")
        print("="*80)
        
        arquitecturas_a_probar = {
            'MINIMA': ARQUITECTURAS_TP2['MINIMA'],
            'COMPACTA': ARQUITECTURAS_TP2['COMPACTA'],
            'DIRECTA_ORIGINAL': ARQUITECTURAS_TP2['DIRECTA_ORIGINAL']
        }
        
        resultados_arquitecturas = {}
        
        for nombre_arq, arquitectura in arquitecturas_a_probar.items():
            print(f"\n🔹 Probando arquitectura {nombre_arq}: {arquitectura}")
            
            resultado = self.entrenador.entrenar_discriminacion_numeros_pares(
                arquitectura=arquitectura,
                digitos_entrenamiento=[0, 2, 4, 6, 1, 3],
                digitos_prueba=[5, 7, 8, 9],
                tasa_aprendizaje=TASA_APRENDIZAJE_DEFECTO,
                max_epocas=5000,
                mostrar_progreso=True
            )
            
            resultados_arquitecturas[nombre_arq] = resultado
        
        self.resultados_experimentos['discriminacion_pares'] = resultados_arquitecturas
        
        self._comparar_arquitecturas_discriminacion(resultados_arquitecturas)
    
    def ejecutar_ejercicio_3_clasificacion_10_clases(self) -> None:

        print("\n" + "="*80)
        print("TP2 - EJERCICIO 3: CLASIFICACIÓN DE DÍGITOS (10 CLASES)")
        print("="*80)
        
        resultado_clasificacion = self.entrenador.entrenar_clasificacion_10_clases(
            arquitectura=ARQUITECTURAS_TP2['CLASIFICACION_10_CLASES'],
            digitos_entrenamiento=[0, 1, 2, 3, 4, 5, 6],
            digitos_prueba=[7, 8, 9],
            tasa_aprendizaje=0.1,
            max_epocas=1000,
            mostrar_progreso=True
        )
        
        self.resultados_experimentos['clasificacion_10_clases'] = resultado_clasificacion
        
        self._analizar_resultados_clasificacion_10_clases(resultado_clasificacion)
    
    def ejecutar_ejercicio_3_evaluacion_ruido(self) -> None:

        print("\n" + "="*80)
        print("TP2 - EJERCICIO 3: EVALUACIÓN CON RUIDO")
        print("="*80)
        
        if 'clasificacion_10_clases' not in self.entrenador.resultados_experimentos:
            print("⚠️ Primero debe ejecutar la clasificación de 10 clases")
            return
        
        resultado_ruido = self.entrenador.evaluar_robustez_ruido(
            nombre_experimento='clasificacion_10_clases',
            probabilidad_ruido=PROBABILIDAD_RUIDO_DEFECTO,
            mostrar_progreso=True
        )
        
        self.resultados_experimentos['evaluacion_ruido'] = resultado_ruido
        
        self._analizar_robustez_ruido(resultado_ruido)
    
    def ejecutar_comparacion_arquitecturas_completa(self) -> None:

        print("\n" + "="*80)
        print("TP2 - COMPARACIÓN COMPLETA DE ARQUITECTURAS")
        print("="*80)
        
        resultados_comparacion = {}
        
        for nombre_arq, arquitectura in ARQUITECTURAS_TP2.items():
            if nombre_arq == 'CLASIFICACION_10_CLASES':
                continue
            
            print(f"\n🔹 Evaluando arquitectura {nombre_arq}: {arquitectura}")
            
            try:
                resultado = self.entrenador.entrenar_discriminacion_numeros_pares(
                    arquitectura=arquitectura,
                    digitos_entrenamiento=[0, 2, 4, 6, 1, 3],
                    digitos_prueba=[5, 7, 8, 9],
                    tasa_aprendizaje=0.1,
                    max_epocas=3000,
                    mostrar_progreso=False
                )
                
                resultados_comparacion[nombre_arq] = resultado
                
                precision_test = resultado.get('precision_prueba', 0) * 100
                convergencia = "✅" if resultado.get('convergencia', False) else "❌"
                epoca = resultado.get('epoca_convergencia', 'N/A')
                
                print(f"    Resultado: {convergencia} | Época: {epoca} | Precisión: {precision_test:.1f}%")
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                resultados_comparacion[nombre_arq] = {'error': str(e)}
        
        self.resultados_experimentos['comparacion_arquitecturas'] = resultados_comparacion
        
        self._mostrar_ranking_arquitecturas(resultados_comparacion)
    
    def _analizar_resultados_xor(self, resultado: dict) -> None:

        print(f"\n🔍 ANÁLISIS DEL PROBLEMA XOR:")
        
        if resultado.get('convergencia', False):
            print(f"  ✅ El perceptrón multicapa resolvió exitosamente XOR")
            print(f"  📈 Precisión alcanzada: {resultado['precision']*100:.1f}%")
            print(f"  ⚡ Convergencia en época: {resultado['epoca_convergencia']}")
            print(f"  🧠 Arquitectura exitosa: {resultado['arquitectura']}")
        else:
            print(f"  ❌ No se logró resolver XOR completamente")
            print(f"  📉 Precisión final: {resultado['precision']*100:.1f}%")
            print(f"  💡 Sugerencia: Probar con más épocas o diferente arquitectura")
        
        print(f"\n💡 CONCLUSIÓN:")
        print(f"  El perceptrón multicapa supera las limitaciones del perceptrón simple")
        print(f"  para problemas no linealmente separables como XOR.")
    
    def _comparar_arquitecturas_discriminacion(self, resultados: dict) -> None:

        print(f"\n🔍 COMPARACIÓN DE ARQUITECTURAS - DISCRIMINACIÓN PARES:")
        print(f"{'Arquitectura':>15} | {'Convergencia':>12} | {'Época':>8} | {'Prec. Train':>12} | {'Prec. Test':>11} | {'Generalización':>14}")
        print(f"{'-'*15} | {'-'*12} | {'-'*8} | {'-'*12} | {'-'*11} | {'-'*14}")
        
        mejor_arquitectura = None
        mejor_precision_test = 0
        
        for nombre, resultado in resultados.items():
            if 'error' not in resultado:
                convergencia = "SÍ" if resultado['convergencia'] else "NO"
                epoca = resultado.get('epoca_convergencia', 'N/A')
                prec_train = resultado['precision_entrenamiento'] * 100
                prec_test = resultado['precision_prueba'] * 100
                
                diferencia = prec_train - prec_test
                if diferencia < 10:
                    generalizacion = "Excelente"
                elif diferencia < 25:
                    generalizacion = "Buena"
                else:
                    generalizacion = "Limitada"
                
                print(f"{nombre:>15} | {convergencia:>12} | {str(epoca):>8} | "
                      f"{prec_train:>11.1f}% | {prec_test:>10.1f}% | {generalizacion:>14}")
                
                if prec_test > mejor_precision_test:
                    mejor_precision_test = prec_test
                    mejor_arquitectura = nombre
            else:
                print(f"{nombre:>15} | {'ERROR':>12} | {'N/A':>8} | {'N/A':>12} | {'N/A':>11} | {'N/A':>14}")
        
        if mejor_arquitectura:
            print(f"\n🏆 MEJOR ARQUITECTURA: {mejor_arquitectura} (Precisión test: {mejor_precision_test:.1f}%)")
    
    def _analizar_resultados_clasificacion_10_clases(self, resultado: dict) -> None:

        print(f"\n🔍 ANÁLISIS CLASIFICACIÓN 10 CLASES:")
        
        prec_train = resultado['precision_entrenamiento'] * 100
        prec_test = resultado['precision_prueba'] * 100
        diferencia = prec_train - prec_test
        
        print(f"  📊 Precisión entrenamiento: {prec_train:.1f}%")
        print(f"  📊 Precisión prueba: {prec_test:.1f}%")
        print(f"  📈 Diferencia: {diferencia:.1f}%")
        
        if diferencia > 50:
            print(f"  ⚠️ SOBREAJUSTE SEVERO detectado")
            print(f"     - La red memorizó los patrones de entrenamiento")
            print(f"     - No logra generalizar a nuevos dígitos")
        elif diferencia > 25:
            print(f"  ⚠️ Sobreajuste moderado detectado")
        else:
            print(f"  ✅ Buena capacidad de generalización")
        
        print(f"\n  📋 Análisis por dígito de prueba:")
        if 'salidas_test' in resultado and 'clases_predichas_test' in resultado:
            for digito in [7, 8, 9]:
                indices = resultado['salidas_test'] == digito
                if np.any(indices):
                    predicciones_digito = resultado['clases_predichas_test'][indices]
                    aciertos = np.sum(predicciones_digito == digito)
                    total = len(predicciones_digito)
                    precision_digito = (aciertos / total) * 100 if total > 0 else 0
                    print(f"     Dígito {digito}: {precision_digito:.1f}% ({aciertos}/{total})")
    
    def _analizar_robustez_ruido(self, resultado: dict) -> None:

        print(f"\n🔍 ANÁLISIS DE ROBUSTEZ AL RUIDO:")
        
        robustez = resultado['robustez_porcentaje']
        degradacion = resultado['degradacion'] * 100
        
        print(f"  🎯 Robustez: {robustez:.1f}%")
        print(f"  📉 Degradación: {degradacion:.1f}%")
        
        if robustez > 90:
            print(f"  🏆 EXCELENTE robustez - La red es muy resistente al ruido")
        elif robustez > 70:
            print(f"  👍 BUENA robustez - La red maneja bien el ruido")
        elif robustez > 50:
            print(f"  ⚠️ MODERADA robustez - Sensibilidad al ruido detectada")
        else:
            print(f"  ❌ BAJA robustez - La red es muy sensible al ruido")
        
        print(f"\n💡 INTERPRETACIÓN:")
        if degradacion < 5:
            print(f"  La red aprendió características robustas de los dígitos")
        elif degradacion < 15:
            print(f"  La red tiene buena tolerancia a pequeñas perturbaciones")
        else:
            print(f"  La red puede estar sobreajustada a patrones específicos")
    
    def _mostrar_ranking_arquitecturas(self, resultados: dict) -> None:

        print(f"\n🏆 RANKING DE ARQUITECTURAS:")
        
        arquitecturas_validas = []
        for nombre, resultado in resultados.items():
            if 'error' not in resultado and 'precision_prueba' in resultado:
                arquitecturas_validas.append((nombre, resultado))
        
        arquitecturas_ordenadas = sorted(
            arquitecturas_validas, 
            key=lambda x: x[1]['precision_prueba'], 
            reverse=True
        )
        
        for i, (nombre, resultado) in enumerate(arquitecturas_ordenadas, 1):
            precision = resultado['precision_prueba'] * 100
            convergencia = "✅" if resultado['convergencia'] else "❌"
            epoca = resultado.get('epoca_convergencia', 'N/A')
            
            medalla = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            
            print(f"  {medalla} {nombre}: {precision:.1f}% {convergencia} (Época: {epoca})")
        
        if arquitecturas_ordenadas:
            mejor = arquitecturas_ordenadas[0]
            print(f"\n🎯 RECOMENDACIÓN: Usar arquitectura {mejor[0]} para mejores resultados")
    
    def ejecutar_todos_los_experimentos(self) -> None:

        print("🚀 INICIANDO EJECUCIÓN COMPLETA DEL TP2")
        print("="*80)
        
        try:
            self.ejecutar_ejercicio_1_xor()
            
            self.ejecutar_ejercicio_2_discriminacion_pares()
            
            self.ejecutar_ejercicio_3_clasificacion_10_clases()
            
            self.ejecutar_ejercicio_3_evaluacion_ruido()
            
            self.ejecutar_comparacion_arquitecturas_completa()
            
            self._mostrar_resumen_final()
            
        except Exception as e:
            print(f"❌ Error durante la ejecución: {str(e)}")
            print("💡 Verifique que los archivos de datos estén disponibles en tp2/datos/")
    
    def _mostrar_resumen_final(self) -> None:

        print("\n" + "="*80)
        print("📋 RESUMEN FINAL DEL TP2")
        print("="*80)
        
        print("\n🎯 OBJETIVOS ALCANZADOS:")
        print("  ✅ Implementación del perceptrón multicapa con retropropagación")
        print("  ✅ Resolución del problema XOR (no linealmente separable)")
        print("  ✅ Discriminación de números pares vs impares")
        print("  ✅ Clasificación multiclase de dígitos (0-9)")
        print("  ✅ Evaluación de robustez ante ruido")
        print("  ✅ Comparación de diferentes arquitecturas")
        
        print("\n🔬 CONCLUSIONES PRINCIPALES:")
        print("  • El perceptrón multicapa supera las limitaciones del perceptrón simple")
        print("  • La retropropagación permite aprender patrones complejos no lineales")
        print("  • Arquitecturas más simples pueden generalizar mejor que las complejas")
        print("  • El sobreajuste es un problema común con datos limitados")
        print("  • La robustez al ruido indica la calidad del aprendizaje")
        
        print("\n📊 EXPERIMENTOS COMPLETADOS:")
        for nombre, _ in self.resultados_experimentos.items():
            print(f"  ✓ {nombre}")
        
        print("\n💡 LECCIONES APRENDIDAS:")
        print("  • Más parámetros no siempre significa mejor rendimiento")
        print("  • La validación independiente es crucial para evaluar generalización")
        print("  • El balance entre capacidad y generalización es fundamental")
        print("  • Las redes neuronales pueden ser robustas a perturbaciones pequeñas")
    
    def obtener_resultados_completos(self) -> dict:

        return self.resultados_experimentos.copy()

def main():

    ejecutor = EjecutorTP2()
    ejecutor.ejecutar_todos_los_experimentos()

if __name__ == "__main__":
    main()
