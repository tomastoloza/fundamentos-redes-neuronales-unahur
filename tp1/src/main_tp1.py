import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tp1.src.entrenador_compuertas import EntrenadorCompuertas
from tp1.src.cargador_datos import CargadorDatos
from tp1.src.perceptron_simple import PerceptronSimple
from tp1.src.visualizador_resultados import VisualizadorResultados
from comun.constantes.constantes_redes_neuronales import (
    TASA_APRENDIZAJE_DEFECTO, EPOCAS_MAXIMAS_DEFECTO
)

class EjecutorTP1:

    def __init__(self):

        self.entrenador = EntrenadorCompuertas()
        self.visualizador = VisualizadorResultados()
        self.resultados_experimentos = {}
    
    def ejecutar_ejercicio_1_compuertas_logicas(self) -> None:

        print("\n" + "="*80)
        print("TP1 - EJERCICIO 1: COMPUERTAS LÓGICAS")
        print("="*80)
        
        print("\n🔹 Entrenando con función ESCALÓN:")
        resultados_escalon = self.entrenador.entrenar_todas_las_compuertas(
            funcion_activacion='escalon',
            tasa_aprendizaje=TASA_APRENDIZAJE_DEFECTO,
            max_epocas=1000,
            mostrar_progreso=True
        )
        
        self.resultados_experimentos['ejercicio_1_escalon'] = resultados_escalon
        
        self._analizar_resultados_compuertas(resultados_escalon)
    
    def ejecutar_ejercicio_1_comparacion_funciones(self) -> None:

        print("\n" + "="*80)
        print("TP1 - COMPARACIÓN DE FUNCIONES DE ACTIVACIÓN (XOR)")
        print("="*80)
        
        resultados_comparacion = self.entrenador.comparar_funciones_activacion(
            tipo_compuerta='xor',
            funciones=['escalon', 'sigmoide', 'lineal'],
            mostrar_progreso=True
        )
        
        self.resultados_experimentos['ejercicio_1_comparacion'] = resultados_comparacion
        
        self._analizar_problema_xor(resultados_comparacion)
    
    def ejecutar_ejercicio_2_datos_archivo(self) -> None:

        print("\n" + "="*80)
        print("TP1 - EJERCICIO 2: DATOS DESDE ARCHIVO")
        print("="*80)
        
        try:
            entradas, salidas = CargadorDatos.cargar_datos_tp1_ejercicio2()
            
            info_datos = CargadorDatos.obtener_informacion_datos(entradas, salidas)
            self._mostrar_informacion_datos(info_datos)
            
            entradas_train, salidas_train, entradas_val, salidas_val = (
                CargadorDatos.dividir_datos_entrenamiento_validacion(
                    entradas, salidas, porcentaje_entrenamiento=0.8
                )
            )
            
            print("\n🔹 Entrenando PERCEPTRÓN LINEAL:")
            resultado_lineal = self._entrenar_perceptron_individual(
                entradas_train, salidas_train, entradas_val, salidas_val,
                funcion_activacion='lineal',
                nombre_experimento='Perceptrón Lineal'
            )
            
            print("\n🔹 Entrenando PERCEPTRÓN NO LINEAL (Sigmoide):")
            resultado_sigmoide = self._entrenar_perceptron_individual(
                entradas_train, salidas_train, entradas_val, salidas_val,
                funcion_activacion='sigmoide',
                nombre_experimento='Perceptrón No Lineal'
            )
            
            self.resultados_experimentos['ejercicio_2'] = {
                'lineal': resultado_lineal,
                'sigmoide': resultado_sigmoide,
                'info_datos': info_datos
            }
            
            self._comparar_perceptrones_lineales_no_lineales(resultado_lineal, resultado_sigmoide)
            
        except Exception as e:
            print(f"❌ Error al ejecutar ejercicio 2: {str(e)}")
            print("💡 Asegúrese de que los archivos de datos estén en tp1/datos/")
    
    def _entrenar_perceptron_individual(self, entradas_train, salidas_train, 
                                      entradas_val, salidas_val,
                                      funcion_activacion: str,
                                      nombre_experimento: str) -> dict:

        perceptron = PerceptronSimple(
            num_entradas=entradas_train.shape[1],
            funcion_activacion=funcion_activacion
        )
        
        convergencia, epoca = perceptron.entrenar(
            entradas=entradas_train,
            salidas_esperadas=salidas_train,
            tasa_aprendizaje=TASA_APRENDIZAJE_DEFECTO,
            max_epocas=EPOCAS_MAXIMAS_DEFECTO,
            mostrar_progreso=True
        )
        
        metricas_train = perceptron.evaluar(entradas_train, salidas_train)
        
        metricas_val = perceptron.evaluar(entradas_val, salidas_val)
        
        resultado = {
            'nombre_experimento': nombre_experimento,
            'funcion_activacion': funcion_activacion,
            'convergencia': convergencia,
            'epoca_convergencia': epoca,
            'metricas_entrenamiento': metricas_train,
            'metricas_validacion': metricas_val,
            'info_entrenamiento': perceptron.obtener_informacion_entrenamiento(),
            'perceptron': perceptron
        }
        
        self._mostrar_resultados_experimento_individual(resultado)
        
        return resultado
    
    def _mostrar_informacion_datos(self, info_datos: dict) -> None:

        print(f"\n📊 INFORMACIÓN DE LOS DATOS:")
        print(f"  Número de muestras: {info_datos['num_muestras']}")
        print(f"  Número de características: {info_datos['num_caracteristicas']}")
        print(f"  Rango de entradas: {info_datos['rango_entradas']}")
        print(f"  Rango de salidas: {info_datos['rango_salidas']}")
        print(f"  Forma de entradas: {info_datos['forma_entradas']}")
        print(f"  Valores únicos en salidas: {info_datos['valores_unicos_salidas']}")
    
    def _mostrar_resultados_experimento_individual(self, resultado: dict) -> None:

        print(f"\n📈 RESULTADOS - {resultado['nombre_experimento']}:")
        
        convergencia = "✅ SÍ" if resultado['convergencia'] else "❌ NO"
        print(f"  Convergencia: {convergencia}")
        print(f"  Época: {resultado['epoca_convergencia']}")
        
        train_mse = resultado['metricas_entrenamiento'].get('error_cuadratico_medio', 0)
        train_precision = resultado['metricas_entrenamiento'].get('precision', 0) * 100
        
        val_mse = resultado['metricas_validacion'].get('error_cuadratico_medio', 0)
        val_precision = resultado['metricas_validacion'].get('precision', 0) * 100
        
        print(f"  MSE Entrenamiento: {train_mse:.6f}")
        print(f"  MSE Validación: {val_mse:.6f}")
        print(f"  Precisión Entrenamiento: {train_precision:.2f}%")
        print(f"  Precisión Validación: {val_precision:.2f}%")
        
        diferencia_precision = train_precision - val_precision
        if diferencia_precision > 10:
            print(f"  ⚠️  Posible sobreajuste detectado (diferencia: {diferencia_precision:.1f}%)")
    
    def _analizar_resultados_compuertas(self, resultados: dict) -> None:

        print(f"\n🔍 ANÁLISIS DE RESULTADOS:")
        
        for compuerta, resultado in resultados.items():
            if 'error' not in resultado:
                convergencia = resultado['convergencia']
                precision = resultado['metricas'].get('precision', 0) * 100
                
                if convergencia:
                    print(f"  ✅ {compuerta.upper()}: Problema linealmente separable (Precisión: {precision:.1f}%)")
                else:
                    print(f"  ❌ {compuerta.upper()}: Problema NO linealmente separable (Precisión: {precision:.1f}%)")
        
        print(f"\n💡 CONCLUSIÓN:")
        print(f"  El perceptrón simple con función escalón puede resolver problemas")
        print(f"  linealmente separables (AND, OR) pero no problemas no lineales (XOR).")
    
    def _analizar_problema_xor(self, resultados: dict) -> None:

        print(f"\n🔍 ANÁLISIS DEL PROBLEMA XOR:")
        
        for funcion, resultado in resultados.items():
            if 'error' not in resultado:
                convergencia = resultado['convergencia']
                precision = resultado['metricas'].get('precision', 0) * 100
                
                if funcion == 'escalon':
                    print(f"  📐 Escalón: No puede resolver XOR (separabilidad lineal)")
                elif funcion == 'sigmoide':
                    if convergencia:
                        print(f"  📈 Sigmoide: Puede aproximar XOR (Precisión: {precision:.1f}%)")
                    else:
                        print(f"  📈 Sigmoide: Dificultad para resolver XOR con una sola capa")
                elif funcion == 'lineal':
                    print(f"  📏 Lineal: Limitado a aproximaciones lineales (Precisión: {precision:.1f}%)")
    
    def _comparar_perceptrones_lineales_no_lineales(self, resultado_lineal: dict, 
                                                  resultado_sigmoide: dict) -> None:

        print(f"\n🔍 COMPARACIÓN LINEAL vs NO LINEAL:")
        
        mse_lineal = resultado_lineal['metricas_validacion'].get('error_cuadratico_medio', 0)
        mse_sigmoide = resultado_sigmoide['metricas_validacion'].get('error_cuadratico_medio', 0)
        
        precision_lineal = resultado_lineal['metricas_validacion'].get('precision', 0) * 100
        precision_sigmoide = resultado_sigmoide['metricas_validacion'].get('precision', 0) * 100
        
        print(f"  📏 Perceptrón Lineal:")
        print(f"    - MSE: {mse_lineal:.6f}")
        print(f"    - Precisión: {precision_lineal:.2f}%")
        print(f"    - Ventajas: Simple, estable, interpretable")
        print(f"    - Limitaciones: Solo relaciones lineales")
        
        print(f"  📈 Perceptrón No Lineal (Sigmoide):")
        print(f"    - MSE: {mse_sigmoide:.6f}")
        print(f"    - Precisión: {precision_sigmoide:.2f}%")
        print(f"    - Ventajas: Puede modelar relaciones no lineales")
        print(f"    - Limitaciones: Más complejo, riesgo de sobreajuste")
        
        if mse_sigmoide < mse_lineal:
            mejora = ((mse_lineal - mse_sigmoide) / mse_lineal) * 100
            print(f"  🏆 El perceptrón no lineal es superior (mejora MSE: {mejora:.1f}%)")
        else:
            print(f"  🏆 El perceptrón lineal es suficiente para este problema")
    
    def ejecutar_todos_los_experimentos(self) -> None:

        print("🚀 INICIANDO EJECUCIÓN COMPLETA DEL TP1")
        print("="*80)
        
        self.ejecutar_ejercicio_1_compuertas_logicas()
        
        self.ejecutar_ejercicio_1_comparacion_funciones()
        
        self.ejecutar_ejercicio_2_datos_archivo()
        
        self._mostrar_resumen_final()
    
    def _mostrar_resumen_final(self) -> None:

        print("\n" + "="*80)
        print("📋 RESUMEN FINAL DEL TP1")
        print("="*80)
        
        print("\n🎯 OBJETIVOS ALCANZADOS:")
        print("  ✅ Implementación del perceptrón simple con diferentes funciones de activación")
        print("  ✅ Evaluación de problemas linealmente separables vs no separables")
        print("  ✅ Comparación entre perceptrón lineal y no lineal")
        print("  ✅ Análisis de capacidad de generalización")
        
        print("\n🔬 CONCLUSIONES PRINCIPALES:")
        print("  • El perceptrón simple escalón resuelve problemas linealmente separables")
        print("  • XOR requiere arquitecturas más complejas o funciones no lineales")
        print("  • La función sigmoide permite mayor flexibilidad que la escalón")
        print("  • La validación cruzada es esencial para evaluar generalización")
        
        print("\n📊 EXPERIMENTOS COMPLETADOS:")
        for nombre, _ in self.resultados_experimentos.items():
            print(f"  ✓ {nombre}")
    
    def obtener_resultados_completos(self) -> dict:

        return self.resultados_experimentos.copy()

def main():

    ejecutor = EjecutorTP1()
    ejecutor.ejecutar_todos_los_experimentos()

if __name__ == "__main__":
    main()
