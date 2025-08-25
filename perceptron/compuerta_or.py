import numpy as np

def entrenar_compuerta_or_lineal():
    from perceptron_unificado import PerceptronUnificado
    
    print("🟡 ENTRENANDO PERCEPTRÓN LINEAL PARA COMPUERTA OR")
    print("=" * 50)
    
    entradas = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    salidas_or = np.array([0, 1, 1, 1])
    
    tasa_aprendizaje = 1.0
    max_epocas = 1000
    error_min = 0.01
    
    print(f"Configuración:")
    print(f"  • Tipo: Lineal (Escalón)")
    print(f"  • Tasa de aprendizaje: {tasa_aprendizaje}")
    print(f"  • Máximo de épocas: {max_epocas}")
    print(f"  • Error mínimo: {error_min}")
    print(f"  • Ejemplos de entrenamiento: {len(entradas)}")
    print()
    
    perceptron = PerceptronUnificado(
        num_entradas=2,
        tasa_aprendizaje=tasa_aprendizaje,
        max_epocas=max_epocas,
        error_min=error_min
    )
    
    resultado = perceptron.entrenar(entradas, salidas_or, tipo_activacion='escalon')
    
    perceptron.mostrar_resultados(entradas, salidas_or, "OR", tipo_activacion='escalon')
    
    return perceptron

def entrenar_compuerta_or_no_lineal():
    from perceptron_unificado import PerceptronUnificado
    
    print("🟡 ENTRENANDO PERCEPTRÓN NO LINEAL PARA COMPUERTA OR")
    print("=" * 50)
    
    entradas = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    salidas_or = np.array([0, 1, 1, 1])
    
    tasa_aprendizaje = 1.0
    max_epocas = 1000
    error_min = 0.01
    
    print(f"Configuración:")
    print(f"  • Tipo: No Lineal (Sigmoide)")
    print(f"  • Tasa de aprendizaje: {tasa_aprendizaje}")
    print(f"  • Máximo de épocas: {max_epocas}")
    print(f"  • Error mínimo: {error_min}")
    print(f"  • Ejemplos de entrenamiento: {len(entradas)}")
    print()
    
    perceptron = PerceptronUnificado(
        num_entradas=2,
        tasa_aprendizaje=tasa_aprendizaje,
        max_epocas=max_epocas,
        error_min=error_min
    )
    
    resultado = perceptron.entrenar(entradas, salidas_or, tipo_activacion='sigmoide')
    
    perceptron.mostrar_resultados(entradas, salidas_or, "OR", tipo_activacion='sigmoide')
    
    return perceptron

def probar_compuerta_or(perceptron, tipo_activacion='escalon'):
    print("🧪 PROBANDO COMPUERTA OR:")
    
    casos_prueba = [
        ([0, 0], "A=0, B=0"),
        ([0, 1], "A=0, B=1"),
        ([1, 0], "A=1, B=0"),
        ([1, 1], "A=1, B=1")
    ]
    
    for entrada, descripcion in casos_prueba:
        salida = perceptron.predecir(np.array(entrada), tipo_activacion)
        if tipo_activacion == 'escalon':
            print(f"  {descripcion} → Salida: {salida}")
        else:
            print(f"  {descripcion} → Salida: {salida:.3f}")

def comparar_tipos_or():
    print("🔄 COMPARANDO PERCEPTRÓN LINEAL vs NO LINEAL - COMPUERTA OR")
    print("=" * 60)
    
    print("\n1️⃣ PERCEPTRÓN LINEAL:")
    perceptron_lineal = entrenar_compuerta_or_lineal()
    
    print("\n2️⃣ PERCEPTRÓN NO LINEAL:")
    perceptron_no_lineal = entrenar_compuerta_or_no_lineal()
    
    print("\n📊 COMPARACIÓN DE RESULTADOS:")
    print("Tipo      | Épocas | Error Final")
    print("----------|--------|------------")
    
    print(f"Lineal    | {perceptron_lineal.epoca_convergencia or 'N/A':6} | {perceptron_lineal.historial_errores[-1]:11.6f}")
    print(f"No Lineal | {perceptron_no_lineal.epoca_convergencia or 'N/A':6} | {perceptron_no_lineal.historial_errores[-1]:11.6f}")
    
    return perceptron_lineal, perceptron_no_lineal

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'comparar':
            perceptron_lineal, perceptron_no_lineal = comparar_tipos_or()
        elif sys.argv[1] == 'no_lineal':
            perceptron = entrenar_compuerta_or_no_lineal()
            probar_compuerta_or(perceptron, tipo_activacion='sigmoide')
        else:
            perceptron = entrenar_compuerta_or_lineal()
            probar_compuerta_or(perceptron, tipo_activacion='escalon')
    else:
        perceptron = entrenar_compuerta_or_lineal()
        probar_compuerta_or(perceptron, tipo_activacion='escalon')