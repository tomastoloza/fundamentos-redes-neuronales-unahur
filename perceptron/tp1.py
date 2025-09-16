import numpy as np
import os
from datetime import datetime
from perceptron import Perceptron
from activation_functions import sigmoid, sigmoid_derivative


def cargar_datos_desde_archivos(archivo_entradas, archivo_salidas, verbose=False):
    try:
        entradas_raw = np.loadtxt(archivo_entradas)
        salidas_raw = np.loadtxt(archivo_salidas)

        num_ejemplos = entradas_raw.shape[0]
        entradas = np.column_stack([entradas_raw, np.ones(num_ejemplos)])

        if verbose:
            print(f"Datos cargados: {num_ejemplos} ejemplos, {entradas_raw.shape[1]} características + sesgo")

        return entradas, salidas_raw

    except FileNotFoundError:
        print("Error: No se encontraron los archivos de datos especificados")
        return None, None
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None, None


def normalizar_datos(entradas, salidas):
    entradas_norm = entradas.copy()
    entradas_stats = {}

    for i in range(entradas.shape[1] - 1):
        col_mean = np.mean(entradas[:, i])
        col_std = np.std(entradas[:, i])
        entradas_norm[:, i] = (entradas[:, i] - col_mean) / col_std
        entradas_stats[f'col_{i}'] = {'mean': col_mean, 'std': col_std}

    salidas_mean = np.mean(salidas)
    salidas_std = np.std(salidas)
    salidas_norm = (salidas - salidas_mean) / salidas_std

    stats = {
        'entradas': entradas_stats,
        'salidas': {'mean': salidas_mean, 'std': salidas_std}
    }

    return entradas_norm, salidas_norm, stats


def dividir_datos(entradas, salidas, porcentaje_entrenamiento=0.8, random_state=None):
    num_ejemplos = len(entradas)
    num_entrenamiento = int(num_ejemplos * porcentaje_entrenamiento)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(num_ejemplos)

    indices_train = indices[:num_entrenamiento]
    indices_test = indices[num_entrenamiento:]

    x_train = entradas[indices_train]
    y_train = salidas[indices_train]
    x_test = entradas[indices_test]
    y_test = salidas[indices_test]

    return x_train, y_train, x_test, y_test


def entrenar_perceptron_regresion(entradas, salidas, tasa_aprendizaje, max_epocas):
    entradas_sin_sesgo = entradas[:, :-1]
    num_entradas = entradas_sin_sesgo.shape[1]

    perceptron = Perceptron(
        num_entradas=num_entradas,
        tasa_aprendizaje=tasa_aprendizaje,
        max_epocas=max_epocas,
        error_min=1e-6
    )

    resultado = perceptron.entrenar(entradas_sin_sesgo, salidas, sigmoid, sigmoid_derivative, "NO LINEAL (Sigmoide)")

    return perceptron.w.flatten(), perceptron.historial_errores, resultado


def evaluar_modelo(pesos, entradas):
    predicciones = []

    for i in range(len(entradas)):
        suma_ponderada = 0
        entrada = entradas[i][:-1]
        for j in range(len(entrada)):
            suma_ponderada += pesos[j] * entrada[j]
        suma_ponderada += pesos[-1]

        predicciones.append(suma_ponderada)

    predicciones = np.array(predicciones)

    return {
        'predicciones': predicciones
    }


def guardar_resultados(y_test, predicciones, pesos_finales, error_final, timestamp,
                      tipo_activacion='sigmoide', prefijo_archivo='resultados'):
    carpeta_resultados = "resultados"
    if not os.path.exists(carpeta_resultados):
        os.makedirs(carpeta_resultados)
        print(f"📁 Carpeta '{carpeta_resultados}' creada")

    nombre_archivo = f"{prefijo_archivo}_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    ruta_completa = os.path.join(carpeta_resultados, nombre_archivo)

    nombres_activacion = {
        'sigmoide': 'Sigmoide (No Lineal)',
        'escalon': 'Escalón (Lineal)',
        'lineal': 'Lineal (Regresión)'
    }
    nombre_activacion = nombres_activacion.get(tipo_activacion, tipo_activacion)

    with open(ruta_completa, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RESULTADOS DE PREDICCIÓN - REGRESIÓN\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp de ejecución: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Función de activación: {nombre_activacion}\n")
        f.write(f"Error final (MSE): {error_final:.6f}\n")
        f.write(f"Pesos finales: {pesos_finales}\n")
        f.write("\n")

        f.write("TABLA DE PREDICCIONES (Conjunto de Prueba)\n")
        f.write("-" * 60 + "\n")
        f.write("Índice | Real      | Predicción | Error Abs. | Error Rel. (%)\n")
        f.write("-" * 60 + "\n")

        for i in range(len(y_test)):
            real = y_test[i]
            pred = predicciones[i]
            error_abs = abs(real - pred)
            error_rel = (error_abs / abs(real) * 100) if real != 0 else float('inf')

            f.write(f"{i + 1:6d} | {real:9.4f} | {pred:10.4f} | {error_abs:10.4f} | {error_rel:11.2f}\n")

        f.write("-" * 60 + "\n")

        errores_abs = [abs(y_test[i] - predicciones[i]) for i in range(len(y_test))]
        f.write("\nESTADÍSTICAS DE ERROR\n")
        f.write("-" * 30 + "\n")
        f.write(f"Error absoluto promedio: {np.mean(errores_abs):.6f}\n")
        f.write(f"Error absoluto mínimo:   {np.min(errores_abs):.6f}\n")
        f.write(f"Error absoluto máximo:   {np.max(errores_abs):.6f}\n")
        f.write(f"Desviación estándar:     {np.std(errores_abs):.6f}\n")

        errores_relativos = []
        for i in range(len(y_test)):
            if y_test[i] != 0:
                error_rel = abs(y_test[i] - predicciones[i]) / abs(y_test[i]) * 100
                errores_relativos.append(error_rel)

        if errores_relativos:
            excelentes = sum(1 for e in errores_relativos if e < 5)
            buenas = sum(1 for e in errores_relativos if 5 <= e < 20)
            regulares = sum(1 for e in errores_relativos if 20 <= e < 50)
            malas = sum(1 for e in errores_relativos if e >= 50)

            f.write("\nCALIDAD DE PREDICCIONES\n")
            f.write("-" * 30 + "\n")
            f.write(f"Excelentes (< 5%):     {excelentes:3d} ({excelentes / len(errores_relativos) * 100:.1f}%)\n")
            f.write(f"Buenas (5-20%):        {buenas:3d} ({buenas / len(errores_relativos) * 100:.1f}%)\n")
            f.write(f"Regulares (20-50%):    {regulares:3d} ({regulares / len(errores_relativos) * 100:.1f}%)\n")
            f.write(f"Malas (≥ 50%):         {malas:3d} ({malas / len(errores_relativos) * 100:.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"📄 Resultados guardados en: {ruta_completa}")
    return ruta_completa


def entrenar_tp1_legacy():
    timestamp_inicio = datetime.now()

    print("🔵 ENTRENANDO PERCEPTRÓN PARA TP1-EJ2 (REGRESIÓN)")
    print("=" * 60)
    print(f"🕐 Iniciado: {timestamp_inicio.strftime('%Y-%m-%d %H:%M:%S')}")

    entradas, salidas = cargar_datos_desde_archivos(
        'TP1-ej2-Conjunto-entrenamiento.txt', 
        'TP1-ej2-Salida-deseada.txt', 
        verbose=True
    )
    if entradas is None or salidas is None:
        print("❌ Error: No se pudieron cargar los datos")
        return None

    print("\n📊 Normalizando datos...")
    entradas_norm, salidas_norm, stats = normalizar_datos(entradas, salidas)

    print("🔄 Dividiendo datos...")
    porcentaje_entrenamiento = 0.8
    x_train, y_train, x_test, y_test = dividir_datos(
        entradas_norm, salidas_norm, porcentaje_entrenamiento, random_state=42
    )

    print(f"   • Entrenamiento: {len(x_train)} ejemplos ({porcentaje_entrenamiento * 100:.0f}%)")
    print(f"   • Prueba: {len(x_test)} ejemplos ({(1 - porcentaje_entrenamiento) * 100:.0f}%)")

    tasa_aprendizaje = 0.1
    max_epocas = 100000

    print("Configuración:")
    print(f"   • Tasa de aprendizaje: {tasa_aprendizaje}")
    print(f"   • Épocas máximas: {max_epocas}")

    print("\n🚀 Iniciando entrenamiento...")
    pesos_finales, historial_errores, resultado_entrenamiento = entrenar_perceptron_regresion(
        x_train, y_train, tasa_aprendizaje, max_epocas
    )

    metricas_test = evaluar_modelo(pesos_finales, x_test)

    print("\n🔍 EJEMPLOS DE PREDICCIONES (primeros 10 del conjunto de prueba):")
    print("Real      | Predicción")
    print("----------|------------")
    for i in range(min(10, len(y_test))):
        real = y_test[i]
        pred = metricas_test['predicciones'][i]
        print(f"{real:9.4f} | {pred:10.4f}")

    print(f"\n🏆 Pesos finales: {pesos_finales}")
    print(f"📉 Error final (MSE): {historial_errores[-1]:.6f}")

    # Guardar resultados en archivo
    timestamp_fin = datetime.now()
    print("Guardando resultados...")
    archivo_guardado = guardar_resultados(
        y_test,
        metricas_test['predicciones'],
        pesos_finales,
        historial_errores[-1],
        timestamp_fin,
        tipo_activacion='sigmoide',
        prefijo_archivo='tp1_resultados'
    )

    duracion = timestamp_fin - timestamp_inicio
    print(f"⏱️  Duración total: {duracion.total_seconds():.2f} segundos")

    return pesos_finales, metricas_test, stats


def entrenar_desde_archivos(archivo_entradas, archivo_salidas, 
                           tasa_aprendizaje=0.1, max_epocas=100000, 
                           porcentaje_entrenamiento=0.8, verbose=True):
    """Entrena un perceptrón para regresión usando datos de archivos.
    
    Args:
        archivo_entradas: Ruta al archivo con datos de entrada
        archivo_salidas: Ruta al archivo con datos de salida esperados
        tasa_aprendizaje: Tasa de aprendizaje para el entrenamiento
        max_epocas: Número máximo de épocas de entrenamiento
        porcentaje_entrenamiento: Porcentaje de datos para entrenamiento (resto para prueba)
        verbose: Si True, muestra información detallada del proceso
    
    Returns:
        tuple: (pesos_finales, metricas_test, stats) o None si hay error
    """
    timestamp_inicio = datetime.now()
    
    if verbose:
        print("🔵 ENTRENANDO PERCEPTRÓN PARA REGRESIÓN")
        print(f"🕐 Iniciado: {timestamp_inicio.strftime('%Y-%m-%d %H:%M:%S')}")

    # Cargar datos
    entradas, salidas = cargar_datos_desde_archivos(archivo_entradas, archivo_salidas, verbose)
    if entradas is None or salidas is None:
        print("❌ Error: No se pudieron cargar los datos")
        return None

    # Normalizar datos
    if verbose:
        print("📊 Normalizando datos...")
    entradas_norm, salidas_norm, stats = normalizar_datos(entradas, salidas)

    # Dividir datos
    if verbose:
        print("🔄 Dividiendo datos...")
    x_train, y_train, x_test, y_test = dividir_datos(
        entradas_norm, salidas_norm, porcentaje_entrenamiento, random_state=42
    )

    if verbose:
        print(f"   • Entrenamiento: {len(x_train)} ejemplos ({porcentaje_entrenamiento * 100:.0f}%)")
        print(f"   • Prueba: {len(x_test)} ejemplos ({(1 - porcentaje_entrenamiento) * 100:.0f}%)")
        print(f"   • Tasa de aprendizaje: {tasa_aprendizaje}")
        print(f"   • Épocas máximas: {max_epocas}")

    # Entrenar
    if verbose:
        print("🚀 Iniciando entrenamiento...")
    pesos_finales, historial_errores, resultado_entrenamiento = entrenar_perceptron_regresion(
        x_train, y_train, tasa_aprendizaje, max_epocas
    )

    # Evaluar
    metricas_test = evaluar_modelo(pesos_finales, x_test)

    if verbose:
        print(f"🏆 Pesos finales: {pesos_finales}")
        print(f"📉 Error final (MSE): {historial_errores[-1]:.6f}")
        
        # Mostrar algunas predicciones de ejemplo
        print("\n🔍 EJEMPLOS DE PREDICCIONES (primeros 5):")
        print("Real      | Predicción")
        print("----------|------------")
        for i in range(min(5, len(y_test))):
            real = y_test[i]
            pred = metricas_test['predicciones'][i]
            print(f"{real:9.4f} | {pred:10.4f}")

    # Guardar resultados
    timestamp_fin = datetime.now()
    if verbose:
        print("💾 Guardando resultados...")
    
    archivo_guardado = guardar_resultados(
        y_test,
        metricas_test['predicciones'],
        pesos_finales,
        historial_errores[-1],
        timestamp_fin,
        tipo_activacion='sigmoide',
        prefijo_archivo='entrenamiento'
    )

    if verbose:
        duracion = timestamp_fin - timestamp_inicio
        print(f"⏱️  Duración: {duracion.total_seconds():.2f}s")

    return pesos_finales, metricas_test, stats


if __name__ == "__main__":
    np.random.seed(42)
    
    # Ejemplo de uso con archivos específicos del TP1
    resultado = entrenar_desde_archivos(
        'TP1-ej2-Conjunto-entrenamiento.txt',
        'TP1-ej2-Salida-deseada.txt',
        verbose=True
    )

    if resultado is not None:
        pesos, metricas, stats = resultado
        print("\n✅ Entrenamiento completado exitosamente!")
    else:
        print("\n❌ El entrenamiento falló.")
