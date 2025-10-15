import numpy as np
from tp3.comun.procesador_datos import ProcesadorDatos
from tp3.comun.generador_ruido import GeneradorRuido


def debug_caracter_7():
    print("=== DEBUG: CARÁCTER #7 ===\n")
    
    procesador = ProcesadorDatos()
    datos = procesador.obtener_datos_procesados()
    generador = GeneradorRuido()
    
    patron_original = datos[7:8]
    
    print("PATRÓN ORIGINAL:")
    procesador.mostrar_patron_ascii(7, "Carácter 7")
    print(f"Píxeles activos: {int(np.sum(patron_original))}/35\n")
    print(f"Array original:\n{patron_original}\n")
    
    print("="*60)
    print("GENERANDO 20 MUESTRAS CON RUIDO BINARIO 0.05")
    print("="*60)
    
    cambios_totales = []
    agregados_totales = []
    eliminados_totales = []
    
    for i in range(20):
        patron_ruidoso = generador.generar_conjunto_ruidoso(
            patron_original, 'binario', 0.05
        )
        
        pixeles_cambiados = np.sum(patron_original != patron_ruidoso)
        cambios_0_a_1 = int(np.sum((patron_original == 0) & (patron_ruidoso == 1)))
        cambios_1_a_0 = int(np.sum((patron_original == 1) & (patron_ruidoso == 0)))
        
        cambios_totales.append(pixeles_cambiados)
        agregados_totales.append(cambios_0_a_1)
        eliminados_totales.append(cambios_1_a_0)
        
        pixeles_activos_original = int(np.sum(patron_original))
        pixeles_activos_ruidoso = int(np.sum(patron_ruidoso))
        
        print(f"Muestra {i+1:2d}: {int(pixeles_cambiados):2d} cambios ({pixeles_cambiados/35*100:5.1f}%) | "
              f"0→1: {cambios_0_a_1:2d} | 1→0: {cambios_1_a_0:2d} | "
              f"Activos: {pixeles_activos_original}→{pixeles_activos_ruidoso} (Δ{pixeles_activos_ruidoso - pixeles_activos_original:+2d})")
        
        if pixeles_cambiados >= 5:
            print(f"  ⚠️  CASO EXTREMO: {pixeles_cambiados} cambios es {pixeles_cambiados/35*100:.1f}%")
            print("  Visual:")
            patron_2d = patron_ruidoso.reshape(7, 5)
            for fila in patron_2d:
                print('  ' + ''.join(['██' if pixel else '  ' for pixel in fila]))
    
    print("\n" + "="*60)
    print("ESTADÍSTICAS")
    print("="*60)
    promedio_cambios = np.mean(cambios_totales)
    max_cambios = np.max(cambios_totales)
    min_cambios = np.min(cambios_totales)
    
    print(f"Cambios promedio: {promedio_cambios:.2f} píxeles ({promedio_cambios/35*100:.1f}%)")
    print(f"Cambios mínimo: {min_cambios} píxeles ({min_cambios/35*100:.1f}%)")
    print(f"Cambios máximo: {max_cambios} píxeles ({max_cambios/35*100:.1f}%)")
    print(f"Agregados promedio (0→1): {np.mean(agregados_totales):.2f}")
    print(f"Eliminados promedio (1→0): {np.mean(eliminados_totales):.2f}")
    
    print(f"\nEsperado con 5%: {35 * 0.05:.2f} píxeles")
    
    if promedio_cambios > 2.5:
        print(f"\n⚠️  PROBLEMA: El promedio ({promedio_cambios:.2f}) es mayor que el esperado (1.75)")
        print("Esto sugiere que el generador de ruido está aplicando MÁS del 5%")
    else:
        print(f"\n✓ El promedio está dentro del rango esperado")
    
    print("\n" + "="*60)
    print("VERIFICACIÓN DEL GENERADOR")
    print("="*60)
    
    np.random.seed(42)
    test_data = np.array([[0, 1, 0, 1, 0]])
    test_ruidoso = generador.aplicar_ruido_binario(test_data, 0.05)
    
    print("Test simple con seed fija:")
    print(f"Original: {test_data}")
    print(f"Ruidoso:  {test_ruidoso}")
    print(f"Cambios:  {np.sum(test_data != test_ruidoso)}")
    
    conteo_cambios = []
    for _ in range(1000):
        test_ruidoso = generador.aplicar_ruido_binario(test_data, 0.05)
        conteo_cambios.append(np.sum(test_data != test_ruidoso))
    
    print(f"\nEn 1000 iteraciones con 5 píxeles:")
    print(f"Promedio de cambios: {np.mean(conteo_cambios):.3f}")
    print(f"Esperado: {5 * 0.05:.3f}")
    print(f"Diferencia: {abs(np.mean(conteo_cambios) - 5*0.05):.3f}")


if __name__ == "__main__":
    debug_caracter_7()
