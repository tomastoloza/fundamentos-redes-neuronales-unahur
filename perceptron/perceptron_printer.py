
class PerceptronPrinter:
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def imprimir_configuracion(self, tipo_activacion, tasa_aprendizaje, max_epocas, error_min, num_ejemplos):
        if self.verbose:
            print("Configuración:")
            print(f"  • Tipo: {tipo_activacion}")
            print(f"  • Tasa de aprendizaje: {tasa_aprendizaje}")
            print(f"  • Máximo de épocas: {max_epocas}")
            print(f"  • Error mínimo: {error_min}")
            print(f"  • Ejemplos de entrenamiento: {num_ejemplos}")
            print()
    
    def imprimir_inicio_entrenamiento(self, nombre_tipo, pesos_iniciales):
        if self.verbose:
            print(f"ENTRENANDO PERCEPTRÓN {nombre_tipo}")
            print(f"Pesos iniciales: {pesos_iniciales.flatten()}")
            print()
    
    def imprimir_progreso(self, epoca, error_valor):
        if self.verbose and (epoca % 20 == 0 or epoca == 0):
            print(f"Época: {epoca+1}, Error: {error_valor:.6f}")
    
    def imprimir_convergencia(self, error_valor, epoca):
        if self.verbose:
            print(f"¡Convergencia alcanzada en época {epoca+1}! Error = {error_valor:.6f}")
    
    def mostrar_resultados_entrenamiento(self, resultado_entrenamiento, entradas, salidas_esperadas, 
                                        nombre_modelo, es_entero=False, predecir_func=None):
        if not self.verbose:
            return
            
        print(f"\n=== RESULTADOS PERCEPTRÓN - {nombre_modelo} ===")
        print(f"Función de activación: {resultado_entrenamiento['nombre_activacion']}")
        print(f"Pesos finales: {resultado_entrenamiento['pesos_finales'].flatten()}")
        
        if resultado_entrenamiento['epoca_convergencia']:
            print(f"Convergencia en época: {resultado_entrenamiento['epoca_convergencia']}")
        
        if resultado_entrenamiento['error_final'] is not None:
            print(f"Error final: {resultado_entrenamiento['error_final']:.6f}")
        
        print("--------------------------------")
        print(f"Tabla de resultados {nombre_modelo}:")
        print("A | B | Salida | Esperada")
        print("--|---|--------|----------")
        
        funcion_activacion = resultado_entrenamiento['funcion_activacion']
        for idx in range(len(entradas)):
            entrada = entradas[idx]
            salida_obtenida = predecir_func(entrada, funcion_activacion)
            a, b = entrada[0], entrada[1]
            
            if es_entero:
                print(f"{int(a)} | {int(b)} |   {salida_obtenida:2d}   |    {salidas_esperadas[idx]:2d}")
            else:
                print(f"{int(a)} | {int(b)} | {salida_obtenida:6.3f} |    {salidas_esperadas[idx]:2d}")
        
        print("--------------------------------\n")

    def mostrar_resultados(self, entradas, salidas_esperadas, nombre_modelo, 
                          funcion_activacion, nombre_activacion, pesos_finales, 
                          epoca_convergencia, es_entero=False, predecir_func=None):
        print(f"\n=== RESULTADOS PERCEPTRÓN - {nombre_modelo} ===")
        print(f"Función de activación: {nombre_activacion}")
        print(f"Pesos finales: {pesos_finales.flatten()}")
        
        if epoca_convergencia:
            print(f"Convergencia en época: {epoca_convergencia}")
        
        print("--------------------------------")
        print(f"Tabla de resultados {nombre_modelo}:")
        print("A | B | Salida | Esperada")
        print("--|---|--------|----------")
        
        for idx in range(len(entradas)):
            entrada = entradas[idx]
            salida_obtenida = predecir_func(entrada, funcion_activacion)
            a, b = entrada[0], entrada[1]
            
            if es_entero:
                print(f"{int(a)} | {int(b)} |   {salida_obtenida:2d}   |    {salidas_esperadas[idx]:2d}")
            else:
                print(f"{int(a)} | {int(b)} | {salida_obtenida:6.3f} |    {salidas_esperadas[idx]:2d}")
        
        print("--------------------------------\n")
    
    def set_verbose(self, verbose):
        self.verbose = verbose
