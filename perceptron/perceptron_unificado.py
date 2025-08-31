import numpy as np

class PerceptronUnificado:
    
    def __init__(self, num_entradas, tasa_aprendizaje=1.0, max_epocas=1000, 
                 error_min=0.01, verbose=True, random_seed=None):
        self.num_entradas = num_entradas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_epocas = max_epocas
        self.error_min = error_min
        self.verbose = verbose
        self.rng = np.random.default_rng(random_seed)
        self.w = self.rng.uniform(-1, 1, (num_entradas + 1, 1))
        self.historial_errores = []
        self.epoca_convergencia = None
    
    def sigmoide(self, x):
        return 1 / (1 + np.exp(-2 * x))
    
    def derivada_sigmoide(self, x):
        s = self.sigmoide(x)
        return 2 * s * (1 - s)
    
    def escalon(self, x):
        return 1 if x >= 0 else 0
    
    def derivada_escalon(self, x):
        return 1
    
    def lineal(self, x):
        return x
    
    def derivada_lineal(self, x):
        return 1
    
    def calcular_error(self, entradas, salidas_deseadas, funcion_activacion):
        total_error = 0
        for j in range(len(entradas)):
            entrada_con_sesgo = np.append(entradas[j], 1)
            h = np.dot(entrada_con_sesgo, self.w)
            h_valor = h[0] if isinstance(h, np.ndarray) else h
            O = funcion_activacion(h_valor)
            total_error += (salidas_deseadas[j] - O) ** 2
        
        return (1/2) * total_error
    
    def entrenar(self, entradas, salidas_deseadas, tipo_activacion='sigmoide'):
        if tipo_activacion == 'sigmoide':
            funcion_activacion = self.sigmoide
            derivada_activacion = self.derivada_sigmoide
            nombre_tipo = "NO LINEAL (Sigmoide)"
        elif tipo_activacion == 'escalon':
            funcion_activacion = self.escalon
            derivada_activacion = self.derivada_escalon
            nombre_tipo = "LINEAL (Escalón)"
        elif tipo_activacion == 'lineal':
            funcion_activacion = self.lineal
            derivada_activacion = self.derivada_lineal
            nombre_tipo = "REGRESIÓN (Lineal)"
        else:
            raise ValueError("tipo_activacion debe ser 'sigmoide', 'escalon' o 'lineal'")
        
        if self.verbose:
            print(f"ENTRENANDO PERCEPTRÓN {nombre_tipo}")
            print(f"Pesos iniciales: {self.w.flatten()}")
            print()
        
        self.historial_errores = []
        i = 0
        
        while True:
            error_global = self.calcular_error(entradas, salidas_deseadas, funcion_activacion)
            error_valor = error_global[0] if isinstance(error_global, np.ndarray) else error_global
            self.historial_errores.append(error_valor)
            
            if self.verbose and (i % 20 == 0 or i == 0):
                print(f"Época: {i+1}, Error: {error_valor:.6f}")
            
            if error_valor < self.error_min or i >= self.max_epocas:
                if error_valor < self.error_min and self.verbose:
                    print(f"¡Convergencia alcanzada en época {i+1}! Error = {error_valor:.6f}")
                self.epoca_convergencia = i + 1
                break
            
            indice = self.rng.integers(0, len(entradas))
            entrada = np.append(entradas[indice], 1)
            
            h = np.dot(entrada, self.w)
            h_valor = h[0] if isinstance(h, np.ndarray) else h
            O = funcion_activacion(h_valor)
            
            M = salidas_deseadas[indice] - O
            
            if tipo_activacion == 'escalon' and M == 0:
                i += 1
                continue
            
            delta_W = self.tasa_aprendizaje * M * derivada_activacion(h_valor) * entrada.reshape(-1, 1)
            
            self.w += delta_W
            
            i += 1
        
        return self._generar_resultado_entrenamiento(i, tipo_activacion)
    
    def _generar_resultado_entrenamiento(self, epoca_final, tipo_activacion):
        nombres_activacion = {
            'sigmoide': 'Sigmoide (No Lineal)',
            'escalon': 'Escalón (Lineal)', 
            'lineal': 'Lineal (Regresión)'
        }
        nombre_activacion = nombres_activacion.get(tipo_activacion, tipo_activacion)
        
        return {
            'pesos_finales': self.w.copy(),
            'epoca_final': epoca_final,
            'epoca_convergencia': self.epoca_convergencia,
            'historial_errores': self.historial_errores.copy(),
            'error_final': self.historial_errores[-1] if self.historial_errores else None,
            'tipo_activacion': tipo_activacion,
            'nombre_activacion': nombre_activacion
        }
    
    def predecir(self, entrada, tipo_activacion='sigmoide'):
        if tipo_activacion == 'sigmoide':
            funcion_activacion = self.sigmoide
        elif tipo_activacion == 'escalon':
            funcion_activacion = self.escalon
        elif tipo_activacion == 'lineal':
            funcion_activacion = self.lineal
        else:
            raise ValueError("tipo_activacion debe ser 'sigmoide', 'escalon' o 'lineal'")
        
        entrada_con_sesgo = np.append(entrada, 1)
        h = np.dot(entrada_con_sesgo, self.w)
        h_valor = h[0] if isinstance(h, np.ndarray) else h
        return funcion_activacion(h_valor)
    
    def evaluar(self, entradas, salidas_esperadas, tipo_activacion='sigmoide'):
        predicciones = []
        errores = 0
        
        for i in range(len(entradas)):
            prediccion = self.predecir(entradas[i], tipo_activacion)
            predicciones.append(prediccion)
            
            if tipo_activacion == 'escalon':
                if prediccion != salidas_esperadas[i]:
                    errores += 1
            else:
                prediccion_binaria = 1 if prediccion >= 0.5 else 0
                if prediccion_binaria != salidas_esperadas[i]:
                    errores += 1
        
        return {
            'predicciones': predicciones,
            'errores': errores,
            'total_ejemplos': len(entradas)
        }
    
    def mostrar_resultados(self, entradas, salidas_esperadas, nombre_modelo, tipo_activacion='sigmoide'):
        nombres_activacion = {
            'sigmoide': 'Sigmoide (No Lineal)',
            'escalon': 'Escalón (Lineal)', 
            'lineal': 'Lineal (Regresión)'
        }
        nombre_activacion = nombres_activacion.get(tipo_activacion, tipo_activacion)
        tipo_nombre = "LINEAL" if tipo_activacion == 'escalon' else "NO LINEAL"
        
        print(f"\n=== RESULTADOS PERCEPTRÓN {tipo_nombre} - {nombre_modelo} ===")
        print(f"Función de activación: {nombre_activacion}")
        print(f"Pesos finales: {self.w.flatten()}")
        
        if self.epoca_convergencia:
            print(f"Convergencia en época: {self.epoca_convergencia}")
        
        print("--------------------------------")
        print(f"Tabla de resultados {nombre_modelo}:")
        print("A | B | Salida | Esperada")
        print("--|---|--------|----------")
        
        for idx in range(len(entradas)):
            entrada = entradas[idx]
            salida_obtenida = self.predecir(entrada, tipo_activacion)
            a, b = entrada[0], entrada[1]
            
            if tipo_activacion == 'escalon':
                print(f"{int(a)} | {int(b)} |   {salida_obtenida:2d}   |    {salidas_esperadas[idx]:2d}")
            else:
                print(f"{int(a)} | {int(b)} | {salida_obtenida:6.3f} |    {salidas_esperadas[idx]:2d}")
        
        print("--------------------------------\n")

def entrenar_perceptron_unificado(entradas, salidas_deseadas, tipo_activacion='sigmoide', 
                                 tasa_aprendizaje=1.0, max_epocas=1000, error_min=0.01):
    num_entradas = entradas.shape[1]
    
    perceptron = PerceptronUnificado(
        num_entradas=num_entradas,
        tasa_aprendizaje=tasa_aprendizaje,
        max_epocas=max_epocas,
        error_min=error_min
    )
    
    resultado = perceptron.entrenar(entradas, salidas_deseadas, tipo_activacion)
    
    return perceptron, resultado
