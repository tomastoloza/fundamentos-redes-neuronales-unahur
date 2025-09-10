import numpy as np
from activation_functions import get_activation_function
from perceptron_printer import PerceptronPrinter


class Perceptron:

    def __init__(self, num_entradas, tasa_aprendizaje=1.0, max_epocas=1000,
                 error_min=0.01, verbose=True, random_seed=None):
        self.num_entradas = num_entradas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_epocas = max_epocas
        self.error_min = error_min
        self.rng = np.random.default_rng(random_seed)
        self.w = self.rng.uniform(-1, 1, (num_entradas + 1, 1))
        self.historial_errores = []
        self.epoca_convergencia = None
        self.printer = PerceptronPrinter(verbose)

    def calcular_error(self, entradas, salidas_deseadas, funcion_activacion):
        total_error = 0
        for j in range(len(entradas)):
            entrada_con_sesgo = np.append(entradas[j], 1)
            h = np.dot(entrada_con_sesgo, self.w)
            h_valor = h[0] if isinstance(h, np.ndarray) else h
            O = funcion_activacion(h_valor)
            total_error += (salidas_deseadas[j] - O) ** 2

        return (1 / 2) * total_error

    def entrenar(self, entradas, salidas_deseadas, funcion_activacion, derivada_activacion,
                 nombre_tipo="PERSONALIZADA"):
        self.printer.imprimir_inicio_entrenamiento(nombre_tipo, self.w)
        self.historial_errores = []
        i = 0

        while True:
            error_valor = self._calcular_y_registrar_error(entradas, salidas_deseadas, funcion_activacion)
            self.printer.imprimir_progreso(i, error_valor)

            if self._debe_terminar_entrenamiento(error_valor, i):
                self._manejar_convergencia(error_valor, i)
                break

            self._actualizar_pesos(entradas, salidas_deseadas, funcion_activacion, derivada_activacion)
            i += 1

        return self._generar_resultado_entrenamiento(i, funcion_activacion, nombre_tipo)

    def _calcular_y_registrar_error(self, entradas, salidas_deseadas, funcion_activacion):
        error_global = self.calcular_error(entradas, salidas_deseadas, funcion_activacion)
        error_valor = error_global[0] if isinstance(error_global, np.ndarray) else error_global
        self.historial_errores.append(error_valor)
        return error_valor

    def _debe_terminar_entrenamiento(self, error_valor, epoca):
        return error_valor < self.error_min or epoca >= self.max_epocas

    def _manejar_convergencia(self, error_valor, epoca):
        if error_valor < self.error_min:
            self.printer.imprimir_convergencia(error_valor, epoca)
        self.epoca_convergencia = epoca + 1

    def _actualizar_pesos(self, entradas, salidas_deseadas, funcion_activacion, derivada_activacion):
        indice = self.rng.integers(0, len(entradas))
        entrada = np.append(entradas[indice], 1)

        h = np.dot(entrada, self.w)
        h_valor = h[0] if isinstance(h, np.ndarray) else h
        O = funcion_activacion(h_valor)

        M = salidas_deseadas[indice] - O
        delta_W = self.tasa_aprendizaje * M * derivada_activacion(h_valor) * entrada.reshape(-1, 1)
        self.w += delta_W

    def _generar_resultado_entrenamiento(self, epoca_final, funcion_activacion, nombre_tipo):
        return {
            'pesos_finales': self.w.copy(),
            'epoca_final': epoca_final,
            'epoca_convergencia': self.epoca_convergencia,
            'historial_errores': self.historial_errores.copy(),
            'error_final': self.historial_errores[-1] if self.historial_errores else None,
            'nombre_activacion': nombre_tipo,
            'funcion_activacion': funcion_activacion
        }

    def predecir(self, entrada, funcion_activacion):
        entrada_con_sesgo = np.append(entrada, 1)
        h = np.dot(entrada_con_sesgo, self.w)
        h_valor = h[0] if isinstance(h, np.ndarray) else h
        return funcion_activacion(h_valor)

    def mostrar_resultados_entrenamiento(self, resultado_entrenamiento, entradas, salidas_esperadas,
                                         nombre_modelo, es_entero=False):
        self.printer.mostrar_resultados_entrenamiento(
            resultado_entrenamiento, entradas, salidas_esperadas, nombre_modelo, es_entero, self.predecir
        )

    def mostrar_resultados(self, entradas, salidas_esperadas, nombre_modelo, funcion_activacion, nombre_activacion,
                           es_entero=False):
        self.printer.mostrar_resultados(
            entradas, salidas_esperadas, nombre_modelo, funcion_activacion,
            nombre_activacion, self.w, self.epoca_convergencia, es_entero, self.predecir
        )
