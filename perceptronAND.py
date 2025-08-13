# x entrada
# y salida
# w pesos
# n tasa de aprendizaje
# p cantidad de ejemplos
# COTA cantidad de epocas
# N cantidad de entradas
# algoritmo de perceptron
# i = 0
# w = zeros(N + 1, 1)
# error = 1
# error_min = p*2
# while error > 0 ^ i < COTA
    # tomar un numero i_x aleatorio entre 1 y p
    # calcular la excitacion h = x[i_x] * w
    # calcular la activacion O = signo(h)
    # dw = n * (y[i_x] - O ) . x[i_x] multiplicacion matricial
    # w = w + dw
    # error = CalcularError(x, y, w, p)
    # if error < error_min
        # error_min = error
        # w_min = w
    # i = i + 1
# return w_min

import random
import numpy as np

"""calcular_error debe calcular cuántos ejemplos de entrada (x) el perceptrón clasifica incorrectamente usando los pesos actuales (w).
Debe recorrer todos los ejemplos (de 0 a p-1), calcular la salida del perceptrón para cada uno, compararla con la salida esperada (y), y contar los errores.
La función debe devolver el número total de errores de clasificación.
"""
def CalcularError(x, y, w, p):
    error = 0
    for i in range(p):
        h = getExcitacion(x[i], w)
        activacion = signo(h)
        if activacion != y[i]:
            error += 1
    return error

"""signo debe devolver 1 si el valor de entrada (h) es mayor o igual a 0, y -1 en caso contrario."""
def signo(h):
    return 1 if h >= 0 else -1

"""getExcitacion debe calcular la excitación (h) del perceptrón para una entrada (x) y pesos (w).
La excitación se calcula como el producto punto (multiplicación matricial) entre la entrada y los pesos."""
def getExcitacion(x, w):
    return np.dot(x, w)

"""perceptron debe implementar el algoritmo de entrenamiento del perceptrón.
Debe iterar hasta que el error sea 0 o se alcance el número máximo de épocas (epocas).
En cada iteración, debe seleccionar un ejemplo aleatorio, calcular la excitación, la activación y el cambio de pesos (dw).
Luego, debe actualizar los pesos y calcular el error.
Si el error es menor que el error mínimo, debe actualizar el error mínimo y los pesos mínimos."""
def perceptron(i, w, entradas, salidas, tasa_aprendizaje, p, epocas):
    error = 1
    error_min = p*2
    w_min = w
    while error > 0 and i < epocas:
        i_x = random.randint(0, p-1)
        excitacion = getExcitacion(entradas[i_x], w)
        activacion = signo(excitacion)
        # dw = n * (y[i_x] - O ) . x[i_x]
        dw = tasa_aprendizaje * (salidas[i_x] - activacion) * entradas[i_x]
        w = w + dw
        error = CalcularError(entradas, salidas, w, p)
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1
    return w_min

def main():
    entradas = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    salidas = np.array([-1, -1, -1, 1])
    tasa_aprendizaje = 0.1
    p = 4
    epocas = 100000
    i = 0
    cantidad_entradas = 2
    w = np.zeros(cantidad_entradas + 1)
    w_min = perceptron(i, w, entradas, salidas, tasa_aprendizaje, p, epocas)
    print("Pesos finales: ", w_min)
    print("--------------------------------")
    for entrada in entradas:
        excitacion = getExcitacion(entrada, w_min)
        print(f"Entrada: {entrada}, Excitacion: {excitacion}")
        print(f"Activacion: {signo(excitacion)}")
        print("--------------------------------")

if __name__ == "__main__":
    main()

