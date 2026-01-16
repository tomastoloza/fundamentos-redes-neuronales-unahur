import numpy as np

def sigmoid(x):
    """Función de activación sigmoide."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivada de la función sigmoide."""
    return x * (1 - x)

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        """Inicializa el perceptrón con pesos y sesgos aleatorios."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicializar pesos y sesgos de la capa oculta
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.rand(1, self.hidden_size)

        # Inicializar pesos y sesgos de la capa de salida
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.random.rand(1, self.output_size)

    def forward(self, X):
        """Calcula la salida de la red para una entrada dada."""
        # Capa oculta
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        # Capa de salida
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_layer_input)
        return self.output

    def backward(self, X, y, output, learning_rate):
        """Ajusta los pesos y sesgos usando retropropagación (backpropagation)."""
        # Calcular el error de la capa de salida
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        # Calcular el error de la capa oculta
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)

        # Actualizar pesos y sesgos de la capa de salida
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        # Actualizar pesos y sesgos de la capa oculta
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        """Entrena la red con los datos de entrada."""
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

            # Opcional: imprimir el error cada 1000 épocas
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Época {epoch}: Pérdida = {loss:.4f}")

    def predict(self, X):
        """Realiza predicciones después del entrenamiento."""
        return self.forward(X)


# Datos de entrada y salida para el problema XOR
X_xor = np.array([[-1, 1],
                  [1, -1],
                  [-1, -1],
                  [1, 1]])

y_xor = np.array([[1],
                  [1],
                  [-1],
                  [-1]])

# Instanciar y entrenar el MLP
mlp = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1)
mlp.train(X_xor, y_xor, epochs=10000, learning_rate=0.1)

# Realizar predicciones y mostrar resultados
predictions = mlp.predict(X_xor)
print("\nSalida esperada (y):")
print(y_xor)
print("\nPredicciones (y_hat):")
print(np.round(predictions))