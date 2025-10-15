from tensorflow import keras
from tensorflow.keras import layers


class ConstructorModelosTipografia:
    def __init__(self, tamaño_imagen=32):
        self.tamaño_imagen = tamaño_imagen
        self.input_size = tamaño_imagen * tamaño_imagen
    
    def crear_autocodificador_desde_config(self, config):
        input_layer = layers.Input(shape=(self.input_size,))
        
        x = input_layer
        for neuronas in config['capas_encoder']:
            x = layers.Dense(neuronas, activation=config['activacion'])(x)
        
        latente = layers.Dense(config['dimension_latente'], activation='linear', name='latente')(x)
        
        x = latente
        for neuronas in config['capas_decoder']:
            x = layers.Dense(neuronas, activation=config['activacion'])(x)
        
        output_layer = layers.Dense(self.input_size, activation=config['activacion_salida'])(x)
        
        modelo = keras.Model(input_layer, output_layer)
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['mse']
        )
        
        return modelo
