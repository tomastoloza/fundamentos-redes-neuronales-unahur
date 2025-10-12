from tensorflow import keras
from tensorflow.keras import layers


class ConstructorModelos:
    def __init__(self):
        pass
    
    def crear_autocodificador_desde_config(self, config):
        input_layer = layers.Input(shape=(35,))
        
        x = input_layer
        # Capas encoder
        for neuronas in config['capas_encoder']:
            x = layers.Dense(neuronas, activation=config['activacion'])(x)
        
        latente = layers.Dense(config['dimension_latente'], activation='linear', name='latente')(x)
        
        x = latente
        for neuronas in config['capas_decoder']:
            x = layers.Dense(neuronas, activation=config['activacion'])(x)
        
        output_layer = layers.Dense(35, activation=config['activacion_salida'])(x)
        
        modelo = keras.Model(input_layer, output_layer)
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['mse']
        )
        
        return modelo