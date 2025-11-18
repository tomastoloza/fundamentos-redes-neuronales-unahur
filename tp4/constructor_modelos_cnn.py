import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam


class ConstructorModelosCNN:
    @staticmethod
    def construir_modelo(configuracion):
        tamaño_imagen = configuracion['tamaño_imagen']
        input_shape = (tamaño_imagen[0], tamaño_imagen[1], 3)
        
        modelo = Sequential()
        
        primera_capa = True
        for i, capa_config in enumerate(configuracion['capas_conv']):
            if primera_capa:
                modelo.add(Conv2D(
                    filters=capa_config['filters'],
                    kernel_size=capa_config['kernel_size'],
                    activation='linear' if capa_config.get('bn', False) else capa_config['activation'],
                    input_shape=input_shape,
                    padding='same'
                ))
                primera_capa = False
            else:
                modelo.add(Conv2D(
                    filters=capa_config['filters'],
                    kernel_size=capa_config['kernel_size'],
                    activation='linear' if capa_config.get('bn', False) else capa_config['activation'],
                    padding='same'
                ))
            
            # Agregar Batch Normalization si está configurado
            if capa_config.get('bn', False):
                modelo.add(BatchNormalization())
                modelo.add(Activation(capa_config['activation']))
            
            # Agregar MaxPooling si está configurado
            if 'pool' in capa_config:
                modelo.add(MaxPooling2D(pool_size=capa_config['pool']))
            else:
                # Mantener comportamiento por defecto para compatibilidad
                modelo.add(MaxPooling2D(pool_size=(2, 2)))
        
        modelo.add(Flatten())
        
        for unidades in configuracion['capas_dense']:
            modelo.add(Dense(unidades, activation='relu'))
            modelo.add(Dropout(configuracion['dropout_rate']))
        
        modelo.add(Dense(
            configuracion['num_clases'], 
            activation=configuracion['activacion_salida']
        ))
        
        optimizer = Adam(learning_rate=configuracion['learning_rate'])
        
        modelo.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return modelo
    
    @staticmethod
    def obtener_resumen_arquitectura(configuracion):
        resumen = []
        resumen.append(f"Entrada: {configuracion['tamaño_imagen']} RGB")
        
        for i, capa in enumerate(configuracion['capas_conv']):
            resumen.append(f"Conv2D_{i+1}: {capa['filters']} filtros {capa['kernel_size']}")
            resumen.append(f"MaxPool2D_{i+1}: (2,2)")
        
        resumen.append("Flatten")
        
        for i, unidades in enumerate(configuracion['capas_dense']):
            resumen.append(f"Dense_{i+1}: {unidades} + Dropout({configuracion['dropout_rate']})")
        
        resumen.append(f"Salida: {configuracion['num_clases']} clases ({configuracion['activacion_salida']})")
        
        return resumen
    
    @staticmethod
    def calcular_parametros_aproximados(configuracion):
        tamaño = configuracion['tamaño_imagen'][0]
        parametros = 0
        
        for capa in configuracion['capas_conv']:
            kernel_params = capa['kernel_size'][0] * capa['kernel_size'][1] * 3 * capa['filters']
            parametros += kernel_params
            tamaño = tamaño // 2
        
        flatten_size = tamaño * tamaño * configuracion['capas_conv'][-1]['filters']
        
        entrada_anterior = flatten_size
        for unidades in configuracion['capas_dense']:
            parametros += entrada_anterior * unidades
            entrada_anterior = unidades
        
        parametros += entrada_anterior * configuracion['num_clases']
        
        return parametros



