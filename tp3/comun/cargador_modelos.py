import os
from tensorflow import keras


class CargadorModelos:
    EXTENSION_MODELO = '.keras'
    
    def __init__(self, directorio_modelos="tp3/modelos"):
        self.directorio_modelos = directorio_modelos

    def extraer_info_nombre(self, nombre_archivo):
        nombre_base = nombre_archivo.replace(self.EXTENSION_MODELO, '')
        partes = nombre_base.split('_')
        
        info = {
            'nombre_completo': nombre_archivo,
            'dimension_latente': None,
            'epochs': None,
            'learning_rate': None
        }
        
        for parte in partes:
            if parte.startswith('lat'):
                info['dimension_latente'] = int(parte[3:])
            elif parte.startswith('ep'):
                info['epochs'] = int(parte[2:])
            elif parte.startswith('lr'):
                lr_str = parte[2:].replace('_', '.')
                info['learning_rate'] = float(lr_str)
        
        return info
    
    def cargar_modelo(self, nombre_modelo):
        if not nombre_modelo.endswith(self.EXTENSION_MODELO):
            nombre_modelo += self.EXTENSION_MODELO
        
        ruta_completa = os.path.join(self.directorio_modelos, nombre_modelo)
        
        if not os.path.exists(ruta_completa):
            raise FileNotFoundError(f"Modelo no encontrado: {ruta_completa}")
        
        modelo = keras.models.load_model(ruta_completa)
        return modelo
    
    def listar_modelos(self):
        if not os.path.exists(self.directorio_modelos):
            return []
        
        archivos = os.listdir(self.directorio_modelos)
        modelos = [archivo for archivo in archivos if archivo.endswith(self.EXTENSION_MODELO)]
        return sorted(modelos)
    
    def obtener_info_modelos(self):
        modelos = self.listar_modelos()
        info_modelos = []
        
        for modelo in modelos:
            info = self.extraer_info_nombre(modelo)
            info_modelos.append(info)
        
        return info_modelos

