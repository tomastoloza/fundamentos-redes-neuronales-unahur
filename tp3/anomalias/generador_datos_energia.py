import numpy as np
from datetime import datetime, timedelta


class GeneradorDatosEnergia:
    def __init__(self, longitud_serie=168, seed=42):
        self.longitud_serie = longitud_serie
        np.random.seed(seed)
        
        self.patrones_normales = {
            'servidor_web': {'base': 45, 'variacion': 15, 'picos_hora': [9, 14, 20]},
            'servidor_bd': {'base': 65, 'variacion': 10, 'picos_hora': [8, 13, 18]},
            'servidor_backup': {'base': 25, 'variacion': 8, 'picos_hora': [2, 22]},
            'servidor_computo': {'base': 80, 'variacion': 20, 'picos_hora': [10, 15]}
        }
        
        self.tipos_anomalias = {
            'pico_extremo': {'multiplicador': 2.5, 'duracion': 3},
            'caida_sistema': {'multiplicador': 0.1, 'duracion': 6},
            'sobrecarga': {'multiplicador': 1.8, 'duracion': 12},
            'fallo_gradual': {'multiplicador': 0.3, 'duracion': 24}
        }
    
    def generar_patron_base_diario(self, tipo_servidor):
        patron = self.patrones_normales[tipo_servidor]
        horas = np.arange(24)
        
        consumo_base = np.full(24, patron['base'], dtype=np.float64)
        
        for hora_pico in patron['picos_hora']:
            inicio = max(0, hora_pico - 2)
            fin = min(24, hora_pico + 3)
            factor_pico = np.exp(-0.5 * ((horas[inicio:fin] - hora_pico) / 1.5) ** 2)
            consumo_base[inicio:fin] += patron['variacion'] * factor_pico
        
        ruido = np.random.normal(0, patron['variacion'] * 0.1, 24)
        return consumo_base + ruido
    
    def generar_serie_normal(self, tipo_servidor, num_dias=7):
        serie_completa = []
        
        for dia in range(num_dias):
            patron_diario = self.generar_patron_base_diario(tipo_servidor)
            
            factor_dia_semana = 1.0
            if dia >= 5:
                factor_dia_semana = 0.7
            
            patron_diario *= factor_dia_semana
            serie_completa.extend(patron_diario)
        
        return np.array(serie_completa[:self.longitud_serie])
    
    def insertar_anomalia(self, serie_normal, tipo_anomalia, posicion=None):
        serie_anomala = serie_normal.copy()
        anomalia = self.tipos_anomalias[tipo_anomalia]
        
        if posicion is None:
            max_pos = max(1, len(serie_normal) - anomalia['duracion'])
            posicion = np.random.randint(0, max_pos)
        
        fin_anomalia = min(posicion + anomalia['duracion'], len(serie_normal))
        
        if tipo_anomalia == 'fallo_gradual':
            factor_degradacion = np.linspace(1.0, anomalia['multiplicador'], fin_anomalia - posicion)
            serie_anomala[posicion:fin_anomalia] *= factor_degradacion
        else:
            serie_anomala[posicion:fin_anomalia] *= anomalia['multiplicador']
        
        return serie_anomala, posicion, fin_anomalia
    
    def generar_conjunto_entrenamiento(self, num_muestras=1000):
        tipos_servidor = list(self.patrones_normales.keys())
        datos_entrenamiento = []
        metadatos = []
        
        for i in range(num_muestras):
            tipo_servidor = np.random.choice(tipos_servidor)
            serie_normal = self.generar_serie_normal(tipo_servidor)
            
            datos_entrenamiento.append(serie_normal)
            metadatos.append({
                'id': i,
                'tipo_servidor': tipo_servidor,
                'es_anomalo': False,
                'tipo_anomalia': None
            })
        
        return np.array(datos_entrenamiento), metadatos
    
    def generar_conjunto_prueba(self, num_normales=200, num_anomalas=50):
        tipos_servidor = list(self.patrones_normales.keys())
        tipos_anomalia = list(self.tipos_anomalias.keys())
        
        datos_prueba = []
        metadatos = []
        
        for i in range(num_normales):
            tipo_servidor = np.random.choice(tipos_servidor)
            serie_normal = self.generar_serie_normal(tipo_servidor)
            
            datos_prueba.append(serie_normal)
            metadatos.append({
                'id': i,
                'tipo_servidor': tipo_servidor,
                'es_anomalo': False,
                'tipo_anomalia': None
            })
        
        for i in range(num_anomalas):
            if len(tipos_servidor) > 0:
                tipo_servidor = np.random.choice(tipos_servidor)
            else:
                tipo_servidor = 'servidor_web'
            
            if len(tipos_anomalia) > 0:
                tipo_anomalia = np.random.choice(tipos_anomalia)
            else:
                tipo_anomalia = 'pico_extremo'
            
            serie_normal = self.generar_serie_normal(tipo_servidor)
            serie_anomala, pos_inicio, pos_fin = self.insertar_anomalia(serie_normal, tipo_anomalia)
            
            datos_prueba.append(serie_anomala)
            metadatos.append({
                'id': num_normales + i,
                'tipo_servidor': tipo_servidor,
                'es_anomalo': True,
                'tipo_anomalia': tipo_anomalia,
                'posicion_anomalia': pos_inicio,
                'fin_anomalia': pos_fin
            })
        
        if len(datos_prueba) > 0:
            indices = np.random.permutation(len(datos_prueba))
            datos_prueba = np.array(datos_prueba)[indices]
            metadatos = [metadatos[i] for i in indices]
        else:
            datos_prueba = np.array(datos_prueba)
        
        return datos_prueba, metadatos
    
    def normalizar_datos(self, datos):
        media = np.mean(datos, axis=1, keepdims=True)
        std = np.std(datos, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)
        return (datos - media) / std, media, std
    
    def desnormalizar_datos(self, datos_normalizados, media, std):
        return datos_normalizados * std + media
    
    def guardar_datos(self, datos, metadatos, nombre_archivo):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_datos = f"{nombre_archivo}_datos_{timestamp}.csv"
        archivo_metadatos = f"{nombre_archivo}_metadatos_{timestamp}.csv"
        
        np.savetxt(archivo_datos, datos, delimiter=',', fmt='%.6f')
        
        with open(archivo_metadatos, 'w') as f:
            if metadatos:
                headers = list(metadatos[0].keys())
                f.write(','.join(headers) + '\n')
                
                for meta in metadatos:
                    values = [str(meta.get(h, '')) for h in headers]
                    f.write(','.join(values) + '\n')
        
        return archivo_datos, archivo_metadatos
