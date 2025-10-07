import os
import pickle
import numpy as np
from typing import Dict, Any

from tp3.src.autocodificador_caracteres import AutocodificadorCaracteres


def entrenar_modelo(
    arquitectura: str,
    dim_latente: int,
    datos: np.ndarray,
    tasa_aprendizaje: float,
    max_epocas: int,
    error_objetivo: float,
    paciencia: int,
    tipo_perdida: str = 'bce',
    monitor: str = 'val_loss',
    usar_scheduler: bool = True,
    batch_size: int = 32
) -> Dict[str, Any]:
    
    modelo = AutocodificadorCaracteres(
        dimension_latente=dim_latente,
        arquitectura=arquitectura
    )
    modelo.datos_entrenamiento = datos
    modelo.datos_cargados = True
    
    convergio, epocas = modelo.entrenar_autocodificador(
        tasa_aprendizaje=tasa_aprendizaje,
        max_epocas=max_epocas,
        error_objetivo=error_objetivo,
        paciencia=paciencia,
        tipo_perdida=tipo_perdida,
        monitor=monitor,
        usar_scheduler=usar_scheduler,
        batch_size=batch_size
    )
    
    reconstrucciones = modelo.reconstruir_caracteres(datos)
    mse = np.mean((datos - reconstrucciones) ** 2)
    precision = np.mean((datos > 0.5) == (reconstrucciones > 0.5)) * 100
    
    val_loss_final = modelo.historial_entrenamiento.history['val_loss'][-1]
    val_mse_final = modelo.historial_entrenamiento.history.get('val_mse', [mse])[-1]
    
    return {
        'modelo': modelo,
        'convergio': convergio,
        'epocas': epocas,
        'mse': mse,
        'precision': precision,
        'val_loss': val_loss_final,
        'val_mse': val_mse_final,
        'parametros': modelo.autocodificador.count_params()
    }


def guardar_modelo(modelo: AutocodificadorCaracteres, nombre_base: str, directorio: str = None):
    if directorio is None:
        directorio = os.path.join(os.path.dirname(__file__), '..', '..', 'modelos')
    
    os.makedirs(directorio, exist_ok=True)
    
    ruta_base = os.path.join(directorio, nombre_base)
    ruta_auto = f"{ruta_base}_autocodificador.keras"
    ruta_cod = f"{ruta_base}_codificador.keras"
    ruta_dec = f"{ruta_base}_decodificador.keras"
    ruta_meta = f"{ruta_base}_metadatos.pkl"
    
    modelo.autocodificador.save(ruta_auto)
    modelo.codificador.save(ruta_cod)
    modelo.decodificador.save(ruta_dec)
    
    metadatos = {
        'dimension_entrada': modelo.dimension_entrada,
        'dimension_latente': modelo.dimension_latente,
        'arquitectura': modelo.arquitectura,
        'datos_entrenamiento': modelo.datos_entrenamiento,
        'representaciones_latentes': modelo.representaciones_latentes,
        'historial_entrenamiento': modelo.historial_entrenamiento.history if modelo.historial_entrenamiento else None,
        'conjunto_datos': getattr(modelo, 'conjunto_datos_actual', None)
    }
    
    with open(ruta_meta, 'wb') as f:
        pickle.dump(metadatos, f)
    
    return {
        'autocodificador': ruta_auto,
        'codificador': ruta_cod,
        'decodificador': ruta_dec,
        'metadatos': ruta_meta
    }


def imprimir_encabezado_entrenamiento(titulo: str, detalles: Dict[str, Any] = None):
    print(f"\n{'='*60}")
    print(f"{titulo}")
    if detalles:
        for clave, valor in detalles.items():
            print(f"{clave}: {valor}")
    print(f"{'='*60}")
