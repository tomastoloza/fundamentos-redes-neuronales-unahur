import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

TAMAÑO_IMAGEN = 32

SIMBOLOS_PREDEFINIDOS_30 = [
    '■', '●', '▲', '▼', '◆', '★',
    '☀', '☁', '☂', '☃', '☎', '☕',
    '✈', '✉', '✏', '✒', '✓', '✗',
    '✚', '✝', '✠', '✦', '✪', '✰',
    '✶', '✿', '❤', '❥', '❦', '❧'
]

RANGOS_UNICODE = {
    'geometricos': (0x25A0, 0x25FF),
    'flechas': (0x2190, 0x21FF),
    'simbolos_misc': (0x2600, 0x26FF),
    'dingbats': (0x2700, 0x27BF),
    'matematicos': (0x2200, 0x22FF)
}

def obtener_caracteres_unicode(rango_nombre='geometricos', cantidad=10):
    if rango_nombre not in RANGOS_UNICODE:
        print(f"Rango '{rango_nombre}' no encontrado. Usando 'geometricos'.")
        rango_nombre = 'geometricos'
    
    inicio, fin = RANGOS_UNICODE[rango_nombre]
    caracteres = []
    
    for codigo in range(inicio, min(inicio + cantidad * 3, fin)):
        try:
            char = chr(codigo)
            if char.isprintable() and not char.isspace():
                caracteres.append(char)
                if len(caracteres) >= cantidad:
                    break
        except:
            continue
    
    return caracteres

def obtener_simbolos_entrenamiento(tamaño_imagen=32):
    return obtener_bitmaps_wingdings(SIMBOLOS_PREDEFINIDOS_30, tamaño_imagen)

def obtener_fuente_unicode(tamaño):
    return FontProperties(family='DejaVu Sans', size=tamaño * 1.5, weight='bold')

def obtener_simbolos_predefinidos():
    return SIMBOLOS_PREDEFINIDOS_30.copy()

def renderizar_caracter_a_imagen(caracter, fuente, tamaño):
    fig, ax = plt.subplots(figsize=(1, 1), dpi=tamaño)
    ax.text(0.5, 0.5, caracter, ha='center', va='center', fontproperties=fuente)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)
    data_rgb = data[:, :, :3]
    plt.close(fig)
    
    return data_rgb

def convertir_a_escala_grises(imagen_rgb):
    return np.mean(imagen_rgb, axis=2)

def binarizar_imagen(imagen_gris):
    umbral = np.mean(imagen_gris[imagen_gris < 255]) * 0.9
    return (imagen_gris < umbral).astype(int)

def ajustar_tamaño_matriz(matriz, tamaño_objetivo):
    if matriz.shape == (tamaño_objetivo, tamaño_objetivo):
        return matriz
    
    h, w = matriz.shape
    start_h = (h - tamaño_objetivo) // 2
    start_w = (w - tamaño_objetivo) // 2
    
    matriz_recortada = matriz[start_h:start_h + tamaño_objetivo, start_w:start_w + tamaño_objetivo]
    
    if matriz_recortada.shape != (tamaño_objetivo, tamaño_objetivo):
        return np.zeros((tamaño_objetivo, tamaño_objetivo), dtype=int)
    
    return matriz_recortada

def generar_matriz_binaria(caracter, tamaño):
    fuente = obtener_fuente_unicode(tamaño)
    imagen_rgb = renderizar_caracter_a_imagen(caracter, fuente, tamaño)
    imagen_gris = convertir_a_escala_grises(imagen_rgb)
    matriz_binaria = binarizar_imagen(imagen_gris)
    matriz_final = ajustar_tamaño_matriz(matriz_binaria, tamaño)
    
    return matriz_final

def obtener_bitmaps_wingdings(caracteres, tamaño_imagen):
    bitmaps = []
    
    for caracter in caracteres:
        matriz = generar_matriz_binaria(caracter, tamaño_imagen)
        bitmaps.append(matriz)
    
    return bitmaps

def obtener_dataset_entrenamiento(tamaño_imagen=32):
    bitmaps = obtener_simbolos_entrenamiento(tamaño_imagen)
    X = np.array(bitmaps)
    X_flat = X.reshape(X.shape[0], -1)
    return X_flat, SIMBOLOS_PREDEFINIDOS_30