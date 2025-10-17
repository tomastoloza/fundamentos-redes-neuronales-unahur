import os
from PIL import Image


def preprocesar_dataset_suricatas():
    carpeta_origen = 'tp4/datos/suricatas'
    carpeta_destino = 'tp4/datos/suricatas_200x200'
    tamaño_final = (200, 200)
    color_relleno = (0, 0, 0)
    
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    
    archivos_procesados = 0
    archivos_error = 0
    
    print(f"Procesando imágenes de {carpeta_origen} a {carpeta_destino}")
    print(f"Tamaño objetivo: {tamaño_final[0]}x{tamaño_final[1]}px")
    print(f"Color de relleno: RGB{color_relleno}")
    print("="*60)
    
    for nombre_archivo in os.listdir(carpeta_origen):
        ruta_archivo = os.path.join(carpeta_origen, nombre_archivo)
        
        if os.path.isfile(ruta_archivo) and nombre_archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                with Image.open(ruta_archivo) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img.thumbnail(tamaño_final, Image.Resampling.LANCZOS)
                    
                    fondo = Image.new('RGB', tamaño_final, color_relleno)
                    
                    posicion = (
                        (tamaño_final[0] - img.width) // 2,
                        (tamaño_final[1] - img.height) // 2
                    )
                    
                    fondo.paste(img, posicion)
                    
                    ruta_destino = os.path.join(carpeta_destino, nombre_archivo)
                    fondo.save(ruta_destino, quality=95)
                    
                    archivos_procesados += 1
                    if archivos_procesados % 20 == 0:
                        print(f"Procesadas {archivos_procesados} imágenes...")
                    
            except Exception as e:
                print(f"✗ Error procesando {nombre_archivo}: {e}")
                archivos_error += 1
    
    print("="*60)
    print(f"✓ Proceso completado")
    print(f"Imágenes procesadas exitosamente: {archivos_procesados}")
    print(f"Errores: {archivos_error}")
    print(f"Dataset final disponible en: {carpeta_destino}")


def redimensionar_con_padding(imagen_pil, tamaño_objetivo=(200, 200), color_relleno=(0, 0, 0)):
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    
    imagen_pil.thumbnail(tamaño_objetivo, Image.Resampling.LANCZOS)
    
    fondo = Image.new('RGB', tamaño_objetivo, color_relleno)
    
    posicion = (
        (tamaño_objetivo[0] - imagen_pil.width) // 2,
        (tamaño_objetivo[1] - imagen_pil.height) // 2
    )
    
    fondo.paste(imagen_pil, posicion)
    
    return fondo


def procesar_imagen_individual(ruta_entrada, ruta_salida, tamaño=(200, 200)):
    try:
        with Image.open(ruta_entrada) as img:
            imagen_procesada = redimensionar_con_padding(img, tamaño)
            imagen_procesada.save(ruta_salida, quality=95)
            return True
    except Exception as e:
        print(f"Error procesando {ruta_entrada}: {e}")
        return False


def verificar_dataset_procesado():
    carpeta_original = 'tp4/datos/suricatas'
    carpeta_procesada = 'tp4/datos/suricatas_200x200'
    
    if not os.path.exists(carpeta_procesada):
        print(f"El dataset procesado no existe en {carpeta_procesada}")
        return False
    
    archivos_original = len([f for f in os.listdir(carpeta_original) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    archivos_procesada = len([f for f in os.listdir(carpeta_procesada) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    print(f"Dataset original: {archivos_original} imágenes")
    print(f"Dataset procesado: {archivos_procesada} imágenes")
    
    if archivos_original == archivos_procesada:
        print("✓ Todos los archivos fueron procesados correctamente")
        
        imagen_muestra = os.listdir(carpeta_procesada)[0]
        ruta_muestra = os.path.join(carpeta_procesada, imagen_muestra)
        
        with Image.open(ruta_muestra) as img:
            print(f"✓ Verificación de tamaño: {img.size} (esperado: (200, 200))")
            print(f"✓ Modo de color: {img.mode}")
        
        return True
    else:
        print(f"⚠️  Faltan {archivos_original - archivos_procesada} archivos")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocesador de imágenes para TP4')
    parser.add_argument('--procesar', action='store_true',
                       help='Procesar todas las imágenes del dataset')
    parser.add_argument('--verificar', action='store_true',
                       help='Verificar el dataset procesado')
    parser.add_argument('--tamaño', type=str, default='200x200',
                       help='Tamaño objetivo (ej: 200x200)')
    
    args = parser.parse_args()
    
    if args.verificar:
        verificar_dataset_procesado()
    elif args.procesar:
        try:
            ancho, alto = map(int, args.tamaño.split('x'))
            print(f"Procesando con tamaño {ancho}x{alto}")
            preprocesar_dataset_suricatas()
        except:
            print(f"Error: Formato de tamaño inválido '{args.tamaño}'")
    else:
        print("Uso: python3 -m tp4.preprocesar_imagenes --procesar [--tamaño 200x200]")
        print("     python3 -m tp4.preprocesar_imagenes --verificar")
