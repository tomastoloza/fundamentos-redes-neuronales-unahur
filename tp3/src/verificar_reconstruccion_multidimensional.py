import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import argparse
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tp3.src.autocodificador_caracteres import AutocodificadorCaracteres
from tp3.src.cargador_datos_caracteres import CargadorDatosCaracteres
from tp3.src.configuraciones_arquitecturas import obtener_arquitecturas_disponibles


def escanear_modelos():
    modelos_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'modelos')
    
    if not os.path.exists(modelos_dir):
        return {}
    
    archivos = glob.glob(os.path.join(modelos_dir, '*_autocodificador.keras'))
    modelos = {}
    arquitecturas_validas = list(obtener_arquitecturas_disponibles().keys())
    
    for archivo in archivos:
        nombre_archivo = os.path.basename(archivo)
        nombre_base = nombre_archivo.replace('_autocodificador.keras', '')
        
        partes = nombre_base.split('_')
        
        dim_latente = None
        for parte in partes:
            if parte.startswith('lat'):
                try:
                    dim_latente = int(parte[3:])
                    break
                except:
                    pass
        
        if dim_latente is None:
            continue
        
        arquitectura = None
        for parte in partes:
            if parte in arquitecturas_validas:
                arquitectura = parte
                break
        
        if arquitectura is None:
            for i in range(len(partes)):
                candidato = '_'.join(partes[1:i+2])
                if candidato in arquitecturas_validas:
                    arquitectura = candidato
                    break
        
        if arquitectura is None:
            continue
        
        epocas = None
        lr = None
        for parte in partes:
            if parte.startswith('ep'):
                try:
                    epocas = int(parte[2:])
                except:
                    pass
            elif parte.startswith('lr'):
                try:
                    lr_str = parte[2:].replace('_', '.')
                    lr = float(lr_str)
                except:
                    pass
        
        info = {
            'nombre_base': nombre_base,
            'arquitectura': arquitectura,
            'dim_latente': dim_latente,
            'epocas': epocas,
            'lr': lr,
            'ruta': archivo
        }
        
        if dim_latente not in modelos:
            modelos[dim_latente] = []
        
        modelos[dim_latente].append(info)
    
    for dim in modelos:
        modelos[dim].sort(key=lambda x: (x['arquitectura'], x['epocas'] or 0, x['lr'] or 0))
    
    return modelos


def cargar_modelo(info_modelo):
    nombre_base = info_modelo['nombre_base']
    dim_latente = info_modelo['dim_latente']
    arquitectura = info_modelo['arquitectura']
    
    modelos_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'modelos')
    
    ruta_autocodificador = os.path.join(modelos_dir, f"{nombre_base}_autocodificador.keras")
    ruta_codificador = os.path.join(modelos_dir, f"{nombre_base}_codificador.keras")
    ruta_decodificador = os.path.join(modelos_dir, f"{nombre_base}_decodificador.keras")
    
    if not all(os.path.exists(r) for r in [ruta_autocodificador, ruta_codificador, ruta_decodificador]):
        return None
    
    autocodificador_obj = AutocodificadorCaracteres(
        dimension_latente=dim_latente,
        arquitectura=arquitectura
    )
    
    autocodificador_obj.autocodificador = keras.models.load_model(ruta_autocodificador)
    autocodificador_obj.codificador = keras.models.load_model(ruta_codificador)
    autocodificador_obj.decodificador = keras.models.load_model(ruta_decodificador)
    autocodificador_obj.entrenado = True
    
    return autocodificador_obj


def verificar_reconstruccion_todos(autocodificador, datos, info_modelo):
    caracteres_nombres = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
                          ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                          '8', '9', ':', ';', '<', '=', '>', '?']
    
    reconstrucciones = autocodificador.reconstruir_caracteres(datos)
    
    num_caracteres = len(datos)
    num_cols = 8
    num_rows = (num_caracteres + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows * 2, num_cols, figsize=(20, num_rows * 5))
    
    for idx in range(num_caracteres):
        row = (idx // num_cols) * 2
        col = idx % num_cols
        
        original = datos[idx].reshape(7, 5)
        reconstruido = reconstrucciones[idx].reshape(7, 5)
        reconstruido_bin = (reconstruido > 0.5).astype(float)
        
        ax_orig = axes[row, col] if num_rows > 1 else axes[0, col]
        ax_orig.imshow(original, cmap='binary', interpolation='nearest', vmin=0, vmax=1)
        ax_orig.set_title(f'{caracteres_nombres[idx]} - Original', fontsize=10, fontweight='bold')
        ax_orig.axis('off')
        
        ax_recon = axes[row + 1, col] if num_rows > 1 else axes[1, col]
        ax_recon.imshow(reconstruido_bin, cmap='binary', interpolation='nearest', vmin=0, vmax=1)
        
        mse = np.mean((original - reconstruido) ** 2)
        precision = np.mean((original > 0.5) == (reconstruido > 0.5)) * 100
        
        ax_recon.set_title(f'Reconstruido\nMSE:{mse:.3f} Prec:{precision:.1f}%', 
                          fontsize=9)
        ax_recon.axis('off')
    
    for idx in range(num_caracteres, num_rows * num_cols):
        row = (idx // num_cols) * 2
        col = idx % num_cols
        if num_rows > 1:
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')
        else:
            axes[0, col].axis('off')
            axes[1, col].axis('off')
    
    titulo = f"Comparación Original vs Reconstruido (L={info_modelo['dim_latente']})\n"
    titulo += f"Modelo: {info_modelo['arquitectura']}"
    if info_modelo['epocas']:
        titulo += f" | Épocas: {info_modelo['epocas']}"
    if info_modelo['lr']:
        titulo += f" | LR: {info_modelo['lr']}"
    
    plt.suptitle(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    resultados_dir = os.path.join(os.path.dirname(__file__), '..', 'resultados')
    os.makedirs(resultados_dir, exist_ok=True)
    nombre_archivo = f"verificacion_{info_modelo['nombre_base']}.png"
    ruta_imagen = os.path.join(resultados_dir, nombre_archivo)
    plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
    print(f"\n💾 Guardado: {ruta_imagen}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Verificar reconstrucción de caracteres')
    parser.add_argument('--dim', type=int, help='Dimensión latente del modelo')
    parser.add_argument('--arquitectura', type=str, choices=['TinyAE', 'BalancedAE', 'DeepSparseAE'],
                       help='Arquitectura específica')
    parser.add_argument('--modelo', type=str, help='Nombre completo del modelo')
    parser.add_argument('--listar', action='store_true', help='Listar modelos disponibles')
    parser.add_argument('--comparar', action='store_true', help='Comparar múltiples dimensiones')
    parser.add_argument('--todos', action='store_true', help='Verificar todos los modelos')
    
    args = parser.parse_args()
    
    if args.listar:
        print("="*60)
        print("MODELOS DISPONIBLES")
        print("="*60)
        modelos = escanear_modelos()
        
        total = sum(len(v) for v in modelos.values())
        print(f"\nTotal: {total} modelos encontrados\n")
        
        for dim in sorted(modelos.keys()):
            print(f"\n{'='*60}")
            print(f"Dimensión Latente = {dim} ({len(modelos[dim])} modelos)")
            print(f"{'='*60}")
            for info in modelos[dim]:
                epocas_str = f"ep{info['epocas']}" if info['epocas'] else ""
                lr_str = f"lr{info['lr']}" if info['lr'] else ""
                print(f"  {info['arquitectura']:15s} {epocas_str:8s} {lr_str:12s} | {info['nombre_base']}")
        return
    
    if args.comparar:
        print("="*60)
        print("COMPARACIÓN DE RECONSTRUCCIÓN (TODAS LAS DIMENSIONES)")
        print("="*60)
        
        cargador = CargadorDatosCaracteres()
        datos, etiquetas = cargador.cargar_datos_desde_modulo(1)
        
        modelos_disponibles = escanear_modelos()
        resultados_comparacion = []
        
        arquitectura_filtro = args.arquitectura if args.arquitectura else None
        
        for dim in sorted(modelos_disponibles.keys()):
            modelos_dim = modelos_disponibles[dim]
            
            if arquitectura_filtro:
                modelos_dim = [m for m in modelos_dim if m['arquitectura'] == arquitectura_filtro]
            
            if len(modelos_dim) > 0:
                info_modelo = modelos_dim[0]
                print(f"\n📊 Procesando L={dim}: {info_modelo['arquitectura']}")
                
                autocodificador = cargar_modelo(info_modelo)
                if autocodificador:
                    autocodificador.datos_entrenamiento = datos
                    autocodificador.datos_cargados = True
                    autocodificador.obtener_representacion_latente()
                    
                    reconstrucciones = autocodificador.reconstruir_caracteres(datos)
                    mse = np.mean((datos - reconstrucciones) ** 2)
                    precision = np.mean((datos > 0.5) == (reconstrucciones > 0.5)) * 100
                    
                    resultados_comparacion.append({
                        'dim': dim,
                        'arquitectura': info_modelo['arquitectura'],
                        'mse': mse,
                        'precision': precision
                    })
                    
                    print(f"   MSE: {mse:.4f}, Precisión: {precision:.1f}%")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        dims = [r['dim'] for r in resultados_comparacion]
        mses = [r['mse'] for r in resultados_comparacion]
        precs = [r['precision'] for r in resultados_comparacion]
        
        ax1.plot(dims, mses, 'o-', linewidth=2, markersize=10)
        ax1.set_xlabel('Dimensión Latente', fontweight='bold', fontsize=12)
        ax1.set_ylabel('MSE', fontweight='bold', fontsize=12)
        ax1.set_title('MSE vs Dimensión Latente', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(dims)
        
        ax2.plot(dims, precs, 'o-', linewidth=2, markersize=10, color='green')
        ax2.set_xlabel('Dimensión Latente', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Precisión (%)', fontweight='bold', fontsize=12)
        ax2.set_title('Precisión vs Dimensión Latente', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(dims)
        
        plt.tight_layout()
        
        resultados_dir = os.path.join(os.path.dirname(__file__), '..', 'resultados')
        ruta_grafico = os.path.join(resultados_dir, 'comparacion_reconstruccion_dimensiones.png')
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        print(f"\n💾 Gráfico comparativo guardado: {ruta_grafico}")
        
        plt.show()
        return
    
    if args.todos:
        print("="*60)
        print("VERIFICACIÓN DE TODOS LOS MODELOS")
        print("="*60)
        
        cargador = CargadorDatosCaracteres()
        datos, etiquetas = cargador.cargar_datos_desde_modulo(1)
        
        modelos_disponibles = escanear_modelos()
        
        for dim in sorted(modelos_disponibles.keys()):
            for info_modelo in modelos_disponibles[dim]:
                print(f"\n{'='*60}")
                print(f"Verificando: {info_modelo['nombre_base']}")
                print(f"{'='*60}")
                
                autocodificador = cargar_modelo(info_modelo)
                if autocodificador:
                    autocodificador.datos_entrenamiento = datos
                    autocodificador.datos_cargados = True
                    autocodificador.obtener_representacion_latente()
                    
                    verificar_reconstruccion_todos(autocodificador, datos, info_modelo)
                    plt.close()
        
        print(f"\n✅ Verificación completa de todos los modelos")
        return
    
    modelos_disponibles = escanear_modelos()
    
    info_modelo = None
    
    if args.modelo:
        for dim in modelos_disponibles:
            for info in modelos_disponibles[dim]:
                if info['nombre_base'] == args.modelo:
                    info_modelo = info
                    break
            if info_modelo:
                break
        
        if not info_modelo:
            print(f"❌ No se encontró el modelo: {args.modelo}")
            print("\nUsa --listar para ver modelos disponibles")
            return
    else:
        if not args.dim:
            print("❌ Debes especificar --dim o --modelo")
            print("\nUsa --listar para ver modelos disponibles")
            return
        
        if args.dim not in modelos_disponibles or len(modelos_disponibles[args.dim]) == 0:
            print(f"❌ No hay modelos disponibles para L={args.dim}")
            print("\nUsa --listar para ver modelos disponibles")
            return
        
        modelos_filtrados = modelos_disponibles[args.dim]
        
        if args.arquitectura:
            modelos_filtrados = [m for m in modelos_filtrados if m['arquitectura'] == args.arquitectura]
            if not modelos_filtrados:
                print(f"❌ No hay modelos {args.arquitectura} para L={args.dim}")
                return
        
        info_modelo = modelos_filtrados[0]
    
    print("="*60)
    print(f"VERIFICACIÓN DE RECONSTRUCCIÓN")
    print("="*60)
    print(f"Modelo: {info_modelo['nombre_base']}")
    print(f"Arquitectura: {info_modelo['arquitectura']}")
    print(f"Dimensión Latente: {info_modelo['dim_latente']}")
    if info_modelo['epocas']:
        print(f"Épocas: {info_modelo['epocas']}")
    if info_modelo['lr']:
        print(f"Learning Rate: {info_modelo['lr']}")
    print("="*60)
    
    autocodificador = cargar_modelo(info_modelo)
    if autocodificador is None:
        print(f"❌ No se pudo cargar el modelo")
        return
    
    cargador = CargadorDatosCaracteres()
    datos, etiquetas = cargador.cargar_datos_desde_modulo(1)
    
    autocodificador.datos_entrenamiento = datos
    autocodificador.datos_cargados = True
    
    print(f"\n📊 Generando representaciones latentes...")
    autocodificador.obtener_representacion_latente()
    
    print(f"\n🔍 Verificando reconstrucción de todos los caracteres...")
    fig = verificar_reconstruccion_todos(autocodificador, datos, info_modelo)
    
    reconstrucciones = autocodificador.reconstruir_caracteres(datos)
    
    mse_total = np.mean((datos - reconstrucciones) ** 2)
    precision_total = np.mean((datos > 0.5) == (reconstrucciones > 0.5)) * 100
    
    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS GLOBALES")
    print(f"{'='*60}")
    print(f"Modelo: {info_modelo['nombre_base']}")
    print(f"Dimensión Latente: {info_modelo['dim_latente']}")
    print(f"MSE promedio: {mse_total:.4f}")
    print(f"Precisión binaria promedio: {precision_total:.2f}%")
    
    caracteres_nombres = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
                          ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                          '8', '9', ':', ';', '<', '=', '>', '?']
    
    print(f"\n{'='*60}")
    print("DETALLE POR CARÁCTER")
    print(f"{'='*60}")
    
    for idx in range(len(datos)):
        original = datos[idx]
        reconstruido = reconstrucciones[idx]
        
        mse = np.mean((original - reconstruido) ** 2)
        precision = np.mean((original > 0.5) == (reconstruido > 0.5)) * 100
        
        punto_latente = autocodificador.representaciones_latentes[idx]
        latente_str = ', '.join([f'{v:.3f}' for v in punto_latente[:min(3, len(punto_latente))]])
        if len(punto_latente) > 3:
            latente_str += '...'
        
        print(f"{caracteres_nombres[idx]:2s} | MSE: {mse:.4f} | Precisión: {precision:5.1f}% | "
              f"Latente: [{latente_str}]")
    
    plt.show()


if __name__ == "__main__":
    main()
