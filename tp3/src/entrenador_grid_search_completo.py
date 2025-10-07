import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tp3.src.cargador_datos_caracteres import CargadorDatosCaracteres
from tp3.src.configuraciones_arquitecturas import obtener_arquitecturas_disponibles
from tp3.src.configuraciones_entrenamiento import obtener_configuracion_entrenamiento
from tp3.src.utilidades_entrenamiento import entrenar_modelo, guardar_modelo, imprimir_encabezado_entrenamiento


def entrenar_con_dimension_latente(dim_latente, arquitectura, epocas, lr, datos, config):
    imprimir_encabezado_entrenamiento(
        f"ENTRENANDO",
        {
            'Arquitectura': arquitectura,
            'Dim Latente': dim_latente,
            'Épocas': epocas,
            'Learning Rate': lr
        }
    )
    
    resultado = entrenar_modelo(
        arquitectura=arquitectura,
        dim_latente=dim_latente,
        datos=datos,
        tasa_aprendizaje=lr,
        max_epocas=epocas,
        error_objetivo=config['error_objetivo'],
        paciencia=config['paciencia'],
        tipo_perdida=config['tipo_perdida'],
        monitor=config['monitor'],
        usar_scheduler=config['usar_scheduler'],
        batch_size=config['batch_size']
    )
    
    nombre_base = f"tp3_{arquitectura}_lat{dim_latente}_ep{epocas}_lr{str(lr).replace('.', '_')}"
    guardar_modelo(resultado['modelo'], nombre_base)
    
    return {
        'nombre_modelo': nombre_base,
        'dim_latente': dim_latente,
        'arquitectura': arquitectura,
        'epocas_max': epocas,
        'learning_rate': lr,
        'convergio': resultado['convergio'],
        'epocas_reales': resultado['epocas'],
        'mse': resultado['mse'],
        'precision': resultado['precision'],
        'parametros': resultado['parametros']
    }


def cargar_modelos_entrenados():
    resultados_dir = os.path.join(os.path.dirname(__file__), '..', 'resultados')
    ruta_csv = os.path.join(resultados_dir, 'grid_search_completo.csv')
    
    if not os.path.exists(ruta_csv):
        return {}
    
    modelos_entrenados = {}
    with open(ruta_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nombre_modelo = row['nombre_modelo']
            modelos_entrenados[nombre_modelo] = {
                'nombre_modelo': nombre_modelo,
                'dim_latente': int(row['dim_latente']),
                'arquitectura': row['arquitectura'],
                'epocas_max': int(row['epocas_max']),
                'learning_rate': float(row['learning_rate']),
                'convergio': row['convergio'] == 'True',
                'epocas_reales': int(row['epocas_reales']),
                'mse': float(row['mse']),
                'precision': float(row['precision']),
                'parametros': int(row['parametros'])
            }
    
    return modelos_entrenados


def guardar_csv(resultados, mostrar_mensaje=True):
    resultados_dir = os.path.join(os.path.dirname(__file__), '..', 'resultados')
    os.makedirs(resultados_dir, exist_ok=True)
    
    ruta_csv = os.path.join(resultados_dir, 'grid_search_completo.csv')
    
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'nombre_modelo', 'dim_latente', 'arquitectura', 'epocas_max', 'learning_rate',
            'convergio', 'epocas_reales', 'mse', 'precision', 'parametros'
        ])
        writer.writeheader()
        for r in sorted(resultados, key=lambda x: (x['dim_latente'], x['mse'])):
            writer.writerow(r)
    
    if mostrar_mensaje:
        print(f"💾 CSV guardado: {ruta_csv}")
    return ruta_csv




def visualizar_comparacion(resultados):
    resultados_dir = os.path.join(os.path.dirname(__file__), '..', 'resultados')
    
    df = pd.DataFrame(resultados)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    dimensiones = sorted(df['dim_latente'].unique())
    arquitecturas = sorted(df['arquitectura'].unique())
    lrs = sorted(df['learning_rate'].unique())
    
    ax1 = fig.add_subplot(gs[0, 0])
    for arq in arquitecturas:
        df_arq = df[df['arquitectura'] == arq]
        mse_por_dim = df_arq.groupby('dim_latente')['mse'].mean()
        ax1.plot(mse_por_dim.index, mse_por_dim.values, marker='o', label=arq, linewidth=2)
    ax1.set_xlabel('Dimensión Latente', fontweight='bold')
    ax1.set_ylabel('MSE Promedio', fontweight='bold')
    ax1.set_title('MSE vs Dimensión Latente por Arquitectura', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    for arq in arquitecturas:
        df_arq = df[df['arquitectura'] == arq]
        prec_por_dim = df_arq.groupby('dim_latente')['precision'].mean()
        ax2.plot(prec_por_dim.index, prec_por_dim.values, marker='o', label=arq, linewidth=2)
    ax2.set_xlabel('Dimensión Latente', fontweight='bold')
    ax2.set_ylabel('Precisión Promedio (%)', fontweight='bold')
    ax2.set_title('Precisión vs Dimensión Latente por Arquitectura', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    df_boxplot = [df[df['dim_latente'] == dim]['mse'].values for dim in dimensiones]
    bp = ax3.boxplot(df_boxplot, labels=dimensiones, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_xlabel('Dimensión Latente', fontweight='bold')
    ax3.set_ylabel('MSE', fontweight='bold')
    ax3.set_title('Distribución de MSE por Dimensión', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    for lr in lrs:
        df_lr = df[df['learning_rate'] == lr]
        mse_por_dim = df_lr.groupby('dim_latente')['mse'].mean()
        ax4.plot(mse_por_dim.index, mse_por_dim.values, marker='s', label=f'LR={lr}', linewidth=2)
    ax4.set_xlabel('Dimensión Latente', fontweight='bold')
    ax4.set_ylabel('MSE Promedio', fontweight='bold')
    ax4.set_title('MSE vs Dimensión por Learning Rate', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(df['dim_latente'], df['mse'], 
                         c=df['learning_rate'], cmap='viridis', 
                         s=100, alpha=0.6, edgecolors='black')
    ax5.set_xlabel('Dimensión Latente', fontweight='bold')
    ax5.set_ylabel('MSE', fontweight='bold')
    ax5.set_title('MSE vs Dimensión (color = LR)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Learning Rate', fontweight='bold')
    
    ax6 = fig.add_subplot(gs[1, 2])
    mejores_por_dim = df.loc[df.groupby('dim_latente')['mse'].idxmin()]
    ax6.bar(mejores_por_dim['dim_latente'], mejores_por_dim['precision'], 
            color='green', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Dimensión Latente', fontweight='bold')
    ax6.set_ylabel('Precisión (%)', fontweight='bold')
    ax6.set_title('Mejor Precisión por Dimensión', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    for i, row in mejores_por_dim.iterrows():
        ax6.text(row['dim_latente'], row['precision'] + 1, 
                f"{row['precision']:.1f}%", ha='center', fontsize=9)
    
    ax7 = fig.add_subplot(gs[2, 0])
    convergidos = df.groupby('dim_latente')['convergio'].sum()
    total_por_dim = df.groupby('dim_latente').size()
    tasa_convergencia = (convergidos / total_por_dim * 100)
    ax7.bar(tasa_convergencia.index, tasa_convergencia.values, 
            color='orange', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Dimensión Latente', fontweight='bold')
    ax7.set_ylabel('Tasa de Convergencia (%)', fontweight='bold')
    ax7.set_title('Convergencia por Dimensión', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    ax8 = fig.add_subplot(gs[2, 1])
    for arq in arquitecturas:
        df_arq = df[df['arquitectura'] == arq]
        epocas_por_dim = df_arq.groupby('dim_latente')['epocas_reales'].mean()
        ax8.plot(epocas_por_dim.index, epocas_por_dim.values, marker='o', label=arq, linewidth=2)
    ax8.set_xlabel('Dimensión Latente', fontweight='bold')
    ax8.set_ylabel('Épocas Promedio', fontweight='bold')
    ax8.set_title('Épocas de Entrenamiento por Arquitectura', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, 2])
    top10 = df.nsmallest(10, 'mse')
    labels = [f"{row['arquitectura'][:8]}\nL={row['dim_latente']}" for _, row in top10.iterrows()]
    colors = plt.cm.RdYlGn_r(top10['mse'] / top10['mse'].max())
    ax9.barh(range(len(top10)), top10['mse'], color=colors, edgecolor='black')
    ax9.set_yticks(range(len(top10)))
    ax9.set_yticklabels(labels, fontsize=8)
    ax9.set_xlabel('MSE', fontweight='bold')
    ax9.set_title('Top 10 Mejores Modelos', fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='x')
    ax9.invert_yaxis()
    
    plt.suptitle('Análisis Completo - Grid Search de Dimensiones Latentes', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    ruta_grafico = os.path.join(resultados_dir, 'grid_search_completo.png')
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    print(f"💾 Gráfico guardado: {ruta_grafico}")
    plt.close()
    
    return ruta_grafico


def main():
    print("="*60)
    print("ENTRENADOR GRID SEARCH COMPLETO - TODAS LAS COMBINACIONES")
    print("="*60)
    
    print("\n🔍 Cargando modelos ya entrenados...")
    modelos_entrenados = cargar_modelos_entrenados()
    print(f"   ✅ {len(modelos_entrenados)} modelos encontrados en CSV")
    
    cargador = CargadorDatosCaracteres()
    datos, etiquetas = cargador.cargar_datos_desde_modulo(1)
    
    config = obtener_configuracion_entrenamiento()
    arquitecturas = obtener_arquitecturas_disponibles()
    
    total_experimentos = (len(config['dimensiones_latentes']) * 
                         len(arquitecturas) * 
                         len(config['epocas_lista']) * 
                         len(config['tasas_aprendizaje']))
    
    print(f"\n📊 Configuración del Grid Search:")
    print(f"   - Dimensiones latentes: {config['dimensiones_latentes']}")
    print(f"   - Arquitecturas: {list(arquitecturas.keys())}")
    print(f"   - Épocas: {config['epocas_lista']}")
    print(f"   - Learning rates: {config['tasas_aprendizaje']}")
    print(f"   - TOTAL EXPERIMENTOS: {total_experimentos}")
    
    resultados = list(modelos_entrenados.values())
    
    experimentos_pendientes = []
    experimentos_skipeados = 0
    
    for dim_latente in config['dimensiones_latentes']:
        for nombre_arq in arquitecturas.keys():
            for epocas in config['epocas_lista']:
                for lr in config['tasas_aprendizaje']:
                    nombre_modelo = f"tp3_{nombre_arq}_lat{dim_latente}_ep{epocas}_lr{str(lr).replace('.', '_')}"
                    
                    if nombre_modelo in modelos_entrenados:
                        experimentos_skipeados += 1
                    else:
                        experimentos_pendientes.append((dim_latente, nombre_arq, epocas, lr, datos, config))
    
    print(f"\n📋 Estado de experimentos:")
    print(f"   ✅ Ya entrenados: {experimentos_skipeados}")
    print(f"   ⏳ Pendientes: {len(experimentos_pendientes)}")
    print(f"   📊 Total: {total_experimentos}")
    
    if len(experimentos_pendientes) == 0:
        print(f"\n🎉 ¡Todos los modelos ya están entrenados!")
        print(f"\n{'='*60}")
        print("GENERANDO RESULTADOS")
        print(f"{'='*60}")
        guardar_csv(resultados)
        visualizar_comparacion(resultados)
        return
    
    max_workers = max(1, min(len(experimentos_pendientes), os.cpu_count() or 2))
    print(f"   🔧 Workers paralelos: {max_workers}")
    
    print(f"\n🚀 Iniciando entrenamiento paralelo...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futuros = {
            executor.submit(entrenar_con_dimension_latente, dim, arq, ep, lr, datos, config): (dim, arq, ep, lr)
            for dim, arq, ep, lr, datos, config in experimentos_pendientes
        }
        
        completados = 0
        for futuro in as_completed(futuros):
            dim, arq, ep, lr = futuros[futuro]
            completados += 1
            
            try:
                resultado = futuro.result()
                resultados.append(resultado)
                
                print(f"\n✅ [{completados}/{len(experimentos_pendientes)}] L={dim}, {arq}, ep={ep}, lr={lr}")
                print(f"   MSE: {resultado['mse']:.4f}, Precisión: {resultado['precision']:.1f}%, Convergió: {'Sí' if resultado['convergio'] else 'No'}")
                
                guardar_csv(resultados, mostrar_mensaje=False)
                print(f"   💾 CSV actualizado ({len(resultados)} modelos totales)")
            except Exception as e:
                print(f"\n❌ [{completados}/{len(experimentos_pendientes)}] Error en L={dim}, {arq}, ep={ep}, lr={lr}: {e}")
    
    print(f"\n{'='*60}")
    print("GENERANDO RESULTADOS")
    print(f"{'='*60}")
    
    guardar_csv(resultados)
    visualizar_comparacion(resultados)
    
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    
    mejor_resultado = min(resultados, key=lambda x: x['mse'])
    print(f"\n🏆 Mejor modelo global:")
    print(f"   Dimensión latente: {mejor_resultado['dim_latente']}")
    print(f"   Arquitectura: {mejor_resultado['arquitectura']}")
    print(f"   Épocas: {mejor_resultado['epocas_max']}")
    print(f"   Learning Rate: {mejor_resultado['learning_rate']}")
    print(f"   MSE: {mejor_resultado['mse']:.4f}")
    print(f"   Precisión: {mejor_resultado['precision']:.1f}%")
    print(f"   Convergió: {'Sí' if mejor_resultado['convergio'] else 'No'}")
    
    print(f"\n📊 Comparación por dimensión latente:")
    for dim in sorted(set(r['dim_latente'] for r in resultados)):
        resultados_dim = [r for r in resultados if r['dim_latente'] == dim]
        mejor_dim = min(resultados_dim, key=lambda x: x['mse'])
        mse_promedio = np.mean([r['mse'] for r in resultados_dim])
        prec_promedio = np.mean([r['precision'] for r in resultados_dim])
        print(f"   L={dim}: MSE promedio={mse_promedio:.4f}, Precisión promedio={prec_promedio:.1f}%")
        print(f"         Mejor: {mejor_dim['arquitectura']} (ep={mejor_dim['epocas_max']}, lr={mejor_dim['learning_rate']}) - MSE={mejor_dim['mse']:.4f}")


if __name__ == "__main__":
    main()
