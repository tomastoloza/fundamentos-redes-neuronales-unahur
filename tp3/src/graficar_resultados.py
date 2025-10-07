import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def cargar_csv():
    ruta_csv = os.path.join(os.path.dirname(__file__), '..', 'resultados', 'grid_search_completo.csv')
    if not os.path.exists(ruta_csv):
        print(f"❌ No se encontró el archivo: {ruta_csv}")
        print("Ejecuta primero: python -m tp3.src.entrenador_grid_search_completo")
        return None
    
    df = pd.read_csv(ruta_csv)
    print(f"✅ CSV cargado: {len(df)} experimentos")
    return df


def graficar_heatmap_mse(df):
    resultados_dir = os.path.join(os.path.dirname(__file__), '..', 'resultados')
    
    arquitecturas = sorted(df['arquitectura'].unique())
    n_arqs = len(arquitecturas)
    n_cols = min(3, n_arqs)
    n_rows = (n_arqs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_arqs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, arq in enumerate(arquitecturas):
        df_arq = df[df['arquitectura'] == arq]
        pivot = df_arq.pivot_table(values='mse', index='learning_rate', columns='dim_latente', aggfunc='mean')
        
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                   ax=axes[idx], cbar_kws={'label': 'MSE'})
        axes[idx].set_title(f'{arq}', fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Dimensión Latente', fontweight='bold')
        axes[idx].set_ylabel('Learning Rate', fontweight='bold')
    
    for idx in range(n_arqs, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Heatmap de MSE por Arquitectura', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    ruta = os.path.join(resultados_dir, 'heatmap_mse.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    print(f"💾 Heatmap guardado: {ruta}")
    plt.close()


def graficar_precision_por_config(df):
    resultados_dir = os.path.join(os.path.dirname(__file__), '..', 'resultados')
    
    arquitecturas = sorted(df['arquitectura'].unique())
    n_arqs = len(arquitecturas)
    n_cols = min(3, n_arqs)
    n_rows = (n_arqs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_arqs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, arq in enumerate(arquitecturas):
        df_arq = df[df['arquitectura'] == arq]
        pivot = df_arq.pivot_table(values='precision', index='learning_rate', columns='dim_latente', aggfunc='mean')
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGn', 
                   ax=axes[idx], cbar_kws={'label': 'Precisión (%)'})
        axes[idx].set_title(f'{arq}', fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Dimensión Latente', fontweight='bold')
        axes[idx].set_ylabel('Learning Rate', fontweight='bold')
    
    for idx in range(n_arqs, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Heatmap de Precisión por Arquitectura', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    ruta = os.path.join(resultados_dir, 'heatmap_precision.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    print(f"💾 Heatmap guardado: {ruta}")
    plt.close()


def graficar_comparacion_lr(df):
    resultados_dir = os.path.join(os.path.dirname(__file__), '..', 'resultados')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for arq in sorted(df['arquitectura'].unique()):
        df_arq = df[df['arquitectura'] == arq]
        for lr in sorted(df_arq['learning_rate'].unique()):
            df_lr = df_arq[df_arq['learning_rate'] == lr]
            mse_por_dim = df_lr.groupby('dim_latente')['mse'].mean()
            axes[0, 0].plot(mse_por_dim.index, mse_por_dim.values, 
                          marker='o', label=f'{arq} LR={lr}', linewidth=2)
    
    axes[0, 0].set_xlabel('Dimensión Latente', fontweight='bold')
    axes[0, 0].set_ylabel('MSE', fontweight='bold')
    axes[0, 0].set_title('MSE vs Dimensión Latente', fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    mejores = df.loc[df.groupby(['arquitectura', 'dim_latente'])['mse'].idxmin()]
    for arq in sorted(mejores['arquitectura'].unique()):
        df_arq = mejores[mejores['arquitectura'] == arq]
        axes[0, 1].plot(df_arq['dim_latente'], df_arq['precision'], 
                       marker='s', label=arq, linewidth=2, markersize=8)
    
    axes[0, 1].set_xlabel('Dimensión Latente', fontweight='bold')
    axes[0, 1].set_ylabel('Mejor Precisión (%)', fontweight='bold')
    axes[0, 1].set_title('Mejor Precisión por Dimensión', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    for arq in sorted(df['arquitectura'].unique()):
        df_arq = df[df['arquitectura'] == arq]
        epocas_por_dim = df_arq.groupby('dim_latente')['epocas_reales'].mean()
        axes[1, 0].plot(epocas_por_dim.index, epocas_por_dim.values, 
                       marker='o', label=arq, linewidth=2)
    
    axes[1, 0].set_xlabel('Dimensión Latente', fontweight='bold')
    axes[1, 0].set_ylabel('Épocas Promedio', fontweight='bold')
    axes[1, 0].set_title('Épocas de Entrenamiento', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    top15 = df.nsmallest(15, 'mse')
    labels = [f"{row['arquitectura'][:8]}\nL={row['dim_latente']}\nLR={row['learning_rate']}" 
              for _, row in top15.iterrows()]
    colors = plt.cm.RdYlGn_r(top15['mse'] / top15['mse'].max())
    axes[1, 1].barh(range(len(top15)), top15['precision'], color=colors, edgecolor='black')
    axes[1, 1].set_yticks(range(len(top15)))
    axes[1, 1].set_yticklabels(labels, fontsize=7)
    axes[1, 1].set_xlabel('Precisión (%)', fontweight='bold')
    axes[1, 1].set_title('Top 15 Modelos por Precisión', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()
    
    plt.suptitle('Análisis Detallado de Configuraciones', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    ruta = os.path.join(resultados_dir, 'analisis_detallado.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    print(f"💾 Análisis guardado: {ruta}")
    plt.close()


def main():
    print("="*60)
    print("GRAFICADOR DE RESULTADOS")
    print("="*60)
    
    df = cargar_csv()
    if df is None:
        return
    
    print("\n📊 Generando gráficos...")
    
    graficar_heatmap_mse(df)
    graficar_precision_por_config(df)
    graficar_comparacion_lr(df)
    
    print("\n✅ Todos los gráficos generados exitosamente")
    print("\nArchivos generados:")
    print("  - heatmap_mse.png")
    print("  - heatmap_precision.png")
    print("  - analisis_detallado.png")


if __name__ == "__main__":
    main()
