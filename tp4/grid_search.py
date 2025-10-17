import argparse
import pandas as pd
from datetime import datetime
from tp3.comun.grid_search_base import GridSearchBase
from .entrenador import EntrenadorAutocodificadorImagenes
from .configuraciones import CONFIGURACIONES_AUTOCODIFICADOR, CONFIGURACIONES_ENTRENAMIENTO


class GridSearchAutocodificadorImagenes(GridSearchBase):
    def __init__(self):
        super().__init__()
        self.entrenador = EntrenadorAutocodificadorImagenes()
        self.tamaño_imagen = (64, 64)
        self.max_imagenes = None
        
    def configurar_experimento(self, tamaño_imagen=(64, 64), max_imagenes=None):
        self.tamaño_imagen = tamaño_imagen
        self.max_imagenes = max_imagenes
        
        print(f"Configurando experimentos para imágenes {tamaño_imagen[0]}x{tamaño_imagen[1]}")
        if max_imagenes:
            print(f"Limitando a {max_imagenes} imágenes")
    
    def ejecutar_experimento_individual(self, config_autocodificador, config_entrenamiento, experimento_id):
        nombre_autocodificador = config_autocodificador['nombre']
        nombre_entrenamiento = config_entrenamiento['nombre']
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENTO {experimento_id}")
        print(f"Autocodificador: {nombre_autocodificador}")
        print(f"Entrenamiento: {nombre_entrenamiento}")
        print(f"Tamaño imagen: {self.tamaño_imagen}")
        print(f"{'='*60}")
        
        try:
            inicio = datetime.now()
            
            modelo, historia, nombre_modelo = self.entrenador.entrenar(
                nombre_autocodificador,
                nombre_entrenamiento,
                self.tamaño_imagen,
                self.max_imagenes
            )
            
            fin = datetime.now()
            tiempo_entrenamiento = (fin - inicio).total_seconds()
            
            loss_final = min(historia.history['loss'])
            val_loss_final = min(historia.history.get('val_loss', [float('inf')]))
            mae_final = min(historia.history.get('mae', [float('inf')]))
            val_mae_final = min(historia.history.get('val_mae', [float('inf')]))
            
            epochs_ejecutadas = len(historia.history['loss'])
            epochs_configuradas = config_entrenamiento['config']['epochs']
            convergio = epochs_ejecutadas < epochs_configuradas
            
            mse_evaluacion, mae_evaluacion = self.entrenador.evaluar_reconstruccion(modelo, num_muestras=10)
            
            config_auto = config_autocodificador['config']
            tamaño_str = f"{self.tamaño_imagen[0]}x{self.tamaño_imagen[1]}"
            
            resultado = {
                'experimento': experimento_id,
                'arquitectura': nombre_autocodificador,
                'entrenamiento': nombre_entrenamiento,
                'tamaño_imagen': tamaño_str,
                'dimension_latente': config_auto['dimension_latente'],
                'capas_encoder': str(config_auto['capas_encoder']),
                'capas_decoder': str(config_auto['capas_decoder']),
                'activacion': config_auto['activacion'],
                'learning_rate': config_auto['learning_rate'],
                'batch_size': config_auto['batch_size'],
                'epochs_configuradas': epochs_configuradas,
                'epochs_ejecutadas': epochs_ejecutadas,
                'convergio': convergio,
                'loss_final': loss_final,
                'val_loss_final': val_loss_final,
                'mae_final': mae_final,
                'val_mae_final': val_mae_final,
                'mse_evaluacion': mse_evaluacion,
                'mae_evaluacion': mae_evaluacion,
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'nombre_modelo': nombre_modelo,
                'num_imagenes': len(self.entrenador.datos_entrenamiento) if self.entrenador.datos_entrenamiento is not None else 0
            }
            
            print(f"✓ Experimento {experimento_id} completado")
            print(f"  Loss final: {loss_final:.6f}")
            print(f"  MSE evaluación: {mse_evaluacion:.6f}")
            print(f"  Tiempo: {tiempo_entrenamiento:.1f}s")
            
            return resultado
            
        except Exception as e:
            print(f"✗ Error en experimento {experimento_id}: {e}")
            return {
                'experimento': experimento_id,
                'arquitectura': nombre_autocodificador,
                'entrenamiento': nombre_entrenamiento,
                'error': str(e),
                'completado': False
            }
    
    def ejecutar_grid_search_completo(self, tamaño_imagen=(64, 64), max_imagenes=None):
        self.configurar_experimento(tamaño_imagen, max_imagenes)
        
        configuraciones_autocodificador = [
            {'nombre': nombre, 'config': config}
            for nombre, config in CONFIGURACIONES_AUTOCODIFICADOR.items()
        ]
        
        configuraciones_entrenamiento = [
            {'nombre': nombre, 'config': config}
            for nombre, config in CONFIGURACIONES_ENTRENAMIENTO.items()
        ]
        
        return self.ejecutar_grid_search(
            configuraciones_autocodificador,
            configuraciones_entrenamiento
        )
    
    def ejecutar_grid_search_arquitecturas(self, arquitecturas, entrenamiento='normal', 
                                          tamaño_imagen=(64, 64), max_imagenes=None):
        self.configurar_experimento(tamaño_imagen, max_imagenes)
        
        configuraciones_autocodificador = [
            {'nombre': arq, 'config': CONFIGURACIONES_AUTOCODIFICADOR[arq]}
            for arq in arquitecturas if arq in CONFIGURACIONES_AUTOCODIFICADOR
        ]
        
        configuraciones_entrenamiento = [
            {'nombre': entrenamiento, 'config': CONFIGURACIONES_ENTRENAMIENTO[entrenamiento]}
        ]
        
        return self.ejecutar_grid_search(
            configuraciones_autocodificador,
            configuraciones_entrenamiento
        )
    
    def analizar_resultados(self, archivo_csv):
        df = pd.read_csv(archivo_csv)
        
        if 'error' in df.columns:
            df_exitosos = df[df['error'].isna()]
        else:
            df_exitosos = df
        
        if len(df_exitosos) == 0:
            print("No hay experimentos exitosos para analizar")
            return
        
        print("=== ANÁLISIS DE RESULTADOS - AUTOCODIFICADORES IMÁGENES ===\n")
        
        print("📊 MEJORES MODELOS POR MÉTRICA:")
        
        mejor_loss = df_exitosos.loc[df_exitosos['loss_final'].idxmin()]
        print(f"🏆 Mejor Loss: {mejor_loss['arquitectura']} (Loss: {mejor_loss['loss_final']:.6f})")
        
        mejor_mse = df_exitosos.loc[df_exitosos['mse_evaluacion'].idxmin()]
        print(f"🏆 Mejor MSE: {mejor_mse['arquitectura']} (MSE: {mejor_mse['mse_evaluacion']:.6f})")
        
        mejor_mae = df_exitosos.loc[df_exitosos['mae_evaluacion'].idxmin()]
        print(f"🏆 Mejor MAE: {mejor_mae['arquitectura']} (MAE: {mejor_mae['mae_evaluacion']:.6f})")
        
        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"Experimentos totales: {len(df)}")
        print(f"Experimentos exitosos: {len(df_exitosos)}")
        print(f"Tasa de éxito: {len(df_exitosos)/len(df)*100:.1f}%")
        
        if 'convergio' in df_exitosos.columns:
            convergencias = df_exitosos['convergio'].sum()
            print(f"Modelos que convergieron: {convergencias}/{len(df_exitosos)} ({convergencias/len(df_exitosos)*100:.1f}%)")
        
        print(f"\n⏱️  TIEMPOS DE ENTRENAMIENTO:")
        tiempo_promedio = df_exitosos['tiempo_entrenamiento'].mean()
        tiempo_min = df_exitosos['tiempo_entrenamiento'].min()
        tiempo_max = df_exitosos['tiempo_entrenamiento'].max()
        print(f"Promedio: {tiempo_promedio:.1f}s")
        print(f"Mínimo: {tiempo_min:.1f}s")
        print(f"Máximo: {tiempo_max:.1f}s")
        
        print(f"\n🏗️  ANÁLISIS POR ARQUITECTURA:")
        analisis_arquitectura = df_exitosos.groupby('arquitectura').agg({
            'loss_final': ['mean', 'min'],
            'mse_evaluacion': ['mean', 'min'],
            'tiempo_entrenamiento': 'mean'
        }).round(6)
        
        for arquitectura in analisis_arquitectura.index:
            stats = analisis_arquitectura.loc[arquitectura]
            print(f"  {arquitectura}:")
            print(f"    Loss promedio: {stats[('loss_final', 'mean')]:.6f}")
            print(f"    MSE promedio: {stats[('mse_evaluacion', 'mean')]:.6f}")
            print(f"    Tiempo promedio: {stats[('tiempo_entrenamiento', 'mean')]:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Grid Search para autocodificadores de imágenes')
    parser.add_argument('--arquitecturas', nargs='+', 
                       help='Lista de arquitecturas a probar')
    parser.add_argument('--entrenamiento', type=str, default='normal',
                       help='Tipo de entrenamiento')
    parser.add_argument('--tamaño', type=str, default='64x64',
                       help='Tamaño de imagen (ej: 64x64)')
    parser.add_argument('--max-imagenes', type=int, default=None,
                       help='Número máximo de imágenes')
    parser.add_argument('--completo', action='store_true',
                       help='Ejecutar grid search completo')
    parser.add_argument('--analizar', type=str,
                       help='Analizar resultados de archivo CSV')
    
    args = parser.parse_args()
    
    if args.analizar:
        grid_search = GridSearchAutocodificadorImagenes()
        grid_search.analizar_resultados(args.analizar)
        return
    
    try:
        ancho, alto = map(int, args.tamaño.split('x'))
        tamaño_imagen = (ancho, alto)
    except:
        print(f"Error: Formato de tamaño inválido '{args.tamaño}'")
        return
    
    grid_search = GridSearchAutocodificadorImagenes()
    
    if args.completo:
        print("Ejecutando grid search completo...")
        resultados = grid_search.ejecutar_grid_search_completo(tamaño_imagen, args.max_imagenes)
    elif args.arquitecturas:
        print(f"Ejecutando grid search para arquitecturas: {args.arquitecturas}")
        resultados = grid_search.ejecutar_grid_search_arquitecturas(
            args.arquitecturas, args.entrenamiento, tamaño_imagen, args.max_imagenes
        )
    else:
        print("Debe especificar --completo o --arquitecturas")
        return
    
    if resultados:
        print(f"\n✓ Grid search completado. Resultados guardados en: {resultados}")


if __name__ == "__main__":
    main()
