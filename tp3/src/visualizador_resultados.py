import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


class VisualizadorResultados:
    def __init__(self):
        pass
    
    def mostrar_resultados_completos(self, modelo, datos, historial, config):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        self._mostrar_perdida(axes[0, 0], historial)
        self._mostrar_mse(axes[0, 1], historial)
        self._mostrar_espacio_latente(axes[0, 2], modelo, datos, config)
        self._mostrar_ejemplos_reconstruccion(axes[1, :], modelo, datos)
        
        plt.suptitle(f'Resultados - Latente {config["dimension_latente"]}D', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def _mostrar_perdida(self, ax, historial):
        ax.plot(historial.history['loss'], label='Entrenamiento')
        ax.plot(historial.history['val_loss'], label='Validación')
        ax.set_title('Pérdida')
        ax.set_xlabel('Época')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _mostrar_mse(self, ax, historial):
        ax.plot(historial.history['mse'], label='MSE Train')
        ax.plot(historial.history['val_mse'], label='MSE Val')
        ax.set_title('Error Cuadrático Medio')
        ax.set_xlabel('Época')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _mostrar_espacio_latente(self, ax, modelo, datos, config):
        if config['dimension_latente'] == 2:
            encoder = keras.Model(modelo.input, modelo.get_layer('latente').output)
            latentes = encoder.predict(datos, verbose=0)
            
            scatter = ax.scatter(latentes[:, 0], latentes[:, 1], 
                               c=range(len(datos)), cmap='tab20', s=80)
            ax.set_title('Espacio Latente 2D')
            ax.set_xlabel('Dimensión 1')
            ax.set_ylabel('Dimensión 2')
            ax.grid(True, alpha=0.3)
            
            for i, (x, y) in enumerate(latentes[:5]):
                ax.annotate(f'{i}', (x, y), xytext=(3, 3), 
                           textcoords='offset points', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'Espacio Latente\n{config["dimension_latente"]}D', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Dimensión Latente: {config["dimension_latente"]}D')
    
    def _mostrar_ejemplos_reconstruccion(self, axes, modelo, datos):
        reconstrucciones = modelo.predict(datos, verbose=0)
        reconstrucciones_bin = (reconstrucciones > 0.5).astype(int)
        
        ejemplos = [1, 10, 20]
        for i, idx in enumerate(ejemplos):
            ax = axes[i]
            
            original = datos[idx].reshape(7, 5)
            reconstruido = reconstrucciones_bin[idx].reshape(7, 5)
            
            combinado = np.zeros((7, 10))
            combinado[:, :5] = original
            combinado[:, 5:] = reconstruido
            
            ax.imshow(combinado, cmap='gray_r', interpolation='nearest')
            ax.set_title(f'Patrón {idx}: Original | Reconstruido')
            ax.axis('off')
            
            error = np.mean(np.abs(original - reconstruido)) * 100
            ax.text(0.5, -0.1, f'Error: {error:.1f}%', 
                   ha='center', transform=ax.transAxes, fontsize=10)
    
    def mostrar_metricas_entrenamiento(self, modelo, datos, historial):
        loss_final, mse_final = modelo.evaluate(datos, datos, verbose=0)
        reconstrucciones = modelo.predict(datos, verbose=0)
        reconstrucciones_bin = (reconstrucciones > 0.5).astype(int)
        precision = np.mean(reconstrucciones_bin == datos) * 100
        
        print(f"\nResultados finales:")
        print(f"  Loss: {loss_final:.4f}")
        print(f"  MSE: {mse_final:.4f}")
        print(f"  Precisión: {precision:.1f}%")
        print(f"  Épocas entrenadas: {len(historial.history['loss'])}")
        
        return {
            'loss': loss_final,
            'mse': mse_final,
            'precision': precision,
            'epochs': len(historial.history['loss'])
        }
    
    def mostrar_comparacion_arquitecturas(self, resultados):
        if not resultados:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        nombres = [f"{r['arquitectura']}-{r['dimension']}D" for r in resultados]
        precisiones = [r['precision'] for r in resultados]
        mses = [r['mse'] for r in resultados]
        
        ax1.bar(range(len(precisiones)), precisiones, color='skyblue')
        ax1.set_title('Precisión por Configuración')
        ax1.set_xlabel('Configuración')
        ax1.set_ylabel('Precisión (%)')
        ax1.set_xticks(range(len(nombres)))
        ax1.set_xticklabels(nombres, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(range(len(mses)), mses, color='lightcoral')
        ax2.set_title('MSE por Configuración')
        ax2.set_xlabel('Configuración')
        ax2.set_ylabel('MSE')
        ax2.set_xticks(range(len(nombres)))
        ax2.set_xticklabels(nombres, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
