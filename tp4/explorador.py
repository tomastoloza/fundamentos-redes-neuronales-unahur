import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, classification_report
from tp3.comun.cargador_modelos import CargadorModelos
from .procesador_imagenes import ProcesadorImagenes


class ExploradorCNN:
    def __init__(self, modelo_path=None):
        self.procesador = ProcesadorImagenes()
        self.cargador = CargadorModelos()
        self.modelo = None
        self.datos_test = None
        self.clases = None
        self.predicciones = None
        self.indice_actual = 0
        
        if modelo_path:
            self.cargar_modelo(modelo_path)
    
    def cargar_modelo(self, modelo_path):
        try:
            from tensorflow.keras.models import load_model
            self.modelo = load_model(modelo_path)
            if self.modelo:
                print(f"✓ Modelo CNN cargado: {modelo_path}")
                return True
        except Exception as e:
            print(f"✗ Error cargando modelo: {e}")
        return False
    
    def cargar_datos(self, tamaño_imagen=(200, 200), max_imagenes_por_clase=None):
        print("Cargando datos para exploración...")
        imagenes, labels = self.procesador.cargar_imagenes_multiclase(
            tamaño_imagen, max_imagenes_por_clase
        )
        
        datos_entrenamiento, datos_validacion, self.datos_test = self.procesador.preparar_datos_entrenamiento()
        self.clases = self.procesador.obtener_nombres_clases()
        
        print(f"✓ Datos cargados: {len(self.clases)} clases")
        return True
    
    def generar_predicciones(self):
        if not self.modelo or not self.datos_test:
            print("❌ Debe cargar modelo y datos primero")
            return False
        
        X_test, y_test = self.datos_test
        print("Generando predicciones...")
        
        self.predicciones = self.modelo.predict(X_test, verbose=0)
        self.y_test_classes = np.argmax(y_test, axis=1)
        self.y_pred_classes = np.argmax(self.predicciones, axis=1)
        
        accuracy = np.mean(self.y_test_classes == self.y_pred_classes)
        print(f"✓ Predicciones generadas. Accuracy: {accuracy:.4f}")
        return True
    
    def mostrar_matriz_confusion(self):
        if self.predicciones is None:
            self.generar_predicciones()
        
        cm = confusion_matrix(self.y_test_classes, self.y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.clases, yticklabels=self.clases)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.tight_layout()
        plt.show()
        
        print("\nReporte de clasificación:")
        print(classification_report(self.y_test_classes, self.y_pred_classes, 
                                  target_names=self.clases))
    
    def mostrar_predicciones_por_clase(self, num_ejemplos=3):
        if self.predicciones is None:
            self.generar_predicciones()
        
        X_test, _ = self.datos_test
        
        fig, axes = plt.subplots(len(self.clases), num_ejemplos * 2, 
                               figsize=(num_ejemplos * 6, len(self.clases) * 3))
        
        for i, clase in enumerate(self.clases):
            indices_clase = np.where(self.y_test_classes == i)[0]
            
            correctos = [idx for idx in indices_clase if self.y_pred_classes[idx] == i]
            incorrectos = [idx for idx in indices_clase if self.y_pred_classes[idx] != i]
            
            for j in range(num_ejemplos):
                if j < len(correctos):
                    idx = correctos[j]
                    axes[i, j].imshow(X_test[idx])
                    pred_prob = self.predicciones[idx][i]
                    axes[i, j].set_title(f'✓ Correcto\nConf: {pred_prob:.3f}')
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
                
                if j < len(incorrectos):
                    idx = incorrectos[j]
                    axes[i, j + num_ejemplos].imshow(X_test[idx])
                    pred_clase = self.clases[self.y_pred_classes[idx]]
                    pred_prob = self.predicciones[idx][self.y_pred_classes[idx]]
                    axes[i, j + num_ejemplos].set_title(f'✗ Pred: {pred_clase}\nConf: {pred_prob:.3f}')
                    axes[i, j + num_ejemplos].axis('off')
                else:
                    axes[i, j + num_ejemplos].axis('off')
            
            axes[i, 0].set_ylabel(f'{clase}', rotation=90, fontsize=12)
        
        plt.suptitle('Predicciones: Correctas (izq) vs Incorrectas (der)', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def explorar_imagen_individual(self, indice=None):
        if self.predicciones is None:
            self.generar_predicciones()
        
        X_test, _ = self.datos_test
        
        if indice is None:
            indice = self.indice_actual
        
        if indice >= len(X_test):
            print(f"Índice {indice} fuera de rango. Máximo: {len(X_test)-1}")
            return
        
        imagen = X_test[indice]
        clase_real = self.clases[self.y_test_classes[indice]]
        clase_pred = self.clases[self.y_pred_classes[indice]]
        
        prediccion_probs = self.predicciones[indice]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(imagen)
        ax1.set_title(f'Imagen {indice}\nReal: {clase_real}\nPredicción: {clase_pred}')
        ax1.axis('off')
        
        ax2.bar(range(len(self.clases)), prediccion_probs)
        ax2.set_xticks(range(len(self.clases)))
        ax2.set_xticklabels(self.clases, rotation=45)
        ax2.set_ylabel('Probabilidad')
        ax2.set_title('Distribución de Probabilidades')
        
        max_prob_idx = np.argmax(prediccion_probs)
        ax2.bar(max_prob_idx, prediccion_probs[max_prob_idx], color='red', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Imagen {indice}:")
        print(f"  Clase real: {clase_real}")
        print(f"  Predicción: {clase_pred}")
        print(f"  Confianza: {prediccion_probs[max_prob_idx]:.4f}")
        print(f"  ¿Correcto?: {'Sí' if clase_real == clase_pred else 'No'}")
    
    def mostrar_activaciones_capa(self, indice_imagen, nombre_capa=None):
        if not self.modelo:
            print("❌ Debe cargar un modelo primero")
            return
        
        X_test, _ = self.datos_test
        imagen = X_test[indice_imagen:indice_imagen+1]
        
        if nombre_capa is None:
            capas_conv = [layer for layer in self.modelo.layers if 'conv' in layer.name.lower()]
            if capas_conv:
                capa = capas_conv[0]
            else:
                print("No se encontraron capas convolucionales")
                return
        else:
            capa = self.modelo.get_layer(nombre_capa)
        
        from tensorflow.keras.models import Model
        modelo_activacion = Model(inputs=self.modelo.input, outputs=capa.output)
        activaciones = modelo_activacion.predict(imagen, verbose=0)
        
        num_filtros = min(16, activaciones.shape[-1])
        filas = 4
        columnas = 4
        
        fig, axes = plt.subplots(filas, columnas, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_filtros):
            if i < activaciones.shape[-1]:
                axes[i].imshow(activaciones[0, :, :, i], cmap='viridis')
                axes[i].set_title(f'Filtro {i+1}')
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.suptitle(f'Activaciones de {capa.name} - Imagen {indice_imagen}')
        plt.tight_layout()
        plt.show()
    
    def analizar_errores_por_clase(self):
        if self.predicciones is None:
            self.generar_predicciones()
        
        print("=== ANÁLISIS DE ERRORES POR CLASE ===\n")
        
        for i, clase in enumerate(self.clases):
            indices_clase = np.where(self.y_test_classes == i)[0]
            correctos = np.sum(self.y_pred_classes[indices_clase] == i)
            total = len(indices_clase)
            accuracy_clase = correctos / total if total > 0 else 0
            
            print(f"{clase}:")
            print(f"  Total: {total}")
            print(f"  Correctos: {correctos}")
            print(f"  Accuracy: {accuracy_clase:.4f}")
            
            if total > correctos:
                errores = indices_clase[self.y_pred_classes[indices_clase] != i]
                predicciones_erroneas = self.y_pred_classes[errores]
                
                print(f"  Errores más comunes:")
                for pred_clase in np.unique(predicciones_erroneas):
                    count = np.sum(predicciones_erroneas == pred_clase)
                    print(f"    Confundido con {self.clases[pred_clase]}: {count} veces")
            print()
    
    def prediccion_automatica_aleatoria(self, intervalo=2.0, num_predicciones=20):
        """
        Cambia automáticamente de foto aleatoria y muestra predicciones en tiempo real
        
        Args:
            intervalo: Segundos entre cambios de imagen
            num_predicciones: Número total de predicciones a mostrar (0 = infinito)
        """
        if self.predicciones is None:
            self.generar_predicciones()
        
        X_test, _ = self.datos_test
        
        print("=== PREDICCIÓN AUTOMÁTICA ALEATORIA ===")
        print(f"Cambiando cada {intervalo}s")
        print("Presiona Ctrl+C para detener")
        print("-" * 50)
        
        contador = 0
        aciertos = 0
        
        try:
            while num_predicciones == 0 or contador < num_predicciones:
                # Seleccionar imagen aleatoria
                self.indice_actual = np.random.randint(0, len(X_test))
                
                # Obtener datos de la imagen
                imagen = X_test[self.indice_actual]
                clase_real = self.clases[self.y_test_classes[self.indice_actual]]
                clase_pred = self.clases[self.y_pred_classes[self.indice_actual]]
                confianza = self.predicciones[self.indice_actual][self.y_pred_classes[self.indice_actual]]
                
                # Verificar si es correcto
                es_correcto = clase_real == clase_pred
                if es_correcto:
                    aciertos += 1
                
                contador += 1
                accuracy_actual = (aciertos / contador) * 100
                
                # Mostrar información
                estado = "✅ CORRECTO" if es_correcto else "❌ ERROR"
                print(f"[{contador:3d}] {estado} | Real: {clase_real:12s} | Pred: {clase_pred:12s} | Conf: {confianza:.3f} | Acc: {accuracy_actual:.1f}%")
                
                # Mostrar imagen cada 5 predicciones o si hay error
                if contador % 5 == 0 or not es_correcto:
                    self.mostrar_prediccion_rapida(self.indice_actual)
                
                # Esperar intervalo
                time.sleep(intervalo)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Detenido por usuario")
        
        print(f"\n📊 RESUMEN FINAL:")
        print(f"   Total predicciones: {contador}")
        print(f"   Aciertos: {aciertos}")
        print(f"   Accuracy final: {accuracy_actual:.1f}%")
    
    def mostrar_prediccion_rapida(self, indice):
        """Muestra una predicción de forma rápida y compacta"""
        X_test, _ = self.datos_test
        
        imagen = X_test[indice]
        clase_real = self.clases[self.y_test_classes[indice]]
        clase_pred = self.clases[self.y_pred_classes[indice]]
        prediccion_probs = self.predicciones[indice]
        
        # Crear figura compacta
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Mostrar imagen
        ax1.imshow(imagen)
        color = 'green' if clase_real == clase_pred else 'red'
        ax1.set_title(f'Real: {clase_real}\nPred: {clase_pred}', color=color, fontweight='bold')
        ax1.axis('off')
        
        # Mostrar probabilidades
        colores = ['red' if i == self.y_pred_classes[indice] else 'lightblue' for i in range(len(self.clases))]
        bars = ax2.bar(range(len(self.clases)), prediccion_probs, color=colores)
        ax2.set_xticks(range(len(self.clases)))
        ax2.set_xticklabels(self.clases, rotation=45, ha='right')
        ax2.set_ylabel('Probabilidad')
        ax2.set_title('Confianza por Clase')
        ax2.set_ylim(0, 1)
        
        # Añadir valores en las barras
        for bar, prob in zip(bars, prediccion_probs):
            if prob > 0.1:  # Solo mostrar si es significativo
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)  # Pausa breve para que se renderice
    
    def navegacion_interactiva(self):
        if self.predicciones is None:
            self.generar_predicciones()
        
        print("=== NAVEGACIÓN INTERACTIVA ===")
        print("Comandos:")
        print("  n: siguiente imagen")
        print("  p: imagen anterior")
        print("  r: imagen aleatoria")
        print("  auto: modo automático aleatorio")
        print("  m: mostrar matriz de confusión")
        print("  a: análisis de errores")
        print("  q: salir")
        
        X_test, _ = self.datos_test
        
        while True:
            self.explorar_imagen_individual(self.indice_actual)
            
            comando = input(f"\nImagen {self.indice_actual}/{len(X_test)-1} - Comando: ").lower().strip()
            
            if comando == 'q':
                break
            elif comando == 'n':
                self.indice_actual = (self.indice_actual + 1) % len(X_test)
            elif comando == 'p':
                self.indice_actual = (self.indice_actual - 1) % len(X_test)
            elif comando == 'r':
                self.indice_actual = np.random.randint(0, len(X_test))
            elif comando == 'auto':
                try:
                    intervalo = float(input("Intervalo en segundos (default 2.0): ") or "2.0")
                    num_pred = int(input("Número de predicciones (0=infinito): ") or "20")
                    self.prediccion_automatica_aleatoria(intervalo, num_pred)
                except ValueError:
                    print("Valores inválidos, usando defaults")
                    self.prediccion_automatica_aleatoria()
            elif comando == 'm':
                self.mostrar_matriz_confusion()
            elif comando == 'a':
                self.analizar_errores_por_clase()
            else:
                try:
                    nuevo_indice = int(comando)
                    if 0 <= nuevo_indice < len(X_test):
                        self.indice_actual = nuevo_indice
                    else:
                        print(f"Índice fuera de rango. Use 0-{len(X_test)-1}")
                except ValueError:
                    print("Comando no reconocido")


def main():
    parser = argparse.ArgumentParser(description='Explorador CNN para clasificación de animales')
    parser.add_argument('modelo', help='Ruta al modelo CNN (.keras)')
    parser.add_argument('--tamaño', default='200x200', help='Tamaño de imagen (ej: 200x200)')
    parser.add_argument('--max-imagenes', type=int, help='Máximo de imágenes por clase')
    parser.add_argument('--listar', action='store_true', help='Listar modelos disponibles')
    parser.add_argument('--matriz', action='store_true', help='Mostrar solo matriz de confusión')
    parser.add_argument('--predicciones', action='store_true', help='Mostrar predicciones por clase')
    parser.add_argument('--interactivo', action='store_true', help='Modo interactivo')
    parser.add_argument('--auto', action='store_true', help='Modo automático aleatorio')
    parser.add_argument('--intervalo', type=float, default=2.0, help='Intervalo en segundos para modo auto')
    parser.add_argument('--num-pred', type=int, default=20, help='Número de predicciones en modo auto')
    
    args = parser.parse_args()
    
    if args.listar:
        cargador = CargadorModelos()
        cargador.listar_modelos_disponibles()
        return
    
    try:
        ancho, alto = map(int, args.tamaño.split('x'))
        tamaño_imagen = (ancho, alto)
    except:
        print(f"Error: Formato de tamaño inválido '{args.tamaño}'")
        return
    
    explorador = ExploradorCNN()
    
    if not explorador.cargar_modelo(args.modelo):
        return
    
    if not explorador.cargar_datos(tamaño_imagen, args.max_imagenes):
        return
    
    if args.matriz:
        explorador.mostrar_matriz_confusion()
    elif args.predicciones:
        explorador.mostrar_predicciones_por_clase()
    elif args.interactivo:
        explorador.navegacion_interactiva()
    elif args.auto:
        explorador.prediccion_automatica_aleatoria(args.intervalo, args.num_pred)
    else:
        explorador.mostrar_matriz_confusion()
        explorador.mostrar_predicciones_por_clase()
        explorador.analizar_errores_por_clase()


if __name__ == "__main__":
    main()