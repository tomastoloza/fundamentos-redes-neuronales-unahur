import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

try:
    from .autocodificador import AutocodificadorAnomalias
    from .generador_datos_energia import GeneradorDatosEnergia
except ImportError:
    from autocodificador import AutocodificadorAnomalias
    from generador_datos_energia import GeneradorDatosEnergia

from tp3.comun.cargador_modelos import CargadorModelos


class ExploradorAnomalias:
    def __init__(self, modelo_path=None):
        self.autocodificador = AutocodificadorAnomalias()
        self.generador_datos = GeneradorDatosEnergia()
        self.cargador_modelos = CargadorModelos()
        
        self.modelo_cargado = False
        self.datos_prueba = None
        self.metadatos_prueba = None
        self.errores_reconstruccion = None
        self.reconstrucciones = None
        
        self.muestra_actual = 0
        self.modo_generacion = 'aleatorio'
        
        if modelo_path:
            self.cargar_modelo(modelo_path)
    
    def cargar_modelo(self, modelo_path):
        try:
            # Intentar cargar como nombre de archivo primero (sin ruta completa)
            try:
                # Si es solo el nombre del archivo, usar CargadorModelos
                if not os.path.sep in modelo_path and not modelo_path.startswith('tp3/'):
                    modelo_keras = self.cargador_modelos.cargar_modelo(modelo_path)
                    # Cargar el modelo Keras en el autocodificador
                    self.autocodificador.modelo = modelo_keras
                    # Intentar cargar metadatos
                    ruta_completa = os.path.join(self.cargador_modelos.directorio_modelos, modelo_path)
                    if not ruta_completa.endswith('.keras'):
                        ruta_completa += '.keras'
                    try:
                        metadatos = np.load(f"{ruta_completa}_metadatos.npy", allow_pickle=True).item()
                        self.autocodificador.longitud_serie = metadatos['longitud_serie']
                        self.autocodificador.dimension_latente = metadatos['dimension_latente']
                        self.autocodificador.umbral_anomalia = metadatos['umbral_anomalia']
                        self.autocodificador.datos_normalizacion = metadatos['datos_normalizacion']
                    except:
                        print("Advertencia: No se pudieron cargar los metadatos")
                else:
                    # Si es una ruta completa, usar el método original
                    self.autocodificador.cargar_modelo(modelo_path)
                
                self.modelo_cargado = True
                print(f"Modelo cargado: {modelo_path}")
                if self.autocodificador.umbral_anomalia:
                    print(f"Umbral de anomalía: {self.autocodificador.umbral_anomalia:.6f}")
                return True
            except Exception as e:
                # Si falla, intentar como ruta completa
                self.autocodificador.cargar_modelo(modelo_path)
                self.modelo_cargado = True
                print(f"Modelo cargado: {modelo_path}")
                if self.autocodificador.umbral_anomalia:
                    print(f"Umbral de anomalía: {self.autocodificador.umbral_anomalia:.6f}")
                return True
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            return False
    
    def cargar_datos_prueba(self, num_normales=100, num_anomalas=25):
        print("Generando datos de prueba...")
        self.datos_prueba, self.metadatos_prueba = self.generador_datos.generar_conjunto_prueba(
            num_normales=num_normales,
            num_anomalas=num_anomalas
        )
        
        datos_norm, _, _ = self.generador_datos.normalizar_datos(self.datos_prueba)
        self.datos_prueba = datos_norm
        
        if self.modelo_cargado:
            predicciones, errores, reconstrucciones = self.autocodificador.detectar_anomalias(self.datos_prueba)
            self.errores_reconstruccion = errores
            self.reconstrucciones = reconstrucciones
        
        print(f"Datos cargados: {len(self.datos_prueba)} muestras")
        print(f"- Normales: {sum(1 for m in self.metadatos_prueba if not m['es_anomalo'])}")
        print(f"- Anomalías: {sum(1 for m in self.metadatos_prueba if m['es_anomalo'])}")
    
    def explorar_interactivo(self):
        if not self.modelo_cargado:
            print("Error: Debe cargar un modelo primero")
            return
        
        if self.datos_prueba is None:
            self.cargar_datos_prueba()
        
        self.fig = plt.figure(figsize=(16, 10))
        
        ax_original = plt.subplot2grid((3, 4), (0, 0), colspan=2)
        ax_reconstruido = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        ax_error = plt.subplot2grid((3, 4), (1, 0), colspan=2)
        ax_sintetico = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        ax_controles = plt.subplot2grid((3, 4), (2, 0), colspan=4)
        
        self.axes = {
            'original': ax_original,
            'reconstruido': ax_reconstruido,
            'error': ax_error,
            'sintetico': ax_sintetico,
            'controles': ax_controles
        }
        
        self.actualizar_visualizacion()
        
        self.configurar_controles()
        
        plt.tight_layout()
        plt.show()
    
    def actualizar_visualizacion(self):
        muestra = self.datos_prueba[self.muestra_actual]
        reconstruccion = self.reconstrucciones[self.muestra_actual]
        error = self.errores_reconstruccion[self.muestra_actual]
        metadatos = self.metadatos_prueba[self.muestra_actual]
        
        self.axes['original'].clear()
        self.axes['original'].plot(muestra, 'b-', linewidth=2, label='Original')
        self.axes['original'].set_title(f'Serie Original - Muestra {self.muestra_actual}')
        self.axes['original'].set_ylabel('Consumo Normalizado')
        self.axes['original'].grid(True)
        self.axes['original'].legend()
        
        self.axes['reconstruido'].clear()
        self.axes['reconstruido'].plot(muestra, 'b-', alpha=0.7, label='Original')
        self.axes['reconstruido'].plot(reconstruccion, 'r--', linewidth=2, label='Reconstruido')
        
        es_anomalia = error > self.autocodificador.umbral_anomalia
        estado = "ANOMALÍA" if es_anomalia else "NORMAL"
        color_titulo = 'red' if es_anomalia else 'green'
        
        self.axes['reconstruido'].set_title(f'Reconstrucción - {estado}\nError: {error:.6f}', 
                                          color=color_titulo)
        self.axes['reconstruido'].set_ylabel('Consumo Normalizado')
        self.axes['reconstruido'].grid(True)
        self.axes['reconstruido'].legend()
        
        self.axes['error'].clear()
        diferencia = np.abs(muestra - reconstruccion)
        self.axes['error'].plot(diferencia, 'orange', linewidth=2)
        self.axes['error'].axhline(y=np.mean(diferencia), color='red', linestyle='--', 
                                 label=f'Error Medio: {np.mean(diferencia):.4f}')
        self.axes['error'].set_title('Error de Reconstrucción por Punto')
        self.axes['error'].set_xlabel('Tiempo (horas)')
        self.axes['error'].set_ylabel('Error Absoluto')
        self.axes['error'].grid(True)
        self.axes['error'].legend()
        
        self.generar_y_mostrar_sintetico()
        
        info_texto = f"""
Información de la Muestra:
- ID: {metadatos['id']}
- Tipo Servidor: {metadatos['tipo_servidor']}
- Es Anomalía Real: {'Sí' if metadatos['es_anomalo'] else 'No'}
- Predicción: {'Anomalía' if es_anomalia else 'Normal'}
- Error Reconstrucción: {error:.6f}
- Umbral: {self.autocodificador.umbral_anomalia:.6f}
"""
        
        if metadatos['es_anomalo']:
            info_texto += f"- Tipo Anomalía: {metadatos['tipo_anomalia']}\n"
            info_texto += f"- Posición: {metadatos.get('posicion_anomalia', 'N/A')}\n"
        
        self.axes['controles'].clear()
        self.axes['controles'].text(0.02, 0.5, info_texto, transform=self.axes['controles'].transAxes,
                                  fontsize=10, verticalalignment='center',
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.axes['controles'].set_xlim(0, 1)
        self.axes['controles'].set_ylim(0, 1)
        self.axes['controles'].axis('off')
        
        plt.draw()
    
    def generar_y_mostrar_sintetico(self):
        if self.modo_generacion == 'aleatorio':
            muestra_sintetica, _ = self.autocodificador.generar_muestra_sintetica(1)
            titulo = 'Muestra Sintética (Aleatoria)'
        else:
            indices_normales = [i for i, m in enumerate(self.metadatos_prueba) if not m['es_anomalo']]
            if len(indices_normales) >= 2:
                idx1, idx2 = np.random.choice(indices_normales, 2, replace=False)
                interpolaciones = self.autocodificador.generar_desde_interpolacion(
                    self.datos_prueba[idx1], self.datos_prueba[idx2], 5
                )
                muestra_sintetica = interpolaciones[2:3]
                titulo = f'Interpolación (muestras {idx1}-{idx2})'
            else:
                muestra_sintetica, _ = self.autocodificador.generar_muestra_sintetica(1)
                titulo = 'Muestra Sintética (Aleatoria)'
        
        self.axes['sintetico'].clear()
        self.axes['sintetico'].plot(muestra_sintetica[0], 'g-', linewidth=2)
        self.axes['sintetico'].set_title(titulo)
        self.axes['sintetico'].set_xlabel('Tiempo (horas)')
        self.axes['sintetico'].set_ylabel('Consumo Normalizado')
        self.axes['sintetico'].grid(True)
    
    def configurar_controles(self):
        print("\nControles disponibles:")
        print("← →: Navegar entre muestras")
        print("N: Ir a siguiente muestra normal")
        print("A: Ir a siguiente muestra anómala")
        print("G: Cambiar modo de generación (aleatorio/interpolación)")
        print("R: Regenerar muestra sintética")
        print("Q: Salir")
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def on_key_press(self, event):
        if event.key == 'right':
            self.muestra_actual = (self.muestra_actual + 1) % len(self.datos_prueba)
            self.actualizar_visualizacion()
        
        elif event.key == 'left':
            self.muestra_actual = (self.muestra_actual - 1) % len(self.datos_prueba)
            self.actualizar_visualizacion()
        
        elif event.key == 'n':
            indices_normales = [i for i, m in enumerate(self.metadatos_prueba) if not m['es_anomalo']]
            if indices_normales:
                siguiente = min([i for i in indices_normales if i > self.muestra_actual], 
                              default=indices_normales[0])
                self.muestra_actual = siguiente
                self.actualizar_visualizacion()
        
        elif event.key == 'a':
            indices_anomalos = [i for i, m in enumerate(self.metadatos_prueba) if m['es_anomalo']]
            if indices_anomalos:
                siguiente = min([i for i in indices_anomalos if i > self.muestra_actual], 
                              default=indices_anomalos[0])
                self.muestra_actual = siguiente
                self.actualizar_visualizacion()
        
        elif event.key == 'g':
            self.modo_generacion = 'interpolacion' if self.modo_generacion == 'aleatorio' else 'aleatorio'
            print(f"Modo de generación cambiado a: {self.modo_generacion}")
            self.generar_y_mostrar_sintetico()
            plt.draw()
        
        elif event.key == 'r':
            self.generar_y_mostrar_sintetico()
            plt.draw()
        
        elif event.key == 'q':
            plt.close()
    
    def analizar_distribucion_errores(self):
        if self.errores_reconstruccion is None:
            print("No hay errores calculados")
            return
        
        etiquetas_reales = [m['es_anomalo'] for m in self.metadatos_prueba]
        errores_normales = [e for e, real in zip(self.errores_reconstruccion, etiquetas_reales) if not real]
        errores_anomalos = [e for e, real in zip(self.errores_reconstruccion, etiquetas_reales) if real]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(errores_normales, bins=30, alpha=0.7, label='Normal', color='green')
        plt.hist(errores_anomalos, bins=30, alpha=0.7, label='Anomalía', color='red')
        plt.axvline(self.autocodificador.umbral_anomalia, color='black', linestyle='--', 
                   label=f'Umbral: {self.autocodificador.umbral_anomalia:.4f}')
        plt.xlabel('Error de Reconstrucción')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Errores')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.scatter(range(len(errores_normales)), errores_normales, alpha=0.6, color='green', label='Normal')
        plt.scatter(range(len(errores_anomalos)), errores_anomalos, alpha=0.6, color='red', label='Anomalía')
        plt.axhline(self.autocodificador.umbral_anomalia, color='black', linestyle='--', 
                   label=f'Umbral: {self.autocodificador.umbral_anomalia:.4f}')
        plt.xlabel('Índice de Muestra')
        plt.ylabel('Error de Reconstrucción')
        plt.title('Errores por Muestra')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        tipos_anomalia = {}
        for meta, error in zip(self.metadatos_prueba, self.errores_reconstruccion):
            if meta['es_anomalo']:
                tipo = meta['tipo_anomalia']
                if tipo not in tipos_anomalia:
                    tipos_anomalia[tipo] = []
                tipos_anomalia[tipo].append(error)
        
        if tipos_anomalia:
            tipos = list(tipos_anomalia.keys())
            errores_por_tipo = [tipos_anomalia[tipo] for tipo in tipos]
            plt.boxplot(errores_por_tipo, labels=tipos)
            plt.axhline(self.autocodificador.umbral_anomalia, color='black', linestyle='--', 
                       label=f'Umbral: {self.autocodificador.umbral_anomalia:.4f}')
            plt.ylabel('Error de Reconstrucción')
            plt.title('Errores por Tipo de Anomalía')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
        
        plt.subplot(2, 2, 4)
        predicciones = self.errores_reconstruccion > self.autocodificador.umbral_anomalia
        tp = sum(1 for real, pred in zip(etiquetas_reales, predicciones) if real and pred)
        fp = sum(1 for real, pred in zip(etiquetas_reales, predicciones) if not real and pred)
        tn = sum(1 for real, pred in zip(etiquetas_reales, predicciones) if not real and not pred)
        fn = sum(1 for real, pred in zip(etiquetas_reales, predicciones) if real and not pred)
        
        matriz = np.array([[tn, fp], [fn, tp]])
        plt.imshow(matriz, interpolation='nearest', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.colorbar()
        
        for i in range(2):
            for j in range(2):
                plt.text(j, i, matriz[i, j], ha="center", va="center", fontsize=14)
        
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.xticks([0, 1], ['Normal', 'Anomalía'])
        plt.yticks([0, 1], ['Normal', 'Anomalía'])
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Explorador interactivo de detección de anomalías')
    parser.add_argument('modelo', type=str, nargs='?',
                       help='Nombre o ruta del modelo a cargar')
    parser.add_argument('--listar', action='store_true',
                       help='Listar modelos disponibles')
    parser.add_argument('--mejor', action='store_true',
                       help='Usar el mejor modelo disponible')
    parser.add_argument('--analisis', action='store_true', 
                       help='Mostrar análisis de distribución de errores')
    
    args = parser.parse_args()
    
    if args.listar:
        cargador = CargadorModelos()
        modelos = cargador.listar_modelos()
        print("Modelos disponibles:")
        anomalias_modelos = [m for m in modelos if 'anomalias' in m]
        if anomalias_modelos:
            for modelo in anomalias_modelos:
                print(f"  - {modelo}")
        else:
            print("  No hay modelos de anomalías disponibles")
        return
    
    modelo_path = None
    if args.mejor:
        # Buscar el mejor modelo de anomalías
        cargador = CargadorModelos()
        modelos = cargador.listar_modelos()
        anomalias_modelos = [m for m in modelos if 'anomalias' in m]
        if anomalias_modelos:
            modelo_path = anomalias_modelos[-1]  # Usar el más reciente
            print(f"Usando modelo más reciente: {modelo_path}")
        else:
            print("No hay modelos de anomalías disponibles")
            return
    elif args.modelo:
        modelo_path = args.modelo
    
    if not modelo_path:
        print("Debe especificar un modelo o usar --listar para ver modelos disponibles")
        print("Ejemplo de uso:")
        print("  python3 -m tp3.anomalias.explorador tp3_anomalias_20251012_201643_anomalias_lat16_20251012_201643")
        print("  python3 -m tp3.anomalias.explorador --mejor")
        print("  python3 -m tp3.anomalias.explorador --listar")
        return
    
    print("=== EXPLORADOR DETECCIÓN DE ANOMALÍAS ===")
    print("TP3 - Detección de anomalías en consumo energético")
    print()
    
    explorador = ExploradorAnomalias(modelo_path)
    if explorador.modelo_cargado:
        if args.analisis:
            explorador.cargar_datos_prueba()
            explorador.analizar_distribucion_errores()
        else:
            explorador.explorar_interactivo()
    else:
        print("No se pudo cargar el modelo especificado.")


if __name__ == "__main__":
    main()
