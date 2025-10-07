"""
Comparador de Conjuntos de Entrenamiento para Discriminación de Números Pares

Este módulo implementa un sistema de comparación sistemática que evalúa cómo diferentes
combinaciones de conjuntos de entrenamiento afectan el rendimiento de la red neuronal
en el problema de discriminación de números pares vs impares.

Autor: Sistema de IA
Fecha: 2025-09-23
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
from itertools import combinations

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .entrenador_tp2 import EntrenadorTP2
from comun.constantes.constantes_redes_neuronales import ARQUITECTURAS_TP2


class ComparadorConjuntosEntrenamiento:
    """
    Clase para comparar sistemáticamente diferentes combinaciones de conjuntos 
    de entrenamiento en el problema de discriminación de números pares.
    """

    def __init__(self):
        """Inicializa el comparador con las configuraciones base."""
        self.entrenador = EntrenadorTP2()
        self.resultados_comparacion = {}
        self.digitos_disponibles = list(range(10))  # 0-9
        
        # Configuraciones de experimentos predefinidas
        self.configuraciones_experimento = self._generar_configuraciones_experimento()
    
    def _generar_configuraciones_experimento(self) -> Dict[str, Dict]:
        """
        Genera diferentes configuraciones de experimentos para evaluar.
        
        Returns:
            Dict con configuraciones de experimentos organizadas por categoría
        """
        configuraciones = {
            # Experimentos con balance par/impar
            'balance_equilibrado': {
                'descripcion': 'Conjuntos balanceados con igual cantidad de pares e impares',
                'conjuntos': [
                    {
                        'nombre': 'balance_2_2',
                        'entrenamiento': [0, 2, 1, 3],  # 2 pares, 2 impares
                        'prueba': [4, 6, 8, 5, 7, 9]
                    },
                    {
                        'nombre': 'balance_3_3',
                        'entrenamiento': [0, 2, 4, 1, 3, 5],  # 3 pares, 3 impares
                        'prueba': [6, 8, 7, 9]
                    },
                    {
                        'nombre': 'balance_4_4',
                        'entrenamiento': [0, 2, 4, 6, 1, 3, 5, 7],  # 4 pares, 4 impares
                        'prueba': [8, 9]
                    }
                ]
            },
            
            # Experimentos con desbalance par/impar
            'desbalance_controlado': {
                'descripcion': 'Conjuntos desbalanceados para evaluar sesgo',
                'conjuntos': [
                    {
                        'nombre': 'mas_pares_4_2',
                        'entrenamiento': [0, 2, 4, 6, 1, 3],  # 4 pares, 2 impares
                        'prueba': [8, 5, 7, 9]
                    },
                    {
                        'nombre': 'mas_impares_2_4',
                        'entrenamiento': [0, 2, 1, 3, 5, 7],  # 2 pares, 4 impares
                        'prueba': [4, 6, 8, 9]
                    },
                    {
                        'nombre': 'solo_pares',
                        'entrenamiento': [0, 2, 4, 6],  # Solo pares
                        'prueba': [8, 1, 3, 5, 7, 9]
                    },
                    {
                        'nombre': 'solo_impares',
                        'entrenamiento': [1, 3, 5, 7],  # Solo impares
                        'prueba': [0, 2, 4, 6, 8, 9]
                    }
                ]
            },
            
            # Experimentos con diferentes rangos numéricos
            'rangos_numericos': {
                'descripcion': 'Evaluación por rangos numéricos (bajos vs altos)',
                'conjuntos': [
                    {
                        'nombre': 'numeros_bajos',
                        'entrenamiento': [0, 1, 2, 3],  # Números 0-3
                        'prueba': [4, 5, 6, 7, 8, 9]
                    },
                    {
                        'nombre': 'numeros_medios',
                        'entrenamiento': [2, 3, 4, 5, 6, 7],  # Números 2-7
                        'prueba': [0, 1, 8, 9]
                    },
                    {
                        'nombre': 'numeros_altos',
                        'entrenamiento': [6, 7, 8, 9],  # Números 6-9
                        'prueba': [0, 1, 2, 3, 4, 5]
                    },
                    {
                        'nombre': 'extremos',
                        'entrenamiento': [0, 1, 8, 9],  # Extremos
                        'prueba': [2, 3, 4, 5, 6, 7]
                    }
                ]
            },
            
            # Experimentos con patrones específicos
            'patrones_especificos': {
                'descripcion': 'Patrones específicos para evaluar generalización',
                'conjuntos': [
                    {
                        'nombre': 'alternados',
                        'entrenamiento': [0, 2, 1, 5],  # Patrón alternado
                        'prueba': [3, 4, 6, 7, 8, 9]
                    },
                    {
                        'nombre': 'secuenciales',
                        'entrenamiento': [0, 1, 2, 3, 4, 5],  # Secuencial
                        'prueba': [6, 7, 8, 9]
                    },
                    {
                        'nombre': 'salteados',
                        'entrenamiento': [0, 3, 6, 9],  # Cada 3
                        'prueba': [1, 2, 4, 5, 7, 8]
                    }
                ]
            }
        }
        
        return configuraciones
    
    def ejecutar_comparacion_completa(self, 
                                    arquitecturas: List[str] = None,
                                    mostrar_progreso: bool = True) -> Dict:
        """
        Ejecuta una comparación completa con todas las configuraciones.
        
        Args:
            arquitecturas: Lista de nombres de arquitecturas a probar
            mostrar_progreso: Si mostrar el progreso durante la ejecución
            
        Returns:
            Dict con todos los resultados organizados por categoría y configuración
        """
        if arquitecturas is None:
            arquitecturas = ['MINIMA', 'COMPACTA', 'DIRECTA_ORIGINAL']
        
        if mostrar_progreso:
            print("🚀 INICIANDO COMPARACIÓN SISTEMÁTICA DE CONJUNTOS DE ENTRENAMIENTO")
            print("=" * 70)
        
        resultados_completos = {}
        
        for categoria, config_categoria in self.configuraciones_experimento.items():
            if mostrar_progreso:
                print(f"\n📂 CATEGORÍA: {categoria.upper()}")
                print(f"   {config_categoria['descripcion']}")
                print("-" * 50)
            
            resultados_categoria = {}
            
            for config_conjunto in config_categoria['conjuntos']:
                nombre_config = config_conjunto['nombre']
                entrenamiento = config_conjunto['entrenamiento']
                prueba = config_conjunto['prueba']
                
                if mostrar_progreso:
                    print(f"\n🔸 Configuración: {nombre_config}")
                    print(f"   Entrenamiento: {entrenamiento}")
                    print(f"   Prueba: {prueba}")
                
                resultados_config = {}
                
                for nombre_arq in arquitecturas:
                    if nombre_arq not in ARQUITECTURAS_TP2:
                        continue
                    
                    arquitectura = ARQUITECTURAS_TP2[nombre_arq]
                    
                    if mostrar_progreso:
                        print(f"     🏗️ Arquitectura {nombre_arq}: {arquitectura}")
                    
                    # Ejecutar experimento
                    resultado = self.entrenador.entrenar_discriminacion_numeros_pares(
                        arquitectura=arquitectura,
                        digitos_entrenamiento=entrenamiento,
                        digitos_prueba=prueba,
                        mostrar_progreso=False
                    )
                    
                    # Calcular métricas adicionales
                    metricas_adicionales = self._calcular_metricas_adicionales(
                        resultado, entrenamiento, prueba
                    )
                    
                    resultado.update(metricas_adicionales)
                    resultados_config[nombre_arq] = resultado
                    
                    if mostrar_progreso:
                        self._mostrar_resumen_resultado(resultado, nombre_arq)
                
                resultados_categoria[nombre_config] = resultados_config
            
            resultados_completos[categoria] = resultados_categoria
        
        self.resultados_comparacion = resultados_completos
        
        if mostrar_progreso:
            print("\n" + "=" * 70)
            print("✅ COMPARACIÓN COMPLETA FINALIZADA")
            self._generar_resumen_comparacion()
        
        return resultados_completos
    
    def _calcular_metricas_adicionales(self, 
                                     resultado: Dict, 
                                     entrenamiento: List[int], 
                                     prueba: List[int]) -> Dict:
        """
        Calcula métricas adicionales para el análisis.
        
        Args:
            resultado: Resultado del experimento
            entrenamiento: Lista de dígitos de entrenamiento
            prueba: Lista de dígitos de prueba
            
        Returns:
            Dict con métricas adicionales
        """
        # Análisis de balance
        pares_train = len([d for d in entrenamiento if d % 2 == 0])
        impares_train = len([d for d in entrenamiento if d % 2 == 1])
        total_train = len(entrenamiento)
        
        pares_test = len([d for d in prueba if d % 2 == 0])
        impares_test = len([d for d in prueba if d % 2 == 1])
        total_test = len(prueba)
        
        # Calcular balance como ratio
        balance_train = min(pares_train, impares_train) / max(pares_train, impares_train) if max(pares_train, impares_train) > 0 else 0
        balance_test = min(pares_test, impares_test) / max(pares_test, impares_test) if max(pares_test, impares_test) > 0 else 0
        
        # Calcular diferencia de rendimiento (overfitting)
        diferencia_rendimiento = resultado.get('precision_entrenamiento', 0) - resultado.get('precision_prueba', 0)
        
        # Evaluar calidad de generalización
        if diferencia_rendimiento <= 0.1:
            calidad_generalizacion = 'Excelente'
        elif diferencia_rendimiento <= 0.3:
            calidad_generalizacion = 'Buena'
        elif diferencia_rendimiento <= 0.5:
            calidad_generalizacion = 'Regular'
        else:
            calidad_generalizacion = 'Pobre'
        
        return {
            'metricas_conjunto': {
                'pares_entrenamiento': pares_train,
                'impares_entrenamiento': impares_train,
                'total_entrenamiento': total_train,
                'pares_prueba': pares_test,
                'impares_prueba': impares_test,
                'total_prueba': total_test,
                'balance_entrenamiento': balance_train,
                'balance_prueba': balance_test,
                'diferencia_rendimiento': diferencia_rendimiento,
                'calidad_generalizacion': calidad_generalizacion
            }
        }
    
    def _mostrar_resumen_resultado(self, resultado: Dict, nombre_arq: str) -> None:
        """Muestra un resumen conciso del resultado."""
        if 'error' in resultado:
            print(f"       ❌ {nombre_arq}: Error - {resultado.get('mensaje', 'Desconocido')}")
            return
        
        prec_train = resultado.get('precision_entrenamiento', 0) * 100
        prec_test = resultado.get('precision_prueba', 0) * 100
        epoca = resultado.get('epoca_convergencia', 0)
        calidad = resultado.get('metricas_conjunto', {}).get('calidad_generalizacion', 'N/A')
        
        print(f"       ✅ {nombre_arq}: Train={prec_train:.1f}% Test={prec_test:.1f}% "
              f"Épocas={epoca} Generalización={calidad}")
    
    def _generar_resumen_comparacion(self) -> None:
        """Genera un resumen ejecutivo de todos los resultados."""
        print("\n📊 RESUMEN EJECUTIVO DE LA COMPARACIÓN")
        print("=" * 50)
        
        # Encontrar mejores configuraciones por métrica
        mejor_generalizacion = self._encontrar_mejor_configuracion('calidad_generalizacion')
        mejor_precision_test = self._encontrar_mejor_configuracion('precision_prueba')
        menor_overfitting = self._encontrar_menor_overfitting()
        
        print(f"\n🏆 MEJORES CONFIGURACIONES:")
        print(f"   Mejor Generalización: {mejor_generalizacion['nombre']} "
              f"({mejor_generalizacion['calidad']})")
        print(f"   Mejor Precisión Test: {mejor_precision_test['nombre']} "
              f"({mejor_precision_test['precision']:.1f}%)")
        print(f"   Menor Overfitting: {menor_overfitting['nombre']} "
              f"(Diferencia: {menor_overfitting['diferencia']:.1f}%)")
        
        # Análisis por categoría
        print(f"\n📈 ANÁLISIS POR CATEGORÍA:")
        for categoria in self.resultados_comparacion:
            promedio_test = self._calcular_promedio_categoria(categoria, 'precision_prueba')
            print(f"   {categoria}: Precisión promedio test = {promedio_test:.1f}%")
    
    def _encontrar_mejor_configuracion(self, metrica: str) -> Dict:
        """Encuentra la mejor configuración según una métrica específica."""
        mejor = {'nombre': 'N/A', 'valor': -1}
        
        for categoria, configs in self.resultados_comparacion.items():
            for nombre_config, arquitecturas in configs.items():
                for nombre_arq, resultado in arquitecturas.items():
                    if 'error' in resultado:
                        continue
                    
                    if metrica == 'calidad_generalizacion':
                        calidades = ['Excelente', 'Buena', 'Regular', 'Pobre']
                        calidad = resultado.get('metricas_conjunto', {}).get(metrica, 'Pobre')
                        valor = len(calidades) - calidades.index(calidad)
                        if valor > mejor['valor']:
                            mejor = {'nombre': f"{categoria}/{nombre_config}/{nombre_arq}", 
                                   'valor': valor, 'calidad': calidad}
                    else:
                        valor = resultado.get(metrica, 0)
                        if valor > mejor['valor']:
                            mejor = {'nombre': f"{categoria}/{nombre_config}/{nombre_arq}", 
                                   'valor': valor, 'precision': valor * 100}
        
        return mejor
    
    def _encontrar_menor_overfitting(self) -> Dict:
        """Encuentra la configuración con menor overfitting."""
        menor = {'nombre': 'N/A', 'diferencia': float('inf')}
        
        for categoria, configs in self.resultados_comparacion.items():
            for nombre_config, arquitecturas in configs.items():
                for nombre_arq, resultado in arquitecturas.items():
                    if 'error' in resultado:
                        continue
                    
                    diferencia = resultado.get('metricas_conjunto', {}).get('diferencia_rendimiento', float('inf'))
                    if diferencia < menor['diferencia']:
                        menor = {
                            'nombre': f"{categoria}/{nombre_config}/{nombre_arq}",
                            'diferencia': diferencia * 100
                        }
        
        return menor
    
    def _calcular_promedio_categoria(self, categoria: str, metrica: str) -> float:
        """Calcula el promedio de una métrica para una categoría."""
        valores = []
        
        for nombre_config, arquitecturas in self.resultados_comparacion[categoria].items():
            for nombre_arq, resultado in arquitecturas.items():
                if 'error' not in resultado:
                    valor = resultado.get(metrica, 0)
                    valores.append(valor)
        
        return np.mean(valores) * 100 if valores else 0
    
    def generar_reporte_detallado(self, archivo_salida: str = None) -> str:
        """
        Genera un reporte detallado de todos los experimentos.
        
        Args:
            archivo_salida: Ruta del archivo donde guardar el reporte
            
        Returns:
            String con el reporte completo
        """
        reporte = []
        reporte.append("REPORTE DETALLADO - COMPARACIÓN DE CONJUNTOS DE ENTRENAMIENTO")
        reporte.append("=" * 80)
        reporte.append("")
        
        for categoria, configs in self.resultados_comparacion.items():
            reporte.append(f"CATEGORÍA: {categoria.upper()}")
            reporte.append("-" * 60)
            
            for nombre_config, arquitecturas in configs.items():
                reporte.append(f"\nConfiguración: {nombre_config}")
                
                # Obtener info del conjunto
                config_info = None
                for cat_config in self.configuraciones_experimento[categoria]['conjuntos']:
                    if cat_config['nombre'] == nombre_config:
                        config_info = cat_config
                        break
                
                if config_info:
                    reporte.append(f"  Entrenamiento: {config_info['entrenamiento']}")
                    reporte.append(f"  Prueba: {config_info['prueba']}")
                
                reporte.append("  Resultados por arquitectura:")
                
                for nombre_arq, resultado in arquitecturas.items():
                    if 'error' in resultado:
                        reporte.append(f"    {nombre_arq}: ERROR - {resultado.get('mensaje', 'Desconocido')}")
                        continue
                    
                    prec_train = resultado.get('precision_entrenamiento', 0) * 100
                    prec_test = resultado.get('precision_prueba', 0) * 100
                    epoca = resultado.get('epoca_convergencia', 0)
                    metricas = resultado.get('metricas_conjunto', {})
                    
                    reporte.append(f"    {nombre_arq}:")
                    reporte.append(f"      Precisión Entrenamiento: {prec_train:.1f}%")
                    reporte.append(f"      Precisión Prueba: {prec_test:.1f}%")
                    reporte.append(f"      Épocas hasta convergencia: {epoca}")
                    reporte.append(f"      Balance entrenamiento: {metricas.get('balance_entrenamiento', 0):.2f}")
                    reporte.append(f"      Calidad generalización: {metricas.get('calidad_generalizacion', 'N/A')}")
                
                reporte.append("")
        
        reporte_texto = "\n".join(reporte)
        
        if archivo_salida:
            with open(archivo_salida, 'w', encoding='utf-8') as f:
                f.write(reporte_texto)
            print(f"📄 Reporte guardado en: {archivo_salida}")
        
        return reporte_texto
    
    def obtener_mejores_configuraciones(self, top_n: int = 5) -> List[Dict]:
        """
        Obtiene las mejores configuraciones ordenadas por rendimiento.
        
        Args:
            top_n: Número de mejores configuraciones a retornar
            
        Returns:
            Lista de las mejores configuraciones con sus métricas
        """
        configuraciones = []
        
        for categoria, configs in self.resultados_comparacion.items():
            for nombre_config, arquitecturas in configs.items():
                for nombre_arq, resultado in arquitecturas.items():
                    if 'error' in resultado:
                        continue
                    
                    config = {
                        'categoria': categoria,
                        'configuracion': nombre_config,
                        'arquitectura': nombre_arq,
                        'precision_entrenamiento': resultado.get('precision_entrenamiento', 0),
                        'precision_prueba': resultado.get('precision_prueba', 0),
                        'diferencia_rendimiento': resultado.get('metricas_conjunto', {}).get('diferencia_rendimiento', 1),
                        'calidad_generalizacion': resultado.get('metricas_conjunto', {}).get('calidad_generalizacion', 'Pobre'),
                        'epoca_convergencia': resultado.get('epoca_convergencia', float('inf'))
                    }
                    configuraciones.append(config)
        
        # Ordenar por precisión de prueba (descendente) y luego por diferencia de rendimiento (ascendente)
        configuraciones.sort(key=lambda x: (-x['precision_prueba'], x['diferencia_rendimiento']))
        
        return configuraciones[:top_n]


def main():
    """Función principal para ejecutar la comparación completa."""
    print("🚀 Iniciando Comparación de Conjuntos de Entrenamiento")
    
    comparador = ComparadorConjuntosEntrenamiento()
    
    # Ejecutar comparación con arquitecturas seleccionadas
    arquitecturas_a_probar = ['MINIMA', 'COMPACTA', 'DIRECTA_ORIGINAL']
    
    resultados = comparador.ejecutar_comparacion_completa(
        arquitecturas=arquitecturas_a_probar,
        mostrar_progreso=True
    )
    
    # Generar reporte detallado
    print("\n📄 Generando reporte detallado...")
    archivo_reporte = "/Users/ttoloza/git/personal/unahur/fundamentos-redes-neuronales/tp2/resultados/reporte_conjuntos_entrenamiento.txt"
    comparador.generar_reporte_detallado(archivo_reporte)
    
    # Mostrar mejores configuraciones
    print("\n🏆 TOP 5 MEJORES CONFIGURACIONES:")
    mejores = comparador.obtener_mejores_configuraciones(5)
    for i, config in enumerate(mejores, 1):
        print(f"{i}. {config['categoria']}/{config['configuracion']}/{config['arquitectura']}")
        print(f"   Precisión Test: {config['precision_prueba']*100:.1f}% | "
              f"Generalización: {config['calidad_generalizacion']} | "
              f"Épocas: {config['epoca_convergencia']}")


if __name__ == "__main__":
    main()
