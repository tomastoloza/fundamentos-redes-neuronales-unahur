#!/usr/bin/env python3

import sys
import os
import numpy as np

def mostrar_banner():
    print("=" * 60)
    print("    PERCEPTRÓN SIMPLE - MÚLTIPLES APLICACIONES")
    print("=" * 60)
    print("Implementación del algoritmo del perceptrón siguiendo el pseudocódigo")
    print("Aplicaciones: Compuertas lógicas AND/OR y Regresión")
    print("=" * 60)

def mostrar_menu():
    print("\n🎯 Seleccione una opción:")
    print("1. 🔴 Compuerta AND (Lineal)")
    print("2. 🟡 Compuerta OR (Lineal)") 
    print("3. 🔵 TP1-EJ2 (Regresión)")
    print("4. 🧠 Compuerta AND (No Lineal)")
    print("5. 🧠 Compuerta OR (No Lineal)")
    print("6. 🚪 Salir")

def ejecutar_compuerta_and_lineal():
    try:
        from compuerta_and import entrenar_compuerta_and_lineal
        return entrenar_compuerta_and_lineal()
    except ImportError as e:
        print(f"❌ Error: No se pudo importar compuerta_and.py: {e}")
        return None
    except Exception as e:
        print(f"❌ Error ejecutando compuerta AND lineal: {e}")
        return None

def ejecutar_compuerta_and_no_lineal():
    try:
        from compuerta_and import entrenar_compuerta_and_no_lineal
        return entrenar_compuerta_and_no_lineal()
    except ImportError as e:
        print(f"❌ Error: No se pudo importar compuerta_and.py: {e}")
        return None
    except Exception as e:
        print(f"❌ Error ejecutando compuerta AND no lineal: {e}")
        return None

def ejecutar_compuerta_or_lineal():
    try:
        from compuerta_or import entrenar_compuerta_or_lineal
        return entrenar_compuerta_or_lineal()
    except ImportError as e:
        print(f"❌ Error: No se pudo importar compuerta_or.py: {e}")
        return None
    except Exception as e:
        print(f"❌ Error ejecutando compuerta OR lineal: {e}")
        return None

def ejecutar_compuerta_or_no_lineal():
    try:
        from compuerta_or import entrenar_compuerta_or_no_lineal
        return entrenar_compuerta_or_no_lineal()
    except ImportError as e:
        print(f"❌ Error: No se pudo importar compuerta_or.py: {e}")
        return None
    except Exception as e:
        print(f"❌ Error ejecutando compuerta OR no lineal: {e}")
        return None

def ejecutar_tp1():
    try:
        from tp1 import entrenar_tp1
        return entrenar_tp1()
    except ImportError as e:
        print(f"❌ Error: No se pudo importar tp1.py: {e}")
        return None
    except Exception as e:
        print(f"❌ Error ejecutando TP1: {e}")
        return None

def validar_archivos():
    archivos_requeridos = [
        'perceptron_unificado.py',
        'compuerta_and.py', 
        'compuerta_or.py',
        'tp1.py'
    ]
    
    archivos_faltantes = []
    for archivo in archivos_requeridos:
        if not os.path.exists(archivo):
            archivos_faltantes.append(archivo)
    
    if archivos_faltantes:
        print("⚠️  ADVERTENCIA: Archivos faltantes:")
        for archivo in archivos_faltantes:
            print(f"   • {archivo}")
        print("   Algunas funcionalidades pueden no estar disponibles.\n")
        return False
    
    return True

def main():
    np.random.seed(42)
    
    mostrar_banner()
    validar_archivos()
    
    while True:
        try:
            mostrar_menu()
            opcion = input("\nIngrese su opción (1-6): ").strip()
            
            if opcion == "1":
                print("\n" + "="*60)
                ejecutar_compuerta_and_lineal()
                
            elif opcion == "2":
                print("\n" + "="*60)
                ejecutar_compuerta_or_lineal()
                
            elif opcion == "3":
                print("\n" + "="*60)
                ejecutar_tp1()
                
            elif opcion == "4":
                print("\n" + "="*60)
                ejecutar_compuerta_and_no_lineal()
                
            elif opcion == "5":
                print("\n" + "="*60)
                ejecutar_compuerta_or_no_lineal()
                
            elif opcion == "6":
                print("\n👋 ¡Gracias por usar el sistema de perceptrón!")
                print("🎓 Esperamos que haya sido útil para su aprendizaje.")
                break
                
            else:
                print("❌ Opción inválida. Por favor, seleccione 1-6.")
            
            if opcion in ["1", "2", "3", "4", "5"]:
                input("\n⏸️  Presione Enter para continuar...")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Programa interrumpido por el usuario.")
            print("👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            print("🔄 Regresando al menú principal...")
            input("\n⏸️  Presione Enter para continuar...")

if __name__ == "__main__":
    main()