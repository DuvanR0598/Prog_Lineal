import numpy as np
import sys
from src.utils import load_problem
from src.simplex import to_standard_form, simplex_method, interpret_solution
from src.sensitivity import analyze_sensitivity
from src.graphical import plot_solution  # Importar la función gráfica

def main():
    # 1. Cargar problema
    if len(sys.argv) < 2:
        print("Usa el comando: python main.py <ruta_al_problema>")
        return
    ruta_problema = sys.argv[1]
    try:
        problem = load_problem(ruta_problema)
    except FileNotFoundError:
        print("Error: Archivo no encontrado. Verifica la ruta.")
        return
    except Exception as e:
        print(f"Error al cargar el problema: {str(e)}")
        return

    print("\nProblema cargado:")
    print(f"Función Objetivo: {'Maximizar' if problem['type'] == 'max' else 'Minimizar'} Z = {problem['c']}")
    print("Restricciones:")
    for i, (a, b, sign) in enumerate(zip(problem['A'], problem['b'], problem['signs'])):
        vars_str = " + ".join([f"{a[j]}x_{j+1}" for j in range(len(a))])
        print(f"{i+1}. {vars_str} {sign} {b}")

    # 2. Convertir a forma estándar
    try:
        std_problem = to_standard_form(problem)
    except Exception as e:
        print(f"Error al convertir a forma estándar: {str(e)}")
        return

    # 3. Resolver con Simplex
    print("\nIniciando método Simplex...")
    try:
        final_tableau = simplex_method(std_problem)
        if final_tableau is None:
            print("El problema no tiene solución factible.")
            return
    except Exception as e:
        print(f"Error durante la ejecución del Simplex: {str(e)}")
        return

    # 4. Interpretar resultados
    solution = interpret_solution(final_tableau, std_problem['num_vars'])
    
    # 5. Mostrar solución
    print("\n--- Solución Óptima ---")
    if problem['type'] == 'max':
        solution['z'] = -solution['z']  # Ajustar para maximización (corregido el signo)

    # Mostrar todas las variables
    for j in range(len(solution['x'])):
        print(f"x_{j+1} = {solution['x'][j]:.2f}")
    print(f"Valor óptimo: Z = {solution['z']:.2f}")

    # 6. Visualización gráfica (solo para problemas 2D)
    if 2 <= len(problem['c']) <= 3:
        try:
            plot_solution(
                problem['A'],
                problem['b'],
                problem['signs'],
                solution,
                problem['c']  # Coeficientes de la FO
            )
        except Exception as e:
            print(f"\nError al generar visualización: {str(e)}")
    else:
        print(f"\nVisualización no disponible para {len(problem['c'])} variables")

    # 7. Análisis de sensibilidad
    try:
        sensitivity = analyze_sensitivity(
            final_tableau,
            std_problem['num_vars'],
            std_problem['num_slack']
        )

        print("\n--- Análisis de Sensibilidad ---")
        print("Precios sombra (valores duales):")
        for i, price in enumerate(sensitivity['shadow_prices']):
            print(f"Restricción {i+1} ({problem['signs'][i]}): {price:.2f}")

        print("\nIntervalos de factibilidad para lados derechos (b):")
        for change in sensitivity['allowable_increases']:
            print(f"Restricción {change['constraint']+1} ({problem['signs'][change['constraint']]}):")
            if change['increase'] == np.inf:
                print("  - Puede aumentar ilimitadamente")
            else:
                print(f"  - Puede aumentar hasta {change['increase']:.2f}")
            if change['decrease'] == np.inf:
                print("  - Puede disminuir ilimitadamente")
            else:
                print(f"  - Puede disminuir hasta {change['decrease']:.2f}")

        print("\nCostos reducidos para variables no básicas:")
        for j, cost in enumerate(sensitivity['reduced_costs']):
            if not np.isclose(cost, 0, atol=1e-6):
                print(f"Variable x_{j+1}: {cost:.2f}")
    except Exception as e:
        print(f"\nError en análisis de sensibilidad: {str(e)}")

if __name__ == "__main__":
    main()