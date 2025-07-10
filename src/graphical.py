import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_solution(A, b, signs, solution, obj_coeff):
    """Función unificada para 2D y 3D con manejo robusto de errores"""
    if not A or not solution or 'x' not in solution:
        print("Error: Datos insuficientes para graficar")
        return

    num_vars = len(solution['x'])
    
    try:
        if num_vars == 2:
            plot_2d_solution(A, b, signs, solution, obj_coeff)
        elif num_vars == 3:
            plot_3d_solution(A, b, signs, solution, obj_coeff)
        else:
            print(f"Visualización no soportada para {num_vars} variables")
    except Exception as e:
        print(f"Error al generar visualización: {str(e)}")

def plot_2d_solution(A, b, signs, solution, obj_coeff):
    """Visualización 2D mejorada con validación de datos"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Validar dimensiones
    for a in A:
        if len(a) < 2:
            raise ValueError("Coeficientes de restricción incompletos para 2D")
    
    x = np.linspace(0, max(b)*1.2, 400)
    
    # Graficar restricciones
    for i, (a, bi, sign) in enumerate(zip(A, b, signs)):
        if len(a) >= 2 and a[1] != 0:  # Validación adicional
            y = (bi - a[0]*x) / a[1]
            label = f'{a[0]}x₁ + {a[1]}x₂ {sign} {bi}'
            ax.plot(x, y, label=label)
            
            if sign == '<=':
                ax.fill_between(x, y, 0, alpha=0.1)
            elif sign == '>=':
                ax.fill_between(x, y, max(b)*1.2, alpha=0.1)
    
    # Graficar solución óptima
    if len(solution['x']) >= 2:
        x_opt, y_opt = solution['x'][0], solution['x'][1]
        ax.scatter(x_opt, y_opt, color='red', s=100, 
                  label=f'Óptimo: ({x_opt:.2f}, {y_opt:.2f})')
        
        # Curvas de nivel (solo si tenemos coeficientes)
        if obj_coeff and len(obj_coeff) >= 2:
            X, Y = np.meshgrid(np.linspace(0, max(b)*1.2, 100), 
                             np.linspace(0, max(b)*1.2, 100))
            Z = obj_coeff[0]*X + obj_coeff[1]*Y
            ax.contour(X, Y, Z, levels=10, alpha=0.3, linestyles='dashed')
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    plt.title('Solución Óptima 2D')
    plt.grid()
    plt.show()

def plot_3d_solution(A, b, signs, solution, obj_coeff):
    """Visualización 3D mejorada con validación de datos"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Validar dimensiones
    for a in A:
        if len(a) < 3:
            raise ValueError("Coeficientes de restricción incompletos para 3D")
    
    max_val = max(b) * 1.5
    x = y = np.linspace(0, max_val, 20)
    X, Y = np.meshgrid(x, y)
    
    # Graficar restricciones
    for i, (a, bi, sign) in enumerate(zip(A, b, signs)):
        if len(a) >= 3 and a[2] != 0:  # Validación adicional
            Z = (bi - a[0]*X - a[1]*Y) / a[2]
            surf = ax.plot_surface(X, Y, Z, alpha=0.3, 
                                 label=f'{a[0]}x₁ + {a[1]}x₂ + {a[2]}x₃ {sign} {bi}')
            
            # Compatibilidad con versiones de Matplotlib
            if hasattr(surf, '_facecolors'):
                surf._facecolors2d = surf._facecolors
                surf._edgecolors2d = surf._edgecolors
    
    # Graficar solución óptima
    if len(solution['x']) >= 3:
        x_opt, y_opt, z_opt = solution['x'][0], solution['x'][1], solution['x'][2]
        ax.scatter(x_opt, y_opt, z_opt, color='red', s=100,
                  label=f'Óptimo: ({x_opt:.2f}, {y_opt:.2f}, {z_opt:.2f})')
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_zlim(0, max_val)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('x₃')
    ax.legend()
    plt.title('Solución Óptima 3D')
    plt.tight_layout()
    plt.show()