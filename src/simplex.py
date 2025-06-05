import numpy as np
import pandas as pd  # Necesario para print_tableau()

def to_standard_form(problem):
    """Convierte el problema a forma estándar para Simplex"""
    c = np.array(problem['c'])
    A = np.array(problem['A'])
    b = np.array(problem['b'])
    signs = problem['signs']
    
    # Convertir a problema de minimización
    if problem['type'] == 'max':
        c = -c
    
    # Añadir variables de holgura/exceso
    num_slack = len(signs)
    slack_matrix = np.eye(num_slack)
    
    for i, sign in enumerate(signs):
        if sign == '>=':
            slack_matrix[i] *= -1
    
    A_std = np.hstack([A, slack_matrix])
    c_std = np.hstack([c, np.zeros(num_slack)])
    
    return {
        'A': A_std,
        'b': b,
        'c': c_std,
        'num_vars': len(c),
        'num_slack': num_slack
    }

# --- NUEVAS FUNCIONES A AGREGAR ---
def initialize_tableau(std_problem):
    """Crea el tableau inicial para Simplex"""
    m, n = std_problem['A'].shape
    tableau = np.zeros((m+1, n+1))
    
    # Filas de restricciones
    tableau[:-1, :-1] = std_problem['A']
    tableau[:-1, -1] = std_problem['b']
    
    # Fila objetivo
    tableau[-1, :-1] = std_problem['c']
    
    return tableau

def print_tableau(tableau, decimals=2):
    """Muestra el tableau formateado"""
    df = pd.DataFrame(tableau)
    print(df.round(decimals))