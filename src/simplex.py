import numpy as np
import pandas as pd

def to_standard_form(problem):
    """
    Convierte el problema a forma estándar con variables artificiales si es necesario.
    Devuelve:
        - std_problem: Problema en forma estándar
        - artificial_vars: Índices de columnas de variables artificiales
    """
    c = np.array(problem['c'])
    A = np.array(problem['A'])
    b = np.array(problem['b'])
    signs = problem['signs']
    
    # Convertir a minimización
    if problem['type'] == 'max':
        c = -c
    
    num_vars = len(c)
    num_slack = 0
    num_artificial = 0
    slack_cols = []
    artificial_cols = []
    
    # Preparar matrices para holgura, exceso y artificiales
    for i, sign in enumerate(signs):
        if sign == '<=':
            slack_cols.append((i, 1))  # +holgura
            num_slack += 1
        elif sign == '>=':
            slack_cols.append((i, -1)) # -exceso
            num_slack += 1
            artificial_cols.append(i)  # +artificial
            num_artificial += 1
        elif sign == '=':
            artificial_cols.append(i)  # +artificial
            num_artificial += 1
    
    # Construir matriz aumentada
    A_aug = A.copy()
    
    # Añadir holgura/exceso
    if num_slack > 0:
        slack_matrix = np.zeros((len(signs), num_slack))
        pos = 0
        for i, mult in slack_cols:
            slack_matrix[i, pos] = mult
            pos += 1
        A_aug = np.hstack([A_aug, slack_matrix])
    
    # Añadir artificiales
    if num_artificial > 0:
        artificial_matrix = np.zeros((len(signs), num_artificial))
        pos = 0
        for i in artificial_cols:
            artificial_matrix[i, pos] = 1
            pos += 1
        A_aug = np.hstack([A_aug, artificial_matrix])
    
    # Vector de costos extendido
    c_aug = np.hstack([c, np.zeros(num_slack + num_artificial)])
    
    return {
        'A': A_aug,
        'b': b,
        'c': c_aug,
        'num_vars': num_vars,
        'num_slack': num_slack,
        'num_artificial': num_artificial,
        'artificial_cols': list(range(num_vars + num_slack, num_vars + num_slack + num_artificial))
    }

def initialize_tableau(std_problem):
    """
    Inicializa el tableau para el método Simplex.
    
    Args:
        std_problem: Problema en forma estándar (salida de to_standard_form)
    
    Returns:
        numpy.ndarray: Tableau inicial
    """
    m, n = std_problem['A'].shape
    tableau = np.zeros((m+1, n+1))
    
    # Filas de restricciones
    tableau[:-1, :-1] = std_problem['A']
    tableau[:-1, -1] = std_problem['b']
    
    # Fila objetivo
    tableau[-1, :-1] = std_problem['c']
    
    return tableau

def print_tableau(tableau, decimals=2):
    """
    Imprime el tableau de forma legible.
    
    Args:
        tableau: Tableau a imprimir
        decimals: Número de decimales a mostrar
    """
    df = pd.DataFrame(tableau)
    print(df.round(decimals))

def select_entering_variable(tableau):
    """
    Selecciona la variable entrante (columna pivote)."""
    last_row = tableau[-1, :-1]  # Excluir la columna LD
    return np.argmin(last_row)   # Columna con valor más negativo

def select_leaving_variable(tableau, entering_idx):
    """
    Selecciona la variable saliente (fila pivote) usando la regla de razón mínima."""
    ratios = []
    for i in range(len(tableau) - 1):  # Excluir la fila FO
        if tableau[i, entering_idx] > 0:
            ratio = tableau[i, -1] / tableau[i, entering_idx]  # LD / Coef. columna pivote
            ratios.append(ratio)
        else:
            ratios.append(np.inf)  # Evitar división por cero o negativa
    return np.argmin(ratios)  # Fila con la razón mínima

def pivot(tableau, leaving_idx, entering_idx):
    """
    Realiza la operación de pivoteo sobre el tableau.
    
    Args:
        tableau: Tableau actual
        leaving_idx: Índice de la fila pivote
        entering_idx: Índice de la columna pivote
    """
    pivot_val = tableau[leaving_idx, entering_idx]
    tableau[leaving_idx] = tableau[leaving_idx] / pivot_val  # Normalizar fila pivote
    
    # Eliminación gaussiana en otras filas
    for i in range(len(tableau)):
        if i != leaving_idx:
            multiplier = tableau[i, entering_idx]
            tableau[i] = tableau[i] - multiplier * tableau[leaving_idx]

def simplex_method(std_problem, max_iter=100):
    """Método Simplex de dos fases"""
    # Fase 1 (si hay variables artificiales)
    if std_problem['num_artificial'] > 0:
        phase1_tableau = phase_one_simplex(std_problem)
        if phase1_tableau is None:
            return None  # Problema infactible
        
        # Preparar tableau para Fase 2
        # 1. Eliminar columnas artificiales
        tableau = np.delete(phase1_tableau, std_problem['artificial_cols'], axis=1)
        # 2. Resetear FO
        tableau[-1, :] = 0
        # 3. Restaurar FO original solo para variables originales
        tableau[-1, :std_problem['num_vars']] = std_problem['c'][:std_problem['num_vars']]
        
        # Recalcular FO considerando variables básicas
        basic_vars = []
        for col in range(tableau.shape[1]-1):
            if np.sum(tableau[:-1, col] == 1) == 1:  # Variable básica
                row = np.where(tableau[:-1, col] == 1)[0][0]
                basic_vars.append((row, col))
        
        for row, col in basic_vars:
            if col < std_problem['num_vars']:  # Si es variable original
                tableau[-1, :] -= tableau[-1, col] * tableau[row, :]
    else:
        tableau = initialize_tableau(std_problem)
    
    print("\n--- Fase 2: Optimizando FO original ---")
    print_tableau(tableau)
    
    for iteration in range(max_iter):
        if all(tableau[-1, :-1] >= -1e-8):  # Optimalidad
            print(f"\nSolución óptima encontrada en {iteration} iteraciones")
            break
            
        entering = select_entering_variable(tableau)
        leaving = select_leaving_variable(tableau, entering)
        pivot(tableau, leaving, entering)
        print(f"\nIteración {iteration+1}:")
        print_tableau(tableau)
    
    return tableau

def interpret_solution(tableau, num_vars):
    solution = np.zeros(num_vars)
    
    # Identificar variables básicas
    for col in range(num_vars):
        column = tableau[:-1, col]
        if np.sum(column == 1) == 1 and np.sum(column != 0) == 1:  # Columna canónica
            row = np.argwhere(column == 1)[0][0]
            solution[col] = tableau[row, -1]
    
    z = tableau[-1, -1]
    return {'x': solution, 'z': z}


########################################################

def phase_one_simplex(std_problem):
    # Crear FO para Fase 1 (minimizar suma de artificiales)
    c_phase1 = np.zeros(std_problem['A'].shape[1])
    for col in std_problem['artificial_cols']:
        c_phase1[col] = 1
    
    # Tableau para Fase 1
    tableau = initialize_tableau({
        'A': std_problem['A'],
        'b': std_problem['b'],
        'c': c_phase1
    })
    
    print("\n--- Fase 1: Minimizando variables artificiales ---")
    for iteration in range(100):
        if all(tableau[-1, :-1] >= -1e-8):  # Optimalidad
            break
            
        entering = select_entering_variable(tableau)
        leaving = select_leaving_variable(tableau, entering)
        pivot(tableau, leaving, entering)
        print(f"Iteración {iteration+1}:")
        print_tableau(tableau)
    
    # Verificar factibilidad
    if abs(tableau[-1, -1]) > 1e-6:
        print("\n¡Problema infactible! (Suma de artificiales > 0)")
        return None
    
    print("\nFase 1 completada. Problema factible. Preparando Fase 2...")
    return tableau