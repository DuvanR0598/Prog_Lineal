import numpy as np

def analyze_sensitivity(final_tableau, num_vars, num_slack):
    """
    Realiza análisis de sensibilidad a partir del tableau óptimo.
    
    Args:
        final_tableau: Tableau final del método Simplex
        num_vars: Número de variables originales
        num_slack: Número de variables de holgura
    
    Returns:
        dict: Resultados del análisis con:
            - shadow_prices: Precios sombra
            - allowable_increases: Intervalos de optimalidad
            - reduced_costs: Costos reducidos
    """
    # Precios sombra (coeficientes de variables de holgura en FO)
    shadow_prices = final_tableau[-1, num_vars:num_vars+num_slack]
    
    # Costos reducidos (variables no básicas originales)
    reduced_costs = final_tableau[-1, :num_vars]
    
    # Intervalos de optimalidad (para lados derechos)
    allowable_changes = calculate_allowable_changes(final_tableau, num_vars)
    
    return {
        'shadow_prices': shadow_prices,
        'reduced_costs': reduced_costs,
        'allowable_increases': allowable_changes
    }

def calculate_allowable_changes(tableau, num_vars):
    """
    Calcula los intervalos en los que pueden variar los lados derechos (b)
    sin cambiar la base óptima.
    """
    m, n = tableau.shape
    basis = []
    allowable = []
    
    # Identificar variables básicas
    for col in range(num_vars, n-1):
        if np.sum(tableau[:-1, col] == 1) == 1:  # Es variable básica
            row = np.where(tableau[:-1, col] == 1)[0][0]
            basis.append((row, col))
    
    # Calcular intervalos para cada restricción
    for row, col in basis:
        ratios = []
        for i in range(m-1):
            if tableau[i, col] != 0:
                ratio = tableau[i, -1] / tableau[i, col]
                ratios.append(ratio)
        
        allowable.append({
            'constraint': row,
            'increase': min([r for r in ratios if r > 0], default=np.inf),
            'decrease': abs(max([r for r in ratios if r < 0], default=np.inf))
        })
    
    return allowable