### función para leer archivos de entrada ###

def load_problem(filepath):
    """Carga un problema de PL desde archivo"""
    with open(filepath, 'r') as f:
        # Leer cabecera (tipo de problema y función objetivo)
        problem_type = f.readline().strip()  # 'max' o 'min'
        c = list(map(float, f.readline().strip().split(',')))  # Coeficientes FO
        
        # Leer restricciones
        A = []
        b = []
        signs = []
        for line in f:
            parts = line.strip().split(',')
            A.append(list(map(float, parts[:-2])))
            signs.append(parts[-2])
            b.append(float(parts[-1]))
            
    return {
        'type': problem_type,
        'c': c,
        'A': A,
        'b': b,
        'signs': signs
    }