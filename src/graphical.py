import matplotlib.pyplot as plt
import numpy as np

def plot_feasible_region(A, b, signs, xlim=(0,10), ylim=(0,10)):
    fig, ax = plt.subplots()
    x = np.linspace(xlim[0], xlim[1], 400)
    
    for i in range(len(A)):
        a1, a2 = A[i]
        bi = b[i]
        sign = signs[i]
        
        if a2 != 0:
            y = (bi - a1*x)/a2
            if sign == '<=':
                ax.fill_between(x, y, ylim[0], alpha=0.1)
            ax.plot(x, y, label=f'{a1}x1 + {a2}x2 {sign} {bi}')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    plt.grid()
    plt.show()