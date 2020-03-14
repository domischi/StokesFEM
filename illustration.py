import numpy as np
import matplotlib.pyplot as plt

def get_io_colors(X,Y, U,V, _config):
    C = U*X+V*Y
    if 'normalize_io_color_scheme' in _config and _config['normalize_io_color_scheme']:
        mi=min(C.flatten())
        mx=max(C.flatten())
        assert(mi<0 and mx>0)
        C = [r/mi if r<0 else r/mx for r in C]
    return C

def plot_fluid(u,_config, fix_frame=True):
    L = _config['L']
    plt.figure(figsize=(5,5))
    plt.title('Fluids')
    X, Y = np.meshgrid(np.linspace(-L,L,_config['plot_res']),np.linspace(-L,L,_config['plot_res']))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    C = np.ones_like(Y)
    C[0,0]=0
    for i in range(len(X)):
        for j in range(len(X[0])):
            uv = u(X[i,j],Y[i,j])
            U[i,j]=uv[0]
            V[i,j]=uv[1]
    if _config['color_scheme']=='io':
        C = get_io_colors(X,Y, U,V, _config)
        cmap = 'bwr'
    elif _config['color_scheme']=='vabs':
        C = np.sqrt(U**2+V**2)
        cmap = 'viridis'
    else:
        cmap = 'binary'
    if _config['plot_type'] == 'quiver':
        p=plt.quiver(X,Y, U,V, C, cmap=cmap, pivot='mid', scale=10)
    elif _config['plot_type'] == 'streamplot':
        p=plt.streamplot(X,Y, U,V, color=C, cmap=cmap)
    else:
        print('Illegal type of fluid plot. Check the code.')
        return
    if fix_frame:
        plt.xlim([-L,L])
        plt.ylim([-L,L])
    plt.show(block=True)
