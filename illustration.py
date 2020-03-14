import numpy as np
import matplotlib.pyplot as plt

def plot_fluid(u,_config, fix_frame=True):
    L = _config['L']
    plt.figure(figsize=(5,5))
    plt.title('Fluids')
    X, Y = np.meshgrid(np.linspace(-L,L,_config['plot_res']),np.linspace(-L,L,_config['plot_res']))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(len(X)):
        for j in range(len(X[0])):
            uv = u(X[i,j],Y[i,j])
            U[i,j]=uv[0]
            V[i,j]=uv[1]
    if _config['plot_type'] == 'quiver':
        p=plt.quiver(X,Y, U,V , pivot='mid', scale=10)
    elif _config['plot_type'] == 'streamplot':
        p=plt.streamplot(X,Y, U,V)
    else:
        print('Illegal type of fluid plot. Check the code.')
        return
    if fix_frame:
        plt.xlim([-L,L])
        plt.ylim([-L,L])
    plt.show(block=True)
