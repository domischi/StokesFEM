import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rectangle import cross, active_rect
import time

def get_io_colors(X,Y, U,V, _config):
    C = U*X+V*Y
    if 'normalize_io_color_scheme' in _config and _config['normalize_io_color_scheme']:
        mi=min(C.flatten())
        mx=max(C.flatten())
        assert(mi<0 and mx>0)
        C = [r/mi if r<0 else r/mx for r in C]
    return C

def get_domain(f,X,Y, _config):
    ind=np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            if f((X[i,j], Y[i,j]), False, _config['AR'], _config['bar_width']):
                ind[i,j]=1
    return ind

def sample_velocity(u, _config):
    L = _config['L']
    X, Y = np.meshgrid(np.linspace(-L,L,_config['plot_res']),np.linspace(-L,L,_config['plot_res']))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(len(X)):
        for j in range(len(X[0])):
            uv = u(X[i,j],Y[i,j])
            U[i,j]=uv[0]
            V[i,j]=uv[1]
    return X,Y,U,V

def plot_fluid(u,_config, already_sampled_values = None, fix_frame=True):
    fig = plt.figure(figsize=(5,5))
    plt.title('Fluids')
    filename = f'/tmp/fem-res-{int(time.time())}.png'
    L = _config['L']
    if _config['plot_rectangle']:
        resb = 128
        Xb, Yb = np.meshgrid(np.linspace(-L,L,resb,resb),np.linspace(-L,L,resb,resb))
        ind=get_domain(active_rect, Xb,Yb, _config)
        plt.pcolormesh(Xb,Yb,ind, cmap='Greys', alpha=.2, edgecolor='none')
    if _config['plot_cross']:
        resb = 128
        Xb, Yb = np.meshgrid(np.linspace(-L,L,resb,resb),np.linspace(-L,L,resb,resb))
        ind=get_domain(cross, Xb,Yb, _config)
        plt.pcolormesh(Xb,Yb,ind, cmap='Greys', alpha=.2, edgecolor='none')
    if already_sampled_values == None:
        X, Y, U, V = sample_velocity(u, _config)
    else:
        X, Y, U, V = already_sampled_values
    C = np.ones_like(Y)
    C[0,0]=0
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
    plt.savefig(filename)
    plt.close(fig)
    return filename
