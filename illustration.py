import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometries.rectangle import cross_rect, active_rect
from geometries.hexagon import active_hexagon, corner_hexagon, cross_hexagon
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
            if f((X[i,j], Y[i,j])):
                ind[i,j]=1
    return ind

def sample_velocity(u, _config, L=None):
    if L == None:
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

def plot_active_areas(ax, _config):
    plt.sca(ax)
    L = _config['L']
    if _config['plot_active']:
        resb = 128
        Xb, Yb = np.meshgrid(np.linspace(-L,L,resb,resb),np.linspace(-L,L,resb,resb))
        if  _config['Geometry']=='rectangle':
            active_domain = lambda x: active_rect(x, False, _config['AR'], 0.)
        elif  _config['Geometry']=='hexagon':
            active_domain = active_hexagon
        else:
            raise RuntimeError("Unrecognized geometry in plot_pressure: {_config['Geometry']}")
        ind=get_domain(active_domain, Xb,Yb, _config)
        print(active_domain)
        plt.pcolormesh(Xb,Yb,ind, cmap='Greys', alpha=.2, edgecolor='none')
    if _config['plot_cross']:
        resb = 128
        Xb, Yb = np.meshgrid(np.linspace(-L,L,resb,resb),np.linspace(-L,L,resb,resb))
        if  _config['Geometry']=='rectangle':
            cross = lambda x: cross_rect(x,False, AR=_config['AR'], bar_width = _config['bar_width'])
        elif  _config['Geometry']=='hexagon':
            cross = lambda x: cross_hexagon(x, width = _config['bar_width'])
        else:
            raise RuntimeError("Unrecognized geometry in plot_pressure: {_config['Geometry']}")
        ind=get_domain(cross, Xb,Yb, _config)
        plt.pcolormesh(Xb,Yb,ind, cmap='Greys', alpha=.2, edgecolor='none')
    if _config['plot_corner']:
        resb = 128
        Xb, Yb = np.meshgrid(np.linspace(-L,L,resb,resb),np.linspace(-L,L,resb,resb))
        if  _config['Geometry']=='rectangle':
            raise NotImplementedError('Corner plotting for rectangle not implemented')
        elif  _config['Geometry']=='hexagon':
            corner = lambda x: corner_hexagon(x, _config['bar_width'])
        else:
            raise RuntimeError("Unrecognized geometry in plot_pressure: {_config['Geometry']}")
        ind=get_domain(corner, Xb,Yb, _config)
        plt.pcolormesh(Xb,Yb,ind, cmap='Greys', alpha=.2, edgecolor='none')

def plot_fluid(u,_config, already_sampled_values = None, fix_frame=True, title=None):
    fig = plt.figure(figsize=(5,5))
    if title is None:
        plt.title('Fluids')
    else:
        plt.title(title)
    filename = f'/tmp/fem-res-{int(time.time())}.png'
    L = _config['Lplot']
    plot_active_areas(plt.gca(), _config)
    if _config['L'] == _config['Lplot']:
        if already_sampled_values == None:
            X, Y, U, V = sample_velocity(u, _config)
        else:
            X, Y, U, V = already_sampled_values
    else:
        X, Y, U, V = sample_velocity(u, _config, L=_config['Lplot'])
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
        p=plt.quiver(X,Y, U,V, C, cmap=cmap, pivot='mid')
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

def plot_pressure(p,_config, fix_frame=True):
    fig = plt.figure(figsize=(6,5))
    plt.title('Fluids')
    filename = f'/tmp/fem-pressure-{int(time.time())}.png'
    L = _config['Lplot']
    res = _config['plot_res']
    plot_active_areas(plt.gca(), _config)
    X, Y = np.meshgrid(np.linspace(-L,L,res,res),np.linspace(-L,L,res,res))
    C = np.ones_like(Y)
    for i in range(len(Y)):
        for j in range(len(Y[1])):
            C[i,j]=p(X[i,j],Y[i,j])
    ret = plt.pcolormesh(X,Y,C, cmap='viridis', alpha=.8, edgecolor='none')
    plt.colorbar(ret)
    if fix_frame:
        plt.xlim([-L,L])
        plt.ylim([-L,L])
    plt.savefig(filename)
    plt.close(fig)
    return filename
