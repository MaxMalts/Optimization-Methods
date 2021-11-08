import numpy as np


def f(point):
    x, y = point
    return np.sin(y) * np.exp((1 - np.cos(x)) ** 2) +\
        np.cos(x) * np.exp((1 - np.sin(y)) ** 2) + (x - y) ** 2


def nelder_mead(func, p1, p2, p3, alpha=1, beta=0.5, gamma=2, tolerance=1e-10, tolerance_iterations=100):    
    prev_xl = None
    tol_it = 0
    while True:
        # 1
        psorted = [[p1, func(p1)], [p2, func(p2)], [p3, func(p3)]]
        # 2
        psorted.sort(key=lambda x: x[1])
        xl = psorted[0][0]
        xg = psorted[1][0]
        xh = psorted[2][0]
        # 3
        xc = (xg + xl) / 2
        # 4
        xr = (1 + alpha) * xc - alpha * xh
        # 5
        if (func(xr) < func(xl)):
            xe = (1 - gamma) * xc + gamma * xr
            if (func(xe) < func(xr)):
                xh = xe
            else:
                xh = xr
        elif (func(xr) < func(xg)):
            xh = xr
        else:
            if (func(xr) < func(xh)):
                t = xr
                xr = xh
                xh = t
            
            # 6
            xs = beta * xh + (1 - beta) * xc
            # 7
            if (func(xs) < func(xh)):
                xh = xs
            # 8
            else:
                xh = xl + (xh - xl) / 2
                xg = xl + (xg - xl) / 2
        
        # 9
        if ((not (prev_xl is None)) and np.sum(np.abs(prev_xl - xl)) < tolerance):
            tol_it += 1
            if (tol_it > tolerance_iterations):
                return xl
        else:
            tol_it = 0
        
        prev_xl = xl
        
        p1 = xh
        p2 = xg
        p3 = xl