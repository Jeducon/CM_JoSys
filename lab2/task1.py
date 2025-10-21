import math
import sys
import numpy as np
import matplotlib.pyplot as plt

def f_rhs(t, x, y):
    return (t / y, -t / x)

def exact_x(t): return math.exp(0.5*t*t)
def exact_y(t): return math.exp(-0.5*t*t)


def rk3_step(h, t, x, y):
    k1x, k1y = f_rhs(t, x, y)
    k2x, k2y = f_rhs(t + h/3.0, x + h/3.0*k1x, y + h/3.0*k1y)
    k3x, k3y = f_rhs(t + 2.0*h/3.0, x + 2.0*h/3.0*k2x, y + 2.0*h/3.0*k2y)
    x_out = x + h/4.0*(k1x + 3.0*k3x)
    y_out = y + h/4.0*(k1y + 3.0*k3y)
    return x_out, y_out


def am3_newton_step(h, tn, xn, yn, xn1, yn1, eps, maxit=30):

    fxn, fyn = f_rhs(tn,   xn,  yn)
    fx1, fy1 = f_rhs(tn+h, xn1, yn1)
    tnp2 = tn + 2.0*h

    x2, y2 = xn1, yn1

    for _ in range(maxit):
        
        fxp, fyp = tnp2 / y2, -tnp2 / x2

        G1 = x2 - (xn1 + h/12.0*(5.0*fxp + 8.0*fx1 - fxn))
        G2 = y2 - (yn1 + h/12.0*(5.0*fyp + 8.0*fy1 - fyn))

        
        a11 = 1.0
        a12 = (5.0*h/12.0) * (tnp2/(y2*y2))      
        a21 = -(5.0*h/12.0) * (tnp2/(x2*x2))     
        a22 = 1.0

        det = a11*a22 - a12*a21
        if abs(det) < 1e-15: 
            break

        dx = (-G1*a22 + G2*a12) / det
        dy = (-a11*G2 + a21*G1) / det

        x2 += dx
        y2 += dy
        if abs(dx) < eps and abs(dy) < eps:
            break

    return x2, y2

def find_h_opt(eps, h0):
    h = h0
    while True:
        
        x0, y0, t0 = 1.0, 1.0, 0.0
        x1, y1 = rk3_step(h, t0, x0, y0)                           
        x_big, y_big = am3_newton_step(h, t0, x0, y0, x1, y1, eps) 

        
        hh = h/2.0
        xa, ya = rk3_step(hh, t0, x0, y0)                          
        xb, yb = rk3_step(hh, t0+hh, xa, ya)                       
        xc, yc = am3_newton_step(hh, t0,    x0, y0, xa, ya, eps)   
        xd, yd = am3_newton_step(hh, t0+hh, xa, ya, xb, yb, eps)   
        x_half, y_half = am3_newton_step(hh, t0+h, xb, yb, xd, yd, eps)  

        err = max(abs(x_big - x_half), abs(y_big - y_half)) / 7.0  
        if err <= eps:
            return h
        h *= 0.5


def solve_task(eps=1e-5, h0=0.05, T=1.0):
    h = find_h_opt(eps, h0)
    print(f"h_opt = {h:.6g}\n")
    header = "{:>8s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "t","x","y","exact_x","exact_y","err_x","err_y")
    print(header)

   
    t = 0.0
    x = 1.0
    y = 1.0
    def row(tt, xx, yy):
        print("{:8.4f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}".format(
            tt, xx, yy, exact_x(tt), exact_y(tt), abs(xx-exact_x(tt)), abs(yy-exact_y(tt))))
    row(t, x, y)

    
    x1, y1 = rk3_step(h, t, x, y);  t += h
    x2, y2 = rk3_step(h, t, x1, y1); t += h

    
    if abs((t-h)*10 - round((t-h)*10)) < 1e-12: row(t-h, x1, y1)
    if abs( t    *10 - round( t    *10)) < 1e-12: row(t,   x2, y2)

    
    tn = h
    x_prev, y_prev = x1, y1
    x_curr, y_curr = x2, y2
    while t + 1e-12 < T:
        x_next, y_next = am3_newton_step(h, tn, x_prev, y_prev, x_curr, y_curr, eps)
        tn   += h
        t    += h
        x_prev, y_prev = x_curr, y_curr
        x_curr, y_curr = x_next, y_next
        if abs(t*10 - round(t*10)) < 1e-12:
            row(t, x_curr, y_curr)


def am3_sigma(theta):
    import cmath
    rho = cmath.exp(1j*theta)
    alpha = rho*rho - rho
    beta  = (5.0/12.0)*rho*rho + (8.0/12.0)*rho - (1.0/12.0)
    z = alpha / beta
    return (z.real, z.imag)

def table_sigma(n=360):
    import numpy as np
    th = np.linspace(0.0, 2.0*np.pi, n+1)
    rows = []
    for t in th:
        u, v = am3_sigma(t)
        rows.append((t, u, v))
    return rows  

def plot_boundary_and_region():
    
    data = table_sigma(720)
    uu = np.array([r[1] for r in data])
    vv = np.array([r[2] for r in data])

    def stable_mask(U, V):
        Z = U + 1j*V
        M = np.zeros_like(U, dtype=bool)
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                z = Z[i, j]
                c2 = 1.0 - z*(5.0/12.0)
                c1 = -1.0 - z*(8.0/12.0)
                c0 = 0.0 - z*(-1.0/12.0)
                roots = np.roots([c2, c1, c0])
                if np.all(np.abs(roots) < 1.0 - 1e-12):
                    M[i, j] = True
        return M

    umin, umax = -10.0, 3.0
    vmin, vmax = -10.0, 10.0
    U, V = np.meshgrid(np.linspace(umin, umax, 240),
                       np.linspace(vmin, vmax, 240))
    M = stable_mask(U, V)

    plt.figure(figsize=(7,6))
    plt.contourf(U, V, M, levels=[-0.5,0.5,1.5], alpha=0.75)
    plt.plot(uu, vv, lw=1.3)                 # межа σ(θ)
    plt.axhline(0, lw=0.5); plt.axvline(0, lw=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("u = Re(σ)"); plt.ylabel("v = Im(σ)")
    plt.title("AM3: область абсолютної стійкості")
    plt.grid(True, ls='--', lw=0.3)
    plt.show()


if __name__ == "__main__":
    try:
        eps = float(input("eps: "))
        h0  = float(input("h0 : "))
    except Exception:
        eps, h0 = 1e-5, 0.05
        print(f"Using eps={eps}, h0={h0}.")
    solve_task(eps, h0, T=1.0)

    
    print("\nTable σ(θ)=u(θ)+iv(θ):")
    tbl = table_sigma(36)  
    for i in range(min(10, len(tbl))):
        th,u,v = tbl[i]
        print(f"θ={th:8.5f}  u={u:10.6f}  v={v:10.6f}")

plot_boundary_and_region()
    