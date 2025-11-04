import numpy as np
import math

def rhs(t, x, y):
    return 998.0*x + 1998.0*y, -999.0*x - 1999.0*y

def exact_x(t): return 4.0*math.exp(-t) - 3.0*math.exp(-1000.0*t)
def exact_y(t): return -2.0*math.exp(-t) + 3.0*math.exp(-1000.0*t)

def rk3(h, t, x, y):
    k1x, k1y = rhs(t, x, y)
    k2x, k2y = rhs(t + h/3.0, x + h*k1x/3.0, y + h*k1y/3.0)
    k3x, k3y = rhs(t + 2.0*h/3.0, x + 2.0*h*k2x/3.0, y + 2.0*h*k2y/3.0)
    return x + h*(k1x + 3.0*k3x)/4.0, y + h*(k1y + 3.0*k3y)/4.0

def am3_step(h, tn, xn, yn, x1, y1, eps, it=30):
    fxn, fyn = rhs(tn, xn, yn)
    fx1, fy1 = rhs(tn + h, x1, y1)
    t2 = tn + 2.0*h
    x, y = x1, y1
    for _ in range(it):
        fx, fy = rhs(t2, x, y)
        G1 = x - (x1 + h*(5.0*fx + 8.0*fx1 - fxn)/12.0)
        G2 = y - (y1 + h*(5.0*fy + 8.0*fy1 - fyn)/12.0)
        a11 = 1.0 - (5.0*h/12.0)*998.0
        a12 = - (5.0*h/12.0)*1998.0
        a21 = + (5.0*h/12.0)*999.0
        a22 = 1.0 + (5.0*h/12.0)*1999.0
        det = a11*a22 - a12*a21
        if abs(det) < 1e-15: break
        dx = (-G1*a22 + G2*a12)/det
        dy = (-a11*G2 + a21*G1)/det
        x += dx; y += dy
        if abs(dx) < eps and abs(dy) < eps: break
    return x, y

def advance_interval(h, tL, tR, eps, txy):
    t, x, y = txy
    x1, y1 = rk3(h, tL, x, y)
    t1 = tL + h
    x2, y2 = rk3(h, t1, x1, y1)
    t2 = t1 + h
    rows = []
    def push(tt, xx, yy):
        rows.append((tt, xx, yy, exact_x(tt), exact_y(tt),
                     abs(xx-exact_x(tt)), abs(yy-exact_y(tt))))
    if abs(tL - 0.0) < 1e-14 and abs(t - 0.0) < 1e-14:
        push(t, x, y)
    if t1 <= tR + 1e-12 and abs((t1*10)-round(t1*10))<1e-12: push(t1, x1, y1)
    if t2 <= tR + 1e-12 and abs((t2*10)-round(t2*10))<1e-12: push(t2, x2, y2)
    tn = t1; t = t2; x_prev, y_prev = x1, y1; x_cur, y_cur = x2, y2
    while t + 1e-12 < tR:
        x_n, y_n = am3_step(h, tn, x_prev, y_prev, x_cur, y_cur, eps)
        tn += h; t += h
        x_prev, y_prev = x_cur, y_cur
        x_cur, y_cur = x_n, y_n
        if abs((t*10)-round(t*10))<1e-12: push(t, x_cur, y_cur)
    return (t, x_cur, y_cur), rows

def solve(eps):
    s = 5.0
    maxRe = 1000.0
    K = 100
    r = 2
    T = 1.0

    t0 = 0.0
    t1 = s / maxRe
    hmin = t1 / K
    hmax = r * hmin
    rows = []
    txy = (0.0, 1.0, 1.0)

    print(f"\nStart of integration [{t0:.3f},{t1:.3f}] with step h_min = {hmin:.6g}")
    tL, tR = t0, min(T, t1)
    if tL < tR:
        txy, R = advance_interval(hmin, tL, tR, eps, txy)
        rows += R

    if t1 < T:
        print(f"\nStep changed on h_max = {hmax:.6g} for interval [{t1:.3f},{T:.3f}]")
        txy, R = advance_interval(hmax, t1, T, eps, txy)
        rows += R

    print(f"\nt1={t1:.6g}, h_min={hmin:.6g}, h_max={hmax:.6g}\n")
    print("{:>8s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "t","x","y","exact_x","exact_y","err_x","err_y"))
    for r in rows:
        print("{:8.4f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.2e} {:12.2e}".format(*r))

print("Test example 2")

try:
    eps = float(input("eps: "))
except Exception:
    eps = 1e-8
    print("used eps = 1e-8")

solve(eps)
