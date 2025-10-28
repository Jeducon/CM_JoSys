#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

void f_rhs(double t, double x, double y, double &fx, double &fy) {
    fx = t / y;
    fy = -t / x;
}

void rk3_step(double h, double t, double &x, double &y) {
    double fx1, fy1, fx2, fy2, fx3, fy3;
    f_rhs(t, x, y, fx1, fy1);
    f_rhs(t + h/3.0, x + h/3.0*fx1, y + h/3.0*fy1, fx2, fy2);
    f_rhs(t + 2.0*h/3.0, x + 2.0*h/3.0*fx2, y + 2.0*h/3.0*fy2, fx3, fy3);
    x += h/4.0*(fx1 + 3.0*fx3);
    y += h/4.0*(fy1 + 3.0*fy3);
}

void am3_step(double h, double tn,
              double xn, double yn,
              double xn1, double yn1,
              double &xn2, double &yn2,
              double eps, int maxit=50) {
    double fxn, fyn, fxn1, fyn1;
    f_rhs(tn,   xn,  yn,  fxn,  fyn);
    f_rhs(tn+h, xn1, yn1, fxn1, fyn1);
    xn2 = xn1; yn2 = yn1;
    for (int it=0; it<maxit; ++it) {
        double fxp, fyp;
        f_rhs(tn+2*h, xn2, yn2, fxp, fyp);
        double nx = xn1 + h/12.0*(5.0*fxp + 8.0*fxn1 - fxn);
        double ny = yn1 + h/12.0*(5.0*fyp + 8.0*fyn1 - fyn);
        if (fabs(nx-xn2)<eps && fabs(ny-yn2)<eps) { xn2=nx; yn2=ny; break; }
        xn2 = nx; yn2 = ny;
    }
}

int main() {
    double T=1.0, eps, h;
    cout<<"eps: "; cin>>eps;
    cout<<"h0 : "; cin>>h;

    while (true) {
        double x0=1.0, y0=1.0, t0=0.0;
        double x1=x0, y1=y0; rk3_step(h, t0, x1, y1);
        double x2h, y2h;     am3_step(h, t0, x0, y0, x1, y1, x2h, y2h, eps);

        double hh=h/2.0;
        double xa=1.0, ya=1.0; rk3_step(hh, t0, xa, ya);
        rk3_step(hh, t0+hh, xa, ya);
        double xb=xa, yb=ya, xc, yc, xd, yd;
        am3_step(hh, t0,      1.0,1.0, xb, yb, xc, yc, eps);
        am3_step(hh, t0+hh,   xb, yb, xc, yc, xd, yd, eps);
        double x2h2, y2h2;
        am3_step(hh, t0+h,    xc, yc, xd, yd, x2h2, y2h2, eps);

        double ex=fabs(x2h-x2h2), ey=fabs(y2h-y2h2);
        double err=max(ex,ey)/7.0;
        if (err<=eps) break;
        h/=2.0;
    }

    cout<<"h_opt = "<<h<<"\n\n";
    cout<<fixed<<setprecision(10);
    cout<<"   t        x           y       exact_x     exact_y     err_x      err_y\n";

    auto pr=[&](double tt,double xx,double yy){
        double exx=exp(tt*tt/2.0), eyy=exp(-tt*tt/2.0);
        if (fabs(tt*10.0 - round(tt*10.0))<1e-9)
            cout<<setw(6)<<tt<<" "<<setw(10)<<xx<<" "<<setw(10)<<yy<<" "
                <<setw(10)<<exx<<" "<<setw(10)<<eyy<<" "
                <<setw(10)<<fabs(xx-exx)<<" "<<setw(10)<<fabs(yy-eyy)<<"\n";
    };

    double t=0.0, x=1.0,y=1.0;
    pr(t,x,y);

    double x1=x,y1=y; rk3_step(h,t,x1,y1); t+=h; pr(t,x1,y1);
    double x2=x1,y2=y1; rk3_step(h,t,x2,y2); t+=h; pr(t,x2,y2);

    double tn=h;                 
    double x_prev=x1,y_prev=y1;  
    double x_curr=x2,y_curr=y2;  

    while (t + 1e-12 < T) {
        double x_next,y_next;
        am3_step(h, tn, x_prev,y_prev, x_curr,y_curr, x_next,y_next, eps);
        tn   += h;
        t    += h;
        x_prev = x_curr; y_prev = y_curr;
        x_curr = x_next; y_curr = y_next;
        pr(t,x_curr,y_curr);
    }
    return 0;
}

