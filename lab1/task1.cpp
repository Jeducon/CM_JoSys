#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

void f_rhs(double t, double x, double y, double &fx, double &fy) {
    fx = 2*x - y + t*t - 2*(sin(t)+1) + cos(t);
    fy = x + 2*y - sin(t) - 2*t*t + 2*t - 1;
}

void rk3_step(double h, double t, double &x, double &y) {
    double fx1, fy1, fx2, fy2, fx3, fy3;
    f_rhs(t, x, y, fx1, fy1);
    f_rhs(t + h/3.0, x + h/3.0*fx1, y + h/3.0*fy1, fx2, fy2);
    f_rhs(t + 2*h/3.0, x + 2*h/3.0*fx2, y + 2*h/3.0*fy2, fx3, fy3);
    x += h/4.0*(fx1 + 3.0*fx3);
    y += h/4.0*(fy1 + 3.0*fy3);
}

void am3_step(double h, double tn,
              double xn, double yn,
              double xn1, double yn1,
              double &xn2, double &yn2,
              double eps, int maxit=50) {
    double fxn, fyn, fxn1, fyn1;
    f_rhs(tn, xn, yn, fxn, fyn);
    f_rhs(tn+h, xn1, yn1, fxn1, fyn1);
    xn2 = xn1; yn2 = yn1;
    for (int it=0; it<maxit; ++it) {
        double fxp, fyp;
        f_rhs(tn+2*h, xn2, yn2, fxp, fyp);
        double nx = xn1 + h/12.0*(5*fxp + 8*fxn1 - fxn);
        double ny = yn1 + h/12.0*(5*fyp + 8*fyn1 - fyn);
        if (fabs(nx-xn2)<eps && fabs(ny-yn2)<eps) { xn2=nx; yn2=ny; break; }
        xn2 = nx; yn2 = ny;
    }
}

int main() {
    double T=1.0, eps, h;
    cout<<"eps: "; cin>>eps;
    cout<<"h0 : "; cin>>h;

    while (true) {
        double x0=1.0,y0=0.0;
        double x1=x0,y1=y0; rk3_step(h,0.0,x1,y1);
        double x2h,y2h; am3_step(h,0.0,x0,y0,x1,y1,x2h,y2h,eps);

        double xh2=1.0,yh2=0.0;
        rk3_step(h/2,0.0,xh2,yh2);
        rk3_step(h/2,h/2,xh2,yh2);
        double xmid=xh2,ymid=yh2;
        double x2h2,y2h2;
        am3_step(h/2,0.0,1.0,0.0,xmid,ymid,x2h2,y2h2,eps);
        double x4,y4;
        am3_step(h/2,h/2,xmid,ymid,x2h2,y2h2,x4,y4,eps);

        double ex=fabs(x2h-x4), ey=fabs(y2h-y4);
        double err=max(ex,ey)/(pow(2.0,3.0)-1.0);

        if (err<=eps) break;
        h/=2.0;
    }
    cout<<"h_opt = "<<h<<"\n\n";

    cout<<fixed<<setprecision(6);
    cout<<"   t        x           y       exact_x     exact_y     err_x      err_y\n";

    double t=0.0, x=1.0,y=0.0;
    cout<<setw(6)<<t<<" "<<setw(10)<<x<<" "<<setw(10)<<y<<" "
        <<setw(10)<<sin(t)+1<<" "<<setw(10)<<t*t<<" "
        <<setw(10)<<fabs(x-(sin(t)+1))<<" "<<setw(10)<<fabs(y-t*t)<<"\n";

    double x1=x,y1=y; rk3_step(h,t,x1,y1); t+=h;
    double x2=x1,y2=y1; rk3_step(h,t,x2,y2); t+=h;

    cout<<setw(6)<<h<<" "<<setw(10)<<x1<<" "<<setw(10)<<y1<<" "
        <<setw(10)<<sin(h)+1<<" "<<setw(10)<<h*h<<" "
        <<setw(10)<<fabs(x1-(sin(h)+1))<<" "<<setw(10)<<fabs(y1-h*h)<<"\n";
    cout<<setw(6)<<2*h<<" "<<setw(10)<<x2<<" "<<setw(10)<<y2<<" "
        <<setw(10)<<sin(2*h)+1<<" "<<setw(10)<<(2*h)*(2*h)<<" "
        <<setw(10)<<fabs(x2-(sin(2*h)+1))<<" "<<setw(10)<<fabs(y2-(2*h)*(2*h))<<"\n";

    double tn = h;                 
    double x_prev=x1,y_prev=y1;    
    double x_curr=x2,y_curr=y2;    

    while (t<T-1e-12) {
        double x_next,y_next;
        am3_step(h, tn, x_prev,y_prev, x_curr,y_curr, x_next,y_next, eps);
        tn   += h;
        t    += h;
        x_prev = x_curr; y_prev = y_curr;
        x_curr = x_next; y_curr = y_next;

        if (fabs(fmod(t,0.1))<1e-12 || fabs(fmod(t,0.1)-0.1)<1e-12) {
            cout<<setw(6)<<t<<" "<<setw(10)<<x_curr<<" "<<setw(10)<<y_curr<<" "
                <<setw(10)<<sin(t)+1<<" "<<setw(10)<<t*t<<" "
                <<setw(10)<<fabs(x_curr-(sin(t)+1))<<" "<<setw(10)<<fabs(y_curr-t*t)<<"\n";
        }
    }
    return 0;
}



