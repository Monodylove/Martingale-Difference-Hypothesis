
#include <iostream>
#include <armadillo>
#include <math.h>
#include "stats.hpp"

using namespace std;
using namespace arma;
using namespace stats;

double
GenW(int n)
{
    int i;
    double p;
    vec rep;
    rep.ones(n);
    vec zmat;
    zmat.fill(-(sqrt(5) - 1) / 2);
    vec u = randn(n);
    p = (sqrt(5) + 1) / (2 * sqrt(5));
    zmat.elem(find(u > p)).fill(-(sqrt(5) + 1) / 2);
    //zmat.print();
    zmat.save("W_t.txt");
    return (0);
}
//maybe Generate W_t is useful. But for this paper,as I understand ,W_t ought to calculate {sqrt(n)*R^*_n(t_slide)}
//which means if I just want K_pn and C_pn, I don't need W_t because here W_t has been specialised.

double
C_pn(vec Y, int p)
{
    int i, t, j, N;
    double Cpn = 0, Sum = 0, sum = 0, ssum = 0, ymean = 0;
    N = Y.n_elem;
    vec Y1 = Y - mean(Y);
    mat Z;
    umat C;
    for (i = 0; i < N - p; i++)
    {
        Z.insert_rows(i, Y1.subvec(i, i + p - 1).t());
    }

    for (j = 0; j < N - p; j++)
    {
        for (t = 0; t < N - p; t++)
        {
            C = (Z.row(t) <= Z.row(j));
            if (C.is_zero()) { sum = sum + Y(t) - mean(Y); }
            else { sum = sum; }
        }
        ssum = sum * sum;
        Sum = Sum + ssum;
        ssum = 0; sum = 0;
    }
    Cpn = Sum / ((double)N * (double)N);
    return(Cpn);
}

double
K_pn(vec Y, int p)
{
    int N, i, j;
    N = Y.n_elem;
    double Kpn = 0, sum = 0, asum = 0, Max = 0, ymean = 0;
    vec Y1 = Y - mean(Y);
    mat Z;
    umat C;
    for (i = 0; i < N - p; i++)
    {
        Z.insert_rows(i, Y1.subvec(i, i + p - 1).t());
    }
    for (i = 0; i < N - p; i++)
    {
        for (j = 0; j < N - p; j++)
        {
            C = (Z.row(j) <= Z.row(i));
            if (C.is_zero()) { sum = sum + Y(j) - mean(Y); }
            else { sum = sum; }
        }
        asum = fabs(sum / (double)sqrt(N));
        if (asum >= Max) { Max = asum; }
        sum = 0;
    }
    Kpn = Max;
    return(Kpn);
}

double Beta(double alpha, double beta)
{
    double rand_val = stats::rbeta(alpha, beta);
    return rand_val;
}

vec GenData(int n, int m)  //n : generating function  ;m:sample size,m=100 is recommended. 
{
    arma::arma_rng::set_seed_random();//Ensure that the random number generated each time is different 
    int i;
    mat y = linspace(0, m - 1, m);
    vec kesai = randn(m);
    if (n == 1)   //iid N(0,1)
        y = randn(m);
    else if (n == 2)  //GARCH1 : w=0.001,alpha=0.01,beta=0.97
    {
        double w, alpha, beta;
        w = 0.001, alpha = 0.01, beta = 0.97;
        mat sigma = linspace(0, m - 1, m);
        sigma.replace(0, w);
        y.replace(0, (kesai(0) * sigma(0)));
        for (i = 1; i < m; i++)
        {
            sigma.replace(i, (sqrt(w + alpha * y(i - 1) * y(i - 1) + beta * sigma(i - 1) * sigma(i - 1))));
            y.replace(i, (kesai(i) * sigma(i)));
        }
    }
    else if (n == 3)  //GARCH2 : w=0.001,alpha=0.09,beta=0.89
    {
        double w, alpha, beta;
        w = 0.001, alpha = 0.09, beta = 0.89;
        mat sigma = linspace(0, m - 1, m);
        sigma.replace(0, w);
        y.replace(0, (kesai(0) * sigma(0)));
        for (i = 1; i < m; i++)
        {
            sigma.replace(i, (sqrt(w + alpha * y(i - 1) * y(i - 1) + beta * sigma(i - 1) * sigma(i - 1))));
            y.replace(i, (kesai(i) * sigma(i)));
        }
    }
    else if (n == 4)  //GARCH3 : w=0.001,alpha=0.09,beta=0.90
    {
        double w, alpha, beta;
        w = 0.001, alpha = 0.09, beta = 0.90;
        mat sigma = linspace(0, m - 1, m);
        sigma.replace(0, w);
        y.replace(0, (kesai(0) * sigma(0)));
        for (i = 1; i < m; i++)
        {
            sigma.replace(i, (sqrt(w + alpha * y(i - 1) * y(i - 1) + beta * sigma(i - 1) * sigma(i - 1))));
            y.replace(i, (kesai(i) * sigma(i)));
        }
    }
    else if (n == 5)  //NLMA 100; NLMA500 and NLMA100 may need to change m;
    {
        y.replace(0, 1); y.replace(1, 1);
        for (i = 2; i < m; i++)
        {
            y.replace(i, ((kesai(i - 1) * kesai(i - 2) * (kesai(i - 2) + kesai(i) + 1))));
        }
    }
    else if (n == 6)  //Chaotic 100; Chaotic500 and Chaotic1000 may need to change m;
    {
        vec t(1);
        t << Beta(0.5, 0.5);
        y.replace(0, 4 * t(0) * (1 - t(0)));
        for (i = 1; i < m; i++)
        {
            y.replace(i, (4 * y(i - 1) * (1 - y(i - 1))));
        }
    }
    else if (n == 7)  //Bilinear1 100;Bilinear1 500 and Bilinear1 1000 may need to change m;
    {
        y.replace(0, 1); y.replace(1, 1);
        for (i = 2; i < m; i++)
        {
            y.replace(i, kesai(i) + 0.15 * kesai(i - 1) * y(i - 1) + 0.05 * kesai(i - 1) * y(i - 2));
        }
    }
    else if (n == 8)  //Bilinear2 100;Bilinear2 500 and Bilinear2 1000 may need to change m;
    {
        y.replace(0, 1); y.replace(1, 1);
        for (i = 2; i < m; i++)
        {
            y.replace(i, kesai(i) + 0.25 * kesai(i - 1) * y(i - 1) + 0.15 * kesai(i - 1) * y(i - 2));
        }

    }
    else { cout << "change n plz"; }
    return y;
}

int
main()
{
    int i, j, k, n, N, B;
    cout << "please input sample size 'n'" << endl;
    cin >> n;
    cout << "please input Bootstrap replication's number 'B'" << endl;
    cin >> B;
    cout << "please input replication times 'N'" << endl;
    cin >> N;
    mat y_t = linspace(0, n - 1, n);
    mat sigma = linspace(0, n - 1, n);
    mat BootsC; mat BootsK;
    mat RepC10, RepC5, RepC1, RepK10, RepK5, RepK1;
    vec P90 = { 0.10 }; vec P95 = { 0.05 }; vec P99 = { 0.01 };//0.90  0.95  0.99
    mat bootsc = linspace(0, 7, 8), bootsk = linspace(0, 7, 8);
    for (i = 1; i <= N; i++)
    {
        for (j = 1; j <= B; j++)
        {
            for (k = 1; k <= 8; k++)
            {
                y_t = GenData(k, n);
                bootsc.replace(k - 1, C_pn(y_t, 1));
                bootsk.replace(k - 1, K_pn(y_t, 1));
            }
            BootsC.insert_rows(j - 1, bootsc.t());
            BootsK.insert_rows(j - 1, bootsk.t());
            bootsc = linspace(0, 7, 8), bootsk = linspace(0, 7, 8);
        }
        RepC10.insert_rows(i - 1, quantile(BootsC, P90));
        RepC5.insert_rows(i - 1, quantile(BootsC, P95));
        RepC1.insert_rows(i - 1, quantile(BootsC, P99));
        RepK10.insert_rows(i - 1, quantile(BootsK, P90));
        RepK5.insert_rows(i - 1, quantile(BootsK, P95));
        RepK1.insert_rows(i - 1, quantile(BootsK, P99));
    }
    mat Repc10 = mean(RepC10);
    mat Repc5 = mean(RepC5);
    mat Repc1 = mean(RepC1);
    mat Repk10 = mean(RepK10);
    mat Repk5 = mean(RepK5);
    mat Repk1 = mean(RepK1);
    mat RepC, RepK;
    RepC.insert_rows(0, Repc10);
    RepC.insert_rows(1, Repc5);
    RepC.insert_rows(2, Repc1);
    RepK.insert_rows(0, Repk10);
    RepK.insert_rows(1, Repk5);
    RepK.insert_rows(2, Repk1);
    cout << "C_pn for p=1; alpha = 10%,5%,1%:" << endl << RepC << endl;
    cout << "K_pn for p=1; alpha = 10%,5%,1%:" << endl << RepK << endl;
    return 0;
}
