#include <iostream>
#include <math.h>
#include <vector>

void PRINTM(std::vector<std::vector<double>> x){
    for(auto & t : x){
        for(auto & u : t){
            std::cout << u << "\t";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<double>> MMULT(std::vector<std::vector<double>> x,
                                       std::vector<std::vector<double>> y)
{
    std::vector<std::vector<double>> result;
    std::vector<double> temp;
    double total = 0;

    for(int i = 0; i < x.size(); ++i){
        temp.clear();
        for(int j = 0; j < y[0].size(); ++j){
            total = 0;
            for(int k = 0; k < x[0].size(); ++k){
                total += x[i][k]*y[k][j];
            }
            temp.push_back(total);
        }
        result.push_back(temp);
    }
    return result;
}

std::vector<std::vector<double>> TRANSPOSE(std::vector<std::vector<double>> z)
{
    std::vector<std::vector<double>> X;
    std::vector<double> temp;
    for(int i = 0; i < z[0].size(); ++i){
        temp.clear();
        for(int j = 0; j < z.size(); ++j){
            temp.push_back(z[j][i]);
        }
        X.push_back(temp);
    }
    return X;
}

std::vector<std::vector<double>> INVERSE(std::vector<std::vector<double>> x)
{
    std::vector<std::vector<double>> I;
    std::vector<double> temp;
    int n = x.size();

    for(int i = 0; i < n; ++i){
        temp.clear();
        for(int j = 0; j < n; ++j){
            if(i == j){
                temp.push_back(1.0);
            } else {
                temp.push_back(0.0);
            }
        }
        I.push_back(temp);
    }

    double A, B;

    for(int i = 1; i < n; ++i){
        for(int j = 0; j < i; ++j){
            A = x[i][j];
            B = x[j][j];
            for(int k = 0; k < n; ++k){
                x[i][k] = x[i][k] - (A/B)*x[j][k];
                I[i][k] = I[i][k] - (A/B)*I[j][k];
            }
        }
    }

    for(int i = 1; i < n; ++i){
        for(int j = 0; j < i; ++j){
            A = x[j][i];
            B = x[i][i];
            for(int k = 0; k < n; ++k){
                x[j][k] = x[j][k] - (A/B)*x[i][k];
                I[j][k] = I[j][k] - (A/B)*I[i][k];
            }
        }
    }
    
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            I[i][j] = I[i][j] / x[i][i];
        }
    }

    return I;
}

void ANOVA(std::vector<std::vector<double>> BX, std::vector<std::vector<double>> Y)
{
    std::vector<std::vector<double>> X;
    std::vector<double> temp;
    for(int i = 0; i < BX.size(); ++i){
        temp.clear();
        temp.push_back(1.0);
        for(auto & q : BX[i]){
            temp.push_back(q);
        }
        X.push_back(temp);
    }

    
    std::vector<std::vector<double>> beta, XTX, IXTX, XTY, YHat;
    double ymu = 0, rss, tss, ess, F, rsq, adjrsq;

    int n = X.size(), m = X[0].size() - 1;

    XTX = MMULT(TRANSPOSE(X), X);
    IXTX = INVERSE(XTX);
    XTY = MMULT(TRANSPOSE(X), Y);
    beta = MMULT(IXTX, XTY);

    YHat = MMULT(X, beta);
    
    for(int i = 0; i < Y.size(); ++i){
        ymu += Y[i][0];
        rss += pow(Y[i][0] - YHat[i][0], 2);
    }
    ymu /= (double) Y.size();
    for(int i = 0; i < Y.size(); ++i){
        tss += pow(Y[i][0] - ymu, 2);
    }
    ess = tss - rss;

    F = (ess/(m))/(rss/(n - m - 1));
    rsq = 1 - rss/tss;
    adjrsq = 1 - (1 - rsq)*(n - 1)/(n - m - 1);

    double factor = rss / (n - m - 1);
    std::vector<double> se, tstat;
    for(int i = 0; i < IXTX.size(); ++i){
        se.push_back(pow(factor*IXTX[i][i], 0.5));
    }

    for(int i = 0; i < se.size(); ++i){
        tstat.push_back(beta[i][0]/se[i]);
    }
    
    std::cout << "ANOVA TABLE" << std::endl;
    std::cout << "RSquared: " << rsq << std::endl;
    std::cout << "Adj. RSquared: " << adjrsq << std::endl;
    std::cout << "F-Statistic: " << F << std::endl;
    std::cout << std::endl;
    for(int i = 0; i < tstat.size(); ++i){
        std::cout << "Beta: " << beta[i][0] << "\tStdError: " << se[i] << "\tT-Stat: " << tstat[i] << std::endl;
    }

}


int main()
{
    std::vector<std::vector<double>> X, y, M;

    X = {{1, 2, 3},
         {4, 2, 1},
         {5, 7, 2},
         {4, 3, 4},
         {9, 1, 0},
         {2, 3, 1},
         {1, 4, 2},
         {6, 9, 5},
         {8, 2, 1},
         {4, 6, 9}};

    y = {{2},{3},{5},{8},{13},{21},{34},{55},{89},{144}};

    ANOVA(X, y);

    return 0;
}
