#define USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <string>

void PRINTM(std::vector<std::vector<double>> x)
{
    for(auto & u : x){
        for(auto & v : u){
            std::cout << v << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


std::vector<std::vector<double>> TRANSPOSE(const std::vector<std::vector<double>>& matrix) {
    // Get the number of rows and columns in the original matrix
    size_t numRows = matrix.size();
    size_t numCols = matrix[0].size();

    // Create a new matrix with transposed dimensions
    std::vector<std::vector<double>> transposedMatrix(numCols, std::vector<double>(numRows));

    // Transpose the elements of the original matrix to the new matrix
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            transposedMatrix[j][i] = matrix[i][j];
        }
    }

    return transposedMatrix;
}

std::vector<std::vector<double>> MMULT(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    // Get the dimensions of the input matrices
    size_t numRows1 = matrix1.size();
    size_t numCols1 = matrix1[0].size();
    size_t numRows2 = matrix2.size();
    size_t numCols2 = matrix2[0].size();

    // Check if the matrices can be multiplied
    if (numCols1 != numRows2) {
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }

    // Create a new matrix to store the result
    std::vector<std::vector<double>> result(numRows1, std::vector<double>(numCols2));

    // Perform matrix multiplication
    for (size_t i = 0; i < numRows1; i++) {
        for (size_t j = 0; j < numCols2; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < numCols1; k++) {
                sum += matrix1[i][k] * matrix2[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

std::vector<std::vector<double>> INVERSE(std::vector<std::vector<double>> Z)
{
    int m = Z.size();
    std::vector<std::vector<double>> I;
    std::vector<double> temp;

    for(int i = 0; i < m; ++i){
        temp.clear();
        for(int j = 0; j < m; ++j){
            if(i == j){
                temp.push_back(1.0);
            } else {
                temp.push_back(0.0);
            }
        }
        I.push_back(temp);
    }

    double A, B;

    for(int i = 1; i < m; ++i){
        for(int j = 0; j < i; ++j){
            A = Z[i][j];
            B = Z[j][j];
            for(int k = 0; k < m; ++k){
                Z[i][k] -= (A/B)*Z[j][k];
                I[i][k] -= (A/B)*I[j][k];
            }
        }
    }

    for(int i = 1; i < m; ++i){
        for(int j = 0; j < i; ++j){
            A = Z[j][i];
            B = Z[i][i];
            for(int k = 0; k < m; ++k){
                Z[j][k] -= (A/B)*Z[i][k];
                I[j][k] -= (A/B)*I[i][k];
            }
        }
    }

    for(int i = 0; i < m; ++i){
        for(int j = 0; j < m; ++j){
            I[i][j] = I[i][j] / Z[i][i];
        }
    }

    return I;
}

std::vector<std::vector<double>> Regression(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y)
{
    return MMULT(INVERSE(MMULT(TRANSPOSE(x), x)),MMULT(TRANSPOSE(x), y));
}

void ANOVA(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y)
{
    auto mean = [](std::vector<std::vector<double>> yi){
        double total = 0;
        int n = yi.size();
        for(auto & y0 : yi){
            total += y0[0];
        }
        total /= (double) n;
        return total;
    };

    std::vector<std::vector<double>> X, beta, y_pred, XTX;
    std::vector<double> temp;
    for(int i = 0; i < x.size(); ++i){
        temp.clear();
        temp.push_back(1.0);
        for(auto & t : x[i]){
            temp.push_back(t);
        }
        X.push_back(temp);
    }
    beta = Regression(X, y);
    y_pred = MMULT(X, beta);

    double y_mean = mean(y);
    double sst = 0, ssr = 0;
    for(int i = 0; i < y.size(); ++i){
        sst += pow(y[i][0] - y_mean, 2);
        ssr += pow(y[i][0] - y_pred[i][0], 2);
    }
    int n = X.size(), p = X[0].size() - 1;
    int dof_reg = p, dof_res = n - p - 1;

    double ms_reg = ssr/dof_reg, ms_res = ssr / dof_res;
    double F_Stat = ms_reg / ms_res;
    double R_Squared = 1 - ssr / sst;
    double Adj_R_Squared = 1 - (1 - R_Squared)*(n - 1)/(n - p - 1);
    
    XTX = INVERSE(MMULT(TRANSPOSE(x), x));
    std::vector<double> se, tstat;
    for(int i = 0; i < XTX.size(); ++i){
        se.push_back(pow(ms_res*XTX[i][i], 0.5));
    }
    for(int i = 0; i < se.size(); ++i){
        tstat.push_back(beta[i][0]/se[i]);
    }
    std::cout << "ANOVA TABLE" << std::endl;
    std::cout << std::endl;
    std::cout << "R-Squared: " << R_Squared << std::endl;
    std::cout << "Adj R-Squared: " << Adj_R_Squared << std::endl;
    std::cout << "F-Statistic: " << F_Stat << std::endl;
    std::cout << std::endl;

    std::cout << "Beta\tT-Stat" << std::endl;
    for(int i = 0; i < tstat.size(); ++i){
        std::cout << round(beta[i][0]*1000)/1000 << "\t" << round(tstat[i]*1000)/1000 << std::endl;
    }

}