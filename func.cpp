#define USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <string>

double pi()
{
    return M_PI;
}

double N(double x)
{
    auto simps = [](int n){
        std::vector<double> c;
        for(int i = 0; i < n; ++i){
            if(i == 0 || i == n - 1){
                c.push_back(1);
            } else if (i % 2 == 0){
                c.push_back(2);
            } else {
                c.push_back(4);
            }
        }
        return c;
    };
    auto fx = [](double z)
    {
        double top = 1.0/pow(2*pi(), 0.5);
        double eee = exp(-pow(z, 2)/2);
        return top*eee;
    };
    x = abs(x);
    int n = 101;
    double x0 = 0;
    double dx = (x - x0) / (n - 1);
    double total = 0;
    std::vector<double> S = simps(n);
    for(int i = 0; i < n; ++i){
        total += S[i]*fx(x0 + i*dx);
    }
    total *= dx/3.0;
    return 1 - (0.5 + total);
}

void PRINTM(std::vector<std::vector<double>> result)
{
    // Print the result
    for (const auto& row : result) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::vector<std::vector<double>> MMULT(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    std::size_t rows1 = matrix1.size();
    std::size_t cols1 = matrix1[0].size();
    std::size_t cols2 = matrix2[0].size();

    // Check if matrices can be multiplied
    if (cols1 != matrix2.size()) {
        throw std::runtime_error("Cannot multiply matrices: invalid dimensions");
    }

    // Initialize the result matrix with zeros
    std::vector<std::vector<double>> result(rows1, std::vector<double>(cols2, 0.0));

    // Perform matrix multiplication
    for (std::size_t i = 0; i < rows1; ++i) {
        for (std::size_t j = 0; j < cols2; ++j) {
            for (std::size_t k = 0; k < cols1; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

std::vector<std::vector<double>> TRANSPOSE(const std::vector<std::vector<double>>& matrix) {
    std::size_t rows = matrix.size();
    std::size_t cols = matrix[0].size();

    // Create a new matrix with swapped dimensions
    std::vector<std::vector<double>> transpose(cols, std::vector<double>(rows));

    // Transpose the matrix
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            transpose[j][i] = matrix[i][j];
        }
    }

    return transpose;
}

std::vector<std::vector<double>> INVERSE(std::vector<std::vector<double>> X)
{
    int m = X.size();
    int n = X[0].size();

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

    double A, B, C, D;

    for(int i = 1; i < m; ++i){
        for(int j = 0; j < i; ++j){
            A = X[i][j];
            B = X[j][j];
            for(int k = 0; k < m; ++k){
                X[i][k] = X[i][k] - (A/B)*X[j][k];
                I[i][k] = I[i][k] - (A/B)*I[j][k];
            }
        }
    }

    for(int i = 1; i < m; ++i){
        for(int j = 0; j < i; ++j){
            A = X[j][i];
            B = X[i][i];
            for(int k = 0; k < m; ++k){
                X[j][k] = X[j][k] - (A/B)*X[i][k];
                I[j][k] = I[j][k] - (A/B)*I[i][k];
            }
        }
    }

    for(int i = 0; i < m; ++i){
        for(int j = 0; j < m; ++j){
            I[i][j] = I[i][j] /= X[i][i];
        }
    }

    return I;
}

std::vector<std::vector<double>> Regression(std::vector<std::vector<double>> X, std::vector<std::vector<double>> Y)
{
    return MMULT(INVERSE(MMULT(TRANSPOSE(X), X)), MMULT(TRANSPOSE(X), Y));
}

std::vector<std::vector<double>> MESH(std::vector<std::vector<double>> x)
{
    std::vector<std::vector<double>> z;
    std::vector<double> temp;
    for(auto & t : x){
        temp.clear();
        temp.push_back(1.0);
        for(auto & y : t){
            temp.push_back(y);
        }
        z.push_back(temp);
    }
    return z;
}

void ANOVA(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y)
{
    auto mean = [](std::vector<std::vector<double>> q){
        double m = 0;
        double n = q.size();
        for(auto & y : q){
            m += y[0];
        }
        return m / n;
    };
    x = MESH(x);
    std::vector<std::vector<double>> beta = Regression(x, y);
    std::vector<std::vector<double>> yhat = MMULT(x, beta);
    std::vector<std::vector<double>> XTX;
    std::vector<double> residuals;
    for(int i = 0; i < yhat.size(); ++i){
        residuals.push_back(y[i][0] - yhat[i][0]);
    }
    int n = x.size(), p = x[0].size() - 1;
    double df_regression = p, df_residual = n - p - 1;
    double ss_regression = 0, ss_residual = 0, ss_total = 0;
    double mu_y = mean(y);
    for(int i = 0; i < n; ++i){
        ss_regression += pow(yhat[i][0] - mu_y, 2);
        ss_residual += pow(residuals[i], 2);
        ss_total += pow(y[i][0] - mu_y, 2);
    }
    double ms_regression = ss_regression / df_regression;
    double ms_residual = ss_residual / df_residual;
    double F_STAT = ms_regression / ms_residual;
    std::vector<double> test_statistics, p_val;
    for(int i = 0; i < p + 1; ++i){
        XTX = MMULT(TRANSPOSE(x), x);
        for(int j = 0; j < XTX.size(); ++j){
            for(int k = 0; k < XTX[0].size(); ++k){
                XTX[j][k] = pow(XTX[j][k]*ms_residual, 0.5);
            }
        }
        test_statistics.push_back(beta[i][0] / XTX[i][i]);
    }
    for(auto & k : test_statistics){
        p_val.push_back(N(k));
    }
    
    std::cout << std::endl;
    std::cout << "ANOVA TABLE" << std::endl;
    std::cout << std::endl;
    std::cout << "Source of Variation\tRegression\tResidual\tTotal" << std::endl;
    std::cout << "Degrees of Freedom: \t" << df_regression << "\t\t" << df_residual << "\t\t" << n - 1 << std::endl;
    std::cout << "Sum of Squares: \t" << ss_regression << "\t\t" << ss_residual << "\t\t" << ss_total << std::endl;
    std::cout << std::endl;
    std::cout << "Mean Squares: \t" << ms_regression << "\t\t" << ms_residual << std::endl;
    std::cout << "F-Statistic: \t" << F_STAT << std::endl;
    std::cout << std::endl;
    std::cout << "Coef\tBeta\tT-Stat\tPValue" << std::endl;
    for(int i = 0; i < beta.size(); ++i){
        std::cout << "Beta_" + std::to_string(i) << "\t" << round(beta[i][0]*100)/100 << "\t" << round(test_statistics[i]*100)/100 << "\t" << round(p_val[i]*1000)/1000 << std::endl;
    }

}