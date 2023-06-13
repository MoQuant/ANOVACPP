#include <iostream>
#include <vector>
#include "func.cpp"

int main() {
    
    std::vector<std::vector<double>> X = {
        {21, 3, 5, 100},
        {2, 7, 4, 5},
        {2, 6, 6, 2},
        {21, 3, 1, 6},
        {2, 6, 8, 12},
        {24, 7, 5, 7},
        {2, 4, 6, 9},
        {2, 6, 3, 309}
    };

    std::vector<std::vector<double>> y = {
        {10},
        {5},
        {20},
        {6},
        {120},
        {7},
        {90},
        {3}
    };
    
    ANOVA(X, y);

    return 0;
}
