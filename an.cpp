#include <iostream>
#include <vector>
#include "func.cpp"

int main() {
    
    std::vector<std::vector<double>> test = {
        {1, 3, 5, 1},
        {2, 7, 4, 5},
        {6, 6, 6, 2},
        {2, 3, 1, 6},
        {1, 6, 8, 12.1},
        {6, 7, 5, 7},
        {2, 4, 6, 9},
        {5, 6, 3, 3}
    };

    std::vector<std::vector<double>> res = {
        {1},
        {5},
        {2},
        {6},
        {12},
        {7},
        {9},
        {3}
    };

    ANOVA(test, res);

    

    return 0;
}
