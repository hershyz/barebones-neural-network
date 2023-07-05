// clang++ -std=c++17 Main.cpp

#include <iostream>
#include <vector>
#include <string>
#include "NeuralNetwork.cpp"
using namespace std;

int main() {

    // input layer size 2, hidden layer size 2
    vector<int> layers = {2, 2};
    NeuralNetwork nn = NeuralNetwork(layers);

    // weight (lb), height (in)
    // (shifted weight by -135 and height by -65 for data normalization)
    vector<vector<double>> x = { 
        {-2, -1},
        {25, 6},
        {17, 4},
        {-15, -6}
    };

    // 1 = female, 0 = male
    vector<double> y = {1, 0, 0, 1};

    // fit network (1000 epochs)
    nn.fit(x, y, 1000);

    // make predictions
    cout << "-------" << endl;
    cout << nn.predict({-7, -3}) << endl; // 123 pounds, 62 inches (expected value = 1)
    cout << nn.predict({20, 20}) << endl; // 155 pounds, 68 inches (expected value = 0)
}