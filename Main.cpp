// clang++ -std=c++17 Main.cpp

#include <iostream>
#include <vector>
#include <string>
#include "NeuralNetwork.cpp"
#include "Dataframe.cpp"
using namespace std;

int main() {

    // parse csv, shuffle training data, get x and y dataframes
    vector<vector<double>> matrix = raw2D("training-data/iris-cleaned.csv");
    shuffle2D(matrix);
    vector<vector<double>> x = isolateX(matrix);
    vector<double> y = isolateY(matrix);

    // normalize input data
    vector<double> columnMeans = getColumnMeans(x);
    normalize2D(x, columnMeans);

    // construct fit network (10000 epochs)
    vector<int> layers = {4, 3};
    NeuralNetwork nn = NeuralNetwork(layers);
    nn.fit(x, y, 10000);

    // make predictions
    cout << "-------" << endl;
    vector<double> setosa = {5.1, 3.5, 1.4, 0.2};
    normalize1D(setosa, columnMeans);
    cout << nn.predict(setosa) << endl;
}