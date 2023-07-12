#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
using namespace std;

// get raw 2d vector
vector<vector<double>> raw2D(string filepath) {
    
    // read the file, place raw strings into a 1d vector iteratively
    vector<string> raw;
    ifstream file(filepath);
    string line;
    while (getline(file, line)) {
        raw.push_back(line);
    }

    // split raw strings by comma, convert to double, place into 2d float array
    vector<vector<double>> res;
    for (string line : raw) {
        vector<double> row;
        stringstream ss(line);
        string item;
        while (getline(ss, item, ',')) {
            row.push_back(stod(item)); // cast item (string) to double
        }
        res.push_back(row);
    }

    return res;
}

// shuffle a raw 2d vector
void shuffle2D(vector<vector<double>>& vector2D) {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(vector2D.begin(), vector2D.end(), default_random_engine(seed));
}

// print a 2d vector -- debug
void print2D(vector<vector<double>>& matrix) {
        for (vector<double> row : matrix) {
        for (double n : row) {
            cout << n << " ";
        }
        cout << endl;
    }
}

// print a 1d vector -- debug
void print1D(vector<double>& row) {
    for (double n : row) {
        cout << n << endl;
    }
}

// extract inputs from raw matrix
vector<vector<double>> isolateX(vector<vector<double>>& matrix) {
    vector<vector<double>> res;
    for (vector<double> row : matrix) {
        vector<double> transformedRow;
        for (int i = 0; i < row.size() - 1; i++) {
            transformedRow.push_back(row[i]);
        }
        res.push_back(transformedRow);
    }
    return res;
}

// extract outputs from raw matrix
vector<double> isolateY(vector<vector<double>>& matrix) {
    vector<double> res;
    for (vector<double> row : matrix) {
        res.push_back(row[row.size() - 1]);
    }
    return res;
}

// calculate column-wise mean
vector<double> getColumnMeans(const vector<vector<double>>& vec2D) {
    vector<double> mean(vec2D[0].size(), 0.0);
    int numRows = vec2D.size();
    int numCols = vec2D[0].size();

    for (int j = 0; j < numCols; ++j) {
        double sum = 0.0;
        for (int i = 0; i < numRows; ++i) {
            sum += vec2D[i][j];
        }
        mean[j] = sum / numRows;
    }

    return mean;
}

// subtract column-wise mean from the vector -- data normalization for training
void normalize2D(vector<vector<double>>& vec2D, const vector<double>& mean) {
    int numRows = vec2D.size();
    int numCols = vec2D[0].size();
    for (int j = 0; j < numCols; ++j) {
        for (int i = 0; i < numRows; ++i) {
            vec2D[i][j] -= mean[j];
        }
    }
}

// subtract column-wise mean from the vector -- data normalization for individual predictions
void normalize1D(vector<double>& vec1D, const vector<double>& mean) {
    for (int i = 0; i < vec1D.size(); i++) {
        vec1D[i] -= mean[i];
    }
}