#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>
#include <cmath>
using namespace std;

// custom hash function for vector<vector<int>>
struct Vector2DHash {
    std::size_t operator()(const std::vector<std::vector<int>>& v) const {
        std::size_t seed = v.size();
        for (const auto& row : v) {
            seed ^= row.size() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            for (const auto& element : row) {
                seed ^= element + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
        }
        return seed;
    }
};

// custom hash function for vector<int>
struct VectorHash {
    std::size_t operator()(const std::vector<int>& vec) const {
        std::size_t hash = 0;
        for (const auto& element : vec) {
            hash_combine(hash, element);
        }
        return hash;
    }

    // helper function to combine hash values
    template <typename T>
    void hash_combine(std::size_t& seed, const T& value) const {
        seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

class NeuralNetwork {
public:
    
    // methods
    NeuralNetwork(const vector<int>& layers);                                 // constructor declaration: reference to layers vector where each integer is the number of neurons in the layer
    void displayActivations();                                                // debug function
    void displayWeights();                                                    // debug function
    void displayBiases();                                                     // debug function
    double sigmoid(double& x);                                                // nonlinearity/activation function
    double predict(vector<double> inputs);                                   // prediction function
    void fit(vector<vector<double>>& x, vector<double>& y, int epochs);       // training function
    double MSE(double& predicted, double& real);                              // loss function (mean squared error)

    // objects
    vector<vector<double>> activations;                                 // 2d array of activations, each row represents a layer
    unordered_map<vector<vector<int>>, double, Vector2DHash> weights;   // [[r1, c1], [r2, c2]] : weight (between two neurons)
    unordered_map<vector<int>, double, VectorHash> biases;              // [r1, c1] : bias (per neuron)
    double weightLearnConst = 0.001;                                    // learning rate for updating weights during backpropagation
    double biasLearnConst = 0.001;                                      // learning rate for updating biases during backpropagation
};

// initialization sequence
NeuralNetwork::NeuralNetwork(const vector<int>& layers) {

    // initialize blank activations
    for (int n : layers) {
        vector<double> layer;
        for (int i = 0; i < n; i++) {
            layer.push_back(0);
        }
        activations.push_back(layer);
    }
    activations.push_back({0});

    // initialize random weights between every possible neuron connection (densely connect neurons)
    random_device rd;
    default_random_engine engine(rd());
    uniform_real_distribution<double> dist(-1, 1);
    for (int layerLevel = 0; layerLevel < activations.size() - 1; layerLevel++) {
        for (int i = 0; i < activations[layerLevel].size(); i++) {
            for (int j = 0; j < activations[layerLevel + 1].size(); j++) {
                vector<vector<int>> key = {{layerLevel, i}, {layerLevel + 1, j}};
                double weight = dist(engine);
                weights[key] = weight;
            }
        }
    }

    // intialize random biases for every neuron, leave biases for the input layer = 0
    for (int i = 0; i < activations[0].size(); i++) {
        biases[{0, i}] = 0;
    }
    for (int layerLevel = 1; layerLevel < activations.size(); layerLevel++) {
        for (int i = 0; i < activations[layerLevel].size(); i++) {
            biases[{layerLevel, i}] = dist(engine);
        }
    }
}

// display activations (debug)
void NeuralNetwork::displayActivations() {
    int layerLevel = 0;
    for (vector<double> layer : activations) {
        cout << "layer " << layerLevel << ": ";
        for (double n : layer) {
            cout << n << " ";
        }
        cout << endl;
        layerLevel++;
    }
}

// display weights (debug)
void NeuralNetwork::displayWeights() {
    for (const auto& kvp : weights) {
        vector<vector<int>> key = kvp.first;
        double weight = kvp.second;
        cout << "[" << key[0][0] << ", " << key[0][1] << "], " << "[" << key[1][0] << ", " << key[1][1] << "] -> " << weight << endl;
    }
}

// display biases (debug)
void NeuralNetwork::displayBiases() {
    for (const auto& kvp : biases) {
        vector<int> neuronCoord = kvp.first;
        double bias = kvp.second;
        cout << "[" << neuronCoord[0] << ", " << neuronCoord[1] << "] -> " << bias << endl;
    }
}

// nonlinearity/activation function
double NeuralNetwork::sigmoid(double& x) {
    return 1 / (1 + exp(-x));
}

// prediction function (feed-forward)
double NeuralNetwork::predict(vector<double> inputs) {

    // set input layer
    for (int i = 0; i < activations[0].size(); i++) {
        activations[0][i] = inputs[i];
    }

    // feed-forward portion, each neuron depends on all the neurons in the layer before it
    for (int layerLevel = 1; layerLevel < activations.size(); layerLevel++) {
        for (int i = 0; i < activations[layerLevel].size(); i++) {
            
            // add previous activations, scale with weights
            double currActivation = 0;
            for (int j = 0; j < activations[layerLevel - 1].size(); j++) {
                double prevActivation = activations[layerLevel - 1][j];
                double weight = weights[{{layerLevel - 1, j}, {layerLevel, i}}];
                currActivation += (prevActivation * weight);
            }

            // add bias
            currActivation += biases[{layerLevel, i}];
            
            // apply activation function if we aren't on the output neuron
            if (layerLevel != activations.size() - 1) {
                currActivation = sigmoid(currActivation);
            }

            // place the activation in the 2d vector of neurons
            activations[layerLevel][i] = currActivation;
        }
    }

    // return output neuron
    return activations[activations.size() - 1][0];
}

// MSE loss function
double NeuralNetwork::MSE(double& predicted, double& real) {
    return pow((predicted - real), 2);
}

// training function
void NeuralNetwork::fit(vector<vector<double>>& x, vector<double>& y, int epochs) {

    int index = 0;
    for (int i = 0; i < epochs; i++) {

        // make a prediction and calculate loss
        double predicted = predict(x[index]);
        double loss = MSE(predicted, y[index]);

        cout << "predicted: " << predicted << ", real: " << y[index] << ", loss: " << loss << endl;

        // backpropagation
        for (int layerLevel = 1; layerLevel < activations.size(); layerLevel++) {
            for (int i = 0; i < activations[layerLevel].size(); i++) {

                // update bias of the neuron
                double prevBias = biases[{layerLevel, i}];
                double testBias = prevBias + biasLearnConst;
                biases[{layerLevel, i}] = testBias;
                double testPrediction = predict(x[index]);
                double lossDiff = MSE(testPrediction, y[index]) - loss;
                prevBias -= loss * lossDiff;
                biases[{layerLevel, i}] = prevBias;

                // update all weights connecting to the neuron
                for (int j = 0; j < activations[layerLevel - 1].size(); j++) {
                    double prevWeight = weights[{{layerLevel - 1, j}, {layerLevel, i}}];
                    double testWeight = prevWeight + weightLearnConst;
                    weights[{{layerLevel - 1, j}, {layerLevel, i}}] = testWeight;
                    testPrediction = predict(x[index]);
                    lossDiff = MSE(testPrediction, y[index]) - loss;
                    prevWeight -= loss * lossDiff;
                    weights[{{layerLevel - 1, j}, {layerLevel, i}}] = prevWeight;
                }
            }
        }

        // reset input index when we go out of bounds (in cases where # of epochs > # of inputs)
        index++;
        if (index >= x.size()) {
            index = 0;
        }
    }
}