# barebones-neural-network
Barebones, performance-oriented neural network & backpropagation algorithm in &lt; 250 lines of pure C++.

<br>

### Example Usage:
```C++
// required imports
#include <vector>
#include "NeuralNetwork.cpp"
```
```C++
// input layer with 2 neurons, hidden layer with 2 neurons
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
```
```
Output:
0.845303
-0.0957723
```
**NOTE:** Normalizing each input feature as I have done above almost always yields faster loss convergence. This is due to the [sigmoid activation function's](https://en.wikipedia.org/wiki/Sigmoid_function) sensitivity to extreme neuron activations.

<br>

### Compilation:
```
Compiler:  Command:
clang++    clang++ -std=c++17 Main.cpp
g++        g++ -std=c++17 Main.cpp
```
