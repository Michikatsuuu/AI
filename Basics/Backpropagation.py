import random
import math

def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_neuron(n):
    return {
        'weights': [random.uniform(-(1/math.sqrt(n)), 1/math.sqrt(n)) for _ in range(3)],
        'output': None,
        'deltaW_previous': [0, 0, 0]
    }

def calculate_neuron_output(neuron, inputs):
    u = sum(x * w for x, w in zip(inputs, neuron['weights']))
    neuron['output'] = sigmoid_function(u)

def calculate_weight_delta(neuron, learning_rate, error, values):
    neuron['deltaW'] = [learning_rate * error * value for value in values]

def assign_new_weights(neuron, momentum):
    neuron['weights'] = [w + delta + momentum * previous for w, delta, previous in zip(neuron['weights'], neuron['deltaW'], neuron['deltaW_previous'])]
    neuron['deltaW_previous'] = neuron['deltaW'].copy()

def train_neural_network(layer1, layer2, input_data, expected_output, learning_rate, momentum, epochs):
    for epoch in range(epochs):
        for idx, data_set in enumerate(input_data):
            # Forward pass
            for i, neuron in enumerate(layer1):
                calculate_neuron_output(neuron, data_set)

            calculate_neuron_output(layer2, [neuron['output'] for neuron in layer1] + [1])

            # Print results for the first epoch
            if epoch < 1:
                print(f"=== Iteration {epoch + 1}, Input {idx + 1} ===")
                print(f"Inputs: {data_set}")
                print(f"Expected: {expected_output[idx]}, Obtained: {layer2['output']:.5f}")
                print("\n")

            # Backpropagation - Calculate error and delta for Neuron 3
            error3 = (expected_output[idx] - layer2['output']) * layer2['output'] * (1 - layer2['output'])
            calculate_weight_delta(layer2, learning_rate, error3, [neuron['output'] for neuron in layer1] + [1])

            # Backpropagation - Calculate error and delta for Neuron 2
            error2 = error3 * layer2['weights'][1] * layer1[1]['output'] * (1 - layer1[1]['output'])
            calculate_weight_delta(layer1[1], learning_rate, error2, data_set)

            # Backpropagation - Calculate error and delta for Neuron 1
            error1 = error3 * layer2['weights'][0] * layer1[0]['output'] * (1 - layer1[0]['output'])
            calculate_weight_delta(layer1[0], learning_rate, error1, data_set)

            # Update weights
            for neuron in layer1:
                assign_new_weights(neuron, momentum)
            assign_new_weights(layer2, momentum)

            # Print results for the last two epochs
            if epoch > epochs - 2:
                print(f"=== Iteration {epoch + 1}, Input {idx + 1} ===")
                print(f"Inputs: {data_set}")
                print(f"Expected: {expected_output[idx]}, Obtained: {layer2['output']:.5f}")
                print("\n")

def main():
    layer1 = [initialize_neuron(3) for _ in range(2)]
    layer2 = initialize_neuron(3)
    input_data = [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    expected_output = [0, 1, 1, 0]
    momentum = 0.55
    learning_rate = 0.1
    epochs = 35000

    train_neural_network(layer1, layer2, input_data, expected_output, learning_rate, momentum, epochs)

if __name__ == "__main__":
    main()
