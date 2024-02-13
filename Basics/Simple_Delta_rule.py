import random

x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
x3 = [1, 1, 1, 1]
x = [[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]]

w = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

print("Initial randomly generated weights:", w)

learning_rate = 0.5

expected_output = [0, 1, 1, 1]

output = [0, 0, 0, 0]

delta = [1, 1, 1, 1]

while any(delta) != 0:
    for i in range(4):
        weighted_sum = sum(x[j][i] * w[j] for j in range(3))
        output[i] = 1 if weighted_sum > 0 else 0
        delta[i] = expected_output[i] - output[i]

        for j in range(3):
            w[j] += x[j][i] * learning_rate * delta[i]

    print("Delta =", delta, "Output =", output, "Expected =", expected_output)

print("Final weights after learning:", w)
