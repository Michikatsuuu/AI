def input_pattern(prompt):
    pattern = []
    while len(pattern) < 64:
        user_input = input(prompt)

        if all(symbol in ['0', '1'] for symbol in user_input):
            pattern.extend(map(int, user_input))
        else:
            print("You can only input weights as 0 or 1.")

    return pattern

def calculate_weights(train_pattern):
    size = 64
    weights = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            weights[i][j] = 0 if i == j else (2 * train_pattern[i] - 1) * (2 * train_pattern[j] - 1)

    return weights

def calculate_result(weights, test_pattern):
    size = 64
    s = [sum(weights[i][j] * test_pattern[j] for j in range(size)) for i in range(size)]

    y = [1 if s[i] > 0 else (test_pattern[i] if s[i] == 0 else 0) for i in range(size)]

    return y

def display_result(result):
    print("Result:")
    for i in range(64):
        if i % 8 == 0 and i > 0:    
            print()
        print(result[i], end="")

if __name__ == "__main__":
    print("Enter the training pattern (8x8):")
    train_pattern = input_pattern("")

    weights = calculate_weights(train_pattern)

    print("Enter the test pattern:")
    test_pattern = input_pattern("")

    result = calculate_result(weights, test_pattern)

    display_result(result)
