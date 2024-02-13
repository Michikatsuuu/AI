import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, generations, data_points):
        self.population_size = population_size
        self.generations = generations
        self.data_points = data_points
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        return np.random.randint(-15, 16, size=(self.population_size, 4))

    def objective_function(self, coefficients):
        a, b, c, d = coefficients
        x_values = np.array(self.data_points)[:, 0]
        y_values = np.array(self.data_points)[:, 1]
        function_values = a * x_values**3 + b * x_values**2 + c * x_values + d
        differences = function_values - y_values
        
        return np.sum(differences**2)

    def selection(self, temp_population, number_of_parents):
        fitness_results = 1 / (np.array([self.objective_function(individual) for individual in temp_population]) + 1)
        probabilities = fitness_results / np.sum(fitness_results)
        # Randomly choose parent indices based on their fitness probabilities
        selected_parent_indices = np.random.choice(len(temp_population), number_of_parents, p=probabilities)
        selected_parents = temp_population[selected_parent_indices]
        return selected_parents

    def crossover(self, parents, offspring_size):
        # Randomly select crossover points within each chromosome for each offspring
        crossover_points = np.random.randint(1, parents.shape[1] - 1, size=offspring_size)
        # Randomly choose pairs of parents for crossover
        selected_parent_indices = np.random.choice(parents.shape[0], size=(offspring_size, 2))
        offspring = np.zeros((offspring_size, parents.shape[1]))
        for i in range(offspring_size):
            crossover_point = crossover_points[i]
            parent1, parent2 = parents[selected_parent_indices[i]]
            # Crossover parents by combining their features before and after the crossover point
            offspring[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return offspring

    def mutation(self, offspring):
        mutation_indices = np.random.randint(offspring.shape[1], size=offspring.shape[0])
        # Generate random mutation values (-1, 0, or 1) for each offspring
        random_values = np.random.randint(-1, 2, size=offspring.shape)
        # Mutate offspring by adding random mutation values to corresponding indices
        offspring[np.arange(offspring.shape[0]), mutation_indices] += random_values[np.arange(offspring.shape[0]), mutation_indices]
        return offspring

    def run_algorithm(self):
        for generation in range(self.generations):
            # Select parents from the current population
            parents = self.selection(self.population, self.population_size // 2)
            crossover_offspring = self.crossover(parents, self.population_size - len(parents))
            mutated_offspring = self.mutation(crossover_offspring)
            self.population = np.concatenate((parents, mutated_offspring))

        # Find the best coefficients in the final population
        best_coefficients = sorted(self.population, key=lambda x: self.objective_function(x))[0]
        return best_coefficients, self.objective_function(best_coefficients)

# Input data
data_points = [(-5, -150), (-4, -77), (-3, -30), (-2, 0), (-1, 10), (1/2, 131/8), (1, 18), (2, 25), (3, 32), (4, 75), (5, 130)]

# Genetic algorithm parameters
population_size = 100
generations = 430

genetic_algorithm = GeneticAlgorithm(population_size, generations, data_points)
best_coefficients, objective_value = genetic_algorithm.run_algorithm()


print("Best coefficients a, b, c, d:", best_coefficients)
print("Objective function value for the coefficients:", objective_value)
