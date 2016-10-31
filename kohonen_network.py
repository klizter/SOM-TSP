import numpy as np


class KohonenNework:

    weights = []

    def __init__(self, number_of_weights, learning_rate, epochs):
        self.weights = np.random.rand(number_of_weights, 2)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.__class__.weights = self.weights
        self.radius = 1.4142135623730951 / 16.0
        self.radius_delta = self.radius / epochs

    def start_training(self, input_cases):

        for _ in xrange(self.epochs):

            for input_case in input_cases:
                output_signals = self.integrate_and_fire(input_case)
                winner_index = output_signals.tolist().index(min(output_signals))
                self.update_weights(winner_index, input_case)

            self.radius -= self.radius_delta

    def integrate_and_fire(self, input_case):
        output_signals = np.apply_along_axis(self.inverse_euclidean_distance, 1, self.weights, input_case)
        return output_signals

    def inverse_euclidean_distance(self, weight, input_case):
        return abs(np.linalg.norm(input_case - weight))

    def update_weights(self, winner_index, input_case):
        for current_index in xrange(len(self.weights)):

            if winner_index == current_index:
                continue

            self.apply_weight_delta(winner_index, current_index, input_case)

        self.apply_weight_delta(winner_index, winner_index, input_case)

    def apply_weight_delta(self, winner_index, current_index, input_case):
        neighborhood_factor = self.calculate_neighborhood_factor(winner_index, current_index)
        updated_weight = self.calculate_weight_delta(self.weights[current_index], input_case, neighborhood_factor)
        self.weights[current_index] = updated_weight.tolist()

    def calculate_weight_delta(self, weight, input_case, neighborhood_factor):
        return np.array((weight)) + (self.learning_rate * neighborhood_factor * (np.array(input_case) - np.array(weight)))

    # Smoothing kernel a.k.a neighborhood function
    def calculate_neighborhood_factor(self, winner_index, current_index):
        if winner_index == current_index:
            return 1.0

        distance = abs(np.linalg.norm(np.asarray(self.weights[current_index])-np.asarray(self.weights[winner_index])))

        if distance < self.radius:
            return distance
        else:
            return 0.0

